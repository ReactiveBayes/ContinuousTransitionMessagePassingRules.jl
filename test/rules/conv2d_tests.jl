@testsnippet Conv2dReference begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils
    using BayesBase, ExponentialFamily, LinearAlgebra, StableRNGs

    spd(rng, T, d) = (L = randn(rng, T, d, d); L * L' / T(d) + Matrix{T}(I, d, d))
    without(ms, k) = ntuple(i -> i == k ? nothing : ms[i], length(ms))

    # The convolution as one ContinuousTransition: CTVMP with the matrix that repeats the
    # kernel at every position, rows in the order of the y ports, and W repeated per position.
    function reference(algorithm)
        n, m, P, ny, nx, taps = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx, algorithm.taps
        function matrix(a)
            B = zeros(eltype(a), ny, nx)
            for p in 1:P, j in 1:m, c in 1:n
                t = taps[j, p]
                t == 0 || (B[p + (c - 1) * P, t] = a[(j - 1) * n + c])
            end
            return B
        end
        bigW(W) = kron(W, Matrix{eltype(W)}(I, P, P))
        blocks(R) = [sum(R[p + (i - 1) * P, p + (j - 1) * P] for p in 1:P) for i in 1:n, j in 1:n]
        return (; matrix, algorithm = CTVMP(matrix), bigW, blocks)
    end

    # Independent scalar messages as one message with a diagonal precision.
    function diagonal(ms)
        parameters = collect(weightedmean_precision.(ms))
        return MvNormalWeightedMeanPrecision(first.(parameters), Matrix(Diagonal(last.(parameters))))
    end

    # The message towards member k: the posterior marginal without k's own incoming message.
    function cavities(μ, V, ms)
        return map(eachindex(ms)) do k
            ξ, λ = weightedmean_precision(ms[k])
            (μ[k] / V[k, k] - ξ, inv(V[k, k]) - λ)
        end
    end

    function check_marginal(actual, μ, V, ms; tol...)
        expected = cavities(μ, V, ms)
        @test isapprox(mean(actual), μ; tol...)
        @test isapprox(var(actual), diag(V); tol...)
        @test isapprox(actual.ξ, first.(expected); tol...)
        @test isapprox(actual.λ, last.(expected); tol...)
        @test isapprox(entropy(actual), entropy(MvNormalMeanCovariance(μ, Matrix(Symmetric(V)))); tol...)
        return expected
    end

    geometries = (
        ((4, 4, 1), 1, (3, 3), (;)),
        ((4, 4, 1), 2, (2, 2), (; stride = 2)),
        ((4, 4, 2), 2, (3, 3), (;)),
        ((4, 3, 1), 2, (3, 2), (; stride = (1, 2), pad = 1)),
        ((5, 7, 2), 3, (2, 3), (; stride = (2, 1), pad = (0, 1))),
    )
end

@testitem "rules:Conv2dVMP:CTVMP with the convolution matrix, joint q(y,x)" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(511)
    for T in (Float64, Float32), (insize, cout, kernel, options) in (T === Float64 ? geometries : geometries[2:2])
        @testset "$T $insize -> $cout, $kernel, $options" begin
            algorithm = Conv2dVMP(insize, cout, kernel; T, options...)
            n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
            ct = reference(algorithm)
            tol = T === Float32 ? (; atol = 2.0f-3, rtol = 2.0f-3) : (; atol = 1.0e-9, rtol = 1.0e-9)
            my = ntuple(_ -> NormalMeanVariance(randn(rng, T), T(0.5) + rand(rng, T)), ny)
            mx = ntuple(_ -> NormalWeightedMeanPrecision(randn(rng, T), T(0.5) + rand(rng, T)), nx)
            qa = MvNormalMeanCovariance(randn(rng, T, n * m), T(0.2) * spd(rng, T, n * m))
            W = spd(rng, T, n)
            q, qref = (a = qa, W = PointMass(W)), (a = qa, W = PointMass(ct.bigW(W)))

            joint = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = my, x = mx), q))
            dense = getresult(call_marginal_update_rule(ContinuousTransition, (:y, :x); algorithm = ct.algorithm, m = (y = diagonal(my), x = diagonal(mx)), q = qref))
            # CTVMP's precision is symmetric only to roundoff, which Float32 shows.
            ξ, Λ = weightedmean_precision(dense)
            V = inv(Symmetric((Λ + Λ') / 2))
            μ = V * ξ
            @test joint isa Conv2dMarginal{T}
            @test joint.members === :yx && joint.ny == ny && joint.complete && length(joint) == ny + nx
            expected = check_marginal(joint, μ, V, (my..., mx...); tol...)

            # Every port reads its message from the joint, or computes it under the default scheme.
            clusters = ((:y, :x) => joint,)
            for (group, offset, d) in ((:y, 0, ny), (:x, ny, nx)), k in 1:d
                message = NormalWeightedMeanPrecision(joint.ξ[offset + k], joint.λ[offset + k])
                test_message_update_rule(Conv2dCT, (group, k); algorithm, cases = [(clusters = clusters,) => message], check_type_promotion = false, tol...)
                messages = group === :y ? (y = without(my, k), x = mx) : (y = my, x = without(mx, k))
                default = getresult(call_message_update_rule(Conv2dCT, (group, k); algorithm, m = messages, q))
                @test isapprox(collect(weightedmean_precision(default)), collect(expected[offset + k]); pairs(tol)...)
            end
            @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, (:y, ny + 1); algorithm, clusters))
            @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, (:x, 0); algorithm, clusters))

            # Towards a, W and the energy: the same from the joint and from CTVMP's dense one.
            qyx = MvNormalMeanCovariance(μ, Matrix(Symmetric(V)))
            reference_clusters = ((:y, :x) => qyx,)
            towards_a = convert(MvNormalWeightedMeanPrecision, getresult(call_message_update_rule(ContinuousTransition, :a; algorithm = ct.algorithm, clusters = reference_clusters, q = qref)))
            residual = getresult(call_message_update_rule(ContinuousTransition, :W; algorithm = ct.algorithm, clusters = reference_clusters, q = (a = qa,))).invS
            towards_W = ExponentialFamily.WishartFast(T(n + 1 + P), ct.blocks(residual))
            energy = getresult(call_average_energy(ContinuousTransition; algorithm = ct.algorithm, clusters = reference_clusters, q = qref))
            for cluster in (joint, qyx)
                clusters = ((:y, :x) => cluster,)
                test_message_update_rule(Conv2dCT, :a; algorithm, cases = [(clusters = clusters, q = (W = q.W,)) => towards_a], check_type_promotion = false, tol...)
                test_message_update_rule(Conv2dCT, :W; algorithm, cases = [(clusters = clusters, q = (a = qa,)) => towards_W], check_type_promotion = false, tol...)
                test_average_energy(Conv2dCT; algorithm, cases = [(clusters = clusters, q = q) => energy], check_type_promotion = false, tol...)
            end
        end
    end
end

@testitem "rules:Conv2dVMP:CTVMP with the convolution matrix, q(y)q(x)" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(512)
    tol = (; atol = 1.0e-9, rtol = 1.0e-9)
    for (insize, cout, kernel, options) in geometries
        @testset "$insize -> $cout, $kernel, $options" begin
            algorithm = Conv2dVMP(insize, cout, kernel; options...)
            n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
            ct = reference(algorithm)
            my = ntuple(_ -> NormalMeanVariance(randn(rng), 0.5 + rand(rng)), ny)
            mx = ntuple(_ -> NormalWeightedMeanPrecision(randn(rng), 0.5 + rand(rng)), nx)
            qy0 = MvNormalMeanCovariance(randn(rng, ny), spd(rng, Float64, ny))
            qx0 = MvNormalMeanCovariance(randn(rng, nx), spd(rng, Float64, nx))
            qa = MvNormalMeanCovariance(randn(rng, n * m), 0.2 * spd(rng, Float64, n * m))
            W = spd(rng, Float64, n)
            q, qref = (a = qa, W = PointMass(W)), (a = qa, W = PointMass(ct.bigW(W)))
            # The posterior of a group: CTVMP's message times the independent incoming ones.
            posterior(message, ms) = mean_cov(prod(GenericProd(), convert(MvNormalWeightedMeanPrecision, message), diagonal(ms)))

            qx = getresult(call_marginal_update_rule(Conv2dCT, (:x,); algorithm, m = (x = mx,), q, clusters = ((:y,) => qy0,)))
            μx, Vx = posterior(getresult(call_message_update_rule(ContinuousTransition, :x; algorithm = ct.algorithm, q = merge(qref, (y = qy0,)))), mx)
            @test qx.members === :x && qx.ny == 0 && !qx.complete
            expectedx = check_marginal(qx, μx, Vx, mx; tol...)

            qy = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = my,), q, clusters = ((:x,) => qx0,)))
            μy, Vy = posterior(getresult(call_message_update_rule(ContinuousTransition, :y; algorithm = ct.algorithm, q = merge(qref, (x = qx0,)))), my)
            @test qy.members === :y && qy.ny == ny && !qy.complete
            expectedy = check_marginal(qy, μy, Vy, my; tol...)

            for k in 1:ny
                test_message_update_rule(Conv2dCT, (:y, k); algorithm, cases = [(clusters = ((:y,) => qy,),) => NormalWeightedMeanPrecision(qy.ξ[k], qy.λ[k])], check_type_promotion = false, tol...)
                default = getresult(call_message_update_rule(Conv2dCT, (:y, k); algorithm, m = (y = without(my, k),), q, clusters = ((:x,) => qx0,)))
                @test isapprox(collect(weightedmean_precision(default)), collect(expectedy[k]); pairs(tol)...)
            end
            for k in 1:nx
                test_message_update_rule(Conv2dCT, (:x, k); algorithm, cases = [(clusters = ((:x,) => qx,),) => NormalWeightedMeanPrecision(qx.ξ[k], qx.λ[k])], check_type_promotion = false, tol...)
                default = getresult(call_message_update_rule(Conv2dCT, (:x, k); algorithm, m = (x = without(mx, k),), q, clusters = ((:y,) => qy0,)))
                @test isapprox(collect(weightedmean_precision(default)), collect(expectedx[k]); pairs(tol)...)
            end
            @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, (:x, 1); algorithm, clusters = ((:x,) => qy,)))
            @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, (:y, 1); algorithm, clusters = ((:y,) => qx,)))

            # Towards a, W and the energy, with each group as a Conv2dMarginal or a dense Gaussian.
            densey, densex = MvNormalMeanCovariance(μy, Matrix(Symmetric(Vy))), MvNormalMeanCovariance(μx, Matrix(Symmetric(Vx)))
            states = (y = densey, x = densex)
            towards_a = convert(MvNormalWeightedMeanPrecision, getresult(call_message_update_rule(ContinuousTransition, :a; algorithm = ct.algorithm, q = merge(qref, states))))
            residual = getresult(call_message_update_rule(ContinuousTransition, :W; algorithm = ct.algorithm, q = merge((a = qa,), states))).invS
            towards_W = ExponentialFamily.WishartFast(Float64(n + 1 + P), ct.blocks(residual))
            energy = getresult(call_average_energy(ContinuousTransition; algorithm = ct.algorithm, q = merge(qref, states)))
            for y in (qy, densey), x in (qx, densex)
                clusters = ((:y,) => y, (:x,) => x)
                test_message_update_rule(Conv2dCT, :a; algorithm, cases = [(clusters = clusters, q = (W = q.W,)) => towards_a], check_type_promotion = false, tol...)
                test_message_update_rule(Conv2dCT, :W; algorithm, cases = [(clusters = clusters, q = (a = qa,)) => towards_W], check_type_promotion = false, tol...)
                test_average_energy(Conv2dCT; algorithm, cases = [(clusters = clusters, q = q) => energy], check_type_promotion = false, tol...)
            end
        end
    end
end

@testitem "rules:Conv2dVMP:observed and independent inputs" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(513)
    tol = (; atol = 1.0e-9, rtol = 1.0e-9)
    for (insize, cout, kernel, options) in geometries
        algorithm = Conv2dVMP(insize, cout, kernel; options...)
        n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
        ct = reference(algorithm)
        qa = MvNormalMeanCovariance(randn(rng, n * m), 0.2 * spd(rng, Float64, n * m))
        W = spd(rng, Float64, n)
        q, qref = (a = qa, W = PointMass(W)), (a = qa, W = PointMass(ct.bigW(W)))
        my = ntuple(_ -> NormalMeanVariance(randn(rng), 0.5 + rand(rng)), ny)
        latent = ntuple(_ -> NormalMeanVariance(randn(rng), 0.5 + rand(rng)), nx)
        pixels = randn(rng, nx)
        # The engine's form for data, or independent marginals, and the same state as CTVMP sees it.
        for (scalars, dense) in (
                (ntuple(i -> PointMass(pixels[i]), nx), MvNormalMeanCovariance(pixels, zeros(nx, nx))),
                (latent, MvNormalMeanCovariance(collect(mean.(latent)), Matrix(Diagonal(collect(var.(latent)))))),
            )
            qy = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = my,), q = merge(q, (x = scalars,))))
            message = convert(MvNormalWeightedMeanPrecision, getresult(call_message_update_rule(ContinuousTransition, :y; algorithm = ct.algorithm, q = merge(qref, (x = dense,)))))
            μ, V = mean_cov(prod(GenericProd(), message, diagonal(my)))
            expected = check_marginal(qy, μ, V, my; tol...)
            @test qy.complete
            for k in 1:ny
                default = getresult(call_message_update_rule(Conv2dCT, (:y, k); algorithm, m = (y = without(my, k),), q = merge(q, (x = scalars,))))
                @test isapprox(collect(weightedmean_precision(default)), collect(expected[k]); pairs(tol)...)
            end
            states = (y = MvNormalMeanCovariance(μ, Matrix(Symmetric(V))), x = dense)
            towards_a = convert(MvNormalWeightedMeanPrecision, getresult(call_message_update_rule(ContinuousTransition, :a; algorithm = ct.algorithm, q = merge(qref, states))))
            residual = getresult(call_message_update_rule(ContinuousTransition, :W; algorithm = ct.algorithm, q = merge((a = qa,), states))).invS
            towards_W = ExponentialFamily.WishartFast(Float64(n + 1 + P), ct.blocks(residual))
            energy = getresult(call_average_energy(ContinuousTransition; algorithm = ct.algorithm, q = merge(qref, states)))
            # q(y) holds the moments of the patches it was computed with: with or without the pixels.
            for extra in ((x = scalars,), (;))
                isempty(extra) && !(scalars isa Tuple{Vararg{PointMass}}) && continue
                clusters = ((:y,) => qy,)
                test_message_update_rule(Conv2dCT, :a; algorithm, cases = [(clusters = clusters, q = merge((W = q.W,), extra)) => towards_a], check_type_promotion = false, tol...)
                test_message_update_rule(Conv2dCT, :W; algorithm, cases = [(clusters = clusters, q = merge((a = qa,), extra)) => towards_W], check_type_promotion = false, tol...)
                test_average_energy(Conv2dCT; algorithm, cases = [(clusters = clusters, q = merge(q, extra)) => energy], check_type_promotion = false, tol...)
            end
            # A dense q(y) with independent scalar inputs.
            a2 = getresult(call_message_update_rule(Conv2dCT, :a; algorithm, clusters = ((:y,) => states.y,), q = (W = q.W, x = scalars)))
            @test all(isapprox.(params(a2), params(towards_a); tol...))
        end
        # Without q(x), a q(y) that was not computed from independent inputs cannot stand alone.
        partial = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = my,), q, clusters = ((:x,) => MvNormalMeanCovariance(pixels, Matrix(1.0I, nx, nx)),)))
        @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, :a; algorithm, clusters = ((:y,) => partial,), q = (W = q.W,)))
    end
end

@testitem "rules:Conv2dVMP:one LinearCT per position" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(514)
    for (insize, cout, kernel, options) in geometries[1:3]
        algorithm = Conv2dVMP(insize, cout, kernel; options...)
        n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
        qyx = MvNormalMeanCovariance(randn(rng, ny + nx), spd(rng, Float64, ny + nx))
        qa = MvNormalMeanCovariance(randn(rng, n * m), 0.2 * spd(rng, Float64, n * m))
        qW = Wishart(n + 3.0, spd(rng, Float64, n) / (n + 3))
        clusters = ((:y, :x) => qyx,)
        # Every position is a LinearCT on its own patch, sharing a and W: multiply their messages.
        patches = map(1:P) do p
            index = [p .+ (0:(n - 1)) .* P; ny .+ algorithm.taps[:, p]]
            ((:y, :x) => MvNormalMeanCovariance(mean(qyx)[index], cov(qyx)[index, index]),)
        end
        each(target, q) = [getresult(call_message_update_rule(LinearContinuousTransition, target; algorithm = LCTVMP(n, m), clusters = patch, q)) for patch in patches]
        towards_a = getresult(call_message_update_rule(Conv2dCT, :a; algorithm, clusters, q = (W = qW,)))
        ξ, Λ = weightedmean_precision(towards_a)
        @test ξ ≈ sum(first ∘ weightedmean_precision, each(:a, (W = qW,)))
        @test Λ ≈ sum(last ∘ weightedmean_precision, each(:a, (W = qW,)))
        towards_W = getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters, q = (a = qa,)))
        @test towards_W.ν == sum(w -> w.ν, each(:W, (a = qa,))) - (P - 1) * (n + 1)
        @test towards_W.invS ≈ sum(w -> w.invS, each(:W, (a = qa,)))
        energies = [getresult(call_average_energy(LinearContinuousTransition; algorithm = LCTVMP(n, m), clusters = patch, q = (a = qa, W = qW))) for patch in patches]
        @test getresult(call_average_energy(Conv2dCT; algorithm, clusters, q = (a = qa, W = qW))) ≈ sum(energies)
    end
end

@testitem "rules:Conv2dVMP:cache of the default scheme" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(515)
    insize, cout, kernel = (4, 4, 1), 2, (3, 3)
    algorithm = Conv2dVMP(insize, cout, kernel)
    ny, nx, d = algorithm.ny, algorithm.nx, algorithm.n * algorithm.m
    qa = MvNormalMeanCovariance(randn(rng, d), 0.2 * spd(rng, Float64, d))
    W = spd(rng, Float64, 2)
    q = (a = qa, W = PointMass(W))
    my = ntuple(i -> NormalWeightedMeanPrecision(i / ny, 1.0 + i / ny), ny)
    mx = ntuple(i -> NormalWeightedMeanPrecision(i / nx, 1.0 + i / nx), nx)
    qy = MvNormalMeanCovariance(randn(rng, ny), spd(rng, Float64, ny))
    function check(group, k; my = my, mx = mx, clusters = ())
        messages = isempty(clusters) ? (group === :y ? (y = without(my, k), x = mx) : (y = my, x = without(mx, k))) : NamedTuple{(group,)}((without(group === :y ? my : mx, k),))
        actual = getresult(call_message_update_rule(Conv2dCT, (group, k); algorithm, m = messages, q, clusters))
        fresh = getresult(call_message_update_rule(Conv2dCT, (group, k); algorithm = Conv2dVMP(insize, cout, kernel), m = messages, q, clusters))
        @test all(isapprox.(params(actual), params(fresh); rtol = 1.0e-9))
        return actual
    end
    sweep(; kwargs...) = (foreach(k -> check(:x, k; kwargs...), 1:nx); foreach(k -> check(:y, k; kwargs...), 1:ny))
    @test algorithm.cache.yx === nothing && algorithm.cache.x === nothing && algorithm.cache.band === nothing
    sweep()
    solved = algorithm.cache.yx
    @test algorithm.cache.x === nothing
    @test size(algorithm.cache.band.factor) == (algorithm.bandwidth + 1, nx)
    # One solve serves every port once each port's own message has been seen.
    solves = solved.solves
    sweep()
    @test solved.solves == solves
    # An incoming message that changes means a new solve, shared by the ports after it.
    sweep(; my = (NormalWeightedMeanPrecision(13.0, 1.0 + 1 / ny), my[2:end]...))
    @test solves < solved.solves <= solves + 2
    sweep()
    for mutate! in (() -> (mean(qa)[1] += 0.8), () -> (cov(qa)[2, 2] += 0.9), () -> (W[1, 1] += 0.7))
        previous, version = solved.solves, algorithm.cache.version
        mutate!()
        sweep()
        @test solved.solves == previous + 1
        @test algorithm.cache.version > version
    end
    # A W update refreshes q(a)'s snapshot first: the transition must still be rebuilt.
    cov(qa)[3, 3] += 0.5
    getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters = ((:y, :x) => MvNormalMeanCovariance(zeros(ny + nx), Matrix(1.0I, ny + nx, ny + nx)),), q = (a = qa,)))
    sweep()
    # The separate x system has its own solve, which follows the mean of q(y).
    foreach(k -> check(:x, k; clusters = ((:y,) => qy,)), 1:nx)
    separate = algorithm.cache.x
    solves = separate.solves
    foreach(k -> check(:x, k; clusters = ((:y,) => qy,)), 1:nx)
    @test separate.solves == solves
    mean(qy)[1] += 4.0
    foreach(k -> check(:x, k; clusters = ((:y,) => qy,)), 1:nx)
    @test separate.solves == solves + 1
    # Returned marginals do not alias the cache.
    joint = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = my, x = mx), q))
    saved = deepcopy(joint)
    mean(qa)[2] -= 0.3
    sweep()
    @test all(getfield(joint, f) == getfield(saved, f) for f in fieldnames(Conv2dMarginal))
end

@testitem "rules:Conv2dVMP:geometry and validation" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: ispure, dependencies_spec, extends_default_scheme, target_dependencies, IndexedTarget, Target, interfaces, interface_groups

    @test Conv2dCT === Conv2dContinuousTransition
    @test interfaces(Conv2dCT) == (:y, :x, :a, :W)
    @test interface_groups(Conv2dCT) == (:y, :x)
    algorithm = Conv2dVMP((5, 4, 2), 3, (3, 2); stride = (2, 1), pad = (1, 0))
    @test algorithm.outsize == (3, 3, 3)
    @test (algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx) == (3, 12, 9, 27, 40)
    @test !ispure(algorithm)
    @test occursin("Conv2dVMP((5, 4, 2), 3, (3, 2)", sprint(show, algorithm))
    @test occursin("ports = :default", sprint(show, algorithm))
    spec = dependencies_spec(Conv2dCT, algorithm)
    @test all(t -> extends_default_scheme(spec, t), (IndexedTarget(:y, 1), IndexedTarget(:x, 3), Target(:a), Target(:W)))
    # The taps are the padded image's windows, in column-major order.
    padded = zeros(Int, 7, 4, 2)
    padded[2:6, :, :] .= reshape(1:40, 5, 4, 2)
    for o2 in 1:3, o1 in 1:3
        @test algorithm.taps[:, o1 + (o2 - 1) * 3] == vec(padded[(2 * o1 - 1):(2 * o1 + 1), o2:(o2 + 1), :])
    end
    # The banded order is a permutation in which every patch fits the bandwidth.
    @test sort(algorithm.order) == 1:40
    @test all(p -> (t = filter(!iszero, algorithm.ptaps[:, p]); maximum(t) - minimum(t) <= algorithm.bandwidth), 1:9)
    @test Conv2dVMP((28, 28, 1), 1, (3, 3)).bandwidth == 2 * 28 + 2
    @test Conv2dVMP((28, 6, 1), 1, (3, 3)).bandwidth == 2 * 6 + 2
    @test Conv2dVMP((6, 28, 3), 1, (3, 3)).bandwidth == 3 * (2 * 6 + 2) + 2
    @test eltype(Conv2dVMP((3, 3, 1), 1, (2, 2); T = Float32).cache.W) === Float32
    for arguments in (((0, 3, 1), 1, (2, 2)), ((3, 3, 1), 0, (2, 2)), ((3, 3, 1), 1, (0, 2)), ((3, 3, 1), 1, (4, 2)))
        @test_throws ArgumentError Conv2dVMP(arguments...)
    end
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); stride = 0)
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); pad = -1)
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); stride = 1.5)
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); ports = :other)

    # What a port's message is computed from, by `ports`.
    key(target, ports) = map(d -> (d.container, d.key), target_dependencies(dependencies_spec(Conv2dCT, Conv2dVMP((3, 3, 1), 1, (2, 2); ports)), target))
    @test key(IndexedTarget(:y, 1), :joint) == ((:q, (:y, :x)),) == key(IndexedTarget(:x, 1), :joint)
    @test key(IndexedTarget(:y, 1), :separate) == ((:q, (:y,)),) && key(IndexedTarget(:x, 1), :separate) == ((:q, (:x,)),)
    @test key(IndexedTarget(:y, 1), :observed) == ((:q, (:y,)),) && key(IndexedTarget(:x, 1), :observed) == ()
    @test key(Target(:a), :observed) == ((:q, (:y,)), (:q, :W)) && key(Target(:W), :observed) == ((:q, (:y,)), (:q, :a))
    @test occursin("ports = :joint", sprint(show, Conv2dVMP((3, 3, 1), 1, (2, 2); ports = :joint)))

    algorithm = Conv2dVMP((2, 2, 1), 2, (1, 2))
    normal(d) = MvNormalMeanCovariance(ones(d), Matrix(1.0I, d, d))
    s = NormalMeanVariance(0.0, 1.0)
    q = (a = normal(4), W = Wishart(4.0, Matrix(1.0I, 2, 2)))
    y, x = (s, s, s, s), (s, s, s, s)
    valid = (y = (nothing, s, s, s), x = x)
    message(; target = (:y, 1), m = valid, q = q, clusters = ()) = getresult(call_message_update_rule(Conv2dCT, target; algorithm, m, q, clusters))
    @test message() isa NormalWeightedMeanPrecision
    for m in ((y = (nothing, s, s), x = x), (y = (nothing, s, s, s), x = (s, s, s)))
        @test_throws DimensionMismatch message(; m)
    end
    for m in ((x = x,), (y = y, x = x), (y = (nothing, s, s, PointMass(1.0)), x = x), (y = (nothing, s, s, s),))
        @test_throws ArgumentError message(; m)
    end
    for k in (0, -1, 5)
        @test_throws ArgumentError message(; target = (:y, k))
    end
    for malformed in ((a = normal(5), W = q.W), (a = q.a, W = Wishart(5.0, Matrix(1.0I, 3, 3))))
        @test_throws DimensionMismatch message(; q = malformed)
    end
    # A group is given once: jointly or as scalars, never both, and never with stray messages.
    @test_throws ArgumentError message(; m = (y = (nothing, s, s, s),), q = merge(q, (x = x,)), clusters = ((:x,) => normal(4),))
    @test_throws ArgumentError message(; m = valid, clusters = ((:x,) => normal(4),))
    @test_throws ArgumentError message(; target = (:x, 1), m = (y = y, x = (nothing, s, s, s)), clusters = ((:y,) => normal(4),))
    @test_throws DimensionMismatch message(; m = (y = (nothing, s, s, s),), clusters = ((:x,) => normal(3),))
    @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, :a; algorithm, m = (y = y,), q = (x = x, W = q.W)))
    @test_throws ArgumentError getresult(call_message_update_rule(Conv2dCT, :a; algorithm, q = (x = x, W = q.W)))
    @test_throws ArgumentError getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = y,), q))
    @test_throws ArgumentError getresult(call_marginal_update_rule(Conv2dCT, (:x,); algorithm, m = (x = x,), q))
    @test_throws ArgumentError getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = x), q, clusters = ((:x,) => normal(4),)))
    # A precision that is not positive definite is an error, not a silent result.
    @test_throws LinearAlgebra.PosDefException getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = ntuple(_ -> NormalWeightedMeanPrecision(0.0, -50.0), 4)), q))

    # A single output and a single input, in the engine's scalar forms: the 1 × 1 LinearCT.
    single = Conv2dVMP((1, 1, 1), 1, (1, 1))
    q1 = (a = MvNormalMeanCovariance([2.0], [0.3;;]), W = Wishart(3.0, [0.5;;]))
    block = MvNormalMeanCovariance([0.0], [1.0;;])
    for (group, other) in ((:y, :x), (:x, :y)), structured in (true, false)
        scalar = structured ? (m = NamedTuple{(other,)}(((s,),)), q = q1) : (q = merge(q1, NamedTuple{(other,)}(((s,),))),)
        dense = structured ? (m = NamedTuple{(other,)}((block,)), q = q1) : (q = merge(q1, NamedTuple{(other,)}((block,))),)
        actual = getresult(call_message_update_rule(Conv2dCT, (group, 1); algorithm = single, scalar...))
        expected = getresult(call_message_update_rule(LinearContinuousTransition, group; algorithm = LCTVMP(1, 1), dense...))
        @test mean(actual) ≈ only(mean(expected))
        @test var(actual) ≈ only(cov(expected))
    end
end

@testitem "rules:Conv2dVMP:banded solver" begin
    using ContinuousTransitionMessagePassingRules, LinearAlgebra, StableRNGs
    import ContinuousTransitionMessagePassingRules: banded_cholesky!, banded_solve!, banded_selected_inverse!, banded_logdet, band_entry

    rng = StableRNG(9)
    for T in (Float64, Float32), (N, b) in ((1, 0), (5, 0), (6, 2), (9, 8), (40, 7))
        A = zeros(T, N, N)
        for j in 1:N, i in j:min(N, j + b)
            A[i, j] = A[j, i] = randn(rng, T) / T(2)
        end
        A += (b + 2) * I
        band = zeros(T, b + 1, N)
        for j in 1:N, i in j:min(N, j + b)
            band[i - j + 1, j] = A[i, j]
        end
        factor = banded_cholesky!(copy(band))
        tol = T === Float32 ? 1.0f-4 : 1.0e-11
        L = cholesky(Symmetric(A)).L
        @test all(isapprox(factor[i - j + 1, j], L[i, j]; atol = tol) for j in 1:N for i in j:min(N, j + b))
        @test banded_logdet(factor) ≈ logdet(A) rtol = tol
        h = randn(rng, T, N)
        @test banded_solve!(copy(h), factor) ≈ A \ h rtol = 10 * tol
        selected = banded_selected_inverse!(similar(factor), factor)
        dense = inv(A)
        @test all(isapprox(band_entry(selected, i, j), dense[i, j]; atol = 10 * tol) for j in 1:N for i in max(1, j - b):min(N, j + b))
    end
    @test_throws PosDefException banded_cholesky!([1.0 1.0; 2.0 0.0])
    @test_throws DimensionMismatch banded_selected_inverse!(zeros(2, 3), zeros(3, 3))
end

@testitem "rules:Conv2dVMP:the marginal of a group" begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, LinearAlgebra

    algorithm = Conv2dVMP((4, 4, 1), 2, (3, 3))
    d = algorithm.n * algorithm.m
    result = getresult(
        @call_marginal_update_rule(
            node = Conv2dCT, target = (:y, :x), algorithm = algorithm,
            m = (y = ntuple(i -> NormalMeanVariance(i / 8, 0.5), algorithm.ny), x = ntuple(i -> NormalMeanVariance(i / 16, 2.0), algorithm.nx)),
            q = (a = MvNormalMeanCovariance(collect(1.0:d) ./ d, Matrix(0.1I, d, d)), W = Wishart(4.0, Matrix(1.0I, 2, 2) .+ 0.2)),
        )
    )
    @test result isa Conv2dMarginal{Float64}
    @test length(result) == algorithm.ny + algorithm.nx == length(mean(result)) == length(var(result))
    @test std(result) == sqrt.(var(result)) && all(>(0), var(result)) && all(>(0), result.λ)
    @test entropy(result) == result.entropy && isfinite(entropy(result))
    @test size(result.Exx) == (9, 9) && size(result.Eyx) == (2, 9) && size(result.Eyy) == (2, 2)
    @test isposdef(Symmetric(result.Exx))
    @test paramfloattype(result) === Float64 && eltype(typeof(result)) === Float64
    single = convert_paramfloattype(Float32, result)
    @test single isa Conv2dMarginal{Float32} && mean(single) ≈ mean(result) && single.members === :yx
    @test sprint(show, result) == "Conv2dMarginal{Float64}(:yx, 24 members)"
end

@testitem "rules:Conv2dVMP:large incoming precisions" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(515)
    algorithm = Conv2dVMP((3, 3, 1), 2, (2, 2); stride = (1, 2), pad = (0, 1))
    n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
    qa = MvNormalMeanCovariance(randn(rng, n * m), 0.2 * spd(rng, Float64, n * m))
    W = spd(rng, Float64, n)
    q = (a = qa, W = PointMass(W))
    my, mx = randn(rng, ny), randn(rng, nx)

    # Assemble E[-∇² log f] independently in high precision, before any incoming messages.
    A, Va, Wb = reshape(BigFloat.(mean(qa)), n, m), BigFloat.(cov(qa)), BigFloat.(W)
    J = zeros(BigFloat, ny + nx, ny + nx)
    for p in 1:P
        ys = p .+ (0:(n - 1)) .* P
        J[ys, ys] .+= Wb
        for j in 1:m
            tj = algorithm.taps[j, p]
            tj == 0 && continue
            for c in 1:n
                cross = sum(Wb[c, d] * A[d, j] for d in 1:n)
                J[ys[c], ny + tj] -= cross
                J[ny + tj, ys[c]] -= cross
            end
            for l in 1:m
                tl = algorithm.taps[l, p]
                tl == 0 && continue
                J[ny + tj, ny + tl] += sum(
                    Wb[c, d] * (A[c, j] * A[d, l] + Va[(j - 1) * n + c, (l - 1) * n + d]) for c in 1:n, d in 1:n
                )
            end
        end
    end

    function check_dense(actual, factor, h, messages)
        ξ = BigFloat[first(weightedmean_precision(v)) for v in messages]
        λ = BigFloat[last(weightedmean_precision(v)) for v in messages]
        Q = factor + Diagonal(λ)
        V = inv(Symmetric(Q))
        @test mean(actual) ≈ Float64.(V * (h + ξ)) rtol = 1.0e-10 atol = 1.0e-11
        @test var(actual) ≈ Float64.(diag(V)) rtol = 1.0e-10
        # Eliminate all other coordinates; never subtract the target's incoming parameters.
        for k in eachindex(ξ)
            others = setdiff(eachindex(ξ), (k,))
            cross = factor[k, others]
            F = cholesky(Symmetric(Q[others, others]))
            expected_λ = factor[k, k] - dot(cross, F \ cross)
            expected_ξ = h[k] - dot(cross, F \ (h[others] + ξ[others]))
            @test actual.λ[k] ≈ Float64(expected_λ) rtol = 1.0e-10 atol = 1.0e-11
            @test actual.ξ[k] ≈ Float64(expected_ξ) rtol = 1.0e-10 atol = 1.0e-11
        end
    end

    for scale in (1.0, 1.0e6, 1.0e12), strong in (:x, :y, :both)
        @testset "precision $scale on $strong" begin
            λy, λx = strong === :x ? 1.3 : scale, strong === :y ? 0.7 : scale
            y = ntuple(i -> NormalWeightedMeanPrecision(λy * my[i], λy), ny)
            x = ntuple(i -> NormalWeightedMeanPrecision(λx * mx[i], λx), nx)
            joint = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = x), q))
            check_dense(joint, J, zeros(BigFloat, ny + nx), (y..., x...))
            for (group, offset, messages) in ((:y, 0, y), (:x, ny, x)), k in (1, length(messages))
                incoming = group === :y ? (y = without(y, k), x = x) : (y = y, x = without(x, k))
                message = getresult(call_message_update_rule(Conv2dCT, (group, k); algorithm, m = incoming, q))
                @test collect(weightedmean_precision(message)) ≈ [joint.ξ[offset + k], joint.λ[offset + k]] rtol = 1.0e-10 atol = 1.0e-11
            end
            for (group, indices, others, means, messages) in (
                    (:x, (ny + 1):(ny + nx), 1:ny, my, x),
                    (:y, 1:ny, (ny + 1):(ny + nx), mx, y),
                )
                othergroup = group === :x ? :y : :x
                state = MvNormalMeanCovariance(means, Matrix(1.0I, length(means), length(means)))
                separate = getresult(call_marginal_update_rule(Conv2dCT, (group,); algorithm, m = NamedTuple{(group,)}((messages,)), q, clusters = ((othergroup,) => state,)))
                check_dense(separate, J[indices, indices], -J[indices, others] * BigFloat.(means), messages)
            end
        end
    end
end

@testitem "rules:Conv2dVMP:uncovered pixels and partial observations" tags = [:rules] setup = [Conv2dReference] begin
    algorithm = Conv2dVMP((4, 4, 1), 1, (1, 1); stride = 2)
    ny, nx = algorithm.ny, algorithm.nx
    q = (a = MvNormalMeanCovariance([1.3], [0.2;;]), W = PointMass([2.0;;]))
    y = ntuple(i -> NormalMeanVariance(i / 3, 0.7), ny)
    x = ntuple(i -> NormalMeanVariance(i / 7, 1.3), nx)
    joint = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = x), q))
    qy = MvNormalMeanCovariance(collect(mean.(y)), Matrix(1.0I, ny, ny))
    separate = getresult(call_marginal_update_rule(Conv2dCT, (:x,); algorithm, m = (x = x,), q, clusters = ((:y,) => qy,)))
    for k in setdiff(1:nx, vec(algorithm.taps))
        for (state, offset) in ((joint, ny), (separate, 0))
            @test state.ξ[offset + k] == 0.0
            @test state.λ[offset + k] == 0.0
            @test mean(state)[offset + k] ≈ mean(x[k])
            @test var(state)[offset + k] ≈ var(x[k])
        end
        message = getresult(call_message_update_rule(Conv2dCT, (:x, k); algorithm, m = (y = y, x = without(x, k)), q))
        @test weightedmean_precision(message) == (0.0, 0.0)
    end
    mixed = (PointMass(0.2), x[2:end]...)
    @test_throws ArgumentError getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = mixed), q))
    # Independent scalar marginals may still mix observed and uncertain values.
    observed = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = y,), q = merge(q, (x = mixed,))))
    densex = MvNormalMeanCovariance(collect(mean.(mixed)), Matrix(Diagonal([0.0; collect(var.(x[2:end]))])))
    dense = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = y,), q, clusters = ((:x,) => densex,)))
    @test mean(observed) ≈ mean(dense)
    @test var(observed) ≈ var(dense)
    # Explicitly supplied observed pixels override the moments stored in an older q(y).
    changed = ntuple(i -> PointMass(i / 3), nx)
    densey = MvNormalMeanCovariance(mean(observed), Matrix(Diagonal(var(observed))))
    clusters = ((:y,) => observed,)
    towards_a = getresult(call_message_update_rule(Conv2dCT, :a; algorithm, clusters, q = (W = q.W, x = changed)))
    towards_W = getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters, q = (a = q.a, x = changed)))
    energy = getresult(call_average_energy(Conv2dCT; algorithm, clusters, q = merge(q, (x = changed,))))
    expected_clusters = ((:y,) => densey,)
    expected_a = getresult(call_message_update_rule(Conv2dCT, :a; algorithm, clusters = expected_clusters, q = (W = q.W, x = changed)))
    expected_W = getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters = expected_clusters, q = (a = q.a, x = changed)))
    expected_energy = getresult(call_average_energy(Conv2dCT; algorithm, clusters = expected_clusters, q = merge(q, (x = changed,))))
    @test all(isapprox.(params(towards_a), params(expected_a)))
    @test towards_W.invS ≈ expected_W.invS
    @test energy ≈ expected_energy
end

@testitem "rules:Conv2dVMP:cache recovery after errors" tags = [:rules] setup = [Conv2dReference] begin
    q = (a = MvNormalMeanCovariance(ones(4), Matrix(0.1I, 4, 4)), W = PointMass([1.0;;]))
    y = (NormalMeanVariance(0.2, 1.0),)
    x = ntuple(i -> NormalMeanVariance(i / 10, 1.0), 4)
    qy = MvNormalMeanCovariance([0.2], [1.0;;])
    for separate in (false, true)
        algorithm = Conv2dVMP((2, 2, 1), 1, (2, 2))
        inputs = separate ? (clusters = ((:y,) => qy,),) : (;)
        message(x) = getresult(call_message_update_rule(Conv2dCT, (:x, 1); algorithm, m = separate ? (x = x,) : (y = y, x = x), q, inputs...))
        good = without(x, 1)
        expected = weightedmean_precision(message(good))
        bad = (nothing, NormalWeightedMeanPrecision(0.0, -50.0), x[3], x[4])
        for _ in 1:2
            @test_throws PosDefException message(bad)
        end
        @test collect(weightedmean_precision(message(good))) ≈ collect(expected)
        # Validation can fail after some snapshots have already been changed.
        malformed = (nothing, NormalMeanVariance(2.0, 1.0), PointMass(0.0), x[4])
        for _ in 1:2
            @test_throws ArgumentError message(malformed)
        end
        @test collect(weightedmean_precision(message(good))) ≈ collect(expected)
    end
    # The y-only cached system must also stay invalid after a failed factorisation.
    algorithm = Conv2dVMP((1, 1, 1), 2, (1, 1))
    q2 = (a = MvNormalMeanCovariance(ones(2), Matrix(0.1I, 2, 2)), W = PointMass([1.0 0.2; 0.2 1.0]), x = (PointMass(0.4),))
    message(y) = getresult(call_message_update_rule(Conv2dCT, (:y, 1); algorithm, m = (y = y,), q = q2))
    good = (nothing, NormalMeanVariance(0.3, 1.0))
    expected = weightedmean_precision(message(good))
    for _ in 1:2
        @test_throws PosDefException message((nothing, NormalWeightedMeanPrecision(0.0, -50.0)))
    end
    @test collect(weightedmean_precision(message(good))) ≈ collect(expected)
end

@testitem "rules:Conv2dVMP:parameter representations and mutation" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(516)
    algorithm = Conv2dVMP((2, 3, 1), 2, (2, 2))
    n, m, ny, nx = algorithm.n, algorithm.m, algorithm.ny, algorithm.nx
    d = n * m
    μ, V = randn(rng, d), spd(rng, Float64, d)
    Λ = inv(Symmetric(V))
    W = PointMass(spd(rng, Float64, n))
    y = ntuple(_ -> NormalMeanVariance(randn(rng), 0.7), ny)
    x = ntuple(_ -> NormalMeanVariance(randn(rng), 0.9), nx)
    function check(qa)
        q = (a = qa, W = W)
        actual = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = x), q))
        fresh = Conv2dVMP(algorithm.insize, n, algorithm.kernel)
        canonical = (a = MvNormalMeanCovariance(mean(qa), Matrix(cov(qa))), W = W)
        expected = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm = fresh, m = (y = y, x = x), q = canonical))
        for field in (:mean, :var, :ξ, :λ, :Exx, :Eyx, :Eyy, :entropy)
            @test getfield(actual, field) ≈ getfield(expected, field) rtol = 1.0e-10 atol = 1.0e-11
        end
        clusters = ((:y, :x) => actual,)
        aW = getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters, q = (a = qa,)))
        eW = getresult(call_message_update_rule(Conv2dCT, :W; algorithm = fresh, clusters, q = (a = canonical.a,)))
        @test aW.invS ≈ eW.invS
        @test getresult(call_average_energy(Conv2dCT; algorithm, clusters, q)) ≈ getresult(call_average_energy(Conv2dCT; algorithm = fresh, clusters, q = canonical))
        # Repeating unchanged inputs must not rebuild the parameter snapshot.
        version = algorithm.cache.a.version
        getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = y, x = x), q))
        @test algorithm.cache.a.version == version
    end
    for qa in (
            MvNormalMeanCovariance(copy(μ), copy(V)),
            MvNormalMeanPrecision(copy(μ), copy(Λ)),
            MvNormalWeightedMeanPrecision(Λ * μ, copy(Λ)),
            MvNormalMeanScalePrecision(copy(μ), 2.0),
        )
        check(qa)
        first(params(qa))[1] += 0.3
        check(qa)
        if last(params(qa)) isa AbstractMatrix
            last(params(qa))[2, 2] += 0.4
            check(qa)
        end
    end
    check(MvNormalMeanScalePrecision(copy(μ), 3.0))
end

@testsnippet Conv2dReference begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils
    using BayesBase, ExponentialFamily, LinearAlgebra, StableRNGs

    spd(rng, T, d) = (L = randn(rng, T, d, d); L * L' / T(d) + Matrix{T}(I, d, d))
    without(ms, k) = ntuple(i -> i == k ? nothing : ms[i], length(ms))

    # The convolution as one big transition: a_big = vec(B) = tie * a, where B repeats the
    # kernel at every position, and W_big repeats W. Rows of B follow the y ports.
    function tied(algorithm, T)
        n, P, ny, nx = algorithm.n, algorithm.positions, algorithm.ny, algorithm.nx
        tie = zeros(T, ny * nx, n * algorithm.m)
        select = zeros(T, ny, n, P)
        for p in 1:P, c in 1:n
            select[p + (c - 1) * P, c, p] = 1
            for j in 1:algorithm.m
                t = algorithm.taps[j, p]
                t == 0 || (tie[(t - 1) * ny + p + (c - 1) * P, (j - 1) * n + c] = 1)
            end
        end
        bigW(W) = sum(select[:, :, p] * W * select[:, :, p]' for p in 1:P)
        blocks(R) = sum(select[:, :, p]' * R * select[:, :, p] for p in 1:P)
        biga(qa) = MvNormalMeanCovariance(tie * mean(qa), Matrix(Symmetric(tie * cov(qa) * tie')))
        return (; tie, bigW, blocks, biga, reference = LCTVMP(ny, nx))
    end

    geometries = (
        ((4, 4, 1), 1, (3, 3), (;)),
        ((4, 4, 1), 2, (2, 2), (; stride = 2)),
        ((4, 4, 2), 2, (3, 3), (;)),
        ((4, 3, 1), 2, (3, 2), (; stride = (1, 2), pad = 1)),
    )
end

@testitem "rules:Conv2dVMP:tied transition" tags = [:rules] setup = [Conv2dReference] begin
    import ContinuousTransitionMessagePassingRules: Conv2dBlockCovariance

    rng = StableRNG(511)
    for T in (Float64, Float32), (insize, cout, kernel, options) in (T === Float64 ? geometries : geometries[2:2])
        @testset "$T $insize -> $cout, $kernel, $options" begin
            algorithm = Conv2dVMP(insize, cout, kernel; T, options...)
            n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
            big = tied(algorithm, T)
            normal(d) = MvNormalMeanCovariance(randn(rng, T, d), spd(rng, T, d))
            my = ntuple(_ -> NormalMeanVariance(randn(rng, T), T(0.5) + rand(rng, T)), ny)
            mx = ntuple(_ -> NormalWeightedMeanPrecision(randn(rng, T), T(0.5) + rand(rng, T)), nx)
            qy, qx, qyx = normal(ny), normal(nx), normal(ny + nx)
            qa = MvNormalMeanCovariance(randn(rng, T, n * m), T(0.2) * spd(rng, T, n * m))
            W = spd(rng, T, n)
            q = (a = qa, W = PointMass(W))
            ref = (a = big.biga(qa), W = PointMass(big.bigW(W)))
            opts = (; algorithm, check_type_promotion = false, atol = T === Float32 ? 2.0e-3 : 1.0e-8, rtol = T === Float32 ? 2.0e-3 : 1.0e-8)
            # Twice, so that the second pass reads the cache.
            for _ in 1:2, structured in (true, false)
                for (group, incoming, other, otherq) in ((:y, my, :x, qx), (:x, mx, :y, qy))
                    clusters = structured ? () : ((other,) => otherq,)
                    for k in eachindex(incoming)
                        messages = structured ? (group === :y ? (y = without(my, k), x = mx) : (y = my, x = without(mx, k))) : NamedTuple{(group,)}((without(incoming, k),))
                        expected = getresult(call_message_update_rule(ScalarCT, (group, k); algorithm = big.reference, m = messages, q = ref, clusters))
                        test_message_update_rule(Conv2dCT, (group, k); cases = [(m = messages, q = q, clusters = clusters) => expected], opts...)
                    end
                end
                marginal_cases = structured ? (((:y, :x), (y = my, x = mx), ()),) : (((:y,), (y = my,), ((:x,) => qx,)), ((:x,), (x = mx,), ((:y,) => qy,)))
                for (target, messages, clusters) in marginal_cases
                    expected = getresult(call_marginal_update_rule(ScalarCT, target; algorithm = big.reference, m = messages, q = ref, clusters))
                    actual = getresult(call_marginal_update_rule(Conv2dCT, target; algorithm, m = messages, q, clusters))
                    @test mean(actual) ≈ mean(expected) atol = opts.atol rtol = opts.rtol
                    @test cov(actual) ≈ cov(expected) atol = opts.atol rtol = opts.rtol
                    same = if cov(actual) isa Conv2dBlockCovariance
                        # q(y) given q(x) stores one block per position.
                        Σ = cov(expected)
                        blocks = [Σ[p + (i - 1) * P, p + (j - 1) * P] for i in 1:n, j in 1:n, p in 1:P]
                        MvNormalMeanCovariance(mean(expected), Conv2dBlockCovariance(blocks))
                    else
                        convert(typeof(actual), expected)
                    end
                    test_marginal_update_rule(Conv2dCT, target; cases = [(m = messages, q = q, clusters = clusters) => same], opts...)
                end
                clusters = structured ? ((:y, :x) => qyx,) : ((:y,) => qy, (:x,) => qx)
                # Towards a: the big message, projected back onto the shared kernel.
                ξ, Λ = weightedmean_precision(getresult(call_message_update_rule(ScalarCT, :a; algorithm = big.reference, clusters, q = (W = ref.W,))))
                expected = MvNormalWeightedMeanPrecision(big.tie' * ξ, big.tie' * Λ * big.tie)
                test_message_update_rule(Conv2dCT, :a; cases = [(clusters = clusters, q = (W = q.W,)) => expected], opts...)
                # Towards W: the diagonal blocks of the big residual, one degree of freedom each.
                residual = getresult(call_message_update_rule(ScalarCT, :W; algorithm = big.reference, clusters, q = (a = ref.a,))).invS
                expected = ExponentialFamily.WishartFast(T(n + 1 + P), big.blocks(residual))
                test_message_update_rule(Conv2dCT, :W; cases = [(clusters = clusters, q = (a = q.a,)) => expected], opts...)
                expected = getresult(call_average_energy(ScalarCT; algorithm = big.reference, clusters, q = ref))
                test_average_energy(Conv2dCT; cases = [(clusters = clusters, q = q) => expected], opts...)
            end
        end
    end
end

@testitem "rules:Conv2dVMP:one ScalarCT per position" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(512)
    for (insize, cout, kernel, options) in geometries[1:3]
        algorithm = Conv2dVMP(insize, cout, kernel; options...)
        n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
        qyx = MvNormalMeanCovariance(randn(rng, ny + nx), spd(rng, Float64, ny + nx))
        qa = MvNormalMeanCovariance(randn(rng, n * m), 0.2 * spd(rng, Float64, n * m))
        qW = Wishart(n + 3.0, spd(rng, Float64, n) / (n + 3))
        clusters = ((:y, :x) => qyx,)
        # Every position is a ScalarCT on its own patch, sharing a and W: multiply their messages.
        patches = map(1:P) do p
            index = [p .+ (0:(n - 1)) .* P; ny .+ algorithm.taps[:, p]]
            ((:y, :x) => MvNormalMeanCovariance(mean(qyx)[index], cov(qyx)[index, index]),)
        end
        each(target, q) = [getresult(call_message_update_rule(ScalarCT, target; algorithm = LCTVMP(n, m), clusters = patch, q)) for patch in patches]
        towards_a = getresult(call_message_update_rule(Conv2dCT, :a; algorithm, clusters, q = (W = qW,)))
        ξ, Λ = weightedmean_precision(towards_a)
        @test ξ ≈ sum(first ∘ weightedmean_precision, each(:a, (W = qW,)))
        @test Λ ≈ sum(last ∘ weightedmean_precision, each(:a, (W = qW,)))
        towards_W = getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters, q = (a = qa,)))
        @test towards_W.ν == sum(w -> w.ν, each(:W, (a = qa,))) - (P - 1) * (n + 1)
        @test towards_W.invS ≈ sum(w -> w.invS, each(:W, (a = qa,)))
        energies = [getresult(call_average_energy(ScalarCT; algorithm = LCTVMP(n, m), clusters = patch, q = (a = qa, W = qW))) for patch in patches]
        @test getresult(call_average_energy(Conv2dCT; algorithm, clusters, q = (a = qa, W = qW))) ≈ sum(energies)
    end
end

@testitem "rules:Conv2dVMP:observed and independent inputs" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(513)
    for (insize, cout, kernel, options) in geometries
        algorithm = Conv2dVMP(insize, cout, kernel; options...)
        n, m, P, ny, nx = algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx
        qa = MvNormalMeanCovariance(randn(rng, n * m), 0.2 * spd(rng, Float64, n * m))
        q = (a = qa, W = Wishart(n + 3.0, spd(rng, Float64, n) / (n + 3)))
        my = ntuple(_ -> NormalMeanVariance(randn(rng), 0.5 + rand(rng)), ny)
        mx = ntuple(_ -> NormalMeanVariance(randn(rng), 0.5 + rand(rng)), nx)
        pixels = randn(rng, nx)
        qy = MvNormalMeanCovariance(randn(rng, ny), spd(rng, Float64, ny))
        # The engine's form for data and the same state as a joint marginal.
        for (scalars, joint) in (
                (ntuple(i -> PointMass(pixels[i]), nx), MvNormalMeanCovariance(pixels, zeros(nx, nx))),
                (mx, MvNormalMeanCovariance(collect(mean.(mx)), Matrix(Diagonal(collect(var.(mx)))))),
            )
            independent(parameters) = (q = merge(parameters, (x = scalars,)),)
            together(parameters) = (q = parameters, clusters = ((:x,) => joint,))
            for k in 1:ny
                a = getresult(call_message_update_rule(Conv2dCT, (:y, k); algorithm, m = (y = without(my, k),), independent(q)...))
                b = getresult(call_message_update_rule(Conv2dCT, (:y, k); algorithm, m = (y = without(my, k),), together(q)...))
                @test all(isapprox.(params(a), params(b)))
            end
            a = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = my,), independent(q)...))
            b = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = my,), together(q)...))
            @test all(isapprox.(params(a), params(b)))
            for (target, parameters) in ((:a, (W = q.W,)), (:W, (a = q.a,)))
                a = getresult(call_message_update_rule(Conv2dCT, target; algorithm, clusters = ((:y,) => qy,), independent(parameters).q))
                b = getresult(call_message_update_rule(Conv2dCT, target; algorithm, clusters = ((:y,) => qy, (:x,) => joint), q = parameters))
                @test all(isapprox.(params(a), params(b)))
            end
            a = getresult(call_average_energy(Conv2dCT; algorithm, clusters = ((:y,) => qy,), independent(q).q))
            b = getresult(call_average_energy(Conv2dCT; algorithm, clusters = ((:y,) => qy, (:x,) => joint), q))
            @test a ≈ b
        end
        # An observed image: the message is the convolution of the pixels with E[A], in E[W].
        image = reshape(pixels, insize)
        A, W = reshape(mean(qa), n, m), mean(q.W)
        if isempty(options)
            observed = (q = merge(q, (x = ntuple(i -> PointMass(pixels[i]), nx),)),)
            for o2 in 1:algorithm.outsize[2], o1 in 1:algorithm.outsize[1]
                p = o1 + (o2 - 1) * algorithm.outsize[1]
                μ = A * vec(image[o1:(o1 + kernel[1] - 1), o2:(o2 + kernel[2] - 1), :])
                block = getresult(call_message_update_rule(LinearContinuousTransition, :y; algorithm = LCTVMP(n, m), q = (x = PointMass(vec(image[o1:(o1 + kernel[1] - 1), o2:(o2 + kernel[2] - 1), :])), a = qa, W = q.W)))
                @test mean(block) ≈ μ
                wide = ntuple(i -> NormalMeanVariance(0.0, 1.0e12), ny)
                for c in 1:n
                    k = p + (c - 1) * P
                    message = getresult(call_message_update_rule(Conv2dCT, (:y, k); algorithm, m = (y = without(wide, k),), observed...))
                    @test mean(message) ≈ μ[c] rtol = 1.0e-6
                    @test var(message) ≈ inv(W)[c, c] rtol = 1.0e-6
                end
            end
        end
    end
end

@testitem "rules:Conv2dVMP:cache" tags = [:rules] setup = [Conv2dReference] begin
    rng = StableRNG(514)
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
    @test algorithm.cache.yx === nothing && algorithm.cache.x === nothing
    sweep()
    solve = algorithm.cache.yx.solve
    @test algorithm.cache.x === nothing
    @test size(solve.V) == (ny + nx, ny + nx)
    builds = solve.factorizations
    sweep()
    @test solve.factorizations == builds
    # Incoming weighted means do not change the shared solve; precisions do.
    sweep(; my = (NormalWeightedMeanPrecision(13.0, 1.0 + 1 / ny), my[2:end]...))
    @test solve.factorizations == builds
    check(:y, 2; my = (NormalWeightedMeanPrecision(1.0e14, 1.0e14), my[2:end]...))
    sweep()
    @test solve.factorizations > builds
    for mutate! in (() -> (mean(qa)[1] += 0.8), () -> (cov(qa)[2, 2] += 0.9), () -> (W[1, 1] += 0.7))
        previous, version = solve.factorizations, algorithm.cache.version
        mutate!()
        sweep()
        @test solve.factorizations > previous
        @test algorithm.cache.version > version
    end
    # A W update refreshes q(a)'s snapshot first: the transition must still be rebuilt.
    cov(qa)[3, 3] += 0.5
    getresult(call_message_update_rule(Conv2dCT, :W; algorithm, clusters = ((:y, :x) => MvNormalMeanCovariance(zeros(ny + nx), Matrix(1.0I, ny + nx, ny + nx)),), q = (a = qa,)))
    sweep()
    # The separate x system has its own solve.
    foreach(k -> check(:x, k; clusters = ((:y,) => qy,)), 1:nx)
    separate = algorithm.cache.x.solve
    builds = separate.factorizations
    mean(qy)[1] += 4.0
    foreach(k -> check(:x, k; clusters = ((:y,) => qy,)), 1:nx)
    @test separate.factorizations == builds
    # Returned parameters do not alias the cache.
    joint = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = my, x = mx), q))
    saved = deepcopy(params(joint))
    mean(qa)[2] -= 0.3
    sweep()
    @test isequal(params(joint), saved)
end

@testitem "rules:Conv2dVMP:geometry and validation" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: ispure, dependencies_spec, extends_default_scheme, IndexedTarget, Target, interfaces, interface_groups

    @test Conv2dCT === Conv2dContinuousTransition
    @test interfaces(Conv2dCT) == (:y, :x, :a, :W)
    @test interface_groups(Conv2dCT) == (:y, :x)
    algorithm = Conv2dVMP((5, 4, 2), 3, (3, 2); stride = (2, 1), pad = (1, 0))
    @test algorithm.outsize == (3, 3, 3)
    @test (algorithm.n, algorithm.m, algorithm.positions, algorithm.ny, algorithm.nx) == (3, 12, 9, 27, 40)
    @test !ispure(algorithm)
    @test occursin("Conv2dVMP((5, 4, 2), 3, (3, 2)", sprint(show, algorithm))
    spec = dependencies_spec(Conv2dCT, algorithm)
    @test all(t -> extends_default_scheme(spec, t), (IndexedTarget(:y, 1), IndexedTarget(:x, 3), Target(:a), Target(:W)))
    # The taps are the padded image's windows, in column-major order.
    padded = zeros(Int, 7, 4, 2)
    padded[2:6, :, :] .= reshape(1:40, 5, 4, 2)
    for o2 in 1:3, o1 in 1:3
        @test algorithm.taps[:, o1 + (o2 - 1) * 3] == vec(padded[(2 * o1 - 1):(2 * o1 + 1), o2:(o2 + 1), :])
    end
    @test eltype(Conv2dVMP((3, 3, 1), 1, (2, 2); T = Float32).cache.W) === Float32
    for arguments in (((0, 3, 1), 1, (2, 2)), ((3, 3, 1), 0, (2, 2)), ((3, 3, 1), 1, (0, 2)), ((3, 3, 1), 1, (4, 2)))
        @test_throws ArgumentError Conv2dVMP(arguments...)
    end
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); stride = 0)
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); pad = -1)
    @test_throws ArgumentError Conv2dVMP((3, 3, 1), 1, (2, 2); stride = 1.5)

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

    # A single output and a single input: the engine's scalar forms.
    single = Conv2dVMP((1, 1, 1), 1, (1, 1))
    q1 = (a = MvNormalMeanCovariance([2.0], [0.3;;]), W = Wishart(3.0, [0.5;;]))
    for (group, other) in ((:y, :x), (:x, :y)), inputs in ((m = NamedTuple{(other,)}(((s,),)), q = q1), (q = merge(q1, NamedTuple{(other,)}(((s,),))),))
        actual = getresult(call_message_update_rule(Conv2dCT, (group, 1); algorithm = single, inputs...))
        expected = getresult(call_message_update_rule(ScalarCT, (group, 1); algorithm = LCTVMP(1, 1), inputs...))
        @test all(isapprox.(params(actual), params(expected)))
    end
end

@testitem "rules:Conv2dVMP:block covariance of q(y)" begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra, FastCholesky, StableRNGs
    import ContinuousTransitionMessagePassingRules: Conv2dBlockCovariance

    rng = StableRNG(7)
    for cout in (1, 2, 3)
        algorithm = Conv2dVMP((4, 4, 1), cout, (3, 3))
        d = algorithm.n * algorithm.m
        result = getresult(
            @call_marginal_update_rule(
                node = Conv2dCT, target = (:y,), algorithm = algorithm,
                m = (y = ntuple(_ -> NormalMeanVariance(randn(rng), 0.5), algorithm.ny),),
                q = (
                    x = ntuple(_ -> PointMass(randn(rng)), algorithm.nx),
                    a = MvNormalMeanCovariance(randn(rng, d), Matrix(0.1I, d, d)),
                    W = Wishart(cout + 2.0, Matrix(1.0I, cout, cout) .+ 0.2),
                ),
            )
        )
        V = cov(result)
        dense = Matrix(V)
        @test V isa Conv2dBlockCovariance
        @test size(V.blocks) == (cout, cout, algorithm.positions)
        @test issymmetric(dense) && isposdef(dense)
        # Entries that couple two positions are zero and not stored.
        P = algorithm.positions
        @test all(iszero(dense[i, j]) for i in 1:algorithm.ny, j in 1:algorithm.ny if (i - 1) % P != (j - 1) % P)
        @test dense[1, 1] == V.blocks[1, 1, 1] && dense[P * cout, P * cout] == V.blocks[cout, cout, P]
        @test logdet(V) ≈ logdet(dense)
        @test chollogdet(V) ≈ logdet(dense)
        @test cholinv(V) isa Conv2dBlockCovariance
        @test cholinv(V) ≈ inv(dense)
        @test entropy(result) ≈ entropy(MvNormalMeanCovariance(mean(result), dense))
        @test_throws BoundsError V[algorithm.ny + 1, 1]
    end
end

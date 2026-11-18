@testitem "rules:SLCTVMP:equivalence" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils
    using BayesBase, ExponentialFamily, LinearAlgebra, StableRNGs

    rng = StableRNG(704)
    spd(T, d) = (L = randn(rng, T, d, d); L * L' / T(d) + Matrix{T}(I, d, d))
    without(ms, k) = ntuple(i -> i == k ? nothing : ms[i], length(ms))
    for T in (Float32, Float64), (n, m) in ((1, 1), (1, 3), (3, 1), (2, 3), (3, 2))
        @testset "$T ($n, $m)" begin
            algorithm, reference = SLCTVMP(n, m; T), LCTVMP(n, m)
            normal(d) = MvNormalMeanCovariance(randn(rng, T, d), spd(T, d))
            my = ntuple(_ -> NormalMeanVariance(randn(rng, T), T(0.5) + rand(rng, T)), n)
            mx = ntuple(_ -> NormalWeightedMeanPrecision(randn(rng, T), T(0.5) + rand(rng, T)), m)
            qy, qx, qyx = normal(n), normal(m), normal(n + m)
            q = (a = normal(n * m), W = Wishart(T(n + 3), spd(T, n) / T(n + 3)))
            opts = (; algorithm, check_type_promotion = false, atol = T === Float32 ? 3.0e-5 : 1.0e-11, rtol = T === Float32 ? 3.0e-4 : 1.0e-10)
            for structured in (true, false, true)
                for (group, incoming, other, otherq) in ((:y, my, :x, qx), (:x, mx, :y, qy))
                    clusters = structured ? () : ((other,) => otherq,)
                    for k in reverse(eachindex(incoming))
                        messages = structured ? (group === :y ? (y = without(my, k), x = mx) : (y = my, x = without(mx, k))) : NamedTuple{(group,)}((without(incoming, k),))
                        inputs = (m = messages, q = q, clusters = clusters)
                        expected = getresult(call_message_update_rule(ScalarCT, (group, k); algorithm = reference, inputs...))
                        test_message_update_rule(ScalarCT, (group, k); cases = [inputs => expected], opts...)
                    end
                end
                marginal_cases = structured ? (((:y, :x), (m = (y = my, x = mx), q = q)),) : (
                        ((:y,), (m = (y = my,), q = q, clusters = ((:x,) => qx,))),
                        ((:x,), (m = (x = mx,), q = q, clusters = ((:y,) => qy,))),
                    )
                for (target, inputs) in marginal_cases
                    expected = getresult(call_marginal_update_rule(ScalarCT, target; algorithm = reference, inputs...))
                    test_marginal_update_rule(ScalarCT, target; cases = [inputs => expected], opts...)
                end
                clusters = structured ? ((:y, :x) => qyx,) : ((:y,) => qy, (:x,) => qx)
                for (target, parameters) in ((:a, (W = q.W,)), (:W, (a = q.a,)))
                    inputs = (; clusters, q = parameters)
                    expected = getresult(call_message_update_rule(ScalarCT, target; algorithm = reference, inputs...))
                    test_message_update_rule(ScalarCT, target; cases = [inputs => expected], opts...)
                end
                inputs = (; clusters, q)
                expected = getresult(call_average_energy(ScalarCT; algorithm = reference, inputs...))
                test_average_energy(ScalarCT; cases = [inputs => expected], opts...)
            end
        end
    end
end

@testitem "rules:SLCTVMP:invalidation and reuse" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
    using Distributions: MvNormal

    algorithm, reference = SLCTVMP(2, 3), LCTVMP(2, 3)
    normal(d) = MvNormalMeanCovariance(collect(1.0:d) / d, Matrix(1.0I, d, d) + fill(0.1, d, d))
    qa = normal(6)
    W = [2.0 0.4; 0.4 1.5]
    q = (a = qa, W = PointMass(W))
    my = (NormalWeightedMeanPrecision(0.2, 1.5), NormalWeightedMeanPrecision(-0.4, 2.0))
    mx = ntuple(i -> NormalWeightedMeanPrecision(i / 3, 1.0 + i), 3)
    without(ms, k) = ntuple(i -> i == k ? nothing : ms[i], length(ms))
    function check(group, k; q = q, my = my, mx = mx, clusters = ())
        messages = isempty(clusters) ? (group === :y ? (y = without(my, k), x = mx) : (y = my, x = without(mx, k))) : NamedTuple{(group,)}((without(group === :y ? my : mx, k),))
        inputs = (; m = messages, q, clusters)
        actual = getresult(call_message_update_rule(ScalarCT, (group, k); algorithm, inputs...))
        expected = getresult(call_message_update_rule(ScalarCT, (group, k); algorithm = reference, inputs...))
        @test all(isapprox.(params(actual), params(expected); rtol = 1.0e-10, atol = 1.0e-10))
        return actual
    end
    sweep(; kwargs...) = [check(g, k; kwargs...) for (g, d) in ((:x, 3), (:y, 2)) for k in 1:d]
    sweep()
    c = algorithm.cache
    builds, version = c.yx.factorizations, c.a.version
    sweep()
    @test c.yx.factorizations == builds
    @test c.a.version == version
    # Incoming weighted means do not change the shared covariance.
    changed_y = (NormalWeightedMeanPrecision(13.0, 1.5), my[2])
    sweep(; my = changed_y)
    @test c.yx.factorizations == builds
    # The excluded target may retain an old prior; it must cancel exactly.
    huge_y = (NormalWeightedMeanPrecision(1.0e14, 1.0e14), my[2])
    check(:y, 2; my = huge_y)
    check(:y, 1)
    sweep()
    @test c.yx.factorizations > builds
    for mutate! in (() -> (mean(qa)[1] += 0.8), () -> (cov(qa)[2, 2] += 0.9), () -> (W[1, 1] += 0.7))
        previous = c.yx.factorizations
        mutate!()
        sweep()
        @test c.yx.factorizations > previous
    end
    # All Gaussian representations, including changes between representations with the
    # same raw parameter values, and in-place changes of natural parameters.
    for distribution in (
            MvNormalMeanPrecision(copy(mean(qa)), copy(cov(qa))),
            MvNormalWeightedMeanPrecision(copy(mean(qa)), copy(cov(qa))),
            MvNormalMeanCovariance(copy(mean(qa)), copy(cov(qa))),
            MvNormal(copy(mean(qa)), copy(cov(qa))),
            MvNormalMeanScalePrecision(copy(mean(qa)), 2.0),
        )
        parameters = (a = distribution, W = q.W)
        sweep(; q = parameters)
        parameter_version = c.a.version
        sweep(; q = parameters)
        @test c.a.version == parameter_version
        params(distribution)[1][1] += 0.5
        sweep(; q = parameters)
        @test c.a.version > parameter_version
        if distribution isa Union{MvNormalMeanPrecision, MvNormalWeightedMeanPrecision}
            params(distribution)[2][2, 2] += 0.3
            sweep(; q = parameters)
        end
    end
    # A W update can refresh q(a)'s snapshot before a state update: the base precision
    # must still be rebuilt. State marginal changes affect the mean-field RHS as well.
    cov(qa)[3, 3] += 0.5
    clusters = ((:y,) => normal(2), (:x,) => normal(3))
    getresult(call_message_update_rule(ScalarCT, :W; algorithm, clusters, q = (a = qa,)))
    sweep()
    qx = normal(3)
    check(:y, 1; clusters = ((:x,) => qx,))
    builds = c.y.factorizations
    mean(qx)[1] += 4.0
    check(:y, 1; clusters = ((:x,) => qx,))
    @test c.y.factorizations == builds
end

@testitem "rules:SLCTVMP:ownership and singleton groups" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra

    algorithm = SLCTVMP(1, 1)
    other = SLCTVMP(1, 1)
    for s in (:y, :x, :yx), field in (:ξ, :λ, :h, :J, :factor, :V)
        @test !Base.mightalias(getproperty(getproperty(algorithm.cache, s), field), getproperty(getproperty(other.cache, s), field))
    end
    q = (a = MvNormalMeanCovariance([2.0], [0.3;;]), W = Wishart(3.0, [0.5;;]))
    s = NormalMeanVariance(0.5, 1.0)
    messages = (y = (s,), x = (s,))
    marginals = (y = (s,), x = (s,))
    joint = getresult(call_marginal_update_rule(ScalarCT, (:y, :x); algorithm, m = messages, q))
    a = getresult(call_message_update_rule(ScalarCT, :a; algorithm, q = merge(marginals, (W = q.W,))))
    W = getresult(call_message_update_rule(ScalarCT, :W; algorithm, q = merge(marginals, (a = q.a,))))
    saved = deepcopy.((params(joint), params(a), params(W)))
    for group in (:y, :x)
        opposite = group === :y ? :x : :y
        for inputs in ((m = NamedTuple{(opposite,)}(((s,),)), q = q), (q = merge(q, NamedTuple{(opposite,)}(((s,),))),))
            actual = getresult(call_message_update_rule(ScalarCT, (group, 1); algorithm, inputs...))
            expected = getresult(call_message_update_rule(ScalarCT, (group, 1); algorithm = LCTVMP(1, 1), inputs...))
            @test all(isapprox.(params(actual), params(expected)))
        end
    end
    mean(q.a)[1] += 1.0
    getresult(call_marginal_update_rule(ScalarCT, (:y, :x); algorithm, m = messages, q))
    getresult(call_message_update_rule(ScalarCT, :W; algorithm, q = merge(marginals, (a = q.a,))))
    getresult(call_message_update_rule(ScalarCT, :a; algorithm, q = (y = (NormalMeanVariance(1.0, 2.0),), x = (s,), W = q.W)))
    getresult(call_average_energy(ScalarCT; algorithm, q = merge(marginals, q)))
    @test isequal((params(joint), params(a), params(W)), saved)
end

@testitem "rules:SLCTVMP:validation" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: ispure, dependencies_spec, extends_default_scheme, IndexedTarget, Target

    for dims in ((0, 2), (2, 0), (-1, 2), (2, -1))
        @test_throws ArgumentError SLCTVMP(dims...)
    end
    @test_throws MethodError SLCTVMP(2.0, 3.0)
    for T in (Float32, Float64, BigFloat)
        algorithm = SLCTVMP(2, 3; T)
        @test size(algorithm.cache.a.V) == (6, 6)
        @test size(algorithm.cache.yx.V) == (5, 5)
        @test eltype(algorithm.cache.yx.V) === T
        @test !ispure(algorithm)
        spec = dependencies_spec(ScalarCT, algorithm)
        @test all(t -> extends_default_scheme(spec, t), (IndexedTarget(:y, 1), IndexedTarget(:x, 3), Target(:a), Target(:W)))
    end
    algorithm = SLCTVMP(2, 3)
    normal(d) = MvNormalMeanCovariance(ones(d), Matrix(1.0I, d, d))
    s = NormalMeanVariance(0.0, 1.0)
    q = (a = normal(6), W = Wishart(4.0, Matrix(1.0I, 2, 2)))
    valid = (y = (nothing, s), x = (s, s, s))
    for m in ((y = (nothing,), x = (s, s, s)), (y = (nothing, s), x = (s, s)), (y = (nothing, s), x = (s, s, s, s)))
        @test_throws DimensionMismatch getresult(call_message_update_rule(ScalarCT, (:y, 1); algorithm, m, q))
    end
    for m in ((x = (s, s, s),), (y = (s, s), x = (s, s, s)), (y = (nothing, nothing), x = (s, s, s)))
        @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, (:y, 1); algorithm, m, q))
    end
    for k in (0, -1, 3)
        @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, (:y, k); algorithm, m = valid, q))
    end
    for malformed in ((a = normal(5), W = q.W), (a = q.a, W = Wishart(5.0, Matrix(1.0I, 3, 3))))
        @test_throws DimensionMismatch getresult(call_message_update_rule(ScalarCT, (:y, 1); algorithm, m = valid, q = malformed))
    end
    @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, :a; algorithm, q = (y = (s, s), x = (s, s, s), W = q.W)))
    @test_throws ArgumentError getresult(call_marginal_update_rule(ScalarCT, (:y,); algorithm, m = (y = (s, s),), q))
end

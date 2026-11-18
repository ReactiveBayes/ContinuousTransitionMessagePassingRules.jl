@testitem "rules:ScalarCT:block equivalence" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils
    using BayesBase, ExponentialFamily, LinearAlgebra, StableRNGs

    rng = StableRNG(732)
    spd(T, d) = (L = randn(rng, T, d, d); L * L' / T(d) + Matrix{T}(I, d, d))
    block(ms) = MvNormalMeanCovariance(collect(mean.(ms)), Matrix(Diagonal(collect(var.(ms)))))
    without(ms, k) = ntuple(i -> i == k ? nothing : ms[i], length(ms))

    for T in (Float32, Float64), (n, m) in ((1, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
        @testset "$T ($n, $m)" begin
            algorithm = LCTVMP(n, m)
            my = ntuple(_ -> NormalMeanVariance(randn(rng, T), T(0.5) + rand(rng, T)), n)
            mx = ntuple(_ -> NormalMeanPrecision(randn(rng, T), T(0.5) + rand(rng, T)), m)
            iy, ix = block(my), block(mx)
            qy = MvNormalMeanCovariance(randn(rng, T, n), spd(T, n))
            qx = MvNormalMeanCovariance(randn(rng, T, m), spd(T, m))
            qa = MvNormalMeanCovariance(randn(rng, T, n * m), T(0.15) * spd(T, n * m))
            qW = Wishart(T(n + 3), spd(T, n) / T(n + 3))
            q = (a = qa, W = qW)
            opts = (algorithm = algorithm, check_type_promotion = false, atol = T === Float32 ? 2.0e-5 : 1.0e-11, rtol = T === Float32 ? 2.0e-4 : 1.0e-10)
            joint = getresult(call_marginal_update_rule(LinearContinuousTransition, (:y, :x); algorithm, m = (y = iy, x = ix), q))
            test_marginal_update_rule(ScalarCT, (:y, :x); cases = [(m = (y = my, x = mx), q = q) => joint], opts...)

            for structured in (false, true), (group, incoming, otherq, refincoming, offset) in ((:y, my, qx, iy, 0), (:x, mx, qy, ix, n))
                other = group === :y ? :x : :y
                if structured
                    posterior = joint
                    clusters = ()
                else
                    refq = merge(q, NamedTuple{(other,)}((otherq,)))
                    message = getresult(call_message_update_rule(LinearContinuousTransition, group; algorithm, q = refq))
                    posterior = prod(GenericProd(), refincoming, message)
                    clusters = ((other,) => otherq,)
                    minputs = (m = NamedTuple{(group,)}((incoming,)), q = q, clusters = clusters)
                    test_marginal_update_rule(ScalarCT, (group,); cases = [minputs => posterior], opts...)
                end
                # The existing block rules can return Float32 precisions with roundoff
                # asymmetry. Read the symmetric matrix without changing those rules.
                μ, V = mean_cov(MvNormalWeightedMeanPrecision(weightedmean(posterior), Matrix(Symmetric(precision(posterior)))))
                for k in eachindex(incoming)
                    messages = structured ? (group === :y ? (y = without(my, k), x = mx) : (y = my, x = without(mx, k))) : NamedTuple{(group,)}((without(incoming, k),))
                    idx = structured ? offset + k : k
                    ξin, λin = weightedmean_precision(incoming[k])
                    expected = NormalWeightedMeanPrecision(μ[idx] / V[idx, idx] - ξin, inv(V[idx, idx]) - λin)
                    inputs = (m = messages, q = q, clusters = clusters)
                    test_message_update_rule(ScalarCT, (group, k); cases = [inputs => expected], opts...)
                    actual = getresult(call_message_update_rule(ScalarCT, (group, k); algorithm, inputs...))
                    posterior_scalar = prod(GenericProd(), actual, incoming[k])
                    @test mean(posterior_scalar) ≈ μ[idx] atol = opts.atol rtol = opts.rtol
                    @test var(posterior_scalar) ≈ V[idx, idx] atol = opts.atol rtol = opts.rtol
                end
            end

            # Dense, independently chosen state covariances exercise all covariance terms.
            qyx = MvNormalMeanCovariance([mean(qy); mean(qx)], spd(T, n + m))
            for structured in (false, true)
                clusters = structured ? ((:y, :x) => qyx,) : ((:y,) => qy, (:x,) => qx)
                refstates = structured ? (clusters = clusters,) : (q = (y = qy, x = qx),)
                for (target, parameters) in ((:a, (W = qW,)), (:W, (a = qa,)))
                    refinputs = merge(refstates, (q = merge(get(refstates, :q, (;)), parameters),))
                    expected = getresult(call_message_update_rule(LinearContinuousTransition, target; algorithm, refinputs...))
                    test_message_update_rule(ScalarCT, target; cases = [(clusters = clusters, q = parameters) => expected], opts...)
                end
                refinputs = merge(refstates, (q = merge(get(refstates, :q, (;)), q),))
                expected = getresult(call_average_energy(LinearContinuousTransition; algorithm, refinputs...))
                test_average_energy(ScalarCT; cases = [(clusters = clusters, q = q) => expected], opts...)
            end
        end
    end
end

@testitem "rules:ScalarCT:singleton inputs and promotion" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils
    using BayesBase, ExponentialFamily, LinearAlgebra

    for (n, m) in ((1, 1), (1, 3), (3, 1))
        algorithm = LCTVMP(n, m)
        normal(d) = MvNormalMeanPrecision(ones(d), Matrix(2.0I, d, d) + fill(0.1, d, d))
        qy, qx, qa = normal(n), normal(m), normal(n * m)
        qW = Wishart(n + 3.0, Matrix(1.0I, n, n))
        scalar(q) = NormalWeightedMeanPrecision(only(weightedmean(q)), only(precision(q)))
        singles = merge(n == 1 ? (y = (scalar(qy),),) : (;), m == 1 ? (x = (scalar(qx),),) : (;))
        clusters = ((n == 1 ? () : ((:y,) => qy,))..., (m == 1 ? () : ((:x,) => qx,))...)
        reference = ((:y,) => qy, (:x,) => qx)
        opts = (algorithm = algorithm, float_types = (Float32, Float64), atol = 2.0e-5, rtol = 2.0e-4)
        for (target, parameters) in ((:a, (W = qW,)), (:W, (a = qa,)))
            expected = getresult(call_message_update_rule(ScalarCT, target; algorithm, clusters = reference, q = parameters))
            test_message_update_rule(ScalarCT, target; cases = [(clusters = clusters, q = merge(singles, parameters)) => expected], opts...)
        end
        parameters = (a = qa, W = qW)
        expected = getresult(call_average_energy(ScalarCT; algorithm, clusters = reference, q = parameters))
        test_average_energy(ScalarCT; cases = [(clusters = clusters, q = merge(singles, parameters)) => expected], opts...)

        for (group, d, otherq, otherd) in ((:y, n, qx, m), (:x, m, qy, n))
            d == 1 || continue
            other = group === :y ? :x : :y
            expected = getresult(call_message_update_rule(ScalarCT, (group, 1); algorithm, clusters = ((other,) => otherq,), q = parameters))
            q = otherd == 1 ? merge(parameters, NamedTuple{(other,)}(((scalar(otherq),),))) : parameters
            cs = otherd == 1 ? () : ((other,) => otherq,)
            test_message_update_rule(ScalarCT, (group, 1); cases = [(q = q, clusters = cs) => expected], opts...)
            # The own group is absent under structured dependencies as well.
            messages = NamedTuple{(other,)}((ntuple(_ -> NormalMeanVariance(0.5, 1.0), otherd),))
            explicit = merge(messages, NamedTuple{(group,)}(((nothing,),)))
            expected = getresult(call_message_update_rule(ScalarCT, (group, 1); algorithm, m = explicit, q = parameters))
            test_message_update_rule(ScalarCT, (group, 1); cases = [(m = messages, q = parameters) => expected], opts...)
        end
    end
end

@testitem "rules:ScalarCT:scratch ownership" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: Target, IndexedTarget, ClusterTarget, RuleContext, NoAnnotations, find_message_rule, find_marginal_rule, rule_scratch, execute_rule

    algorithm = LCTVMP(2, 3)
    function cases(scale)
        normal(d) = MvNormalMeanCovariance(fill(scale, d), Matrix(1.0I, d, d) + fill(0.1 * scale, d, d))
        q = (a = normal(6), W = Wishart(4.0, Matrix(scale * I, 2, 2)))
        my, mx = ntuple(_ -> NormalMeanVariance(scale, 1.0), 2), ntuple(_ -> NormalMeanVariance(scale, 1.0), 3)
        return (
            (IndexedTarget(:y, 1), (m = (y = (nothing, my[2]), x = mx), q = q)),
            (IndexedTarget(:x, 2), (m = (y = my, x = (mx[1], nothing, mx[3])), q = q)),
            (IndexedTarget(:y, 2), (m = (y = (my[1], nothing),), clusters = ((:x,) => normal(3),), q = q)),
            (IndexedTarget(:x, 1), (m = (x = (nothing, mx[2], mx[3]),), clusters = ((:y,) => normal(2),), q = q)),
            (Target(:a), (clusters = ((:y, :x) => normal(5),), q = (W = q.W,))),
            (Target(:W), (clusters = ((:y,) => normal(2), (:x,) => normal(3)), q = (a = q.a,))),
            (ClusterTarget((:y, :x)), (m = (y = my, x = mx), q = q)),
            (ClusterTarget((:y,)), (m = (y = my,), clusters = ((:x,) => normal(3),), q = q)),
            (ClusterTarget((:x,)), (m = (x = mx,), clusters = ((:y,) => normal(2),), q = q)),
        )
    end
    ruleargs(inputs) = MessagePassingRulesBase.interactive_args(get(inputs, :m, (;)), get(inputs, :q, (;)), get(inputs, :clusters, ()))
    run(spec, scratch, args, target) = execute_rule(spec, nothing, scratch, algorithm, RuleContext(), args, NoAnnotations(), target)
    for ((target, inputs), (_, nextinputs)) in zip(cases(0.5), cases(1.5))
        args, nextargs = ruleargs(inputs), ruleargs(nextinputs)
        spec = target isa ClusterTarget ? find_marginal_rule(ScalarCT, target, algorithm, args) : find_message_rule(ScalarCT, target, algorithm, args)
        scratch = rule_scratch(spec, algorithm, RuleContext(), args, target)
        result = run(spec, scratch, args, target)
        saved = deepcopy(params(result))
        actual = run(spec, scratch, nextargs, target)
        expected = run(spec, nothing, nextargs, target)
        @test all(isapprox.(params(actual), params(expected)))
        @test isequal(params(result), saved)
        other = rule_scratch(spec, algorithm, RuleContext(), args, target)
        @test all(!Base.mightalias(a, b) for a in values(scratch), b in values(other))
        foreach(A -> fill!(A, NaN), values(scratch))
        @test isequal(params(result), saved)
        @test all(isapprox.(params(run(spec, scratch, args, target)), saved))
    end
end

@testitem "rules:ScalarCT:invalid layouts and dimensions" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, LinearAlgebra
    normal(d) = MvNormalMeanCovariance(ones(d), Matrix(1.0I, d, d))
    s = NormalMeanVariance(0.0, 1.0)
    algorithm = LCTVMP(2, 3)
    q = (a = normal(6), W = Wishart(4.0, Matrix(1.0I, 2, 2)))
    valid = (y = (nothing, s), x = (s, s, s))
    for m in ((y = (nothing,), x = (s, s, s)), (y = (nothing, s, s), x = (s, s, s)), (y = (nothing, s), x = (s, s)), (y = (nothing, s), x = (s, s, s, s)))
        @test_throws DimensionMismatch getresult(call_message_update_rule(ScalarCT, (:y, 1); algorithm, m, q))
    end
    for m in ((x = (s, s, s),), (y = (nothing, nothing), x = (s, s, s)), (y = (s, s), x = (s, s, s)), (y = (nothing, s), x = (normal(1), s, s)))
        @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, (:y, 1); algorithm, m, q))
    end
    for k in (0, -1, 3)
        @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, (:y, k); algorithm, m = valid, q))
    end
    for malformed in ((a = normal(5), W = q.W), (a = q.a, W = Wishart(5.0, Matrix(1.0I, 3, 3))))
        @test_throws DimensionMismatch getresult(call_message_update_rule(ScalarCT, (:y, 1); algorithm, m = valid, q = malformed))
    end
    @test_throws DimensionMismatch getresult(call_message_update_rule(ScalarCT, :a; algorithm, clusters = ((:y, :x) => normal(4),), q = (W = q.W,)))
    @test_throws DimensionMismatch getresult(call_message_update_rule(ScalarCT, :W; algorithm, clusters = ((:y,) => normal(2), (:x,) => normal(2)), q = (a = q.a,)))
    @test_throws DimensionMismatch getresult(call_average_energy(ScalarCT; algorithm, clusters = ((:y,) => normal(1), (:x,) => normal(3)), q))
    @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, :a; algorithm, q = (y = (s, s), x = (s, s, s), W = q.W)))
    @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, :a; algorithm, clusters = ((:y, (:x, 1)) => normal(3),), q = (W = q.W,)))
    @test_throws ArgumentError getresult(call_message_update_rule(ScalarCT, :a; algorithm, clusters = ((:y, :x) => normal(5), (:x,) => normal(3)), q = (W = q.W,)))
    @test_throws ArgumentError getresult(call_marginal_update_rule(ScalarCT, (:y,); algorithm, m = (y = (s, s),), q))
end

@testitem "rules:LinearContinuousTransition:reshape equivalence" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    rng = StableRNG(194)
    function spd(T, d)
        L = randn(rng, T, d, d)
        return L * L' / T(d) + Matrix{T}(I, d, d)
    end

    for T in (Float32, Float64), (n, m) in ((1, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)), scale in (T(0.15), T(1.0e-10))
        @testset "$T, ($n, $m), covariance scale $scale" begin
            algorithm = LCTVMP(n, m)
            reference = CTVMP(a -> reshape(a, n, m))
            qy = MvNormalMeanCovariance(randn(rng, T, n), spd(T, n))
            qx = MvNormalMeanCovariance(randn(rng, T, m), spd(T, m))
            # A dense covariance, not restricted to a Kronecker product.
            qa = MvNormalMeanCovariance(randn(rng, T, n * m), scale * spd(T, n * m))
            qW = Wishart(T(n + 3), spd(T, n) / T(n + 3))
            qyx = MvNormalMeanCovariance([mean(qy); mean(qx)], spd(T, n + m))
            tol = T === Float32 ? (atol = 1.0e-5, rtol = 1.0e-4) : (atol = 1.0e-11, rtol = 1.0e-10)

            cases = (
                (:y, (m = (x = qx,), q = (a = qa, W = qW))),
                (:y, (q = (x = qx, a = qa, W = qW),)),
                (:x, (m = (y = qy,), q = (a = qa, W = qW))),
                (:x, (q = (y = qy, a = qa, W = qW),)),
                (:a, (clusters = ((:y, :x) => qyx,), q = (W = qW,))),
                (:a, (q = (y = qy, x = qx, W = qW),)),
                (:W, (clusters = ((:y, :x) => qyx,), q = (a = qa,))),
                (:W, (q = (y = qy, x = qx, a = qa),)),
            )
            for (target, inputs) in cases
                # Only the reference needs q(a) as an expansion point for the a message.
                refinputs = target === :a ? merge(inputs, (q = merge(inputs.q, (a = qa,)),)) : inputs
                expected = getresult(call_message_update_rule(ContinuousTransition, target; algorithm = reference, refinputs...))
                test_message_update_rule(LinearContinuousTransition, target; algorithm, cases = [inputs => expected], check_type_promotion = false, tol...)
            end
            inputs = (m = (y = qy, x = qx), q = (a = qa, W = qW))
            expected = getresult(call_marginal_update_rule(ContinuousTransition, (:y, :x); algorithm = reference, inputs...))
            test_marginal_update_rule(LinearContinuousTransition, (:y, :x); algorithm, cases = [inputs => expected], check_type_promotion = false, tol...)
            for inputs in ((clusters = ((:y, :x) => qyx,), q = (a = qa, W = qW)), (q = (y = qy, x = qx, a = qa, W = qW),))
                expected = getresult(call_average_energy(ContinuousTransition; algorithm = reference, inputs...))
                test_average_energy(LinearContinuousTransition; algorithm, cases = [inputs => expected], check_type_promotion = false, tol...)
                @test getresult(call_average_energy(LinearContinuousTransition; algorithm, inputs...)) isa T
            end
        end
    end
end

@testitem "rules:LinearContinuousTransition:promotion and Gaussian representations" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    algorithm = LCTVMP(2, 3)
    qy = MvNormalMeanPrecision([1.0, -0.5], [2.0 0.3; 0.3 1.0])
    qx = MvNormalWeightedMeanPrecision([0.2, -0.5, 0.7], [1.0 0.2 0.1; 0.2 2.0 0.3; 0.1 0.3 1.5])
    qa = MvNormalMeanPrecision(collect(0.1:0.1:0.6), Matrix(2.0I, 6, 6))
    qW = Wishart(4.0, [0.5 0.1; 0.1 0.3])
    qyx = MvNormalWeightedMeanPrecision([1.0, -0.5, 0.2, -0.5, 0.7], Matrix(1.0I, 5, 5) + fill(0.1, 5, 5))
    opts = (algorithm = algorithm, float_types = (Float32, Float64), atol = 1.0e-5, rtol = 1.0e-4)
    for (target, inputs) in (
            (:y, (m = (x = qx,), q = (a = qa, W = qW))),
            (:y, (q = (x = qx, a = qa, W = qW),)),
            (:x, (m = (y = qy,), q = (a = qa, W = qW))),
            (:x, (q = (y = qy, a = qa, W = qW),)),
            (:a, (clusters = ((:y, :x) => qyx,), q = (W = qW,))),
            (:a, (q = (y = PointMass(mean(qy)), x = qx, W = qW),)),
            (:W, (clusters = ((:y, :x) => qyx,), q = (a = qa,))),
            (:W, (q = (y = qy, x = qx, a = qa),)),
        )
        message_expected = getresult(call_message_update_rule(LinearContinuousTransition, target; algorithm, inputs...))
        test_message_update_rule(LinearContinuousTransition, target; cases = [inputs => message_expected], opts...)
    end
    inputs = (m = (y = qy, x = qx), q = (a = qa, W = qW))
    expected = getresult(call_marginal_update_rule(LinearContinuousTransition, (:y, :x); algorithm, inputs...))
    test_marginal_update_rule(LinearContinuousTransition, (:y, :x); cases = [inputs => expected], opts...)
    for inputs in ((clusters = ((:y, :x) => qyx,), q = (a = qa, W = qW)), (q = (y = qy, x = qx, a = qa, W = qW),))
        energy_expected = getresult(call_average_energy(LinearContinuousTransition; algorithm, inputs...))
        test_average_energy(LinearContinuousTransition; cases = [inputs => energy_expected], opts...)
    end
end

@testitem "rules:LinearContinuousTransition:scratch ownership and allocations" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase
    using BayesBase, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: Target, ClusterTarget, RuleContext, NoAnnotations,
        find_message_rule, find_marginal_rule, rule_scratch, execute_rule

    function inputs(scale)
        normal(d) = MvNormalMeanCovariance(fill(scale, d), Matrix(1.0I, d, d) + fill(0.1 * scale, d, d))
        qy, qx, qa, qyx = normal(2), normal(3), normal(6), normal(5)
        qW = Wishart(4.0, Matrix(scale * I, 2, 2))
        return (
            (Target(:y), (m = (x = qx,), q = (a = qa, W = qW))),
            (Target(:x), (m = (y = qy,), q = (a = qa, W = qW))),
            (Target(:x), (q = (y = qy, a = qa, W = qW),)),
            (Target(:a), (clusters = ((:y, :x) => qyx,), q = (W = qW,))),
            (Target(:a), (q = (y = qy, x = qx, W = qW),)),
            (Target(:W), (clusters = ((:y, :x) => qyx,), q = (a = qa,))),
            (Target(:W), (q = (y = qy, x = qx, a = qa),)),
            (ClusterTarget((:y, :x)), (m = (y = qy, x = qx), q = (a = qa, W = qW))),
        )
    end
    ruleargs(inputs) = MessagePassingRulesBase.interactive_args(get(inputs, :m, (;)), get(inputs, :q, (;)), get(inputs, :clusters, ()))
    run(spec, scratch, algorithm, args, target) = execute_rule(spec, nothing, scratch, algorithm, RuleContext(), args, NoAnnotations(), target)
    allocated(spec, scratch, algorithm, args, target) = @allocated run(spec, scratch, algorithm, args, target)
    algorithm = LCTVMP(2, 3)
    for ((target, first_inputs), (_, next_inputs)) in zip(inputs(0.5), inputs(1.5))
        args, nextargs = ruleargs(first_inputs), ruleargs(next_inputs)
        spec = target isa ClusterTarget ? find_marginal_rule(LinearContinuousTransition, target, algorithm, args) :
            find_message_rule(LinearContinuousTransition, target, algorithm, args)
        scratch = rule_scratch(spec, algorithm, RuleContext(), args, target)
        @test scratch !== nothing
        @test spec.pure
        first_result = run(spec, scratch, algorithm, args, target)
        saved = deepcopy(params(first_result))
        next_result = run(spec, scratch, algorithm, nextargs, target)
        expected = run(spec, nothing, algorithm, nextargs, target)
        @test all(isapprox.(params(next_result), params(expected)))
        @test isequal(params(first_result), saved)
        # A second stream gets independent scratch even with the same algorithm instance.
        other = rule_scratch(spec, algorithm, RuleContext(), args, target)
        @test all(!Base.mightalias(a, b) for a in values(scratch), b in values(other))
        # Poisoning all scratch must not modify a previously returned result either.
        foreach(A -> fill!(A, NaN), values(scratch))
        @test isequal(params(first_result), saved)
        again = run(spec, scratch, algorithm, args, target)
        @test all(isapprox.(params(again), saved))
        allocated(spec, nothing, algorithm, args, target)
        allocated(spec, scratch, algorithm, args, target)
        fresh = allocated(spec, nothing, algorithm, args, target)
        reused = allocated(spec, scratch, algorithm, args, target)
        @test fresh - reused >= sum(sizeof, values(scratch))
    end
end

@testitem "rules:LinearContinuousTransition:independent identities" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    algorithm = LCTVMP(2, 3)
    qy = MvNormalMeanCovariance([1.0, -0.5], [0.6 0.1; 0.1 0.4])
    qx = MvNormalMeanCovariance([0.3, -0.2, 0.7], [1.0 0.2 0.1; 0.2 0.7 0.3; 0.1 0.3 0.8])
    qW = Wishart(4.0, [0.5 0.1; 0.1 0.3])
    qa = MvNormalMeanCovariance(collect(0.1:0.1:0.6), Matrix(0.2I, 6, 6) + fill(0.03, 6, 6))
    Vyx = [0.1 -0.05 0.02; 0.04 0.08 -0.1]
    qyx = MvNormalMeanCovariance([mean(qy); mean(qx)], [cov(qy) Vyx; Vyx' cov(qx)])

    for (inputs, cross) in (((clusters = ((:y, :x) => qyx,), q = (W = qW,)), Vyx), ((q = (y = qy, x = qx, W = qW),), zeros(2, 3)))
        ξ, Λ = weightedmean_precision(getresult(call_message_update_rule(LinearContinuousTransition, :a; algorithm, inputs...)))
        @test ξ ≈ vec(mean(qW) * (cross + mean(qy) * mean(qx)'))
        @test Λ ≈ kron(cov(qx) + mean(qx) * mean(qx)', mean(qW))
    end

    joint = getresult(call_marginal_update_rule(LinearContinuousTransition, (:y, :x); algorithm, m = (y = qy, x = qx), q = (a = qa, W = qW)))
    for (target, input, incoming, block) in ((:y, (x = qx,), qy, 1:2), (:x, (y = qy,), qx, 3:5))
        message = getresult(call_message_update_rule(LinearContinuousTransition, target; algorithm, m = input, q = (a = qa, W = qW)))
        marginal = prod(GenericProd(), incoming, message)
        @test mean(marginal) ≈ mean(joint)[block] rtol = 1.0e-10
        @test cov(marginal) ≈ cov(joint)[block, block] rtol = 1.0e-10
    end
    # With no cross-covariance, the structured energy equals the mean-field energy.
    independent = MvNormalMeanCovariance([mean(qy); mean(qx)], [cov(qy) zeros(2, 3); zeros(3, 2) cov(qx)])
    @test getresult(call_average_energy(LinearContinuousTransition; algorithm, clusters = ((:y, :x) => independent,), q = (a = qa, W = qW))) ≈
        getresult(call_average_energy(LinearContinuousTransition; algorithm, q = (y = qy, x = qx, a = qa, W = qW)))
end

@testitem "rules:LinearContinuousTransition:dimensions" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, LinearAlgebra

    normal(d) = MvNormalMeanCovariance(zeros(d), Matrix(1.0I, d, d))
    noise(d) = Wishart(d + 2.0, Matrix(1.0I, d, d))
    algorithm = LCTVMP(2, 3)
    for (target, inputs) in (
            (:y, (q = (x = normal(3), a = normal(5), W = noise(2)),)),
            (:y, (q = (x = normal(2), a = normal(6), W = noise(2)),)),
            (:y, (m = (x = normal(2),), q = (a = normal(6), W = noise(2)))),
            (:y, (q = (x = normal(3), a = normal(6), W = noise(3)),)),
            (:x, (m = (y = normal(3),), q = (a = normal(6), W = noise(2)))),
            (:a, (q = (y = normal(2), x = normal(2), W = noise(2)),)),
            (:W, (clusters = ((:y, :x) => normal(4),), q = (a = normal(6),))),
            (:W, (q = (y = normal(3), x = normal(2), a = normal(6)),)),
        )
        @test_throws DimensionMismatch getresult(call_message_update_rule(LinearContinuousTransition, target; algorithm, inputs...))
    end
    @test_throws DimensionMismatch getresult(call_marginal_update_rule(LinearContinuousTransition, (:y, :x); algorithm, m = (y = normal(3), x = normal(2)), q = (a = normal(6), W = noise(2))))
    @test_throws DimensionMismatch getresult(call_average_energy(LinearContinuousTransition; algorithm, clusters = ((:y, :x) => normal(4),), q = (a = normal(6), W = noise(2))))
end

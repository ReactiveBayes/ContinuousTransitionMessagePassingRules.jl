# Run from the repository root; setup instructions are in README.md.
using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
using BenchmarkTools, LinearAlgebra, Random, Printf
using MessagePassingRulesBase: Target, ClusterTarget, RuleContext, NoAnnotations,
    find_message_rule, find_marginal_rule, find_average_energy, rule_scratch, execute_rule

# Resolve once and reuse scratch, as an inference engine does. Setting reuse=false measures
# the same execution path with fresh scratch on each call, isolating its allocation cost.
function prepared(node, algorithm, kind, edge, inputs; reuse = true)
    args = MessagePassingRulesBase.interactive_args(get(inputs, :m, (;)), get(inputs, :q, (;)), get(inputs, :clusters, ()))
    target = kind === :message ? Target(edge) : kind === :marginal ? ClusterTarget(edge) : nothing
    spec = kind === :message ? find_message_rule(node, target, algorithm, args) :
        kind === :marginal ? find_marginal_rule(node, target, algorithm, args) : find_average_energy(node, algorithm, args)
    scratch = reuse ? rule_scratch(spec, algorithm, RuleContext(), args, target) : nothing
    return () -> execute_rule(spec, nothing, scratch, algorithm, RuleContext(), args, NoAnnotations(), target)
end

function measure(f)
    f() # exclude compilation
    return minimum(@benchmark $f() seconds = 0.2 samples = 100 evals = 1)
end

function benchmark()
    BLAS.set_num_threads(1)
    rng = MersenneTwister(42)
    spd(d) = (L = randn(rng, d, d); L * L' / d + I)
    println("Julia ", VERSION, "; BLAS threads: ", BLAS.get_num_threads())
    println("Minimum warmed time; bytes include fresh returned parameters. No timing assertions.")
    println("shape  rule       CTVMP μs  LCTVMP μs  speedup  CTVMP bytes  LCTVMP bytes  scratch saved")
    for (n, m) in ((2, 2), (2, 3), (5, 5), (10, 10))
        qy = MvNormalMeanCovariance(randn(rng, n), spd(n))
        qx = MvNormalMeanCovariance(randn(rng, m), spd(m))
        qa = MvNormalMeanCovariance(randn(rng, n * m), 0.1 * spd(n * m))
        qW = Wishart(n + 3.0, spd(n) / (n + 3))
        qyx = MvNormalMeanCovariance([mean(qy); mean(qx)], spd(n + m))
        for (label, kind, edge, inputs) in (
                ("y struct", :message, :y, (m = (x = qx,), q = (a = qa, W = qW))),
                ("y mf", :message, :y, (q = (x = qx, a = qa, W = qW),)),
                ("x struct", :message, :x, (m = (y = qy,), q = (a = qa, W = qW))),
                ("x mf", :message, :x, (q = (y = qy, a = qa, W = qW),)),
                ("a struct", :message, :a, (clusters = ((:y, :x) => qyx,), q = (W = qW,))),
                ("a mf", :message, :a, (q = (y = qy, x = qx, W = qW),)),
                ("W struct", :message, :W, (clusters = ((:y, :x) => qyx,), q = (a = qa,))),
                ("W mf", :message, :W, (q = (y = qy, x = qx, a = qa),)),
                ("joint", :marginal, (:y, :x), (m = (y = qy, x = qx), q = (a = qa, W = qW))),
                ("E struct", :energy, nothing, (clusters = ((:y, :x) => qyx,), q = (a = qa, W = qW))),
                ("E mf", :energy, nothing, (q = (y = qy, x = qx, a = qa, W = qW),)),
            )
            refinputs = edge === :a ? merge(inputs, (q = merge(inputs.q, (a = qa,)),)) : inputs
            old = measure(prepared(ContinuousTransition, CTVMP(a -> reshape(a, n, m)), kind, edge, refinputs))
            new = measure(prepared(LinearContinuousTransition, LCTVMP(n, m), kind, edge, inputs))
            fresh = measure(prepared(LinearContinuousTransition, LCTVMP(n, m), kind, edge, inputs; reuse = false))
            @printf("%2d×%-2d  %-9s  %8.2f  %9.2f  %6.1fx  %11d  %12d  %13d\n", n, m, label, old.time / 1000, new.time / 1000, old.time / new.time, old.memory, new.memory, fresh.memory - new.memory)
        end
    end
    return
end

benchmark()

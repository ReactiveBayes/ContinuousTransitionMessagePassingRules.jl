# Run from the repository root; setup instructions are in README.md.
using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
using BenchmarkTools, LinearAlgebra, Random, Printf
using MessagePassingRulesBase: Target, IndexedTarget, ClusterTarget, RuleContext, NoAnnotations,
    find_message_rule, find_marginal_rule, find_average_energy, rule_scratch, execute_rule

# Resolve once and reuse scratch, as an inference engine does. Setting reuse=false measures
# the same execution path with fresh scratch on each call, isolating its allocation cost.
function prepared(node, algorithm, kind, edge, inputs; reuse = true)
    args = MessagePassingRulesBase.interactive_args(get(inputs, :m, (;)), get(inputs, :q, (;)), get(inputs, :clusters, ()))
    target = kind === :message ? (edge isa Tuple ? IndexedTarget(edge...) : Target(edge)) : kind === :marginal ? ClusterTarget(edge) : nothing
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

function scalar_benchmark()
    BLAS.set_num_threads(1)
    rng = MersenneTwister(73)
    spd(d) = (L = randn(rng, d, d); L * L' / d + I)
    println("\nAll x/y ports once, with fixed incoming messages and marginals.")
    println("Block reference includes conversion to equivalent scalar outgoing messages.")
    println("shape  mode         block μs  scalar μs  block bytes  scalar bytes  scratch saved")
    for (n, m) in ((1, 1), (2, 3), (5, 5), (10, 10))
        algorithm = LCTVMP(n, m)
        my, mx = ntuple(_ -> NormalMeanVariance(randn(rng), 1.0), n), ntuple(_ -> NormalMeanVariance(randn(rng), 1.0), m)
        iy, ix = MvNormalMeanCovariance(collect(mean.(my)), Matrix(1.0I, n, n)), MvNormalMeanCovariance(collect(mean.(mx)), Matrix(1.0I, m, m))
        qy, qx = MvNormalMeanCovariance(randn(rng, n), spd(n)), MvNormalMeanCovariance(randn(rng, m), spd(m))
        q = (a = MvNormalMeanCovariance(randn(rng, n * m), 0.1 * spd(n * m)), W = Wishart(n + 3.0, spd(n) / (n + 3)))
        for structured in (false, true)
            scalar_updates, fresh_updates, block_updates = [], [], []
            for (group, incoming, opposite, oppositeq, oppositein) in ((:y, my, :x, qx, ix), (:x, mx, :y, qy, iy))
                for k in eachindex(incoming)
                    excluded = ntuple(i -> i == k ? nothing : incoming[i], length(incoming))
                    messages = structured ? (group === :y ? (y = excluded, x = mx) : (y = my, x = excluded)) : NamedTuple{(group,)}((excluded,))
                    inputs = (m = messages, q = q, clusters = structured ? () : ((opposite,) => oppositeq,))
                    push!(scalar_updates, prepared(ScalarCT, algorithm, :message, (group, k), inputs))
                    push!(fresh_updates, prepared(ScalarCT, algorithm, :message, (group, k), inputs; reuse = false))
                end
                inputs = structured ? (m = NamedTuple{(opposite,)}((oppositein,)), q = q) : (q = merge(q, NamedTuple{(opposite,)}((oppositeq,))),)
                push!(block_updates, prepared(LinearContinuousTransition, algorithm, :message, group, inputs))
            end
            scalar = () -> map(f -> f(), scalar_updates)
            fresh = () -> map(f -> f(), fresh_updates)
            function blocks()
                return map(zip(block_updates, (iy, ix), (my, mx))) do (update, incoming, scalars)
                    posterior = prod(GenericProd(), update(), incoming)
                    μ, V = mean_cov(posterior)
                    map(eachindex(scalars)) do k
                        ξ, λ = weightedmean_precision(scalars[k])
                        NormalWeightedMeanPrecision(μ[k] / V[k, k] - ξ, inv(V[k, k]) - λ)
                    end
                end
            end
            # Keep the measured workloads numerically equivalent.
            reference, actual = vcat(blocks()...), scalar()
            @assert all(isapprox(mean(a), mean(b); rtol = 1.0e-9, atol = 1.0e-9) && isapprox(var(a), var(b); rtol = 1.0e-9) for (a, b) in zip(actual, reference))
            b, s, f = measure(blocks), measure(scalar), measure(fresh)
            @printf("%2d×%-2d  %-10s  %8.2f  %9.2f  %11d  %12d  %13d\n", n, m, structured ? "structured" : "separate", b.time / 1000, s.time / 1000, b.memory, s.memory, f.memory - s.memory)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark()
    scalar_benchmark()
end

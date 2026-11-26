# Before/after measurements of Conv2d solves, cache hits and complete node updates.
# Run: julia --project=benchmark benchmark/conv2d_cache.jl
# To measure another checkout with the same driver, prepend it to JULIA_LOAD_PATH.
include("linear.jl") # Reuse prepared rule dispatch and the benchmark environment.

function measure_conv(f)
    f()
    result = minimum(@benchmark $f() seconds = 0.2 samples = 30 evals = 1)
    return result.time / 1.0e3, result.memory
end

function conv_cache_benchmark(; configurations = ((7, 1, 1), (14, 1, 1), (28, 1, 1), (7, 2, 2)))
    BLAS.set_num_threads(1)
    rng = MersenneTwister(517)
    println("Julia ", VERSION, "; CPU: ", Sys.CPU_NAME, "; BLAS threads: ", BLAS.get_num_threads())
    println("Package: ", pathof(ContinuousTransitionMessagePassingRules))
    println("Minimum warmed time; complete node = marginal, all outgoing state ports, a, W, energy.")
    println("Individual rules use prepared dispatch; complete node includes public rule lookup and argument packing.")
    println("side cin cout representation input    operation                      us      bytes")
    for (side, cin, cout) in configurations, representation in (:covariance, :precision), input in (:latent, :observed)
        algo = Conv2dVMP((side, side, cin), cout, (3, 3))
        n, m, ny, nx = algo.n, algo.m, algo.ny, algo.nx
        d = n * m
        L = randn(rng, d, d)
        qa0 = MvNormalMeanCovariance(randn(rng, d), Matrix(0.2I, d, d) + 0.01 * L * L')
        qa = representation === :covariance ? qa0 : convert(MvNormalWeightedMeanPrecision, qa0)
        q = (a = qa, W = PointMass(Matrix(2.0I, n, n)))
        y = ntuple(_ -> NormalMeanVariance(randn(rng), 0.1), ny)
        x = ntuple(_ -> NormalMeanVariance(randn(rng), 0.3), nx)
        pixels = ntuple(i -> PointMass(mean(x[i])), nx)
        target = input === :latent ? (:y, :x) : (:y,)
        qs = input === :latent ? q : merge(q, (x = pixels,))
        incoming = input === :latent ? (y = y, x = x) : (y = y,)
        marginal = prepared(Conv2dCT, algo, :marginal, target, (m = incoming, q = qs))
        joint = marginal()
        port = prepared(Conv2dCT, algo, :message, (:y, 1), (clusters = (target => joint,),))
        excluded = ntuple(i -> i == 1 ? nothing : y[i], ny)
        messages = input === :latent ? (y = excluded, x = x) : (y = excluded,)
        cached = prepared(Conv2dCT, algo, :message, (:y, 1), (m = messages, q = qs))
        function node()
            state = getresult(call_marginal_update_rule(Conv2dCT, target; algorithm = algo, m = incoming, q = qs))
            clusters = (target => state,)
            for k in 1:ny
                getresult(call_message_update_rule(Conv2dCT, (:y, k); algorithm = algo, clusters))
            end
            if input === :latent
                for k in 1:nx
                    getresult(call_message_update_rule(Conv2dCT, (:x, k); algorithm = algo, clusters))
                end
            end
            getresult(call_message_update_rule(Conv2dCT, :a; algorithm = algo, clusters, q = (W = q.W,)))
            getresult(call_message_update_rule(Conv2dCT, :W; algorithm = algo, clusters, q = (a = qa,)))
            return getresult(call_average_energy(Conv2dCT; algorithm = algo, clusters, q))
        end
        function changed_mean()
            first(BayesBase.params(qa))[1] = -first(BayesBase.params(qa))[1]
            return cached()
        end
        scale, toggle = last(BayesBase.params(qa))[1, 1], Ref(false)
        function changed_matrix()
            toggle[] = !toggle[]
            last(BayesBase.params(qa))[1, 1] = scale * (toggle[] ? 1.01 : 1.0)
            return cached()
        end
        for (label, f) in (("marginal", marginal), ("port from marginal", port), ("default cache hit", cached), ("complete node", node), ("default changed mean", changed_mean), ("default changed matrix", changed_matrix))
            time, bytes = measure_conv(f)
            @printf("%4d %3d %4d %-14s %-8s %-25s %10.3f %10d\n", side, cin, cout, representation, input, label, time, bytes)
            flush(stdout)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    conv_cache_benchmark()
end

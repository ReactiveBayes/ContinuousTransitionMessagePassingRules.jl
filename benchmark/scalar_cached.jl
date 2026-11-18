# Run with julia --project=benchmark benchmark/scalar_cached.jl.
include("linear.jl")

function cached_sweep(algorithm, n, m, structured, changing; natural = false)
    rng = MersenneTwister(92)
    spd(d) = (L = randn(rng, d, d); L * L' / d + I)
    qa = MvNormalMeanCovariance(randn(rng, n * m), 0.1 * spd(n * m))
    natural && (qa = convert(MvNormalWeightedMeanPrecision, qa))
    q = (a = qa, W = Wishart(n + 3.0, spd(n) / (n + 3)))
    qy, qx = MvNormalMeanCovariance(randn(rng, n), spd(n)), MvNormalMeanCovariance(randn(rng, m), spd(m))
    qyx = MvNormalMeanCovariance(randn(rng, n + m), spd(n + m))
    my = ntuple(i -> NormalWeightedMeanPrecision(i / n, 1.0 + i / n), n)
    mx = ntuple(i -> NormalWeightedMeanPrecision(i / m, 1.0 + i / m), m)
    calls = []
    for (group, incoming, other, otherq) in ((:y, my, :x, qx), (:x, mx, :y, qy)), k in eachindex(incoming)
        excluded = ntuple(i -> i == k ? nothing : incoming[i], length(incoming))
        messages = structured ? (group === :y ? (y = excluded, x = mx) : (y = my, x = excluded)) : NamedTuple{(group,)}((excluded,))
        inputs = (m = messages, q = q, clusters = structured ? () : ((other,) => otherq,))
        push!(calls, prepared(ScalarCT, algorithm, :message, (group, k), inputs))
    end
    clusters = structured ? ((:y, :x) => qyx,) : ((:y,) => qy, (:x,) => qx)
    push!(calls, prepared(ScalarCT, algorithm, :message, :a, (; clusters, q = (W = q.W,))))
    push!(calls, prepared(ScalarCT, algorithm, :message, :W, (; clusters, q = (a = q.a,))))
    push!(calls, prepared(ScalarCT, algorithm, :energy, nothing, (; clusters, q)))
    direction = Ref(1.0)
    return function ()
        if changing
            # Alternate values in the same arrays: caches must detect these edits. Both
            # algorithms see the same changing inputs, with all ports updated each sweep.
            p, P = BayesBase.params(qa)
            p[1] += direction[] * 0.05
            P[1, 1] += direction[] * 0.01
            direction[] = -direction[]
        end
        return map(f -> f(), calls)
    end
end

function cached_benchmark()
    BLAS.set_num_threads(1)
    println("Julia ", VERSION, "; CPU: ", Sys.CPU_NAME, "; BLAS threads: ", BLAS.get_num_threads())
    println("One sweep: every scalar state port, a, W, and average energy. Returned allocations included.")
    println("Minimum warmed timings. Changing inputs alternate q(a)'s parameters once per sweep.")
    println("shape  mode        q(a)        inputs    LCTVMP μs  SLCTVMP μs  speedup  old bytes  cached bytes")
    for (n, m) in ((2, 3), (5, 5), (10, 10)), structured in (false, true), natural in (false, true), changing in (false, true)
        old = cached_sweep(LCTVMP(n, m), n, m, structured, changing; natural)
        new = cached_sweep(SLCTVMP(n, m), n, m, structured, changing; natural)
        for _ in 1:4
            expected, actual = old(), new()
            for (a, b) in zip(expected, actual)
                if a isa Number
                    @assert isapprox(a, b; rtol = 1.0e-9, atol = 1.0e-9)
                else
                    @assert all(isapprox.(BayesBase.params(a), BayesBase.params(b); rtol = 1.0e-9, atol = 1.0e-9))
                end
            end
        end
        o, c = measure(old), measure(new)
        @printf("%2d×%-2d  %-10s  %-10s  %-8s  %9.2f  %10.2f  %6.1fx  %9d  %12d\n", n, m, structured ? "structured" : "separate", natural ? "precision" : "covariance", changing ? "changing" : "fixed", o.time / 1000, c.time / 1000, o.time / c.time, o.memory, c.memory)
    end
    println("\nConstruction excluded above; retained size is the algorithm cache only, not total inference memory.")
    println("shape  construct μs  allocated bytes  retained bytes")
    for (n, m) in ((2, 3), (5, 5), (10, 10))
        cost = measure(() -> SLCTVMP(n, m))
        @printf("%2d×%-2d  %12.2f  %15d  %14d\n", n, m, cost.time / 1000, cost.memory, Base.summarysize(SLCTVMP(n, m)))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cached_benchmark()
end

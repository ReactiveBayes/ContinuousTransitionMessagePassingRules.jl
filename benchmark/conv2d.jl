# Rule by rule: Conv2dCT against the same convolution as one ContinuousTransition under
# CTVMP(a -> the convolution matrix). Run with julia --project=benchmark benchmark/conv2d.jl.
include("linear.jl")

function convolution_matrix(taps, n, ny, nx)
    m, P = size(taps)
    return function (a)
        B = zeros(eltype(a), ny, nx)
        for p in 1:P, j in 1:m
            t = taps[j, p]
            t == 0 && continue
            for c in 1:n
                B[p + (c - 1) * P, t] = a[(j - 1) * n + c]
            end
        end
        return B
    end
end

function diagonal(ms)
    parameters = collect(weightedmean_precision.(ms))
    return MvNormalWeightedMeanPrecision(first.(parameters), Matrix(Diagonal(last.(parameters))))
end

# One measurement, or `nothing` for a reference too large to run.
quick(f) = (f(); minimum(@benchmark $f() seconds = 0.5 samples = 50 evals = 1))

function conv_benchmark(; sides = (4, 7, 14, 28), reference_limit = 14)
    BLAS.set_num_threads(1)
    rng = MersenneTwister(5)
    spd(d) = (L = randn(rng, d, d); L * L' / d + I)
    println("Julia ", VERSION, "; CPU: ", Sys.CPU_NAME, "; BLAS threads: ", BLAS.get_num_threads())
    println("3 × 3 kernel, one channel in and out. Minimum warmed time of one rule call; CTVMP is not run above side ", reference_limit, ".")
    println("One iteration of Conv2dCT is one marginal, one read per port, a, W and the energy; of CTVMP, its y, x, a, W, marginal and energy rules.")
    println("side  ports y/x  rule                 CTVMP ms     Conv ms    speedup   CTVMP bytes   Conv bytes")
    for side in sides
        algorithm = Conv2dVMP((side, side, 1), 1, (3, 3))
        n, m, ny, nx = algorithm.n, algorithm.m, algorithm.ny, algorithm.nx
        ct = CTVMP(convolution_matrix(algorithm.taps, n, ny, nx))
        my = ntuple(_ -> NormalWeightedMeanPrecision(randn(rng), 50.0 + rand(rng)), ny)
        mx = ntuple(_ -> NormalWeightedMeanPrecision(randn(rng), 5.0 + rand(rng)), nx)
        pixels = ntuple(_ -> PointMass(rand(rng)), nx)
        qa = MvNormalMeanCovariance(randn(rng, n * m), 0.1 * spd(n * m))
        W = [4.0;;]
        q, qref = (a = qa, W = PointMass(W)), (a = qa, W = PointMass(Matrix(4.0I, ny, ny)))
        joint = getresult(call_marginal_update_rule(Conv2dCT, (:y, :x); algorithm, m = (y = my, x = mx), q))
        qy = getresult(call_marginal_update_rule(Conv2dCT, (:y,); algorithm, m = (y = my,), q = merge(q, (x = pixels,))))
        referenced = side <= reference_limit
        if referenced
            dy, dx = diagonal(my), diagonal(mx)
            qyx = getresult(call_marginal_update_rule(ContinuousTransition, (:y, :x); algorithm = ct, m = (y = dy, x = dx), q = qref))
            qyx = MvNormalMeanCovariance(mean_cov(qyx)...)
            dense = MvNormalMeanCovariance(collect(mean.(pixels)), zeros(nx, nx))
            densey = MvNormalMeanCovariance(mean(qy), Matrix(Diagonal(var(qy))))
        end
        rows = (
            ("joint q(y,x)", (:marginal, (:y, :x), (m = (y = my, x = mx), q = q)), () -> (:marginal, (:y, :x), (m = (y = dy, x = dx), q = qref))),
            ("port from joint", (:message, (:y, 1), (clusters = ((:y, :x) => joint,),)), () -> (:message, :y, (m = (x = dx,), q = qref))),
            ("a from joint", (:message, :a, (clusters = ((:y, :x) => joint,), q = (W = q.W,))), () -> (:message, :a, (clusters = ((:y, :x) => qyx,), q = qref))),
            ("W from joint", (:message, :W, (clusters = ((:y, :x) => joint,), q = (a = qa,))), () -> (:message, :W, (clusters = ((:y, :x) => qyx,), q = (a = qa,)))),
            ("energy, joint", (:energy, nothing, (clusters = ((:y, :x) => joint,), q = q)), () -> (:energy, nothing, (clusters = ((:y, :x) => qyx,), q = qref))),
            # The reference's posterior of y is its message times the incoming ones, as an engine forms it.
            ("q(y), pixels", (:marginal, (:y,), (m = (y = my,), q = merge(q, (x = pixels,)))), () -> (:message, :y, (q = merge(qref, (x = dense,)),), m -> prod(GenericProd(), convert(MvNormalWeightedMeanPrecision, m), dy))),
            ("a, pixels", (:message, :a, (clusters = ((:y,) => qy,), q = (W = q.W,))), () -> (:message, :a, (q = merge(qref, (y = densey, x = dense)),))),
            ("W, pixels", (:message, :W, (clusters = ((:y,) => qy,), q = (a = qa,))), () -> (:message, :W, (q = (a = qa, y = densey, x = dense),))),
            ("energy, pixels", (:energy, nothing, (clusters = ((:y,) => qy,), q = merge(q, (x = pixels,)))), () -> (:energy, nothing, (q = merge(qref, (y = densey, x = dense)),))),
        )
        for (label, (kind, edge, inputs), reference) in rows
            new = quick(prepared(Conv2dCT, algorithm, kind, edge, inputs))
            if referenced
                rkind, redge, rinputs, after... = reference()
                rule = prepared(ContinuousTransition, ct, rkind, redge, rinputs)
                old = quick(isempty(after) ? rule : () -> only(after)(rule()))
                @printf("%4d  %4d/%-4d  %-17s  %10.3f  %10.4f  %8.0fx  %12d  %11d\n", side, ny, nx, label, old.time / 1.0e6, new.time / 1.0e6, old.time / new.time, old.memory, new.memory)
            else
                @printf("%4d  %4d/%-4d  %-17s  %10s  %10.4f  %9s  %12s  %11d\n", side, ny, nx, label, "-", new.time / 1.0e6, "-", "-", new.memory)
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    conv_benchmark()
end

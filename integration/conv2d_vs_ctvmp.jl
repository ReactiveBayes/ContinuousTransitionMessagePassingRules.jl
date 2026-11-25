# End-to-end RxInfer comparison of Conv2dCT with the same convolution written as one
# ContinuousTransition under CTVMP(a -> the full convolution matrix).
#
# One configuration, in this process (so that the first run includes compilation):
#   julia --project=integration integration/conv2d_vs_ctvmp.jl <conv|ctvmp> <observed|latent> <known|learned> side batch cout stride
# The whole table, one fresh process per configuration:
#   julia --project=integration integration/conv2d_vs_ctvmp.jl all [cap seconds]
using RxInfer, ContinuousTransitionMessagePassingRules, LinearAlgebra, Random, Printf

const KERNEL = (3, 3)
const NOISE = 0.01
const PRIOR_VARIANCE = 0.05 # of a latent pixel, around a noisy copy of the image
const ITERATIONS = parse(Int, get(ENV, "CONV_ITERATIONS", "10"))
# `CONV_PORTS=default` runs Conv2dCT under the engine's default scheme, for comparison.
const PORTS = Ref(Symbol(get(ENV, "CONV_PORTS", "fast")))

# The convolution matrix of the kernel `a`, rows in the order of the node's y ports.
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

@model function conv_observed(x, z, geometry, batch, d, cout, stride, Wknown, ports)
    local y
    ny, nx = geometry.ny, geometry.nx
    a ~ MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
    if Wknown === nothing
        W ~ Wishart(cout + 2, Matrix(1.0I, cout, cout))
    else
        W = Wknown
    end
    for b in 1:batch
        ys, xs = ((b - 1) * ny + 1):(b * ny), ((b - 1) * nx + 1):(b * nx)
        y[ys] ~ Conv2dCT(x = x[xs], a = a, W = W) where {
            algorithm = Conv2dVMP(geometry.insize, cout, KERNEL; stride = stride, ports = ports),
        }
    end
    for i in 1:(batch * ny)
        z[i] ~ NormalMeanVariance(y[i], NOISE)
    end
end

@model function conv_latent(x0, z, geometry, batch, d, cout, stride, Wknown, ports)
    local y, x
    ny, nx = geometry.ny, geometry.nx
    a ~ MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
    if Wknown === nothing
        W ~ Wishart(cout + 2, Matrix(1.0I, cout, cout))
    else
        W = Wknown
    end
    for i in 1:(batch * nx)
        x[i] ~ NormalMeanVariance(x0[i], PRIOR_VARIANCE)
    end
    for b in 1:batch
        ys, xs = ((b - 1) * ny + 1):(b * ny), ((b - 1) * nx + 1):(b * nx)
        y[ys] ~ Conv2dCT(x = x[xs], a = a, W = W) where {
            algorithm = Conv2dVMP(geometry.insize, cout, KERNEL; stride = stride, ports = ports),
        }
    end
    for i in 1:(batch * ny)
        z[i] ~ NormalMeanVariance(y[i], NOISE)
    end
end

@model function ct_observed(x, z, f, ny, d, Wknown)
    a ~ MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
    if Wknown === nothing
        W ~ Wishart(ny + 2, Matrix(1.0I, ny, ny))
    else
        W = Wknown
    end
    for b in eachindex(z)
        y[b] ~ ContinuousTransition(x = x[b], a = a, W = W) where {algorithm = CTVMP(f)}
        z[b] ~ MvNormalMeanCovariance(y[b], Matrix(NOISE * I, ny, ny))
    end
end

@model function ct_latent(x0, z, f, ny, nx, d, Wknown)
    a ~ MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
    if Wknown === nothing
        W ~ Wishart(ny + 2, Matrix(1.0I, ny, ny))
    else
        W = Wknown
    end
    for b in eachindex(z)
        x[b] ~ MvNormalMeanCovariance(x0[b], Matrix(PRIOR_VARIANCE * I, nx, nx))
        y[b] ~ ContinuousTransition(x = x[b], a = a, W = W) where {algorithm = CTVMP(f)}
        z[b] ~ MvNormalMeanCovariance(y[b], Matrix(NOISE * I, ny, ny))
    end
end

function problem(side, batch, cout, stride)
    rng = MersenneTwister(11)
    geometry = Conv2dVMP((side, side, 1), cout, KERNEL; stride)
    n, m, ny, nx = geometry.n, geometry.m, geometry.ny, geometry.nx
    A = randn(rng, n, m)
    f = convolution_matrix(geometry.taps, n, ny, nx)
    B = f(vec(A))
    images = [rand(rng, nx) for _ in 1:batch]
    observed = [B * image .+ sqrt(NOISE) .* randn(rng, ny) for image in images]
    noisy = [image .+ sqrt(PRIOR_VARIANCE) .* randn(rng, nx) for image in images]
    return (; geometry, n, m, ny, nx, d = n * m, A, f, images, observed, noisy)
end

function run(kind, input, noise, side, batch, cout, stride; iterations = ITERATIONS)
    p = problem(side, batch, cout, stride)
    known = noise === :known
    # The true precision of the transition is unknown to the model; a fixed one keeps both
    # models identical, since CTVMP cannot tie a learned W across positions.
    Wsmall = Matrix(4.0I, cout, cout)
    Wbig = Matrix(4.0I, p.ny, p.ny)
    latent = input === :latent
    if kind === :conv
        model = (latent ? conv_latent : conv_observed)(
            geometry = p.geometry, batch = batch, d = p.d, cout = cout, stride = stride, Wknown = known ? Wsmall : nothing,
            ports = PORTS[] === :default ? :default : latent ? :joint : :observed,
        )
        data = latent ? (x0 = reduce(vcat, p.noisy), z = reduce(vcat, p.observed)) : (x = reduce(vcat, p.images), z = reduce(vcat, p.observed))
        Winit = Wishart(cout + 2, Matrix(1.0I, cout, cout))
    else
        model = latent ? ct_latent(f = p.f, ny = p.ny, nx = p.nx, d = p.d, Wknown = known ? Wbig : nothing) :
            ct_observed(f = p.f, ny = p.ny, d = p.d, Wknown = known ? Wbig : nothing)
        data = latent ? (x0 = p.noisy, z = p.observed) : (x = p.images, z = p.observed)
        Winit = Wishart(p.ny + 2, Matrix(1.0I, p.ny, p.ny))
    end
    constraints = if latent
        known ? (
                @constraints begin
                    q(y, x, a) = q(y, x)q(a)
                end
            ) : (
                @constraints begin
                    q(y, x, a, W) = q(y, x)q(a)q(W)
                end
            )
    else
        known ? (
                @constraints begin
                    q(y, a) = q(y)q(a)
                end
            ) : (
                @constraints begin
                    q(y, a, W) = q(y)q(a)q(W)
                end
            )
    end
    qa = MvNormalMeanCovariance(zeros(p.d), Matrix(10.0I, p.d, p.d))
    init = known ? (
            @initialization begin
                q(a) = qa
            end
        ) : (
            @initialization begin
                q(a) = qa
                q(W) = Winit
            end
        )
    once() = infer(
        model = model, data = data, constraints = constraints, initialization = init,
        iterations = iterations, free_energy = true, returnvars = (a = KeepLast(), y = KeepLast()),
    )
    first = @elapsed result = once()
    stats = @timed once()
    # The second call still compiles a little: repeat a fast model and keep the best.
    if stats.time < 2
        for _ in 1:5
            again = @timed once()
            again.time < stats.time && (stats = again)
        end
    end
    estimate = reshape(mean(result.posteriors[:a]), cout, :)
    ymean = kind === :conv ? mean.(result.posteriors[:y]) : reduce(vcat, mean.(result.posteriors[:y]))
    return (;
        first, warm = stats.time, bytes = stats.bytes, free_energy = result.free_energy[end],
        kernel_error = maximum(abs, estimate .- p.A), ysum = sum(abs2, ymean), ports = (p.ny, p.nx),
    )
end

function report(r)
    return @sprintf(
        "RESULT first=%.3f warm=%.4f bytes=%d fe=%.6f kerr=%.5f ysum=%.8f ny=%d nx=%d",
        r.first, r.warm, r.bytes, r.free_energy, r.kernel_error, r.ysum, r.ports...,
    )
end

function versions()
    names = ("RxInfer", "ReactiveMP", "MessagePassingRulesBase", "GraphPPL", "Rocket")
    found = Dict(p.name => p.version for p in values(Base.require(Main, :Pkg).dependencies()) if p.name in names)
    return join(("$name $(found[name])" for name in names if haskey(found, name)), ", ")
end

function child_result(arguments, cap)
    command = `$(Base.julia_cmd()) --project=$(dirname(@__FILE__)) $(@__FILE__) $arguments`
    path = tempname()
    started = time()
    process = Base.run(pipeline(ignorestatus(command); stdout = path, stderr = devnull); wait = false)
    while process_running(process) && time() - started < cap
        sleep(0.25)
    end
    if process_running(process)
        kill(process)
        return nothing, time() - started
    end
    total = time() - started
    lines = filter(l -> startswith(l, "RESULT"), readlines(path))
    isempty(lines) && return nothing, total
    fields = Dict(Pair(split(item, '=')...) for item in split(lines[end])[2:end])
    return Dict(k => parse(Float64, v) for (k, v) in fields), total
end

function table(cap)
    println("Julia ", VERSION, "; ", versions(), "; ", ITERATIONS, " iterations; cap ", cap, " s per process")
    println("process = whole fresh julia process; first = first infer (with compilation); warm = best later infer; seconds")
    println("same = free energy and posterior of y agree to 1e-3 after these iterations (the two schedules differ within an iteration and meet at convergence)")
    println("input     W        side batch cout stride | ports y/x  | CTVMP process first warm   | Conv process first warm    | warm speedup | same")
    # (side, batch, C_out, stride); CTVMP is not started where its Jacobians alone need gigabytes.
    configurations = [(4, 1, 1, 1), (4, 1, 2, 2), (7, 1, 1, 1), (7, 1, 2, 2), (14, 1, 1, 1), (14, 4, 1, 1), (14, 1, 2, 2), (28, 1, 1, 1)]
    for input in ("observed", "latent"), noise in ("known", "learned"), (side, batch, cout, stride) in configurations
        arguments(kind) = [kind, input, noise, string(side), string(batch), string(cout), string(stride)]
        conv, convtotal = child_result(arguments("conv"), cap)
        ct, cttotal = side > 14 ? (nothing, Inf) : child_result(arguments("ctvmp"), cap)
        show3(r, total) = r === nothing ? @sprintf("%7s %6s %7s", total == Inf ? "skipped" : total >= cap ? ">cap" : "failed", "-", "-") : @sprintf("%7.1f %6.2f %7.4f", total, r["first"], r["warm"])
        ports = conv === nothing ? "?" : @sprintf("%d/%d", conv["ny"], conv["nx"])
        speedup = conv === nothing || ct === nothing ? "      -" : @sprintf("%6.1fx", ct["warm"] / conv["warm"])
        same = conv === nothing || ct === nothing ? "-" :
            noise == "known" ? string(isapprox(conv["fe"], ct["fe"]; rtol = 1.0e-3) && isapprox(conv["ysum"], ct["ysum"]; rtol = 1.0e-3)) : "n/a"
        @printf("%-9s %-8s %4d %5d %4d %6d | %-10s | %s | %s | %s | %s\n", input, noise, side, batch, cout, stride, ports, show3(ct, cttotal), show3(conv, convtotal), speedup, same)
        flush(stdout)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS) || ARGS[1] == "all"
        table(length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 900.0)
    else
        kind, input, noise = Symbol.(ARGS[1:3])
        println(report(run(kind, input, noise, parse.(Int, ARGS[4:7])...)))
    end
end

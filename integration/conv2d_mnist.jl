# Learn a convolution kernel from a batch of images with RxInfer.
# Run with: julia -O0 --project=integration integration/conv2d_mnist.jl [size] [batch] [C_out] [picture.png]
# `size` is the image side: 28 for MNIST as is, 14 or 7 for average-pooled MNIST, and any
# other value for random images of that side.
using RxInfer, ContinuousTransitionMessagePassingRules, LinearAlgebra, Random
import Plots

function images(side, batch, rng)
    side in (7, 14, 28) || return [rand(rng, side, side, 1) for _ in 1:batch]
    MNIST = Base.require(Main, :MLDatasets).MNIST
    data = Base.invokelatest(() -> Float64.(MNIST(split = :train).features[:, :, 1:batch]))
    f = 28 ÷ side
    pool(img) = [sum(img[((i - 1) * f + 1):(i * f), ((j - 1) * f + 1):(j * f)]) / f^2 for i in 1:side, j in 1:side, _ in 1:1]
    # MLDatasets stores MNIST as width × height: transpose to see upright digits.
    return [pool(permutedims(data[:, :, b])) for b in 1:batch]
end

# The reference convolution, written without the node's index tables.
function convolve(image, A, kernel)
    H, W, _ = size(image)
    out = zeros(H - kernel[1] + 1, W - kernel[2] + 1, size(A, 1))
    for o2 in axes(out, 2), o1 in axes(out, 1)
        out[o1, o2, :] = A * vec(image[o1:(o1 + kernel[1] - 1), o2:(o2 + kernel[2] - 1), :])
    end
    return out
end

@model function conv2d_batch(x, z, insize, cout, kernel, batch, noise)
    local y
    template = Conv2dVMP(insize, cout, kernel)
    ny, nx, d = template.ny, template.nx, template.n * template.m
    a ~ MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
    W ~ Wishart(cout + 2, Matrix(1.0I, cout, cout))
    for b in 1:batch
        ys, xs = ((b - 1) * ny + 1):(b * ny), ((b - 1) * nx + 1):(b * nx)
        y[ys] ~ Conv2dCT(x = x[xs], a = a, W = W) where {
            algorithm = Conv2dVMP(insize, cout, kernel), # A fresh cache for every image.
        }
    end
    for i in 1:(batch * ny)
        z[i] ~ NormalMeanVariance(y[i], noise)
    end
end

# One row per image: the input, then for every output channel the exact convolution, the
# noisy observation z and the posterior mean of the node's y ports.
function picture(path, batch_images, outputs, observed, inferred)
    rows, cout = min(length(batch_images), 4), size(first(outputs), 3)
    panel(data, title) = Plots.heatmap(
        data; yflip = true, aspect_ratio = :equal, color = :grays, axis = false, ticks = false,
        colorbar = false, title = title, titlefontsize = 8,
    )
    panels = []
    for b in 1:rows
        push!(panels, panel(batch_images[b][:, :, 1], "input x, image $b"))
        for c in 1:cout
            push!(panels, panel(outputs[b][:, :, c], "conv(x), channel $c"))
            push!(panels, panel(observed[b][:, :, c], "observed z"))
            push!(panels, panel(inferred[b][:, :, c], "posterior mean of y"))
        end
    end
    columns = 1 + 3 * cout
    Plots.savefig(Plots.plot(panels...; layout = (rows, columns), size = (190 * columns, 200 * rows)), path)
    return path
end

function main(side = 4, batch = 2, cout = 1, path = nothing; kernel = (3, 3), noise = 0.01, iterations = 15)
    rng = MersenneTwister(7)
    batch_images = images(side, batch, rng)
    insize = size(first(batch_images))
    d = cout * prod(kernel) * insize[3]
    A = randn(rng, cout, d ÷ cout)
    if kernel == (3, 3) && insize[3] == 1
        # Recognisable filters for the first channels: horizontal and vertical edges.
        sobel = [-1.0 0.0 1.0; -2.0 0.0 2.0; -1.0 0.0 1.0]
        A[1, :] = vec(sobel)
        cout >= 2 && (A[2, :] = vec(sobel'))
    end
    outputs = [convolve(img, A, kernel) for img in batch_images]
    x = reduce(vcat, vec.(batch_images))
    z = reduce(vcat, vec.(outputs)) .+ sqrt(noise) .* randn(rng, sum(length, outputs))
    constraints = @constraints begin
        q(y, a, W) = q(y)q(a)q(W)
    end
    init = @initialization begin
        q(a) = MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
        q(W) = Wishart(cout + 2, Matrix(1.0I, cout, cout))
    end
    println("image ", insize, ", batch ", batch, ", ", cout, " output channels: ", length(x) ÷ batch, " input and ", length(z) ÷ batch, " output ports per node")
    elapsed = @elapsed result = infer(
        model = conv2d_batch(insize = insize, cout = cout, kernel = kernel, batch = batch, noise = noise),
        data = (x = x, z = z), constraints = constraints, initialization = init,
        iterations = iterations, free_energy = true, returnvars = (a = KeepLast(), W = KeepLast(), y = KeepLast()),
    )
    estimate = reshape(mean(result.posteriors[:a]), cout, :)
    err = maximum(abs, estimate .- A)
    println("inference took ", round(elapsed; digits = 1), " s; free energy ", round.(result.free_energy; digits = 2))
    println("largest kernel error ", round(err; sigdigits = 3), " (kernel entries up to ", round(maximum(abs, A); digits = 2), ")")
    if path !== nothing
        shape = size(first(outputs))
        split(v) = [reshape(v[((b - 1) * prod(shape) + 1):(b * prod(shape))], shape) for b in 1:batch]
        println("picture saved to ", picture(path, batch_images, outputs, split(z), split(mean.(result.posteriors[:y]))))
    end
    @assert all(isfinite, result.free_energy)
    # Small rises after convergence are tolerated; see the note in the documentation.
    @assert maximum(diff(result.free_energy)) <= 0.01 * (result.free_energy[1] - minimum(result.free_energy))
    @assert err < 0.1
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse.(Int, ARGS[1:min(3, end)])..., ARGS[4:end]...)
end

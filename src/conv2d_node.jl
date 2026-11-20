"""
    Conv2dContinuousTransition

A whole two-dimensional convolution as one stochastic node with scalar state ports:
`y[:, p] ~ N(A * patch(x, p), W⁻¹)` independently for every output position `p`, with the
same `A = reshape(a, C_out, kh * kw * C_in)` and the same `C_out × C_out` noise precision
`W` at every position. Its interfaces are `[:y..., :x..., :a, :W]`.

The image and the output are passed as flat groups of scalars in Julia's column-major order:
`x = vec(image)` for an `H × W × C_in` image and `y = vec(output)` for the
`H_out × W_out × C_out` output. A patch is ordered as `vec` of its `kh × kw × C_in` window,
so `a = vec(A)` with one row of `A` per output channel. [`Conv2dVMP`](@ref) carries this
geometry; the model only indexes `x[1:length(image)]` and `y[1:length(output)]`.

Supports `q(y)q(x)q(a)q(W)` and `q(y,x)q(a)q(W)`, as [`ScalarCT`](@ref) does, and also
independent scalar marginals for a whole state group, such as the `PointMass` marginals of
observed pixels. Incoming state messages are univariate Gaussians. The node is the exact
linear Gaussian transition whose matrix repeats the kernel: its rules agree with
`ScalarCT` supplied with that tied matrix, and overlapping patches are handled inside the
node rather than by loops between nodes.

# Examples

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> algorithm = Conv2dVMP((2, 2, 1), 1, (2, 2));

julia> result = @call_message_update_rule(
           node = Conv2dCT, target = (:y, 1), algorithm = algorithm,
           q = (x = (PointMass(1.0), PointMass(2.0), PointMass(3.0), PointMass(4.0)),
                a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]),
                W = PointMass([2.0;;])),
       );

julia> mean(getresult(result)) ≈ 5.0
true
```
"""
struct Conv2dContinuousTransition end

"""Alias for [`Conv2dContinuousTransition`](@ref)."""
const Conv2dCT = Conv2dContinuousTransition

@define_factor_node(node = Conv2dContinuousTransition, type = Stochastic, interfaces = [:y..., :x..., :a, :W])

# The base precision of one dense state system, rebuilt when q(a) or E[W] changes.
mutable struct Conv2dSystem{T}
    version::Int
    J0::Matrix{T}
    solve::SLCTSystemCache{T}
end

Conv2dSystem(::Type{T}, d) where {T} = Conv2dSystem(0, zeros(T, d, d), SLCTSystemCache(T, d))

mutable struct Conv2dCache{T}
    a::SLCTGaussianCache{T}
    a_version::Int
    version::Int
    W::Matrix{T}
    WA::Matrix{T}
    G::Matrix{T}
    # Dense systems are large and only needed for latent inputs: allocate on first use.
    x::Union{Nothing, Conv2dSystem{T}}
    yx::Union{Nothing, Conv2dSystem{T}}
    # One output position: a C_out × C_out system.
    J::Matrix{T}
    h::Vector{T}
    V::Matrix{T}
    eliminate::Matrix{T}
    rhs::Matrix{T}
    # Second moments summed over positions.
    Exx::Matrix{T}
    Eyx::Matrix{T}
    Eyy::Matrix{T}
    AX::Matrix{T}
    cross::Matrix{T}
    residual::Matrix{T}
end

"""
    Conv2dVMP((H, W, C_in), C_out, (kh, kw); stride = 1, pad = 0, T = Float64)

Cached algorithm for [`Conv2dCT`](@ref): the geometry of the convolution and the working
memory of one node. `stride` and `pad` are integers or `(rows, columns)` pairs; padding is
with zeros. The output has `H_out = (H + 2pad - kh) ÷ stride + 1` rows, likewise for
columns, and `C_out` channels.

The fields `ny`, `nx`, `n = C_out`, `m = kh * kw * C_in`, `positions = H_out * W_out`,
`outsize` and `taps` describe the layout: `taps[j, p]` is the index in `x` of entry `j` of
the patch at position `p`, or `0` for padding, and channel `c` of position `p` is
`y[p + (c - 1) * positions]`.

**Create a separate instance for every node**, inside its `where { algorithm = ... }`
declaration, as for [`SLCTVMP`](@ref): rules of one instance write into the same cache and
must run serially.

The cache snapshots `q(a)` and `E[W]` and shares `E[W]E[A]` and `E[A'WA]` between ports.
Messages towards `y` given `q(x)`, which is the case of an observed image, only solve a
`C_out × C_out` system per position. Messages that involve latent inputs assemble a dense
system over all of `x`, or all of `[y; x]`, and reuse one factorisation across ports while
incoming precisions stay the same; that memory, quadratic in the number of ports, is
allocated on first use.
"""
struct Conv2dVMP{T} <: AbstractAlgorithm
    insize::NTuple{3, Int}
    outsize::NTuple{3, Int}
    kernel::NTuple{2, Int}
    stride::NTuple{2, Int}
    pad::NTuple{2, Int}
    n::Int
    m::Int
    positions::Int
    ny::Int
    nx::Int
    taps::Matrix{Int}
    cache::Conv2dCache{T}
end

conv_pair(v::Integer, name) = conv_pair((v, v), name)
conv_pair(v::Tuple{Integer, Integer}, _) = (Int(v[1]), Int(v[2]))
conv_pair(v, name) = throw(ArgumentError("$name must be an integer or a pair of integers, got $v"))

function Conv2dVMP(
        insize::Tuple{Integer, Integer, Integer}, cout::Integer, kernel::Tuple{Integer, Integer};
        stride = 1, pad = 0, T::Type{<:AbstractFloat} = Float64,
    )
    insize, kernel, n = Int.(insize), Int.(kernel), Int(cout)
    stride, pad = conv_pair(stride, :stride), conv_pair(pad, :pad)
    all(>(0), insize) && all(>(0), kernel) && n > 0 || throw(ArgumentError("Conv2dVMP sizes must be positive, got image $insize, $n output channels and kernel $kernel"))
    all(>(0), stride) && all(>=(0), pad) || throw(ArgumentError("Conv2dVMP needs a positive stride and a non-negative padding, got $stride and $pad"))
    all(insize[1:2] .+ 2 .* pad .>= kernel) || throw(ArgumentError("the kernel $kernel does not fit the padded image $(insize[1:2] .+ 2 .* pad)"))
    out = (insize[1:2] .+ 2 .* pad .- kernel) .÷ stride .+ 1
    m, positions = prod(kernel) * insize[3], prod(out)
    taps = zeros(Int, m, positions)
    patch, image = LinearIndices((kernel..., insize[3])), LinearIndices(insize)
    for o2 in 1:out[2], o1 in 1:out[1], ci in 1:insize[3], i2 in 1:kernel[2], i1 in 1:kernel[1]
        r, c = (o1 - 1) * stride[1] + i1 - pad[1], (o2 - 1) * stride[2] + i2 - pad[2]
        1 <= r <= insize[1] && 1 <= c <= insize[2] || continue
        taps[patch[i1, i2, ci], o1 + (o2 - 1) * out[1]] = image[r, c, ci]
    end
    cache = Conv2dCache(
        SLCTGaussianCache(T, n * m), -1, 0, zeros(T, n, n), zeros(T, n, m), zeros(T, m, m), nothing, nothing,
        zeros(T, n, n), zeros(T, n), zeros(T, n, n), zeros(T, n - 1, n - 1), zeros(T, n - 1, 2),
        zeros(T, m, m), zeros(T, n, m), zeros(T, n, n), zeros(T, n, m), zeros(T, n, n), zeros(T, n, n),
    )
    return Conv2dVMP(insize, (out..., n), kernel, stride, pad, n, m, positions, n * positions, prod(insize), taps, cache)
end

function Base.show(io::IO, algo::Conv2dVMP{T}) where {T}
    return print(io, "Conv2dVMP(", algo.insize, ", ", algo.n, ", ", algo.kernel, "; stride = ", algo.stride, ", pad = ", algo.pad, ", T = ", T, ")")
end
MessagePassingRulesBase.ispure(::Type{<:Conv2dVMP}) = false

@define_dependencies(
    node = Conv2dContinuousTransition, algorithm = Conv2dVMP,
    dependencies = [(:y, k) => (default,), (:x, k) => (default,), :a => (default,), :W => (default,)],
)

conv_dimension(algo, group) = group === :y ? algo.ny : algo.nx
conv_layout_error() = throw(
    ArgumentError("Conv2dCT supports q(y)q(x)q(a)q(W) or q(y,x)q(a)q(W), with whole state groups given jointly or as independent scalars"),
)

const Conv2dScalar = Union{UnivariateNormalDistributionsFamily, PointMass{<:Real}}

# A state group arrives as a joint marginal under a cluster key, or as a tuple of independent
# scalar marginals: what the engine supplies for observed pixels and for singleton groups.
function conv_marginals(algo, @nospecialize(args))
    qy = qx = qyx = nothing
    for group in (:y, :x)
        group in keys(args.q) || continue
        values = slct_count(args.q[group], conv_dimension(algo, group), group)
        all(v -> v isa Conv2dScalar, values) || conv_layout_error()
        group === :y ? (qy = values) : (qx = values)
    end
    for (key, value) in MessagePassingRulesBase.rule_inputs(Conv2dCT, args.q)
        if key === (:y,)
            qy === nothing || conv_layout_error()
            qy = slct_joint(value, algo.ny, :y)
        elseif key === (:x,)
            qx === nothing || conv_layout_error()
            qx = slct_joint(value, algo.nx, :x)
        elseif key === (:y, :x)
            qyx = slct_joint(value, algo.ny + algo.nx, :y_x)
        elseif key isa Tuple{Symbol, Int} && key[1] in (:y, :x)
            key[1] in keys(args.q) || conv_layout_error()
        elseif key !== :a && key !== :W
            conv_layout_error()
        end
    end
    qyx === nothing || (qy === nothing && qx === nothing) || conv_layout_error()
    return (; qy, qx, qyx)
end

conv_mean(q::Tuple) = [float(mean(v)) for v in q]
conv_mean(q) = mean(q)
conv_mean_cov(q::Tuple) = (conv_mean(q), Diagonal([float(var(v)) for v in q]))
conv_mean_cov(q) = mean_cov(q)

# Only the entries of one patch are read from an observed image.
conv_means(q::Tuple) = q
conv_means(q) = mean(q)
conv_mean_at(q::Tuple, i) = mean(q[i])
conv_mean_at(μ::AbstractVector, i) = μ[i]

# The incoming messages of one group, with `nothing` at the target `k` (0 for none).
function conv_messages(algo, @nospecialize(args), name, k)
    d = conv_dimension(algo, name)
    if !(name in keys(args.m))
        d == 1 && k == 1 && return (nothing,)
        throw(ArgumentError("missing incoming messages for $name (expected $d members)"))
    end
    return slct_count(args.m[name], d, name)
end

function conv_message_parameters(@nospecialize(value), name, i)
    value isa UnivariateNormalDistributionsFamily || throw(ArgumentError("$name[$i] must be a scalar Gaussian message"))
    return weightedmean_precision(value)
end

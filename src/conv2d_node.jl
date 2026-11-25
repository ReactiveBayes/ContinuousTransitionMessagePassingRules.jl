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
[`ContinuousTransition`](@ref) under `CTVMP(a -> the convolution matrix of a)` given
independent (diagonal) state messages, and overlapping patches are handled inside the node
rather than by loops between nodes. Group marginals are [`Conv2dMarginal`](@ref)s.

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

"""
    Conv2dMarginal{T}

The marginal of a state group of [`Conv2dCT`](@ref): `q(y, x)`, `q(x)` given `q(y)`, or `q(y)`
given `q(x)`. The joint covariance is never formed. It holds what the node's other rules and
the free energy read:

- `mean` and `var`: of every member, `[y; x]` for the joint;
- `ξ` and `λ`: the weighted mean and precision of the message towards every member, which
  excludes that member's own incoming message;
- `Exx`, `Eyx`, `Eyy`: the second moments `E[x x']`, `E[y x']`, `E[y y']` of the patches,
  summed over positions, with `complete` saying whether all three are held;
- `entropy`: the differential entropy, also returned by `entropy(q)`.

`members` is `:yx`, `:x` or `:y`, and `ny` the number of `y` members it starts with.
"""
struct Conv2dMarginal{T}
    members::Symbol
    ny::Int
    mean::Vector{T}
    var::Vector{T}
    ξ::Vector{T}
    λ::Vector{T}
    Exx::Matrix{T}
    Eyx::Matrix{T}
    Eyy::Matrix{T}
    complete::Bool
    entropy::T
end

BayesBase.mean(q::Conv2dMarginal) = q.mean
BayesBase.var(q::Conv2dMarginal) = q.var
BayesBase.std(q::Conv2dMarginal) = sqrt.(q.var)
BayesBase.entropy(q::Conv2dMarginal) = q.entropy
Base.length(q::Conv2dMarginal) = length(q.mean)
Base.eltype(::Type{Conv2dMarginal{T}}) where {T} = T
BayesBase.paramfloattype(::Conv2dMarginal{T}) where {T} = T
function BayesBase.convert_paramfloattype(::Type{T}, q::Conv2dMarginal) where {T <: Real}
    f(v) = convert(AbstractArray{T}, v)
    return Conv2dMarginal{T}(q.members, q.ny, f(q.mean), f(q.var), f(q.ξ), f(q.λ), f(q.Exx), f(q.Eyx), f(q.Eyy), q.complete, T(q.entropy))
end
function Base.show(io::IO, q::Conv2dMarginal{T}) where {T}
    return print(io, "Conv2dMarginal{", T, "}(", repr(q.members), ", ", length(q), " members)")
end

# The last solve of one state system under the default scheme, where every port asks for
# its message in a call of its own: the inputs it was solved for, and the result.
mutable struct Conv2dSolved{T}
    version::Int
    ξ::Vector{T}
    λ::Vector{T}
    other::Vector{T}
    result::Union{Nothing, Conv2dMarginal{T}}
    solves::Int
end

Conv2dSolved(::Type{T}, d, other) where {T} = Conv2dSolved{T}(-1, zeros(T, d), ones(T, d), fill(T(NaN), other), nothing, 0)

# The banded system over x: allocated on first use, since an observed image never needs it.
struct Conv2dBand{T}
    system::Matrix{T}
    factor::Matrix{T}
    inverse::Matrix{T}
    rhs::Vector{T}
end

Conv2dBand(::Type{T}, b, nx) where {T} = Conv2dBand(zeros(T, b + 1, nx), zeros(T, b + 1, nx), zeros(T, b + 1, nx), zeros(T, nx))

mutable struct Conv2dCache{T}
    a::SLCTGaussianCache{T}
    a_version::Int
    version::Int
    W::Matrix{T}
    WA::Matrix{T}
    G::Matrix{T}
    band::Union{Nothing, Conv2dBand{T}}
    # The default scheme's solves, by block.
    yx::Union{Nothing, Conv2dSolved{T}}
    x::Union{Nothing, Conv2dSolved{T}}
    y::Union{Nothing, Conv2dSolved{T}}
    # One output position.
    Bλ::Vector{T}
    Binv::Matrix{T}
    BW::Matrix{T}
    M::Matrix{T}
    logdetB::T
    Vpp::Matrix{T}
    Cyx::Matrix{T}
    Cyy::Matrix{T}
    μp::Vector{T}
    μy::Vector{T}
    # Second moments summed over positions.
    Exx::Matrix{T}
    Eyx::Matrix{T}
    Eyy::Matrix{T}
    AX::Matrix{T}
    cross::Matrix{T}
    residual::Matrix{T}
end

"""
    Conv2dVMP((H, W, C_in), C_out, (kh, kw); stride = 1, pad = 0, T = Float64, ports = :default)

Cached algorithm for [`Conv2dCT`](@ref): the geometry of the convolution and the working
memory of one node. `stride` and `pad` are integers or `(rows, columns)` pairs; padding is
with zeros. The output has `H_out = (H + 2pad - kh) ÷ stride + 1` rows, likewise for
columns, and `C_out` channels.

The fields `ny`, `nx`, `n = C_out`, `m = kh * kw * C_in`, `positions = H_out * W_out`,
`outsize` and `taps` describe the layout: `taps[j, p]` is the index in `x` of entry `j` of
the patch at position `p`, or `0` for padding, and channel `c` of position `p` is
`y[p + (c - 1) * positions]`.

`ports` says what the message towards a state port is computed from, and with it how the
node behaves in an inference engine:

- `:default`: the engine's default scheme, under any supported factorisation. Every port's
  rule receives all the other messages of its group, and one solve is reused across ports
  while the inputs stay the same.
- `:joint`, for `q(y,x)q(a)q(W)`: every port reads its message from the joint marginal, which
  is computed once per iteration. One rule serves all ports.
- `:separate`, for `q(y)q(x)q(a)q(W)` with latent `x`: likewise from `q(y)` and `q(x)`.
- `:observed`, for `q(y)q(a)q(W)` with `x` given as data: likewise from `q(y)`, which also
  carries the moments of the observed patches, so no other rule reads the pixels again.
  No message is sent towards `x`.

**Create a separate instance for every node**, inside its `where { algorithm = ... }`
declaration, as for [`SLCTVMP`](@ref): rules of one instance write into the same cache and
must run serially.

Latent inputs are solved exactly in a banded system over `x`: the outputs of every position
are eliminated in closed form, `x` is ordered so that a patch spans a narrow band, and a
banded Cholesky factorisation with a selected inverse gives every mean, variance and patch
covariance in `O(nx b²)` time and `O(nx b)` memory for bandwidth `b`, the field `bandwidth`,
about `C_in * ((kw - 1) * H + kh)`. That memory is allocated on first use.
"""
struct Conv2dVMP{T, S} <: AbstractAlgorithm
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
    order::Vector{Int}
    ptaps::Matrix{Int}
    bandwidth::Int
    cache::Conv2dCache{T}
end

conv_pair(v::Integer, name) = conv_pair((v, v), name)
conv_pair(v::Tuple{Integer, Integer}, _) = (Int(v[1]), Int(v[2]))
conv_pair(v, name) = throw(ArgumentError("$name must be an integer or a pair of integers, got $v"))

const CONV_PORTS = (:default, :joint, :separate, :observed)

# The position of every pixel in the banded system: channels together, then along the
# shorter side of the image, so that a patch spans as few positions as possible.
function conv_order(insize, taps)
    H, Wd, C = insize
    image = LinearIndices(insize)
    best, bestwidth = Int[], typemax(Int)
    for rowsfirst in (true, false)
        order = zeros(Int, prod(insize))
        for c in 1:Wd, r in 1:H, ci in 1:C
            order[image[r, c, ci]] = ci + C * (rowsfirst ? (r - 1) + (c - 1) * H : (c - 1) + (r - 1) * Wd)
        end
        width = 0
        for p in axes(taps, 2)
            lo, hi = typemax(Int), 0
            for j in axes(taps, 1)
                t = taps[j, p]
                t == 0 && continue
                lo, hi = min(lo, order[t]), max(hi, order[t])
            end
            hi > 0 && (width = max(width, hi - lo))
        end
        width < bestwidth && ((best, bestwidth) = (order, width))
    end
    return best, bestwidth
end

function Conv2dVMP(
        insize::Tuple{Integer, Integer, Integer}, cout::Integer, kernel::Tuple{Integer, Integer};
        stride = 1, pad = 0, T::Type{<:AbstractFloat} = Float64, ports::Symbol = :default,
    )
    insize, kernel, n = Int.(insize), Int.(kernel), Int(cout)
    stride, pad = conv_pair(stride, :stride), conv_pair(pad, :pad)
    ports in CONV_PORTS || throw(ArgumentError("ports must be one of $CONV_PORTS, got $(repr(ports))"))
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
    order, bandwidth = conv_order(insize, taps)
    ptaps = map(t -> t == 0 ? 0 : order[t], taps)
    z(dims...) = zeros(T, dims...)
    cache = Conv2dCache{T}(
        SLCTGaussianCache(T, n * m), -1, 0, z(n, n), z(n, m), z(m, m), nothing, nothing, nothing, nothing,
        fill(T(NaN), n), z(n, n), z(n, m), z(m, m), zero(T), z(m, m), z(n, m), z(n, n), z(m), z(n),
        z(m, m), z(n, m), z(n, n), z(n, m), z(n, n), z(n, n),
    )
    return Conv2dVMP{T, ports}(insize, (out..., n), kernel, stride, pad, n, m, positions, n * positions, prod(insize), taps, order, ptaps, bandwidth, cache)
end

conv_ports(::Conv2dVMP{T, S}) where {T, S} = S

function Base.show(io::IO, algo::Conv2dVMP{T}) where {T}
    return print(io, "Conv2dVMP(", algo.insize, ", ", algo.n, ", ", algo.kernel, "; stride = ", algo.stride, ", pad = ", algo.pad, ", T = ", T, ", ports = ", repr(conv_ports(algo)), ")")
end
MessagePassingRulesBase.ispure(::Type{<:Conv2dVMP}) = false

@define_dependencies(
    node = Conv2dContinuousTransition, algorithm = Conv2dVMP{<:Any, :default},
    dependencies = [(:y, k) => (default,), (:x, k) => (default,), :a => (default,), :W => (default,)],
)
@define_dependencies(
    node = Conv2dContinuousTransition, algorithm = Conv2dVMP{<:Any, :joint},
    dependencies = [(:y, k) => (q[:y, :x],), (:x, k) => (q[:y, :x],), :a => (default,), :W => (default,)],
)
@define_dependencies(
    node = Conv2dContinuousTransition, algorithm = Conv2dVMP{<:Any, :separate},
    dependencies = [(:y, k) => (q[(:y,)],), (:x, k) => (q[(:x,)],), :a => (default,), :W => (default,)],
)
@define_dependencies(
    node = Conv2dContinuousTransition, algorithm = Conv2dVMP{<:Any, :observed},
    dependencies = [(:y, k) => (q[(:y,)],), (:x, k) => (), :a => (q[(:y,)], q[:W]), :W => (q[(:y,)], q[:a])],
)

conv_dimension(algo, group) = group === :y ? algo.ny : algo.nx
conv_layout_error() = throw(
    ArgumentError("Conv2dCT supports q(y)q(x)q(a)q(W) or q(y,x)q(a)q(W), with whole state groups given jointly or as independent scalars"),
)

const Conv2dScalar = Union{UnivariateNormalDistributionsFamily, PointMass{<:Real}}

function conv_group(q, d, name)
    q isa Conv2dMarginal || q isa MultivariateNormalDistributionsFamily || throw(ArgumentError("$name must be a multivariate Gaussian or a Conv2dMarginal"))
    length(q) == d || throw(DimensionMismatch("$name must have dimension $d, got $(length(q))"))
    return q
end

# The joint marginal under a cluster key, or `nothing`: resolved from the container's type.
@generated function conv_cluster(q::MessagePassingRulesBase.Marginals{N, T, J}, ::Val{K}) where {N, T, J, K}
    all(key -> key in ((:y,), (:x,), (:y, :x)), J) || return :(conv_layout_error())
    i = findfirst(==(K), J)
    return i === nothing ? nothing : :(q.joints[$i])
end

# Members of one type are checked at once; a mixed tuple one by one. A long tuple is never
# matched against `Tuple{Vararg{Conv2dScalar}}`, which the union makes very slow to decide.
function conv_all_scalars(@nospecialize(values::Tuple))
    M = eltype(values)
    isconcretetype(M) && M <: Conv2dScalar && return true
    for value in values
        value isa Conv2dScalar || return false
    end
    return true
end

function conv_scalars(algo, q, group)
    group in keys(q) || return nothing
    values = slct_count(q[group], conv_dimension(algo, group), group)
    conv_all_scalars(values) || conv_layout_error()
    return values
end

# A state group arrives as a joint marginal under a cluster key, or as a tuple of independent
# scalar marginals: what the engine supplies for observed pixels and for singleton groups.
function conv_marginals(algo, q::MessagePassingRulesBase.Marginals)
    all(key -> key in (:y, :x, :a, :W), keys(q)) || conv_layout_error()
    qy, qx = conv_scalars(algo, q, :y), conv_scalars(algo, q, :x)
    jy, jx, qyx = conv_cluster(q, Val((:y,))), conv_cluster(q, Val((:x,))), conv_cluster(q, Val((:y, :x)))
    if jy !== nothing
        qy === nothing || conv_layout_error()
        qy = conv_group(jy, algo.ny, :y)
    end
    if jx !== nothing
        qx === nothing || conv_layout_error()
        qx = conv_group(jx, algo.nx, :x)
    end
    qyx === nothing || conv_group(qyx, algo.ny + algo.nx, :y_x)
    qyx === nothing || (qy === nothing && qx === nothing) || conv_layout_error()
    return (; qy, qx, qyx)
end

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

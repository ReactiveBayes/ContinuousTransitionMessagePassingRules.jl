# Snapshot distribution parameters, rather than object identities: Gaussian parameters may
# be mutated in place. In particular, do not invert q(a)'s precision again for every port.
mutable struct SLCTGaussianCache{T}
    representation::Int
    version::Int
    p::Vector{T}
    P::Matrix{T}
    scale::T
    μ::Vector{T}
    V::Matrix{T}
    factor::Matrix{T}
end

SLCTGaussianCache(::Type{T}, d) where {T} = SLCTGaussianCache(
    0, 0, zeros(T, d), zeros(T, d, d), zero(T), zeros(T, d), zeros(T, d, d), zeros(T, d, d),
)

slct_parameters(q::Union{MvNormalMeanCovariance, MvNormal}) = (1, BayesBase.params(q)...)
slct_parameters(q::MvNormalMeanPrecision) = (2, BayesBase.params(q)...)
slct_parameters(q::MvNormalWeightedMeanPrecision) = (3, BayesBase.params(q)...)
slct_parameters(q::MvNormalMeanScalePrecision) = (4, BayesBase.params(q)...)

function slct_equal(a::AbstractArray{T}, b) where {T}
    size(a) == size(b) || return false
    same = true
    # Scan snapshots without allocating, with a vectorisable reduction for dense arrays.
    @inbounds @simd for i in eachindex(a, b)
        same &= isequal(a[i], T(b[i]))
    end
    return same
end

function slct_identity!(A)
    fill!(A, zero(eltype(A)))
    for i in axes(A, 1)
        A[i, i] = one(eltype(A))
    end
    return A
end

function slct_gaussian!(c::SLCTGaussianCache{T}, q) where {T}
    representation, p, P = slct_parameters(q)
    d = length(c.p)
    lct_size(p, (d,), :a)
    representation == 4 || lct_size(P, (d, d), :a)
    same_matrix = c.representation == representation && (representation == 4 ? isequal(c.scale, T(P)) : slct_equal(c.P, P))
    same_matrix && slct_equal(c.p, p) && return c
    c.representation = 0 # A failed factorisation must not leave a valid cache entry.
    if !same_matrix
        if representation == 4
            c.scale = P
            slct_identity!(c.V)
            c.V ./= c.scale
        else
            c.P .= P
            if representation == 1
                c.V .= c.P
            else
                c.factor .= c.P
                slct_identity!(c.V)
                ldiv!(cholesky!(Symmetric(c.factor)), c.V)
            end
        end
    end
    c.p .= p
    if representation == 3
        mul!(c.μ, c.V, c.p)
    else
        c.μ .= c.p
    end
    c.representation = representation
    c.version += 1
    return c
end

# A completed system includes a positive incoming precision on every coordinate, including
# the temporarily excluded target. Its value cancels from the Schur complement below.
mutable struct SLCTSystemCache{T}
    ξ::Vector{T}
    λ::Vector{T}
    h::Vector{T}
    J::Matrix{T}
    factor::Matrix{T}
    V::Matrix{T}
    valid::Bool
    factorizations::Int
end

SLCTSystemCache(::Type{T}, d) where {T} = SLCTSystemCache(
    zeros(T, d), ones(T, d), zeros(T, d), zeros(T, d, d), zeros(T, d, d), zeros(T, d, d), false, 0,
)

mutable struct SLCTCache{T, S}
    a::SLCTGaussianCache{T}
    a_version::Int
    W::Matrix{T}
    WA::Matrix{T}
    J::Matrix{T}
    y::SLCTSystemCache{T}
    x::SLCTSystemCache{T}
    yx::SLCTSystemCache{T}
    moments::S
    residual::Matrix{T}
end

"""
    SLCTVMP(n, m; T = Float64)

Cached algorithm for [`ScalarCT`](@ref), with `n` scalar outputs and `m` scalar inputs.
Supports both `q(y)q(x)q(a)q(W)` and `q(y,x)q(a)q(W)`, with the same rules and Gaussian
messages as [`LCTVMP`](@ref), up to floating-point roundoff. Positive integer dimensions
allocate the cache immediately; `T <: AbstractFloat` fixes its working precision.

**Create a separate instance for every node. Sharing an instance between nodes is not
allowed: their rules would write into the same cache. Calls using one instance must also
run serially, not concurrently.** In a model loop, construct `SLCTVMP(n, m)` inside each
node's `where { algorithm = ... }` declaration. Reuse that instance for successive calls
belonging to the same node.

The cache stores parameter snapshots, transition precision, and Gaussian solves for the
state groups. It checks parameter values, including in-place edits, before reusing results.
Changing only incoming weighted means reuses the solve; changing precisions invalidates it.
Each outgoing scalar message still excludes its own incoming message exactly. Shared
moment buffers serve parameter and energy updates; returned distributions own their arrays.

This trades retained per-node memory for fewer repeated computations across scalar ports.
It does not eliminate allocations for returned distributions, input moment extraction, or
rule dispatch.
"""
struct SLCTVMP{T, S} <: AbstractAlgorithm
    n::Int
    m::Int
    cache::SLCTCache{T, S}
end

function SLCTVMP(n::Integer, m::Integer; T::Type{<:AbstractFloat} = Float64)
    shape = LCTVMP(n, m)
    n, m = shape.n, shape.m
    cache = SLCTCache(
        SLCTGaussianCache(T, n * m), -1, zeros(T, n, n), zeros(T, n, m), zeros(T, n + m, n + m),
        SLCTSystemCache(T, n), SLCTSystemCache(T, m), SLCTSystemCache(T, n + m),
        lct_residual_scratch(shape, T), zeros(T, n, n),
    )
    return SLCTVMP(n, m, cache)
end

Base.show(io::IO, algo::SLCTVMP{T}) where {T} = print(io, "SLCTVMP(", algo.n, ", ", algo.m, "; T = ", T, ")")
MessagePassingRulesBase.ispure(::Type{<:SLCTVMP}) = false

@define_dependencies(
    node = ScalarCT, algorithm = SLCTVMP,
    dependencies = [(:y, k) => (default,), (:x, k) => (default,), :a => (default,), :W => (default,)],
)

function slct_transition!(algo::SLCTVMP, qa, qW)
    c = algo.cache
    a = slct_gaussian!(c.a, qa)
    W = lct_mean(qW, (algo.n, algo.n), :W)
    c.a_version == a.version && slct_equal(c.W, W) && return c
    c.y.valid = c.x.valid = c.yx.valid = false
    c.a_version = -1
    c.W .= W
    A = reshape(a.μ, algo.n, algo.m)
    mul!(c.WA, c.W, A)
    y, x = 1:algo.n, (algo.n + 1):(algo.n + algo.m)
    c.J[y, y] .= c.W
    c.J[y, x] .= .-c.WA
    c.J[x, y] .= .-c.WA'
    Jxx = view(c.J, x, x)
    lct_uncertainty!(Jxx, LCTVMP(algo.n, algo.m), MvNormalMeanCovariance(a.μ, a.V), c.W)
    mul!(Jxx, A', c.WA, true, true)
    for j in axes(c.J, 2), i in 1:(j - 1)
        c.J[i, j] = c.J[j, i] = (c.J[i, j] + c.J[j, i]) / 2
    end
    c.a_version = a.version
    return c
end

function slct_cached_system!(algo, args, group, k)
    if k != 0
        1 <= k <= slct_dimension(algo, group) || throw(ArgumentError("target index $k is outside group $group"))
    end
    qs = slct_marginals(algo, args)
    qs.qyx === nothing || slct_layout_error()
    block = if group === :yx
        qs.qy === nothing && qs.qx === nothing || slct_layout_error()
        :yx
    elseif group === :y && qs.qy === nothing && qs.qx !== nothing
        :y
    elseif group === :x && qs.qx === nothing && qs.qy !== nothing
        :x
    elseif k != 0 && qs.qy === nothing && qs.qx === nothing
        :yx
    else
        slct_layout_error()
    end
    slct_messages(algo, args, block, group, k)
    c = slct_transition!(algo, args.q[:a], args.q[:W])
    s = getproperty(c, block)
    range = block === :y ? (1:algo.n) : block === :x ? ((algo.n + 1):(algo.n + algo.m)) : (1:(algo.n + algo.m))
    J0 = view(c.J, range, range)
    if block === :y
        mul!(s.h, c.WA, slct_mean(qs.qx))
    elseif block === :x
        mul!(s.h, c.WA', slct_mean(qs.qy))
    else
        fill!(s.h, zero(eltype(s.h)))
    end
    for name in keys(args.m)
        offset = block === :yx && name === :x ? algo.n : 0
        for (i, value) in enumerate(args.m[name])
            value === nothing && continue
            ξ, λ = weightedmean_precision(value)
            index = offset + i
            s.ξ[index] = ξ
            if !isequal(s.λ[index], eltype(s.λ)(λ))
                s.valid = false
                s.λ[index] = λ
            end
        end
    end
    index = block === :yx && group === :x ? algo.n + k : k
    # The target's incoming weighted mean must never enter the outgoing message.
    for i in eachindex(s.h)
        i == index && k != 0 && continue
        s.h[i] += s.ξ[i]
    end
    if !s.valid
        s.J .= J0
        for i in eachindex(s.λ)
            s.J[i, i] += s.λ[i]
        end
    end
    return s, J0, index
end

function slct_cached_message(algo, args, group, k)
    k >= 1 || throw(ArgumentError("target index must be positive, got $k"))
    s, J0, index = slct_cached_system!(algo, args, group, k)
    if !s.valid
        s.factor .= s.J
        slct_identity!(s.V)
        ldiv!(cholesky!(Symmetric(s.factor)), s.V)
        s.valid = true
        s.factorizations += 1
    end
    # V[i,k]/V[k,k] = -(J[-k,-k] \ J[-k,k])[i]. This Schur form avoids subtracting
    # a possibly large cached target precision from 1/V[k,k]. Its own prior cancels.
    ξ, λ = s.h[index], J0[index, index]
    for i in eachindex(s.h)
        i == index && continue
        ratio = s.V[i, index] / s.V[index, index]
        ξ += s.h[i] * ratio
        λ += J0[index, i] * ratio
    end
    return NormalWeightedMeanPrecision(ξ, λ)
end

function slct_cached_marginal(algo, args, group)
    s, _, _ = slct_cached_system!(algo, args, group, 0)
    return MvNormalWeightedMeanPrecision(copy(s.h), copy(s.J))
end

function slct_cached_residual!(algo, args)
    shape, c = LCTVMP(algo.n, algo.m), algo.cache
    my, Vy, mx, Vx, Vyx = slct_state_moments(shape, args)
    a = slct_gaussian!(c.a, args.q[:a])
    A = reshape(a.μ, algo.n, algo.m)
    s, R = c.moments, c.residual
    s.Exx .= Vx .+ mx .* mx'
    s.Eyx .= Vyx .+ my .* mx'
    mul!(s.cross, s.Eyx, A')
    R .= Vy .+ my .* my' .- s.cross .- s.cross'
    mul!(s.AX, A, s.Exx)
    mul!(R, s.AX, A', true, true)
    for l in 1:algo.m, k in 1:algo.m
        C = view(a.V, ((k - 1) * algo.n + 1):(k * algo.n), ((l - 1) * algo.n + 1):(l * algo.n))
        R .+= s.Exx[k, l] .* C
    end
    for j in axes(R, 2), i in 1:(j - 1)
        R[i, j] = R[j, i] = (R[i, j] + R[j, i]) / 2
    end
    return R
end

@define_message_update_rule(
    node = ScalarCT, target = (:y, k), algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> slct_cached_message(algo, args, :y, k),
)

@define_message_update_rule(
    node = ScalarCT, target = (:x, k), algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> slct_cached_message(algo, args, :x, k),
)

@define_message_update_rule(
    node = ScalarCT, target = :a, algorithm = SLCTVMP,
    args = (default, q[:W]::Any),
    body = (algo, args) -> begin
        shape = LCTVMP(algo.n, algo.m)
        my, _, mx, Vx, Vyx = slct_state_moments(shape, args)
        lct_towards_a(algo.cache.moments, shape, my, mx, Vx, Vyx, args.q[:W])
    end,
)

@define_message_update_rule(
    node = ScalarCT, target = :W, algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily),
    body = (algo, args) -> WishartFast(algo.n + 2, copy(slct_cached_residual!(algo, args))),
)

@define_marginal_update_rule(
    node = ScalarCT, target = (:y,), algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> slct_cached_marginal(algo, args, :y),
)

@define_marginal_update_rule(
    node = ScalarCT, target = (:x,), algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> slct_cached_marginal(algo, args, :x),
)

@define_marginal_update_rule(
    node = ScalarCT, target = (:y, :x), algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> slct_cached_marginal(algo, args, :yx),
)

@define_average_energy(
    node = ScalarCT, algorithm = SLCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> begin
        W = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        R = slct_cached_residual!(algo, args)
        gaussian_average_energy(algo.n, trace_product(W, R) - mean(logdet, args.q[:W]))
    end,
)

function conv_transition!(algo::Conv2dVMP, qa, qW)
    c = algo.cache
    a = slct_gaussian!(c.a, qa)
    W = lct_mean(qW, (algo.n, algo.n), :W)
    c.a_version == a.version && slct_equal(c.W, W) && return c
    c.a_version = -1
    c.W .= W
    A = reshape(a.μ, algo.n, algo.m)
    mul!(c.WA, c.W, A)
    lct_uncertainty!(c.G, LCTVMP(algo.n, algo.m), MvNormalMeanCovariance(a.μ, a.V), c.W)
    mul!(c.G, A', c.WA, true, true)
    for j in axes(c.G, 2), i in 1:(j - 1)
        c.G[i, j] = c.G[j, i] = (c.G[i, j] + c.G[j, i]) / 2
    end
    c.version += 1
    c.a_version = a.version
    return c
end

# Scatter the per-position precision [W -WA; -WA' G] through the taps of every patch.
function conv_system!(c::Conv2dCache{T}, algo, block) where {T}
    sys = getproperty(c, block)
    if sys === nothing
        sys = Conv2dSystem(T, block === :yx ? algo.ny + algo.nx : algo.nx)
        setproperty!(c, block, sys)
    end
    sys.version == c.version && return sys
    J, n, P, taps = sys.J0, algo.n, algo.positions, algo.taps
    offset = block === :yx ? algo.ny : 0
    fill!(J, zero(T))
    for p in 1:P
        if block === :yx
            for c2 in 1:n, c1 in 1:n
                J[p + (c1 - 1) * P, p + (c2 - 1) * P] = c.W[c1, c2]
            end
        end
        for j in 1:algo.m
            tj = taps[j, p]
            tj == 0 && continue
            if block === :yx
                for c1 in 1:n
                    i = p + (c1 - 1) * P
                    J[i, offset + tj] = J[offset + tj, i] = -c.WA[c1, j]
                end
            end
            for l in 1:algo.m
                tl = taps[l, p]
                tl == 0 && continue
                J[offset + tl, offset + tj] += c.G[l, j]
            end
        end
    end
    sys.version = c.version
    sys.solve.valid = false
    return sys
end

# The dense canonical system over x given q(y), or over [y; x]. k == 0 includes every
# incoming message, for a group marginal.
function conv_dense!(algo, @nospecialize(args), qs, block, group, k)
    block === :yx || !(:y in keys(args.m)) || conv_layout_error()
    c = conv_transition!(algo, args.q[:a], args.q[:W])
    sys = conv_system!(c, algo, block)
    s, P = sys.solve, algo.positions
    fill!(s.h, zero(eltype(s.h)))
    if block === :x
        my = conv_mean(qs.qy)
        for p in 1:P, j in 1:algo.m
            t = algo.taps[j, p]
            t == 0 && continue
            for c1 in 1:algo.n
                s.h[t] += c.WA[c1, j] * my[p + (c1 - 1) * P]
            end
        end
    end
    index = block === :yx && group === :x ? algo.ny + k : k
    for name in (block === :yx ? (:y, :x) : (:x,))
        offset = block === :yx && name === :x ? algo.ny : 0
        target = name === group ? k : 0
        for (i, value) in enumerate(conv_messages(algo, args, name, target))
            if i == target
                value === nothing || throw(ArgumentError("the target's own incoming message must be nothing"))
                continue
            end
            ξ, λ = conv_message_parameters(value, name, i)
            # The target's incoming weighted mean never enters: its entry is skipped above.
            s.h[offset + i] += ξ
            if !isequal(s.λ[offset + i], eltype(s.λ)(λ))
                s.valid = false
                s.λ[offset + i] = λ
            end
        end
    end
    if !s.valid
        s.J .= sys.J0
        for i in eachindex(s.λ)
            s.J[i, i] += s.λ[i]
        end
    end
    return sys, index
end

# One output position given q(x): [W + diag(λ)] and W * A * E[patch] + ξ, over its channels.
function conv_position!(c, algo, messages, x, p, target)
    n, P = algo.n, algo.positions
    c.J .= c.W
    for i in 1:n
        h = zero(eltype(c.h))
        for j in 1:algo.m
            t = algo.taps[j, p]
            t == 0 && continue
            h += c.WA[i, j] * conv_mean_at(x, t)
        end
        k = p + (i - 1) * P
        if k == target
            messages[k] === nothing || throw(ArgumentError("the target's own incoming message must be nothing"))
        else
            ξ, λ = conv_message_parameters(messages[k], :y, k)
            h += ξ
            c.J[i, i] += λ
        end
        c.h[i] = h
    end
    return c
end

function conv_message(algo, @nospecialize(args), group, k)
    1 <= k <= conv_dimension(algo, group) || throw(ArgumentError("target index $k is outside group $group"))
    qs = conv_marginals(algo, args)
    qs.qyx === nothing || conv_layout_error()
    if group === :y && qs.qy === nothing && qs.qx !== nothing
        :x in keys(args.m) && conv_layout_error()
        c = conv_transition!(algo, args.q[:a], args.q[:W])
        p = (k - 1) % algo.positions + 1
        conv_position!(c, algo, conv_messages(algo, args, :y, k), conv_means(qs.qx), p, k)
        return slct_eliminate(c.eliminate, c.rhs, c.J, c.h, (k - 1) ÷ algo.positions + 1)
    end
    block = if group === :x && qs.qx === nothing && qs.qy !== nothing
        :x
    elseif qs.qy === nothing && qs.qx === nothing
        :yx
    else
        conv_layout_error()
    end
    sys, index = conv_dense!(algo, args, qs, block, group, k)
    slct_factorize!(sys.solve)
    return slct_schur(sys.solve.h, sys.J0, sys.solve.V, index)
end

function conv_marginal(algo, @nospecialize(args), group)
    qs = conv_marginals(algo, args)
    qs.qyx === nothing || conv_layout_error()
    if group === :y
        qs.qy === nothing && qs.qx !== nothing && !(:x in keys(args.m)) || conv_layout_error()
        return conv_y_marginal(algo, args, conv_messages(algo, args, :y, 0), conv_means(qs.qx))
    end
    if group === :x
        qs.qx === nothing && qs.qy !== nothing || conv_layout_error()
    else
        qs.qy === nothing && qs.qx === nothing || conv_layout_error()
    end
    sys, _ = conv_dense!(algo, args, qs, group, group, 0)
    return MvNormalWeightedMeanPrecision(copy(sys.solve.h), copy(sys.solve.J))
end

# Positions are independent given q(x): invert each small block instead of the whole system.
"""
    Conv2dBlockCovariance

The covariance of `q(y)` given `q(x)`: outputs at different positions are independent, so
only the `C_out × C_out` block of every position is stored, as `blocks[:, :, p]`. It indexes
as the full `ny × ny` matrix in the layout of [`Conv2dVMP`](@ref), where channel `c` of
position `p` is entry `p + (c - 1) * positions`, and its log-determinant and inverse are
computed block by block.
"""
struct Conv2dBlockCovariance{T} <: AbstractMatrix{T}
    blocks::Array{T, 3}
end

conv_block_count(V::Conv2dBlockCovariance) = (size(V.blocks, 1), size(V.blocks, 3))
function Base.size(V::Conv2dBlockCovariance)
    n, P = conv_block_count(V)
    return (n * P, n * P)
end
Base.@propagate_inbounds function Base.getindex(V::Conv2dBlockCovariance{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(V, i, j)
    P = size(V.blocks, 3)
    ci, pi = divrem(i - 1, P)
    cj, pj = divrem(j - 1, P)
    return pi == pj ? V.blocks[ci + 1, cj + 1, pi + 1] : zero(T)
end

function LinearAlgebra.logdet(V::Conv2dBlockCovariance{T}) where {T}
    return sum(p -> chollogdet(V.blocks[:, :, p]), axes(V.blocks, 3); init = zero(T))
end
FastCholesky.chollogdet(V::Conv2dBlockCovariance) = logdet(V)
function FastCholesky.cholinv(V::Conv2dBlockCovariance)
    out = similar(V.blocks)
    for p in axes(out, 3)
        out[:, :, p] = cholinv(V.blocks[:, :, p])
    end
    return Conv2dBlockCovariance(out)
end

function conv_y_marginal(algo, @nospecialize(args), messages, x)
    c = conv_transition!(algo, args.q[:a], args.q[:W])
    n, P = algo.n, algo.positions
    μ, V = zeros(eltype(c.h), algo.ny), zeros(eltype(c.h), n, n, P)
    for p in 1:P
        conv_position!(c, algo, messages, x, p, 0)
        slct_identity!(c.V)
        ldiv!(cholesky!(Symmetric(c.J)), c.V)
        for j in 1:n, i in 1:n
            v = (c.V[i, j] + c.V[j, i]) / 2
            V[i, j, p] = v
            μ[p + (i - 1) * P] += v * c.h[j]
        end
    end
    return MvNormalMeanCovariance(μ, Conv2dBlockCovariance(V))
end

function conv_accumulate!(c, algo, my, Vy, mx, Vx, Vyx)
    n, P = algo.n, algo.positions
    fill!(c.Exx, zero(eltype(c.Exx)))
    fill!(c.Eyx, zero(eltype(c.Eyx)))
    fill!(c.Eyy, zero(eltype(c.Eyy)))
    for p in 1:P
        for j in 1:algo.m
            tj = algo.taps[j, p]
            tj == 0 && continue
            for l in 1:algo.m
                tl = algo.taps[l, p]
                tl == 0 && continue
                c.Exx[l, j] += Vx[tl, tj] + mx[tl] * mx[tj]
            end
            for i in 1:n
                k = p + (i - 1) * P
                c.Eyx[i, j] += my[k] * mx[tj] + (Vyx === nothing ? zero(eltype(c.Eyx)) : Vyx[k, tj])
            end
        end
        for j in 1:n, i in 1:n
            ki, kj = p + (i - 1) * P, p + (j - 1) * P
            c.Eyy[i, j] += Vy[ki, kj] + my[ki] * my[kj]
        end
    end
    return c
end

# E[x x'], E[y x'] and E[y y'] of the patches, summed over positions: all that the messages
# towards a and W and the energy need, since every position has the same A and W.
function conv_moments!(algo, @nospecialize(args))
    isempty(keys(args.m)) || conv_layout_error()
    qs = conv_marginals(algo, args)
    c = algo.cache
    if qs.qyx !== nothing
        my, Vy, mx, Vx, Vyx = split_y_x(qs.qyx, algo.ny)
        return conv_accumulate!(c, algo, my, Vy, mx, Vx, Vyx)
    end
    qs.qy !== nothing && qs.qx !== nothing || conv_layout_error()
    my, Vy = conv_mean_cov(qs.qy)
    mx, Vx = conv_mean_cov(qs.qx)
    return conv_accumulate!(c, algo, my, Vy, mx, Vx, nothing)
end

function conv_residual!(algo, @nospecialize(args))
    c = conv_moments!(algo, args)
    a = slct_gaussian!(c.a, args.q[:a])
    A, R = reshape(a.μ, algo.n, algo.m), c.residual
    mul!(c.cross, c.Eyx, A')
    R .= c.Eyy .- c.cross .- c.cross'
    mul!(c.AX, A, c.Exx)
    mul!(R, c.AX, A', true, true)
    for l in 1:algo.m, k in 1:algo.m
        C = view(a.V, ((k - 1) * algo.n + 1):(k * algo.n), ((l - 1) * algo.n + 1):(l * algo.n))
        R .+= c.Exx[k, l] .* C
    end
    for j in axes(R, 2), i in 1:(j - 1)
        R[i, j] = R[j, i] = (R[i, j] + R[j, i]) / 2
    end
    return R
end

@define_message_update_rule(
    node = Conv2dCT, target = (:y, k), algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> conv_message(algo, args, :y, k),
)

@define_message_update_rule(
    node = Conv2dCT, target = (:x, k), algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> conv_message(algo, args, :x, k),
)

@define_message_update_rule(
    node = Conv2dCT, target = :a, algorithm = Conv2dVMP,
    args = (default, q[:W]::Any),
    body = (algo, args) -> begin
        mW = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        c = conv_moments!(algo, args)
        MvNormalWeightedMeanPrecision(vec(mW * c.Eyx), kron(c.Exx, mW))
    end,
)

# The product of one Wishart message per position: every position adds one degree of freedom.
@define_message_update_rule(
    node = Conv2dCT, target = :W, algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily),
    body = (algo, args) -> WishartFast(algo.n + 1 + algo.positions, copy(conv_residual!(algo, args))),
)

@define_marginal_update_rule(
    node = Conv2dCT, target = (:y,), algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> conv_marginal(algo, args, :y),
)

@define_marginal_update_rule(
    node = Conv2dCT, target = (:x,), algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> conv_marginal(algo, args, :x),
)

@define_marginal_update_rule(
    node = Conv2dCT, target = (:y, :x), algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> conv_marginal(algo, args, :yx),
)

@define_average_energy(
    node = Conv2dCT, algorithm = Conv2dVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> begin
        W = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        R = conv_residual!(algo, args)
        gaussian_average_energy(algo.ny, trace_product(W, R) - algo.positions * mean(logdet, args.q[:W]))
    end,
)

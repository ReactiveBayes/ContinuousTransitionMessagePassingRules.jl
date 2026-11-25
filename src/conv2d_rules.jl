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
    fill!(c.Bλ, NaN)
    c.version += 1
    c.a_version = a.version
    return c
end

function conv_band!(algo::Conv2dVMP{T}) where {T}
    c = algo.cache
    c.band === nothing && (c.band = Conv2dBand(T, algo.bandwidth, algo.nx))
    return c.band
end

# The weighted means and precisions of a group's scalar messages, as vectors. The tuple may
# be long and of mixed types: it is walked without specialising on it, unless its members
# are Gaussians of one type and none is missing, as an engine delivers them to a marginal
# rule. Signatures never constrain a tuple's members by the family union: that makes method
# tables very slow to build.
function conv_parameters!(ξ::AbstractVector{T}, λ::AbstractVector{T}, @nospecialize(messages::Tuple), name, skip) where {T}
    M = eltype(messages)
    skip == 0 && isconcretetype(M) && M <: UnivariateNormalDistributionsFamily && return conv_uniform_parameters!(ξ, λ, messages)
    changed = false
    for i in eachindex(ξ)
        value = messages[i]
        if i == skip
            value === nothing || throw(ArgumentError("the target's own incoming message must be nothing"))
            continue
        end
        a, b = conv_message_parameters(value, name, i)::Tuple{Real, Real}
        a, b = T(a), T(b)
        changed |= !isequal(ξ[i], a) | !isequal(λ[i], b)
        ξ[i], λ[i] = a, b
    end
    return changed
end

function conv_uniform_parameters!(ξ::AbstractVector{T}, λ::AbstractVector{T}, messages::Tuple) where {T}
    changed = false
    @inbounds for i in eachindex(ξ)
        a, b = weightedmean_precision(messages[i])
        a, b = T(a), T(b)
        changed |= !isequal(ξ[i], a) | !isequal(λ[i], b)
        ξ[i], λ[i] = a, b
    end
    return changed
end

# The means and variances of a group given by its marginal.
function conv_mean_var(q, ::Type{T}) where {T}
    q isa Tuple || return convert(Vector{T}, mean(q)), convert(Vector{T}, var(q))
    μ, v = Vector{T}(undef, length(q)), Vector{T}(undef, length(q))
    M = eltype(q)
    if isconcretetype(M) && M <: PointMass
        conv_pixels!(μ, v, q)
    elseif isconcretetype(M) && M <: UnivariateNormalDistributionsFamily
        conv_uniform_scalars!(μ, v, q)
    else
        conv_mixed_scalars!(μ, v, q)
    end
    return μ, v
end

function conv_mixed_scalars!(μ::Vector{T}, v::Vector{T}, @nospecialize(q::Tuple)) where {T}
    for i in eachindex(μ)
        value = q[i]
        μ[i] = mean(value)::Real
        v[i] = value isa PointMass ? zero(T) : T(var(value)::Real)
    end
    return nothing
end

function conv_pixels!(μ::Vector{T}, v::Vector{T}, q::Tuple) where {T}
    @inbounds for i in eachindex(μ)
        μ[i] = mean(q[i])
    end
    fill!(v, zero(T))
    return nothing
end

function conv_uniform_scalars!(μ::Vector{T}, v::Vector{T}, q::Tuple) where {T}
    @inbounds for i in eachindex(μ)
        μ[i], v[i] = mean_var(q[i])
    end
    return nothing
end

conv_means(q, ::Type{T}) where {T} = q isa Tuple ? first(conv_mean_var(q, T)) : convert(Vector{T}, mean(q))

# The outputs of position p given everything else: B = W + diag(λ) and its inverse, kept
# while the precisions of the incoming messages repeat from position to position.
function conv_block!(c::Conv2dCache{T}, algo, λy, p, joint) where {T}
    n, P = algo.n, algo.positions
    same = true
    @inbounds for i in 1:n
        same &= isequal(c.Bλ[i], λy[p + (i - 1) * P])
    end
    same && return c
    @inbounds for i in 1:n
        c.Bλ[i] = λy[p + (i - 1) * P]
    end
    B = c.Binv
    B .= c.W
    @inbounds for i in 1:n
        B[i, i] += c.Bλ[i]
    end
    if n == 1
        b = B[1, 1]
        (b > 0 && isfinite(b)) || throw(LinearAlgebra.PosDefException(1))
        c.logdetB = log(b)
        B[1, 1] = inv(b)
    else
        factor = cholesky!(Symmetric(B, :L))
        c.logdetB = logdet(factor)
        LinearAlgebra.inv!(factor)
        @inbounds for j in 1:n, i in 1:(j - 1)
            B[i, j] = B[j, i]
        end
    end
    if joint
        # What eliminating these outputs leaves on their patch: G - (WA)' B⁻¹ (WA).
        mul!(c.BW, c.Binv, c.WA)
        c.M .= c.G
        mul!(c.M, c.WA', c.BW, -one(T), one(T))
    end
    return c
end

# The system over x in its banded order: Σ_p scatter(M_p) + diag(λx), and its right-hand side.
# With `ξy` the outputs are eliminated first (the joint); with `my` they are given (q(y)).
function conv_x_system!(algo::Conv2dVMP{T}, c, band, ξy, λy, my, ξx, λx) where {T}
    n, m, P, ptaps, order = algo.n, algo.m, algo.positions, algo.ptaps, algo.order
    J, rhs = band.system, band.rhs
    fill!(J, zero(T))
    @inbounds for i in 1:algo.nx
        rhs[order[i]] = ξx[i]
        J[1, order[i]] = λx[i]
    end
    joint = ξy !== nothing
    fill!(c.Bλ, NaN)
    logdetB = zero(T)
    @inbounds for p in 1:P
        if joint
            conv_block!(c, algo, λy, p, true)
            logdetB += c.logdetB
        end
        M = joint ? c.M : c.G
        for j in 1:m
            tj = ptaps[j, p]
            tj == 0 && continue
            g = zero(T)
            if joint
                for i in 1:n
                    g += c.BW[i, j] * ξy[p + (i - 1) * P]
                end
            else
                for i in 1:n
                    g += c.WA[i, j] * my[p + (i - 1) * P]
                end
            end
            rhs[tj] += g
            for l in 1:m
                tl = ptaps[l, p]
                tl >= tj && tl != 0 && (J[tl - tj + 1, tj] += M[l, j])
            end
        end
    end
    band.factor .= J
    banded_cholesky!(band.factor)
    banded_solve!(rhs, band.factor)
    banded_selected_inverse!(band.inverse, band.factor)
    return logdetB
end

# The patch of position p: its mean in c.μp and its covariance in c.Vpp, zero at padding.
function conv_patch!(c::Conv2dCache{T}, algo, band, p) where {T}
    ptaps = algo.ptaps
    @inbounds for j in 1:algo.m
        tj = ptaps[j, p]
        c.μp[j] = tj == 0 ? zero(T) : band.rhs[tj]
        for l in 1:algo.m
            tl = ptaps[l, p]
            c.Vpp[l, j] = (tj == 0 || tl == 0) ? zero(T) : band_entry(band.inverse, tl, tj)
        end
    end
    return c
end

# The message towards every x port from the solved system: the precision as a Schur
# complement over the port's row, which does not subtract its own incoming precision.
function conv_x_messages!(ξout, λout, μ, v, algo::Conv2dVMP{T}, band, ξx, λx, offset) where {T}
    b, N = algo.bandwidth, algo.nx
    J, Z = band.system, band.inverse
    @inbounds for i in 1:N
        t = algo.order[i]
        vt = Z[1, t]
        total = zero(T)
        for d in 1:min(b, N - t)
            total += J[d + 1, t] * Z[d + 1, t]
        end
        for d in 1:min(b, t - 1)
            total += J[d + 1, t - d] * Z[d + 1, t - d]
        end
        μ[offset + i], v[offset + i] = band.rhs[t], vt
        λout[offset + i] = J[1, t] - λx[i] + total / vt
        ξout[offset + i] = band.rhs[t] / vt - ξx[i]
    end
    return nothing
end

function conv_add_moments!(c::Conv2dCache, algo, hasy::Bool, hasx::Bool, cross::Bool)
    n, m = algo.n, algo.m
    @inbounds begin
        if hasx
            for j in 1:m, l in 1:m
                c.Exx[l, j] += c.Vpp[l, j] + c.μp[l] * c.μp[j]
            end
        end
        if hasy
            for j in 1:n, i in 1:n
                c.Eyy[i, j] += c.Cyy[i, j] + c.μy[i] * c.μy[j]
            end
        end
        if hasx && hasy
            for j in 1:m, i in 1:n
                c.Eyx[i, j] += (cross ? c.Cyx[i, j] : zero(eltype(c.Eyx))) + c.μy[i] * c.μp[j]
            end
        end
    end
    return c
end

function conv_reset_moments!(c::Conv2dCache{T}) where {T}
    fill!(c.Exx, zero(T))
    fill!(c.Eyx, zero(T))
    fill!(c.Eyy, zero(T))
    return c
end

conv_entropy(d, logdetJ::T) where {T} = (d * (log(2 * T(π)) + 1) - logdetJ) / 2

# q(y, x) from the messages on every port.
function conv_solve_joint(algo::Conv2dVMP{T}, c, ξy, λy, ξx, λx) where {T}
    n, m, P, ny, nx = algo.n, algo.m, algo.positions, algo.ny, algo.nx
    band = conv_band!(algo)
    logdetB = conv_x_system!(algo, c, band, ξy, λy, nothing, ξx, λx)
    μ, v, ξout, λout = (Vector{T}(undef, ny + nx) for _ in 1:4)
    conv_x_messages!(ξout, λout, μ, v, algo, band, ξx, λx, ny)
    conv_reset_moments!(c)
    fill!(c.Bλ, NaN)
    W, WA = c.W, c.WA
    @inbounds for p in 1:P
        conv_block!(c, algo, λy, p, true)
        conv_patch!(c, algo, band, p)
        # E[y_p] = B⁻¹ (ξ_p + WA E[patch]); Cov(y_p, patch) = B⁻¹WA V; Cov(y_p) = B⁻¹ + (B⁻¹WA) V (B⁻¹WA)'.
        mul!(c.μy, c.BW, c.μp)
        for i in 1:n, j in 1:n
            c.μy[i] += c.Binv[i, j] * ξy[p + (j - 1) * P]
        end
        mul!(c.Cyx, c.BW, c.Vpp)
        c.Cyy .= c.Binv
        mul!(c.Cyy, c.Cyx, c.BW', one(T), one(T))
        conv_add_moments!(c, algo, true, true, true)
        for i in 1:n
            k = p + (i - 1) * P
            vk = c.Cyy[i, i]
            total = zero(T)
            for j in 1:n
                j == i || (total += W[i, j] * c.Cyy[j, i])
            end
            for j in 1:m
                total -= WA[i, j] * c.Cyx[i, j]
            end
            μ[k], v[k] = c.μy[i], vk
            λout[k] = W[i, i] + total / vk
            ξout[k] = c.μy[i] / vk - ξy[k]
        end
    end
    entropy = conv_entropy(ny + nx, logdetB + banded_logdet(band.factor))
    return Conv2dMarginal{T}(:yx, ny, μ, v, ξout, λout, copy(c.Exx), copy(c.Eyx), copy(c.Eyy), true, entropy)
end

# q(x) from the messages on x, given the mean of q(y).
function conv_solve_x(algo::Conv2dVMP{T}, c, my, ξx, λx) where {T}
    nx = algo.nx
    band = conv_band!(algo)
    conv_x_system!(algo, c, band, nothing, nothing, my, ξx, λx)
    μ, v, ξout, λout = (Vector{T}(undef, nx) for _ in 1:4)
    conv_x_messages!(ξout, λout, μ, v, algo, band, ξx, λx, 0)
    # The outputs' means enter the right-hand side, which is not part of the message's own
    # incoming message: nothing more to exclude.
    conv_reset_moments!(c)
    for p in 1:algo.positions
        conv_patch!(c, algo, band, p)
        conv_add_moments!(c, algo, false, true, false)
    end
    empty = zeros(T, 0, 0)
    return Conv2dMarginal{T}(:x, 0, μ, v, ξout, λout, copy(c.Exx), empty, empty, false, conv_entropy(nx, banded_logdet(band.factor)))
end

# q(y) from the messages on y, given q(x): positions are independent. The patches' moments
# come with it when the inputs are independent scalars with means mx and variances vx.
function conv_solve_y(algo::Conv2dVMP{T}, c, ξy, λy, mx, vx) where {T}
    n, m, P, ny = algo.n, algo.m, algo.positions, algo.ny
    μ, v, ξout, λout = (Vector{T}(undef, ny) for _ in 1:4)
    complete = vx !== nothing
    conv_reset_moments!(c)
    fill!(c.Bλ, NaN)
    fill!(c.Vpp, zero(T))
    logdetB = zero(T)
    W = c.W
    @inbounds for p in 1:P
        conv_block!(c, algo, λy, p, false)
        logdetB += c.logdetB
        for j in 1:m
            t = algo.taps[j, p]
            c.μp[j] = t == 0 ? zero(T) : mx[t]
            complete && (c.Vpp[j, j] = t == 0 ? zero(T) : vx[t])
        end
        # E[y_p] = B⁻¹ (ξ_p + WA E[patch]), with the first column of c.Cyx as working memory.
        h = view(c.Cyx, :, 1)
        mul!(h, c.WA, c.μp)
        for i in 1:n
            h[i] += ξy[p + (i - 1) * P]
        end
        mul!(c.μy, c.Binv, h)
        c.Cyy .= c.Binv
        conv_add_moments!(c, algo, true, complete, false)
        for i in 1:n
            k = p + (i - 1) * P
            vk = c.Binv[i, i]
            total = zero(T)
            for j in 1:n
                j == i || (total += W[i, j] * c.Binv[j, i])
            end
            μ[k], v[k] = c.μy[i], vk
            λout[k] = W[i, i] + total / vk
            ξout[k] = c.μy[i] / vk - ξy[k]
        end
    end
    empty = zeros(T, 0, 0)
    return Conv2dMarginal{T}(
        :y, ny, μ, v, ξout, λout, complete ? copy(c.Exx) : empty, complete ? copy(c.Eyx) : empty, copy(c.Eyy), complete, conv_entropy(ny, logdetB),
    )
end

# Which system a target belongs to, from what the rule was given.
function conv_block(qs, group)
    qs.qyx === nothing || conv_layout_error()
    if group === :y && qs.qy === nothing && qs.qx !== nothing
        return :y
    elseif group === :x && qs.qx === nothing && qs.qy !== nothing
        return :x
    elseif qs.qy === nothing && qs.qx === nothing
        return :yx
    end
    return conv_layout_error()
end

function conv_solved!(c::Conv2dCache{T}, algo, block) where {T}
    entry = getproperty(c, block)
    if entry === nothing
        d = block === :yx ? algo.ny + algo.nx : conv_dimension(algo, block)
        entry = Conv2dSolved(T, d, block === :yx ? 0 : block === :x ? algo.ny : 2 * algo.nx)
        setproperty!(c, block, entry)
    end
    return entry
end

function conv_update!(target::AbstractVector{T}, source) where {T}
    slct_equal(target, source) && return false
    target .= source
    return true
end

# Solve the system of `block`, or reuse the last solve when nothing it depends on changed.
# `k` is the port whose own message is missing (0 for none): it keeps the value last seen
# there, which the message towards it does not depend on.
function conv_solve!(algo::Conv2dVMP{T}, @nospecialize(args), qs, block, group, k) where {T}
    for name in keys(args.m)
        (name === :y || name === :x) && (block === :yx || name === block) || conv_layout_error()
    end
    c = conv_transition!(algo, args.q[:a], args.q[:W])
    s = conv_solved!(c, algo, block)
    ny, nx = algo.ny, algo.nx
    changed = s.version != c.version || s.result === nothing
    if block === :yx
        ξy, λy, ξx, λx = view(s.ξ, 1:ny), view(s.λ, 1:ny), view(s.ξ, (ny + 1):(ny + nx)), view(s.λ, (ny + 1):(ny + nx))
        changed |= conv_parameters!(ξy, λy, conv_messages(algo, args, :y, group === :y ? k : 0), :y, group === :y ? k : 0)
        changed |= conv_parameters!(ξx, λx, conv_messages(algo, args, :x, group === :x ? k : 0), :x, group === :x ? k : 0)
        changed && (s.result = conv_solve_joint(algo, c, ξy, λy, ξx, λx))
    elseif block === :x
        changed |= conv_parameters!(s.ξ, s.λ, conv_messages(algo, args, :x, k), :x, k)
        changed |= conv_update!(s.other, conv_means(qs.qy, T))
        changed && (s.result = conv_solve_x(algo, c, s.other, s.ξ, s.λ))
    else
        changed |= conv_parameters!(s.ξ, s.λ, conv_messages(algo, args, :y, k), :y, k)
        mx, vx = qs.qx isa Tuple ? conv_mean_var(qs.qx, T) : (conv_means(qs.qx, T), nothing)
        changed |= conv_update!(view(s.other, 1:nx), mx)
        changed |= conv_update!(view(s.other, (nx + 1):(2nx)), vx === nothing ? fill(T(-1), nx) : vx)
        changed && (s.result = conv_solve_y(algo, c, s.ξ, s.λ, mx, vx))
    end
    if changed
        s.version = c.version
        s.solves += 1
    end
    return s.result::Conv2dMarginal{T}
end

function conv_parameters(algo::Conv2dVMP{T}, args, name) where {T}
    d = conv_dimension(algo, name)
    ξ, λ = Vector{T}(undef, d), Vector{T}(undef, d)
    conv_parameters!(ξ, λ, conv_messages(algo, args, name, 0), name, 0)
    return ξ, λ
end

# The marginal of a whole group, from every incoming message of its members.
function conv_marginal(algo::Conv2dVMP{T}, args, group) where {T}
    qs = conv_marginals(algo, args.q)
    qs.qyx === nothing || conv_layout_error()
    for name in keys(args.m)
        (name === :y || name === :x) && (group === :yx || name === group) || conv_layout_error()
    end
    c = conv_transition!(algo, args.q[:a], args.q[:W])
    if group === :yx
        qs.qy === nothing && qs.qx === nothing || conv_layout_error()
        return conv_solve_joint(algo, c, conv_parameters(algo, args, :y)..., conv_parameters(algo, args, :x)...)
    elseif group === :x
        qs.qx === nothing && qs.qy !== nothing || conv_layout_error()
        return conv_solve_x(algo, c, conv_means(qs.qy, T), conv_parameters(algo, args, :x)...)
    end
    qs.qy === nothing && qs.qx !== nothing || conv_layout_error()
    mx, vx = qs.qx isa Tuple ? conv_mean_var(qs.qx, T) : (conv_means(qs.qx, T), nothing)
    return conv_solve_y(algo, c, conv_parameters(algo, args, :y)..., mx, vx)
end

# The message towards one port under the default scheme.
function conv_message(algo, @nospecialize(args), group, k)
    1 <= k <= conv_dimension(algo, group) || throw(ArgumentError("target index $k is outside group $group"))
    qs = conv_marginals(algo, args.q)
    block = conv_block(qs, group)
    return conv_port(conv_solve!(algo, args, qs, block, group, k), group, k)
end

# The message towards one port, read from the marginal of its group.
function conv_port(q::Conv2dMarginal, group, k)
    members = group === :y ? q.ny : length(q) - q.ny
    (group === :y || q.members !== :y) && (group === :x || q.members !== :x) || throw(ArgumentError("the marginal of $(q.members) has no member of $group"))
    1 <= k <= members || throw(ArgumentError("target index $k is outside group $group"))
    i = group === :x ? q.ny + k : k
    return NormalWeightedMeanPrecision(q.ξ[i], q.λ[i])
end

# Second moments of the patches from dense marginals, summed over positions.
function conv_accumulate!(c, algo, my, Vy, mx, Vx, Vyx)
    n, P = algo.n, algo.positions
    conv_reset_moments!(c)
    for p in 1:P
        for j in 1:algo.m
            tj = algo.taps[j, p]
            tj == 0 && continue
            if mx !== nothing
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
        end
        Vy === nothing && continue
        for j in 1:n, i in 1:n
            ki, kj = p + (i - 1) * P, p + (j - 1) * P
            c.Eyy[i, j] += Vy[ki, kj] + my[ki] * my[kj]
        end
    end
    return c
end

# The cross moment of independent groups: Σ_p E[y_p] E[patch_p]'.
function conv_cross!(c::Conv2dCache{T}, algo, my, mx) where {T}
    fill!(c.Eyx, zero(T))
    P = algo.positions
    @inbounds for p in 1:P, j in 1:algo.m
        t = algo.taps[j, p]
        t == 0 && continue
        for i in 1:algo.n
            c.Eyx[i, j] += my[p + (i - 1) * P] * mx[t]
        end
    end
    return c
end

function conv_all_pointmass(@nospecialize(q))
    q isa Tuple || return false
    M = eltype(q)
    return (isconcretetype(M) && M <: PointMass) || all(value -> value isa PointMass, q)
end

# E[x x'], E[y x'] and E[y y'] of the patches, summed over positions: all that the messages
# towards a and W and the energy need, since every position has the same A and W.
function conv_moments!(algo::Conv2dVMP{T}, args) where {T}
    isempty(keys(args.m)) || conv_layout_error()
    qs = conv_marginals(algo, args.q)
    c = algo.cache
    qyx, qy, qx = qs.qyx, qs.qy, qs.qx
    if qyx isa Conv2dMarginal
        qyx.complete || conv_layout_error()
        return conv_copy_moments!(c, qyx)
    elseif qyx !== nothing
        my, Vy, mx, Vx, Vyx = split_y_x(qyx, algo.ny)
        return conv_accumulate!(c, algo, my, Vy, mx, Vx, Vyx)
    end
    qy !== nothing || conv_layout_error()
    # q(y) given observed pixels already holds the moments of their patches.
    qy isa Conv2dMarginal && qy.complete && (qx === nothing || conv_all_pointmass(qx)) && return conv_copy_moments!(c, qy)
    qx !== nothing || conv_layout_error()
    my = conv_means(qy, T)
    if qx isa Conv2dMarginal || qx isa Tuple
        mx, vx = conv_mean_var(qx, T)
        Vy = qy isa Conv2dMarginal ? nothing : qy isa Tuple ? Diagonal(last(conv_mean_var(qy, T))) : cov(qy)
        if qx isa Conv2dMarginal
            conv_accumulate!(c, algo, my, Vy, nothing, nothing, nothing)
            c.Exx .= qx.Exx
        else
            conv_accumulate!(c, algo, my, Vy, mx, Diagonal(vx), nothing)
        end
        qy isa Conv2dMarginal && (c.Eyy .= qy.Eyy)
        return conv_cross!(c, algo, my, mx)
    end
    mx, Vx = mean_cov(qx)
    Vy = qy isa Conv2dMarginal ? nothing : qy isa Tuple ? Diagonal(last(conv_mean_var(qy, T))) : cov(qy)
    conv_accumulate!(c, algo, my, Vy, mx, Vx, nothing)
    qy isa Conv2dMarginal && (c.Eyy .= qy.Eyy)
    return c
end

function conv_copy_moments!(c, q::Conv2dMarginal)
    c.Exx .= q.Exx
    c.Eyx .= q.Eyx
    c.Eyy .= q.Eyy
    return c
end

function conv_residual!(algo, args)
    c = conv_moments!(algo, args)
    a = slct_gaussian!(c.a, args.q[:a])
    A, R = reshape(a.μ, algo.n, algo.m), c.residual
    mul!(c.cross, c.Eyx, A')
    R .= c.Eyy .- c.cross .- c.cross'
    mul!(c.AX, A, c.Exx)
    mul!(R, c.AX, A', true, true)
    n = algo.n
    @inbounds for l in 1:algo.m, k in 1:algo.m
        e = c.Exx[k, l]
        for j in 1:n, i in 1:n
            R[i, j] += e * a.V[(k - 1) * n + i, (l - 1) * n + j]
        end
    end
    for j in axes(R, 2), i in 1:(j - 1)
        R[i, j] = R[j, i] = (R[i, j] + R[j, i]) / 2
    end
    return R
end

# Under the default scheme every port's rule receives the other messages of its group.
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

# With `ports` other than `:default` a port reads its message from its group's marginal.
@define_message_update_rule(
    node = Conv2dCT, target = (:y, k), algorithm = Conv2dVMP,
    args = (q[:y, :x]::Conv2dMarginal,), pure = true,
    body = (args) -> conv_port(args.q[:y, :x], :y, k),
)

@define_message_update_rule(
    node = Conv2dCT, target = (:x, k), algorithm = Conv2dVMP,
    args = (q[:y, :x]::Conv2dMarginal,), pure = true,
    body = (args) -> conv_port(args.q[:y, :x], :x, k),
)

@define_message_update_rule(
    node = Conv2dCT, target = (:y, k), algorithm = Conv2dVMP,
    args = (q[(:y,)]::Conv2dMarginal,), pure = true,
    body = (args) -> conv_port(args.q[(:y,)], :y, k),
)

@define_message_update_rule(
    node = Conv2dCT, target = (:x, k), algorithm = Conv2dVMP,
    args = (q[(:x,)]::Conv2dMarginal,), pure = true,
    body = (args) -> conv_port(args.q[(:x,)], :x, k),
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

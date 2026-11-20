# Assemble either the full [y; x] canonical system or one mean-field state block.
# k == 0 includes all incoming messages for a group marginal.
function slct_canonical!(scratch, algo, args, group, k)
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
    n, m = algo.n, algo.m
    d = block === :yx ? n + m : slct_dimension(algo, block)
    J, h = view(scratch.J, 1:d, 1:d), view(scratch.h, 1:d)
    fill!(h, zero(eltype(h)))
    mW = lct_mean(args.q[:W], (n, n), :W)
    mA = ct_matrix(algo, args.q[:a])
    mul!(scratch.WA, mW, mA)
    if block === :y
        J .= mW
        mul!(h, scratch.WA, slct_mean(qs.qx))
    else
        lct_uncertainty!(scratch.K, algo, args.q[:a], mW)
        if block === :x
            J .= scratch.K
            mul!(J, mA', scratch.WA, true, true)
            mul!(h, scratch.WA', slct_mean(qs.qy))
        else
            y, x = 1:n, (n + 1):(n + m)
            J[y, y] .= mW
            J[y, x] .= .-scratch.WA
            J[x, y] .= .-scratch.WA'
            J[x, x] .= scratch.K
            mul!(view(J, x, x), mA', scratch.WA, true, true)
        end
    end
    for name in keys(args.m)
        offset = block === :yx && name === :x ? n : 0
        for (i, value) in enumerate(args.m[name])
            value === nothing && continue
            ξ, λ = weightedmean_precision(value)
            h[offset + i] += ξ
            J[offset + i, offset + i] += λ
        end
    end
    # Roundoff from matrix multiplication must not leak asymmetric precisions to consumers.
    for j in 1:d, i in 1:(j - 1)
        J[i, j] = J[j, i] = (J[i, j] + J[j, i]) / 2
    end
    index = block === :yx && group === :x ? n + k : k
    return J, h, index
end

# Eliminate every coordinate but `target` from the canonical system (J, h).
function slct_eliminate(eliminate, solve, J, h, target)
    d = length(h)
    d == 1 && return NormalWeightedMeanPrecision(h[1], J[1, 1])
    E, rhs = view(eliminate, 1:(d - 1), 1:(d - 1)), view(solve, 1:(d - 1), :)
    for j in 1:(d - 1)
        jj = j < target ? j : j + 1
        rhs[j, 1] = h[jj]
        rhs[j, 2] = J[jj, target]
        for i in 1:(d - 1)
            ii = i < target ? i : i + 1
            E[i, j] = J[ii, jj]
        end
    end
    ldiv!(cholesky!(Symmetric(E)), rhs)
    ξ, λ = h[target], J[target, target]
    for i in 1:(d - 1)
        ii = i < target ? i : i + 1
        ξ -= J[target, ii] * rhs[i, 1]
        λ -= J[target, ii] * rhs[i, 2]
    end
    return NormalWeightedMeanPrecision(ξ, λ)
end

function slct_scalar_message(scratch, algo, args, group, k)
    k >= 1 || throw(ArgumentError("target index must be positive, got $k"))
    J, h, target = slct_canonical!(scratch, algo, args, group, k)
    return slct_eliminate(scratch.eliminate, scratch.rhs, J, h, target)
end

function slct_group_marginal(scratch, algo, args, group)
    J, h, _ = slct_canonical!(scratch, algo, args, group, 0)
    return MvNormalWeightedMeanPrecision(copy(h), copy(J))
end

@define_message_update_rule(
    node = ScalarCT, target = (:y, k), algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> slct_state_scratch(algo, args),
    body = (scratch, algo, args) -> slct_scalar_message(scratch, algo, args, :y, k),
)

@define_message_update_rule(
    node = ScalarCT, target = (:x, k), algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> slct_state_scratch(algo, args),
    body = (scratch, algo, args) -> slct_scalar_message(scratch, algo, args, :x, k),
)

@define_message_update_rule(
    node = ScalarCT, target = :a, algorithm = LCTVMP,
    args = (default, q[:W]::Any),
    scratch = (algo, args) -> lct_moment_scratch(algo, slct_eltype(args)),
    body = (scratch, algo, args) -> begin
        my, _, mx, Vx, Vyx = slct_state_moments(algo, args)
        lct_towards_a(scratch, algo, my, mx, Vx, Vyx, args.q[:W])
    end,
)

@define_message_update_rule(
    node = ScalarCT, target = :W, algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily),
    scratch = (algo, args) -> lct_residual_scratch(algo, slct_eltype(args)),
    body = (scratch, algo, args) -> begin
        my, Vy, mx, Vx, Vyx = slct_state_moments(algo, args)
        WishartFast(algo.n + 2, lct_residual(scratch, algo, my, Vy, mx, Vx, Vyx, args.q[:a]))
    end,
)

@define_marginal_update_rule(
    node = ScalarCT, target = (:y,), algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> slct_state_scratch(algo, args),
    body = (scratch, algo, args) -> slct_group_marginal(scratch, algo, args, :y),
)

@define_marginal_update_rule(
    node = ScalarCT, target = (:x,), algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> slct_state_scratch(algo, args),
    body = (scratch, algo, args) -> slct_group_marginal(scratch, algo, args, :x),
)

@define_marginal_update_rule(
    node = ScalarCT, target = (:y, :x), algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> slct_state_scratch(algo, args),
    body = (scratch, algo, args) -> slct_group_marginal(scratch, algo, args, :yx),
)

@define_average_energy(
    node = ScalarCT, algorithm = LCTVMP,
    args = (default, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    body = (algo, args) -> begin
        my, Vy, mx, Vx, Vyx = slct_state_moments(algo, args)
        lct_mean(args.q[:W], (algo.n, algo.n), :W)
        ct_energy(algo, my, Vy, mx, Vx, Vyx, args.q[:a], args.q[:W])
    end,
)

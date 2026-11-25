# For a = vec(A), block (k, l) of Va is Cov(A[:, k], A[:, l]). Contract these blocks directly;
# neither the identity Jacobian of vec ∘ reshape nor its row-selection matrices are needed.
function lct_uncertainty!(K, algo::LCTVMP, q_a, mW)
    n, m = algo.n, algo.m
    Va = lct_size(cov(q_a), (n * m, n * m), :a)
    lct_size(mW, (n, n), :W)
    lct_size(K, (m, m), :K)
    Base.require_one_based_indexing(K, Va, mW)
    # Accumulate each block contraction locally: repeatedly updating K[k, l]
    # prevents the compiler from keeping the reduction in a scalar register.
    @inbounds for l in 1:m, k in 1:m
        total = zero(eltype(K))
        for j in 1:n, i in 1:n
            total += mW[i, j] * Va[(k - 1) * n + i, (l - 1) * n + j]
        end
        K[k, l] = total
    end
    return K
end

function lct_towards_a(scratch, algo::LCTVMP, my, mx, Vx, Vyx, q_W)
    mW = lct_mean(q_W, (algo.n, algo.n), :W)
    scratch.Exx .= Vx .+ mx .* mx'
    scratch.Eyx .= Vyx .+ my .* mx'
    return MvNormalWeightedMeanPrecision(vec(mW * scratch.Eyx), kron(scratch.Exx, mW))
end

function lct_towards_x(scratch, algo::LCTVMP, W̃, my, q_a, mW)
    mA = ct_matrix(algo, q_a)
    mul!(scratch.WA, W̃, mA)
    W = Matrix{eltype(scratch.WA)}(undef, algo.m, algo.m)
    lct_uncertainty!(W, algo, q_a, mW)
    mul!(W, mA', scratch.WA, true, true)
    return MvNormalWeightedMeanPrecision(scratch.WA' * my, W)
end

function lct_residual(scratch, algo::LCTVMP, my, Vy, mx, Vx, Vyx, q_a)
    n, m = algo.n, algo.m
    mA = ct_matrix(algo, q_a)
    Va = lct_size(cov(q_a), (n * m, n * m), :a)
    Exx, Eyx = scratch.Exx, scratch.Eyx
    Exx .= Vx .+ mx .* mx'
    Eyx .= Vyx .+ my .* mx'
    mul!(scratch.cross, Eyx, mA')
    # Only R escapes the rule. Never return a scratch array as a message parameter.
    R = Matrix{eltype(Exx)}(undef, n, n)
    R .= Vy .+ my .* my' .- scratch.cross .- scratch.cross'
    mul!(scratch.AX, mA, Exx)
    mul!(R, scratch.AX, mA', true, true)
    for l in 1:m, k in 1:m
        C = @view Va[((k - 1) * n + 1):(k * n), ((l - 1) * n + 1):(l * n)]
        R .+= Exx[k, l] .* C
    end
    for j in 1:n, i in 1:(j - 1)
        R[i, j] = R[j, i] = (R[i, j] + R[j, i]) / 2
    end
    return R
end

# Average energies have no persistent scratch slot in the rule API.
function ct_residual(algo::LCTVMP, my, Vy, mx, Vx, Vyx, q_a)
    T = promote_type(eltype(my), eltype(Vy), eltype(mx), eltype(Vx), eltype(Vyx), eltype(mean(q_a)), eltype(cov(q_a)))
    return lct_residual(lct_residual_scratch(algo, T), algo, my, Vy, mx, Vx, Vyx, q_a)
end

@define_message_update_rule(
    node = LinearContinuousTransition, target = :y, algorithm = LCTVMP,
    args = (m[:x]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> lct_y_scratch(algo, lct_eltype(args.m[:x], args.q[:a], args.q[:W])),
    body = (scratch, algo, args) -> begin
        mx, Vx = lct_mean_cov(args.m[:x], algo.m, :x)
        mA = ct_matrix(algo, args.q[:a])
        mW = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        lct_uncertainty!(scratch.K, algo, args.q[:a], mW)
        mul!(scratch.G, Vx, scratch.K)
        for k in 1:algo.m
            scratch.G[k, k] += one(eltype(scratch.G))
        end
        scratch.rhs[:, 1] .= mx
        scratch.rhs[:, 2:end] .= Vx
        ldiv!(lu!(scratch.G), scratch.rhs)
        μ = mA * view(scratch.rhs, :, 1)
        mul!(scratch.AX, mA, view(scratch.rhs, :, 2:(algo.m + 1)))
        V = Matrix{eltype(scratch.AX)}(cholinv(mW))
        mul!(V, scratch.AX, mA', true, true)
        MvNormalMeanCovariance(μ, V)
    end,
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :y, algorithm = LCTVMP,
    args = (q[:x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> MvNormalMeanPrecision(
        ct_matrix(algo, args.q[:a]) * lct_mean(args.q[:x], (algo.m,), :x),
        lct_mean(args.q[:W], (algo.n, algo.n), :W),
    ),
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :x, algorithm = LCTVMP,
    args = (m[:y]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> lct_x_scratch(algo, lct_eltype(args.m[:y], args.q[:a], args.q[:W])),
    body = (scratch, algo, args) -> begin
        my, Wy = mean_precision(args.m[:y])
        lct_size(my, (algo.n,), :y)
        lct_size(Wy, (algo.n, algo.n), :y)
        mW = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        mul!(scratch.tmp, Wy, cholinv(Wy + mW))
        mul!(scratch.Weff, scratch.tmp, Wy)
        scratch.Weff .= Wy .- scratch.Weff
        lct_towards_x(scratch, algo, scratch.Weff, my, args.q[:a], mW)
    end,
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :x, algorithm = LCTVMP,
    args = (q[:y]::Any, q[:a]::Any, q[:W]::Any),
    scratch = (algo, args) -> (WA = Matrix{lct_eltype(args.q[:y], args.q[:a], args.q[:W])}(undef, algo.n, algo.m),),
    body = (scratch, algo, args) -> begin
        mW = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        lct_towards_x(scratch, algo, mW, lct_mean(args.q[:y], (algo.n,), :y), args.q[:a], mW)
    end,
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :a, algorithm = LCTVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:W]::Any),
    scratch = (algo, args) -> lct_moment_scratch(algo, lct_eltype(args.q[:y, :x], args.q[:W])),
    body = (scratch, algo, args) -> begin
        my, _, mx, Vx, Vyx = lct_split_y_x(algo, args.q[:y, :x])
        lct_towards_a(scratch, algo, my, mx, Vx, Vyx, args.q[:W])
    end,
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :a, algorithm = LCTVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:W]::Any),
    scratch = (algo, args) -> lct_moment_scratch(algo, lct_eltype(args.q[:y], args.q[:x], args.q[:W])),
    body = (scratch, algo, args) -> begin
        mx, Vx = lct_mean_cov(args.q[:x], algo.m, :x)
        my = lct_mean(args.q[:y], (algo.n,), :y)
        lct_towards_a(scratch, algo, my, mx, Vx, 0, args.q[:W])
    end,
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :W, algorithm = LCTVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:a]::MultivariateNormalDistributionsFamily),
    scratch = (algo, args) -> lct_residual_scratch(algo, lct_eltype(args.q[:y, :x], args.q[:a])),
    body = (scratch, algo, args) -> begin
        my, Vy, mx, Vx, Vyx = lct_split_y_x(algo, args.q[:y, :x])
        WishartFast(algo.n + 2, lct_residual(scratch, algo, my, Vy, mx, Vx, Vyx, args.q[:a]))
    end,
)

@define_message_update_rule(
    node = LinearContinuousTransition, target = :W, algorithm = LCTVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:a]::Any),
    scratch = (algo, args) -> lct_residual_scratch(algo, lct_eltype(args.q[:y], args.q[:x], args.q[:a])),
    body = (scratch, algo, args) -> begin
        my, Vy = lct_mean_cov(args.q[:y], algo.n, :y)
        mx, Vx = lct_mean_cov(args.q[:x], algo.m, :x)
        WishartFast(algo.n + 2, lct_residual(scratch, algo, my, Vy, mx, Vx, 0, args.q[:a]))
    end,
)

@define_marginal_update_rule(
    node = LinearContinuousTransition, target = (:y, :x), algorithm = LCTVMP,
    args = (m[:y]::MultivariateNormalDistributionsFamily, m[:x]::MultivariateNormalDistributionsFamily, q[:a]::Any, q[:W]::Any),
    scratch = (algo, args) -> begin
        T = lct_eltype(args.m[:y], args.m[:x], args.q[:a], args.q[:W])
        (K = Matrix{T}(undef, algo.m, algo.m), WA = Matrix{T}(undef, algo.n, algo.m))
    end,
    body = (scratch, algo, args) -> begin
        mW = lct_mean(args.q[:W], (algo.n, algo.n), :W)
        mA = ct_matrix(algo, args.q[:a])
        ξy, Wy = weightedmean_precision(args.m[:y])
        ξx, Wx = weightedmean_precision(args.m[:x])
        lct_size(ξy, (algo.n,), :y)
        lct_size(Wy, (algo.n, algo.n), :y)
        lct_size(ξx, (algo.m,), :x)
        lct_size(Wx, (algo.m, algo.m), :x)
        lct_uncertainty!(scratch.K, algo, args.q[:a], mW)
        mul!(scratch.WA, mW, mA)
        W = Matrix{eltype(scratch.K)}(undef, algo.n + algo.m, algo.n + algo.m)
        y, x = 1:algo.n, (algo.n + 1):(algo.n + algo.m)
        W[y, y] .= Wy .+ mW
        W[y, x] .= .-scratch.WA
        W[x, y] .= .-scratch.WA'
        W[x, x] .= Wx .+ scratch.K
        mul!(view(W, x, x), mA', scratch.WA, true, true)
        MvNormalWeightedMeanPrecision([ξy; ξx], W)
    end,
)

@define_average_energy(
    node = LinearContinuousTransition, algorithm = LCTVMP,
    args = (q[:y, :x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> begin
        lct_mean(args.q[:W], (algo.n, algo.n), :W)
        my, Vy, mx, Vx, Vyx = lct_split_y_x(algo, args.q[:y, :x])
        ct_energy(algo, my, Vy, mx, Vx, Vyx, args.q[:a], args.q[:W])
    end,
)

@define_average_energy(
    node = LinearContinuousTransition, algorithm = LCTVMP,
    args = (q[:y]::Any, q[:x]::Any, q[:a]::Any, q[:W]::Any),
    body = (algo, args) -> begin
        lct_mean(args.q[:W], (algo.n, algo.n), :W)
        my, Vy = lct_mean_cov(args.q[:y], algo.n, :y)
        mx, Vx = lct_mean_cov(args.q[:x], algo.m, :x)
        ct_energy(algo, my, Vy, mx, Vx, zeros(eltype(my), algo.n, algo.m), args.q[:a], args.q[:W])
    end,
)

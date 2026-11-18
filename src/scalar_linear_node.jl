"""
    ScalarLinearContinuousTransition

The density `y ~ N(reshape(a, n, m) * x, W⁻¹)` with separate scalar state ports.
Its interfaces are `[:y..., :x..., :a, :W]`: `n` scalar outputs, `m` scalar inputs,
the `n*m`-vector `a = vec(A)`, and the `n × n` noise precision. Supply `LCTVMP(n, m)`.

Supports `q(y)q(x)q(a)q(W)` and `q(y,x)q(a)q(W)`, with full covariance inside each
state group. Incoming state messages are univariate Gaussians; group marginals are
multivariate Gaussians in interface order, `[y; x]` for the full joint. Separate scalar
messages cannot encode arbitrary correlated multivariate incoming messages.

Address scalar ports with `target = (:y, k)` or `(:x, k)`. Pass message tuples with
`nothing` at the excluded target, and group marginals through `clusters`. Wrong group
sizes raise `DimensionMismatch` during rule execution. Singleton groups may use the
engine's one-element tuple of scalar marginals instead of a joint marginal.

Rules use covariance contractions and Gaussian elimination without Jacobians. The engine
reuses scratch per stream, and returned parameters never alias it. Neither the algorithm
nor the existing `LinearContinuousTransition` node is modified.

# Examples

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(
           node = ScalarCT, target = (:y, 1), algorithm = LCTVMP(1, 1),
           m = (y = (nothing,),),
           clusters = ((:x,) => MvNormalMeanCovariance([2.0], [1.0;;]),),
           q = (a = MvNormalMeanCovariance([3.0], [1.0;;]), W = Wishart(2, [0.5;;])),
       );

julia> mean(getresult(result)) ≈ 6.0
true
```
"""
struct ScalarLinearContinuousTransition end

"""Alias for [`ScalarLinearContinuousTransition`](@ref)."""
const ScalarCT = ScalarLinearContinuousTransition

@define_factor_node(node = ScalarLinearContinuousTransition, type = Stochastic, interfaces = [:y..., :x..., :a, :W])
@define_dependencies(
    node = ScalarLinearContinuousTransition, algorithm = LCTVMP,
    dependencies = [(:y, k) => (default,), (:x, k) => (default,), :a => (default,), :W => (default,)],
)

slct_dimension(algo, group) = group === :y ? algo.n : algo.m
slct_layout_error() = throw(ArgumentError("ScalarCT supports only q(y)q(x)q(a)q(W) or q(y,x)q(a)q(W), with whole state groups"))

function slct_count(values, d, group)
    values isa Tuple || throw(ArgumentError("$group must be a tuple of scalar Gaussian messages or marginals"))
    length(values) == d || throw(DimensionMismatch("$group must contain $d members, got $(length(values))"))
    return values
end

function slct_joint(q, d, name)
    q isa MultivariateNormalDistributionsFamily || throw(ArgumentError("$name must be a multivariate Gaussian"))
    length(q) == d || throw(DimensionMismatch("$name must have dimension $d, got $(length(q))"))
    return q
end

# The engine uses scalar singleton marginals, but whole-group joint keys for larger groups.
# Inspect the raw tuples as well as rule_inputs: the latter deliberately omits `nothing`.
function slct_marginals(algo, args)
    for group in (:y, :x)
        if group in keys(args.q)
            values = slct_count(args.q[group], slct_dimension(algo, group), group)
            length(values) == 1 && only(values) isa UnivariateNormalDistributionsFamily || slct_layout_error()
        end
    end
    qy = qx = qyx = nothing
    for (key, value) in MessagePassingRulesBase.rule_inputs(ScalarCT, args.q)
        if key === (:y,)
            qy === nothing || slct_layout_error()
            qy = slct_joint(value, algo.n, :y)
        elseif key === (:x,)
            qx === nothing || slct_layout_error()
            qx = slct_joint(value, algo.m, :x)
        elseif key === (:y, :x)
            qyx = slct_joint(value, algo.n + algo.m, :y_x)
        elseif key === (:y, 1) && algo.n == 1
            qy === nothing || slct_layout_error()
            qy = value
        elseif key === (:x, 1) && algo.m == 1
            qx === nothing || slct_layout_error()
            qx = value
        elseif key !== :a && key !== :W
            slct_layout_error()
        end
    end
    qyx === nothing || (qy === nothing && qx === nothing) || slct_layout_error()
    return (; qy, qx, qyx)
end

slct_mean(q::MultivariateNormalDistributionsFamily) = mean(q)
slct_mean(q::UnivariateNormalDistributionsFamily) = [mean(q)]
slct_mean_cov(q::MultivariateNormalDistributionsFamily) = mean_cov(q)
slct_mean_cov(q::UnivariateNormalDistributionsFamily) = ([mean(q)], reshape([var(q)], 1, 1))

function slct_state_moments(algo, args)
    isempty(keys(args.m)) || slct_layout_error()
    qs = slct_marginals(algo, args)
    if qs.qyx !== nothing
        return lct_split_y_x(algo, qs.qyx)
    end
    qs.qy !== nothing && qs.qx !== nothing || slct_layout_error()
    my, Vy = slct_mean_cov(qs.qy)
    mx, Vx = slct_mean_cov(qs.qx)
    return my, Vy, mx, Vx, 0
end

function slct_eltype(args)
    inputs = (MessagePassingRulesBase.rule_inputs(ScalarCT, args.m)..., MessagePassingRulesBase.rule_inputs(ScalarCT, args.q)...)
    return promote_type(map(p -> eltype(mean(last(p))), inputs)...)
end

function slct_state_scratch(algo, args)
    T = slct_eltype(args)
    d = algo.n + algo.m
    return (
        J = Matrix{T}(undef, d, d), h = Vector{T}(undef, d),
        K = Matrix{T}(undef, algo.m, algo.m), WA = Matrix{T}(undef, algo.n, algo.m),
        eliminate = Matrix{T}(undef, d - 1, d - 1), rhs = Matrix{T}(undef, d - 1, 2),
    )
end

function slct_messages(algo, args, block, group, k)
    for name in keys(args.m)
        name in (:y, :x) || slct_layout_error()
        block === :yx || name === block || slct_layout_error()
    end
    for name in (:y, :x)
        block === :yx || name === block || continue
        d = slct_dimension(algo, name)
        if !(name in keys(args.m))
            name === group && d == 1 && k == 1 && continue
            throw(ArgumentError("missing incoming messages for $name (expected $d members)"))
        end
        values = slct_count(args.m[name], d, name)
        for (i, value) in enumerate(values)
            if name === group && i == k
                value === nothing || throw(ArgumentError("the target's own incoming message must be nothing"))
            else
                value isa UnivariateNormalDistributionsFamily || throw(ArgumentError("$name[$i] must be a scalar Gaussian message"))
            end
        end
    end
    return nothing
end

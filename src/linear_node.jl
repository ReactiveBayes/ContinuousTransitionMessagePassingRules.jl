"""
    LinearContinuousTransition

The stochastic node `y ~ N(A * x, W⁻¹)` with `A = reshape(a, n, m)`. Its interfaces are
`[:y, :x, :a, :W]`: an `n`-vector output, an `m`-vector input, the `n*m`-vector of matrix
entries, and the `n × n` noise precision. Julia's column-major ordering applies: `a = vec(A)`.

Supply [`LCTVMP`](@ref)`(n, m)` to specify the shape. Its rules are the exact reshape case of
[`ContinuousTransition`](@ref) with `CTVMP(a -> reshape(a, n, m))`, computed without Jacobians
or linearisation. They support mean-field `q(y)q(x)q(a)q(W)` and structured `q(y, x)q(a)q(W)`:
messages towards all four interfaces, the joint marginal `q(y, x)`, and both average energies.
Gaussian solves and contractions of the full covariance of `a` are still required.

The distribution families are the same as for `ContinuousTransition`. The message towards
`a` does not read `q(a)` or require an initial expansion point; other messages still need
`q(a)`. There are no rules under `DefaultAlgorithm()` or for belief propagation.

# Examples

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, LinearAlgebra

julia> result = @call_message_update_rule(
           node = LinearContinuousTransition, target = :y, algorithm = LCTVMP(2, 3),
           q = (x = MvNormalMeanCovariance([1.0, 2.0, 3.0], Matrix(1.0I, 3, 3)),
                a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], Matrix(1.0I, 6, 6)),
                W = Wishart(3.0, Matrix(1.0I, 2, 2))),
       );

julia> mean(getresult(result)) ≈ [4.0, 5.0]
true
```
"""
struct LinearContinuousTransition end

"""
    LCTVMP(n, m)

Shape configuration for [`LinearContinuousTransition`](@ref): `A = reshape(a, n, m)` maps an
`m`-vector to an `n`-vector. Both dimensions must be positive integers. The algorithm stores
only these dimensions, not a transformation function or Jacobians; the factorisation selects
ordinary mean-field or structured VMP.

Message and marginal rules declare scratch matrices that the inference engine allocates once
per outbound stream and reuses. Scratch is separate from the algorithm and from returned
message parameters, so sharing this immutable configuration does not share mutable buffers.
Direct calls allocate fresh scratch unless the caller supplies it through the rule execution
API. Average energies do not retain scratch between calls.

Explicit dimensions are needed in particular for the structured message towards `W`, whose
inputs `q(y, x)` and `q(a)` do not determine which dimension belongs to `y`.
"""
struct LCTVMP <: AbstractAlgorithm
    n::Int
    m::Int

    function LCTVMP(n::Integer, m::Integer)
        n > 0 && m > 0 || throw(ArgumentError("LCTVMP dimensions must be positive, got ($n, $m)"))
        return new(n, m)
    end
end

@define_factor_node(node = LinearContinuousTransition, type = Stochastic, interfaces = [:y, :x, :a, :W])

# No expansion point: even the message towards `a` follows the default dependency scheme.
@define_dependencies(
    node = LinearContinuousTransition, algorithm = LCTVMP,
    dependencies = [:y => (default,), :x => (default,), :a => (default,), :W => (default,)],
)

function lct_size(value, dims, name)
    size(value) == dims || throw(DimensionMismatch("$name must have size $dims, got $(size(value))"))
    return value
end

lct_mean(q, dims, name) = lct_size(mean(q), dims, name)

function lct_mean_cov(q, d, name)
    m, V = mean_cov(q)
    return lct_size(m, (d,), name), lct_size(V, (d, d), name)
end

function lct_split_y_x(algo::LCTVMP, q_y_x)
    lct_mean(q_y_x, (algo.n + algo.m,), :y_x)
    return split_y_x(q_y_x, algo.n)
end

ct_matrix(algo::LCTVMP, q_a) = reshape(lct_mean(q_a, (algo.n * algo.m,), :a), algo.n, algo.m)

lct_eltype(qs...) = promote_type(map(q -> eltype(mean(q)), qs)...)

lct_moment_scratch(algo::LCTVMP, ::Type{T}) where {T} =
    (Exx = Matrix{T}(undef, algo.m, algo.m), Eyx = Matrix{T}(undef, algo.n, algo.m))

function lct_residual_scratch(algo::LCTVMP, ::Type{T}) where {T}
    return (; lct_moment_scratch(algo, T)..., AX = Matrix{T}(undef, algo.n, algo.m), cross = Matrix{T}(undef, algo.n, algo.n))
end

function lct_y_scratch(algo::LCTVMP, ::Type{T}) where {T}
    n, m = algo.n, algo.m
    return (K = Matrix{T}(undef, m, m), G = Matrix{T}(undef, m, m), rhs = Matrix{T}(undef, m, m + 1), AX = Matrix{T}(undef, n, m))
end

function lct_x_scratch(algo::LCTVMP, ::Type{T}) where {T}
    n, m = algo.n, algo.m
    return (WA = Matrix{T}(undef, n, m), tmp = Matrix{T}(undef, n, n), Weff = Matrix{T}(undef, n, n))
end

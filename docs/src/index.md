# ContinuousTransitionMessagePassingRules

The [`ContinuousTransition`](@ref) node and its variational rules. Use it for a linear Gaussian
transition `y ~ N(A x, W⁻¹)` whose matrix `A = f(a)` and noise precision `W` are learned along
with the states: a state-space model with an unknown transition, a rotation by an unknown angle,
a regression with an unknown matrix of coefficients.

```@docs
ContinuousTransitionMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

This page follows the node from its definition to its rules, an example
and its limitations, and ends with the API.

## Overview

`ContinuousTransition` carries an `m`-dimensional `x` to an `n`-dimensional `y` through the
`n × m` matrix `f(a)`, with Gaussian noise of precision `W`. The transformation `f` makes the
matrix as free or as structured as the model needs: `a -> reshape(a, n, m)` learns every entry,
while a rotation `a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]` learns one angle.

The node is a [stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node), a
density over its variables, and its [rules](@extref MessagePassingRulesBase glossary-rule) are
those of [variational message passing](@extref MessagePassingRulesBase glossary-vmp). They update
the [marginals](@extref MessagePassingRulesBase glossary-marginal) `q(a)` and `q(W)` from the
states, and the states from them.

## Definition

```math
p(y \mid x, a, W) = \mathcal{N}\big(y \mid f(a)\, x,\; W^{-1}\big)
```

The output `y` is normal around the transformed input `f(a) x`, with covariance `W⁻¹`.

A nonlinear `f` is linearised around the mean of `q(a)`: each row of `f(a)` is taken as linear
in `a`, through its Jacobian, plus an offset that is zero for an `f` linear through the origin.
The docstring of [`ContinuousTransition`](@ref) gives the expansion and the messages towards `a`
and `W` in full.

## Interfaces

```@example ct
using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
MessagePassingRulesBase.nodespec(ContinuousTransition)
```

The node has four [interfaces](@extref MessagePassingRulesBase glossary-interface).

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `y` | the output, `n`-dimensional | `MvNormal` message; `MvNormal` joint `q(y, x)` or marginal `q(y)` |
| `x` | the input, `m`-dimensional | `MvNormal` message; `MvNormal` joint `q(y, x)` or marginal `q(x)` |
| `a` | the parameters of `A = f(a)`, a vector | `MvNormal` marginal `q(a)` |
| `W` | the `n × n` noise precision | a marginal with a mean and `E[logdet W]`, such as a `Wishart` |

The rules send an `MvNormal` [message](@extref MessagePassingRulesBase glossary-message)
towards `y`, `x` and `a`, and a Wishart towards `W`.

## Algorithm

```@docs
CTVMP
```

[`CTVMP`](@ref)`(f)` is the node's [algorithm](@extref MessagePassingRulesBase glossary-algorithm).
It takes the transformation as its only argument and has no keywords. The node declares no
algorithm of its own: a model must name `CTVMP` for every `ContinuousTransition`, and under the
default one, [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), no rule is
found.

Under `CTVMP` the node declares its
[dependencies](@extref MessagePassingRulesBase glossary-dependencies), the inputs each target's
rule takes:

```@example ct
MessagePassingRulesBase.dependencies_spec(ContinuousTransition, CTVMP(a -> reshape(a, 2, 2)))
```

Every target takes the inputs of the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme), which follow the
factorisation. The rule towards `a` also reads `q(a)`, the point it expands `f` around.

## Supported rules

```@example ct
MessagePassingRulesBase.rule_coverage(ContinuousTransition)
```

Each target has two rules under `CTVMP`, one per
[factorisation](@extref MessagePassingRulesBase glossary-factorisation):

- the [structured](@extref MessagePassingRulesBase glossary-structured-vmp) `q(y, x) q(a) q(W)`,
  reading the messages on `y` and `x` or the joint `q(y, x)`;
- the [mean field](@extref MessagePassingRulesBase glossary-mean-field) `q(y) q(x) q(a) q(W)`,
  reading marginals only.

The marginal rule computes the joint `q(y, x)`. The
[average energy](@extref MessagePassingRulesBase glossary-average-energy) exists in both
factorisations, so the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy)
is available. There is no [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
rule.

## Example

The message towards `y` under a known identity transition: the mean is `A` times the input's,
the covariance the input's plus the inverse of the mean of `q(W)`. When `a` is uncertain, its
spread adds to the precision of `x` before the transition, as it does in the joint `q(y, x)`.

```@example ct
@call_message_update_rule(
    node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 2, 2)),
    m = (x = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
    q = (a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], 1e-12 * Matrix(1.0I, 4, 4)), W = Wishart(3, [1.0 0.0; 0.0 1.0])),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws its inputs: the message on `x` and the marginals of `a` and `W`. The mean of
`q(W)` is `3 I`, so the covariance is `I + I/3`, which
[`getresult`](@extref MessagePassingRulesBase.getresult) and `mean_cov` confirm:

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, LinearAlgebra

julia> result = @call_message_update_rule(
           node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 2, 2)),
           m = (x = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
           q = (a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], 1e-12 * Matrix(1.0I, 4, 4)), W = Wishart(3, [1.0 0.0; 0.0 1.0])),
       );

julia> m, V = mean_cov(getresult(result));

julia> m ≈ [1.0, 2.0] && V ≈ (1 + 1 / 3) * I
true
```

## Limitations

- **Variational only.** Every rule needs [`CTVMP`](@ref), which the model must give, and
  marginals on `a` and `W`, which need initialising.
- **Multivariate normals only** on `y`, `x` and `a`: a scalar state is a vector of length one.
- **A nonlinear `f` is linearised**, around the mean of `q(a)`, so its rules and its average
  energy are approximate; for an `f` linear in `a` they are exact.

## Reshape without Jacobians

When every entry of `A` is learned independently as a parameter, use
[`LinearContinuousTransition`](@ref) with [`LCTVMP`](@ref)`(n, m)`. It represents the same
density and computes the same variational updates as `ContinuousTransition` with
`CTVMP(a -> reshape(a, n, m))`, to floating-point precision. The covariance of `a` can be fully
dense: the entries need not be statistically independent.

The four interfaces are still `y`, `x`, `a`, and `W`, with dimensions `n`, `m`, `n*m`, and
`n × n`. Julia's column-major ordering applies: `a = vec(A)`. `LCTVMP` carries only the shape;
the factorisation selects mean-field or structured VMP. Both dimensions are explicit because
the structured message towards `W` cannot determine the split of `q(y, x)` from its inputs.

```@example ct
@call_message_update_rule(
    node = LinearContinuousTransition, target = :y, algorithm = LCTVMP(2, 3),
    q = (x = MvNormalMeanCovariance([1.0, 2.0, 3.0], Matrix(1.0I, 3, 3)),
         a = MvNormalMeanCovariance(vec([1.0 0.0 1.0; 0.0 1.0 1.0]), Matrix(1.0I, 6, 6)),
         W = Wishart(3.0, Matrix(1.0I, 2, 2))),
)
```

This node supports the same eight messages, joint marginal, and two average energies. Its
message towards `a` uses weighted mean `vec(E[W] * E[y*x′])` and precision
`kron(E[x*x′], E[W])`, so it does not read `q(a)` or need an initial expansion point. Other
messages still require `q(a)`.

The rules reshape the parameter mean and contract blocks of its covariance directly.
They construct no Jacobians, including constant identity or selection matrices. Gaussian
solves and covariance contractions still take work. Message and marginal rules declare
scratch matrices that the engine reuses per outbound stream; each call overwrites its scratch
and returns separately owned parameters. Sharing `LCTVMP` across nodes therefore does not
share mutable working memory. Direct calls allocate fresh scratch, and average energies do
not retain it between calls.

```@example ct
MessagePassingRulesBase.rule_coverage(LinearContinuousTransition)
```

The original node remains appropriate for other transformations, including affine and nonlinear
ones. Neither node supplies belief-propagation rules.

## A convolution as one node

[`Conv2dContinuousTransition`](@ref), also exported as [`Conv2dCT`](@ref), is a whole
two-dimensional convolution with scalar ports: every output position `p` has
`y[:, p] ~ N(A * patch(x, p), W⁻¹)` with the same `A = reshape(a, C_out, kh * kw * C_in)` and
the same `C_out × C_out` precision `W`. Its interfaces are `[:y..., :x..., :a, :W]`, with `y`
and `x` groups of scalar ports, and its algorithm [`Conv2dVMP`](@ref) carries the geometry and a cache:

```@example ct
conv = Conv2dVMP((4, 4, 1), 2, (3, 3)) # a 4 × 4 image, 1 → 2 channels, a 3 × 3 kernel
(conv.outsize, conv.ny, conv.nx, conv.n, conv.m, conv.bandwidth)
```

The image and the output are flat, in column-major order: `x = vec(image)` and
`y = vec(output)` with the output `H_out × W_out × C_out`. The model never indexes rows and
columns; `conv.taps[j, p]` holds the index in `x` of entry `j` of the patch at position `p`
(`0` for zero padding), a patch being `vec` of its `kh × kw × C_in` window. `stride` and
`pad` are keywords, each an integer or a `(rows, columns)` pair.

### The same rules as `CTVMP`

The node is the linear Gaussian transition whose matrix repeats the kernel at every position.
With `B(a)` that `ny × nx` matrix, it is [`ContinuousTransition`](@ref) under
`CTVMP(a -> B(a))` with `W` repeated for every position, `kron(W, I)`, and independent
state messages, that is, messages on `y` and `x` with diagonal covariances:

- the joint `q(y, x)`, and `q(y)` and `q(x)` under `q(y)q(x)`, have the same means, variances
  and entropy;
- the message towards a scalar port is that posterior's marginal without the port's own
  incoming message;
- the message towards `a` is the same, since the kernel is `CTVMP`'s own `a`;
- the message towards `W` is `WishartFast(C_out + 1 + positions, R)`, with `R` the sum over
  positions of the diagonal blocks of `CTVMP`'s residual;
- the average energy is the same.

The tests check every one of these against `CTVMP` on convolutions with strides, padding and
several channels. What differs is the cost: `CTVMP` differentiates `B(a)` and works with
dense `ny × nx` and `(ny + nx)²` matrices, where this node never forms either.

### Group marginals

A group's marginal is a [`Conv2dMarginal`](@ref), not a dense Gaussian. It holds the mean and
variance of every member, the message towards every member, the second moments of the
patches summed over positions, and the entropy. Incoming messages for a latent group must
all be univariate Gaussians. Joint inference with a mixture of `PointMass` and Gaussian
incoming messages (a partially observed image) is not supported. Independent scalar
marginals supplied as `q(x)` may contain both, and fully observed images are supported:

```@example ct
messages_y = ntuple(_ -> NormalMeanVariance(0.0, 0.1), conv.ny)
messages_x = ntuple(i -> NormalMeanVariance(i / 16, 1.0), conv.nx)
conv_parameters = (a = MvNormalMeanCovariance(ones(18), Matrix(0.1I, 18, 18)), W = Wishart(4.0, Matrix(0.25I, 2, 2)))
joint = getresult(@call_marginal_update_rule(
    node = Conv2dCT, target = (:y, :x), algorithm = conv,
    m = (y = messages_y, x = messages_x), q = conv_parameters,
))
```

```@example ct
(mean(joint)[1:2], var(joint)[1:2], entropy(joint), size(joint.Exx))
```

The messages towards `a` and `W` and the average energy read `Exx`, `Eyx` and `Eyy` from it
and invert nothing; dense Gaussian marginals are accepted as well. A port's message is read
from it too:

```@example ct
@call_message_update_rule(node = Conv2dCT, target = (:x, 3), algorithm = conv, clusters = ((:y, :x) => joint,))
```

### Latent images: a banded system

With a latent image the outputs of every position are eliminated in closed form, which
leaves a system over `x` alone that couples two pixels only when a patch contains both. `x`
is ordered with the channels of a pixel together and then along the shorter side of the
image, so that system is banded, with bandwidth `conv.bandwidth`, about
`C_in * ((kw - 1) * H + kh)`. A banded Cholesky factorisation and a selected inverse (the
Takahashi recursion) give every mean, every variance and every covariance inside a patch,
which is all the node needs, in `O(nx b²)` time and `O(nx b)` memory. A 28 × 28 image with
a 3 × 3 kernel is 1,460 ports with bandwidth 58: its joint takes 2.4 ms and 0.2 MB, where a
dense solve would factorise a 1,460 × 1,460 matrix. The precision of a port's message is
formed as a Schur complement over the port's own row. The factor's precision and weighted
mean contributions are kept separately from incoming messages, avoiding subtraction of
large incoming natural parameters when forming outgoing messages. A pixel that no patch
covers receives exactly zero weighted mean and precision.

Given `q(x)`, as for an observed image, the positions are independent and only a
`C_out × C_out` system per position is solved.

### In an inference engine: `ports`

An engine asks for the message towards every port in a call of its own. `ports` says what
that call receives:

| `ports` | factorisation | a port's message is read from |
|---|---|---|
| `:joint` | `q(y,x)q(a)q(W)` | the joint `q(y, x)` |
| `:separate` | `q(y)q(x)q(a)q(W)`, latent `x` | `q(y)` or `q(x)` |
| `:observed` | `q(y)q(a)q(W)`, `x` given as data | `q(y)`; no message goes to `x` |
| `:default` | any of these | the other messages of its group, as the engine's default scheme delivers them |

With `:joint`, `:separate` and `:observed` each group marginal update performs one solve,
and every port reads its message from that result in constant time. `:observed` also keeps
the moments of the observed patches in `q(y)`, so the rules towards `a` and `W` never read
the pixels again. The `:default` rules support both factorizations and direct calls: each port's rule receives the other messages of its group, and one
solve is reused across ports while the inputs stay the same. Prefer the other three for
large images. **Create one `Conv2dVMP` per node.**

```julia
@model function conv2d_batch(x, z, insize, cout, kernel, batch, noise)
    local y
    template = Conv2dVMP(insize, cout, kernel)
    ny, nx, d = template.ny, template.nx, template.n * template.m
    a ~ MvNormalMeanCovariance(zeros(d), Matrix(10.0I, d, d))
    W ~ Wishart(cout + 2, Matrix(1.0I, cout, cout))
    for b in 1:batch
        ys, xs = ((b - 1) * ny + 1):(b * ny), ((b - 1) * nx + 1):(b * nx)
        y[ys] ~ Conv2dCT(x = x[xs], a = a, W = W) where {
            algorithm = Conv2dVMP(insize, cout, kernel; ports = :observed) # A fresh cache for every image.
        }
    end
    for i in 1:(batch * ny)
        z[i] ~ NormalMeanVariance(y[i], noise)
    end
end
```

with `constraints = @constraints begin q(y, a, W) = q(y)q(a)q(W) end`, and `x` and `z` the
concatenated `vec`s of the images and of their noisy convolutions. The model needs RxInfer,
which is not a dependency of this package.

For a latent image, give every pixel its own prior, keep `y` and `x` together with
`q(y, x, a, W) = q(y, x)q(a)q(W)`, and use `ports = :joint`.

### Compared with `CTVMP`

`benchmark/conv2d.jl` times one call of every rule against `CTVMP` with the convolution
matrix, for a 3 × 3 kernel and one channel (Julia 1.13.3, Apple M1, one BLAS thread, minimum
warmed time in microseconds; `CTVMP` was not run at 28 × 28):

| rule | 4 × 4: `CTVMP` | Conv | 7 × 7: `CTVMP` | Conv | 14 × 14: `CTVMP` | Conv | 28 × 28: Conv |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint `q(y, x)` | 16.0 | 5.8 | 326 | 30.3 | 42,963 | 233 | 2,399 |
| one port, from the joint | 18.0 | <0.01 | 392 | <0.01 | 45,265 | <0.01 | <0.01 |
| `a`, from the joint | 13.0 | 0.20 | 333 | 0.20 | 41,647 | 0.20 | 0.20 |
| `W`, from the joint | 20.0 | 0.50 | 1,750 | 0.50 | 670,557 | 0.50 | 0.50 |
| energy, joint | 17.0 | 0.60 | 1,729 | 0.60 | 858,935 | 0.60 | 0.60 |
| `q(y)`, observed pixels | 3.0 | 1.3 | 23.0 | 4.6 | 476 | 24.2 | 109 |
| `a`, observed pixels | 12.0 | 0.20 | 330 | 0.20 | 42,766 | 0.20 | 0.20 |
| `W`, observed pixels | 17.0 | 0.50 | 1,723 | 0.50 | 691,995 | 0.50 | 0.50 |
| energy, observed pixels | 17.0 | 0.70 | 1,721 | 0.70 | 674,987 | 0.70 | 0.70 |

The reference of a port's message is `CTVMP`'s whole message towards `y`, which an engine
computes once for all of `y`; the node's row is the read of one port, after the joint. The
reference of `q(y)` is `CTVMP`'s mean-field message towards `y` times the incoming messages.

With `:joint` and `:observed`, port reads, parameter messages and energy updates use the
moments already held by the group marginal, so their cost does not depend on image size.
Separate groups and direct calls that supply explicit scalar marginals also traverse the
patches to assemble the needed moments. Explicit `q(x)` inputs take precedence over moments
retained by an earlier `q(y)`. The group marginal itself grows linearly with the number of
positions for an observed image, and as `nx b²` for a latent one.

## API

The node and its alias.

```@docs
ContinuousTransition
CTransition
LinearContinuousTransition
LCTVMP
Conv2dContinuousTransition
Conv2dCT
Conv2dVMP
Conv2dMarginal
```

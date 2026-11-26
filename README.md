# ContinuousTransitionMessagePassingRules

[![CI](https://github.com/ReactiveBayes/ContinuousTransitionMessagePassingRules.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/ContinuousTransitionMessagePassingRules.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/dev/)
[![Coverage](https://codecov.io/gh/ReactiveBayes/ContinuousTransitionMessagePassingRules.jl/graph/badge.svg)](https://codecov.io/gh/ReactiveBayes/ContinuousTransitionMessagePassingRules.jl)
[![Code style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

The `ContinuousTransition` node, `y ~ N(f(a) x, W⁻¹)` (alias `CTransition`): a linear Gaussian
transition from `x` to `y` through a matrix built from the vector `a`, with `a` and the noise
precision `W` learned along with the states. Its rules are variational, mean-field or structured
`q(y, x)`, with average energies, and run under `CTVMP(f)`, which carries the transformation `f`
and which a model must give. A nonlinear `f` is linearised with ForwardDiff.

```julia
using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily

result = @call_message_update_rule(
    node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 1, 1)),
    q = (x = MvNormalMeanCovariance([2.0], [1.0;;]), a = MvNormalMeanCovariance([3.0], [1.0;;]), W = Wishart(2, [0.5;;])),
)
getresult(result)   # MvNormalMeanPrecision([6.0], [1.0;;])
```

For a matrix learned entry by entry, use `LinearContinuousTransition` with `LCTVMP(n, m)`:
it computes the same rules as `CTVMP(a -> reshape(a, n, m))` without Jacobians or
linearisation. `a = vec(A)` uses Julia's column-major ordering. `LCTVMP` supplies only the
shape; the factorisation still selects ordinary mean-field or structured VMP. Messages to
`a` need no expansion point, and the inference engine reuses scratch matrices per message
stream without overwriting earlier results.

```julia
result = @call_message_update_rule(
    node = LinearContinuousTransition, target = :y, algorithm = LCTVMP(1, 1),
    q = (x = MvNormalMeanCovariance([2.0], [1.0;;]), a = MvNormalMeanCovariance([3.0], [1.0;;]), W = Wishart(2, [0.5;;])),
)
```

For a convolution, `Conv2dContinuousTransition` (alias `Conv2dCT`) takes a flat image
`x = vec(image)` and a flat output `y`, one node per image, with the geometry in its cached
algorithm:

```julia
y[1:ny] ~ Conv2dCT(x = x[1:nx], a = a, W = W) where {
    algorithm = Conv2dVMP((H, W, C_in), C_out, (kh, kw); ports = :joint)
}
```

It agrees with `ContinuousTransition` under `CTVMP` given the matrix that repeats the kernel
and independent state messages, solves latent images in a banded system instead of a dense
one, accepts observed pixels, and is described in the
[convolution documentation](docs/src/index.md#a-convolution-as-one-node).
`benchmark/conv2d.jl` compares it with `CTVMP` rule by rule.

```julia
import Pkg; Pkg.add("ContinuousTransitionMessagePassingRules")
```

- Documentation: <https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/stable/>.
  `make docs` builds it locally, into `docs/build`.
- Tests: `make test` runs the suite as CI does, ending with the rule-coverage gate: every rule
  must be selected by some test. `make test test_args="name:rules"` selects items by tag, by
  name or by file; `make help` lists the other targets.
- Julia 1.10 or later. MIT licence.

To compare all rules against reshape-based `CTVMP`, including runtime and allocations with
reused scratch, and every `Conv2dCT` rule against `CTVMP` with the convolution matrix, run
from the repository root:

```sh
julia --project=benchmark -e 'import Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/linear.jl
julia --project=benchmark benchmark/conv2d.jl
```

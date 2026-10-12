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

```julia
import Pkg; Pkg.add("ContinuousTransitionMessagePassingRules")
```

- Documentation: <https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/stable/>.
  `make docs` builds it locally, into `docs/build`.
- Tests: `make test` runs the suite as CI does, ending with the rule-coverage gate: every rule
  must be selected by some test. `make test test_args="name:rules"` selects items by tag, by
  name or by file; `make help` lists the other targets.
- Depends on MessagePassingRulesBase, BayesBase, ExponentialFamily,
  Distributions, FastCholesky and ForwardDiff.
- MIT licence.

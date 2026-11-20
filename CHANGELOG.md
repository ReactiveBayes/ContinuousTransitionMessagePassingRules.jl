# Changelog

All notable changes to ContinuousTransitionMessagePassingRules.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Conv2dContinuousTransition` (alias `Conv2dCT`) with `Conv2dVMP((H, W, C_in), C_out, (kh, kw); stride, pad)`:
  a whole convolution as one node with flat scalar `y` and `x` ports, a shared kernel `a`
  and noise precision `W`. Same factorizations as `ScalarCT`, plus independent scalar
  marginals for a group, such as the `PointMass` marginals of observed pixels. The cached
  algorithm solves a `C_out × C_out` system per position given `q(x)` and reuses one dense
  factorisation across ports for latent inputs. Tested against `ScalarCT` with the tied
  matrix and against one `ScalarCT` per position. `integration/conv2d_mnist.jl` learns a
  kernel from a batch of MNIST images with RxInfer.
- `SLCTVMP(n, m; T = Float64)` for `ScalarCT`, with a cache allocated at construction,
  shared transition computations and Gaussian solves across scalar ports, and reusable
  moment/residual buffers. Supports both state factorizations with value-based invalidation,
  including in-place parameter changes. Each node must own a separate algorithm instance;
  sharing between nodes or concurrent calls is not supported. Includes equivalence,
  invalidation, ownership tests, and fixed/changing-input benchmarks.
- `ScalarLinearContinuousTransition` (alias `ScalarCT`) with grouped scalar `y` and `x`
  ports under the existing `LCTVMP(n, m)`. Supports separate state groups and a full joint,
  preserving within-group covariance, with scalar messages, parameter messages, group
  marginals, energies, reusable scratch, and rule-time dimension validation. Includes block
  equivalence tests, scalar-sweep benchmarks.
- `LinearContinuousTransition` with `LCTVMP(n, m)` for `A = reshape(a, n, m)`: exact mean-field
  and structured variational messages, joint marginal, and average energies without Jacobians.
  The message towards `a` requires no `q(a)` expansion point. Message and marginal rules reuse
  scratch matrices through the inference engine while keeping returned parameters independent.
- Equivalence, numeric-type, dimension, and scratch-reuse tests, and an isolated benchmark
  comparing every rule with reshape-based `CTVMP`.

## [1.0.0] - 2026-10-12

The first release: the ContinuousTransition node, a transition through a matrix built from a vector, developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/ContinuousTransitionMessagePassingRules.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ReactiveBayes/ContinuousTransitionMessagePassingRules.jl/releases/tag/v1.0.0

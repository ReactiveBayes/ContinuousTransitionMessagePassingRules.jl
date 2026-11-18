# Changelog

All notable changes to ContinuousTransitionMessagePassingRules.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

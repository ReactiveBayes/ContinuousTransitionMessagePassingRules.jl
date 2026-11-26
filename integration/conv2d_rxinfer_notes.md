# Conv2d scheduling handoff for RxInfer developers

This is an integration investigation, not a limitation of the package's separate-group
variational rules. Those rules are checked directly against dense `CTVMP` updates.
The package does not choose RxInfer's subscription order or initialization schedule.

## Reproducer

Use the integration environment with RxInfer 6.1.3. Save the following as a Julia script
and run it with `julia --project=integration script.jl separate separate`. The first argument
selects the factorization (`joint` or `separate`); the second selects `ports`.

```julia
using RxInfer, ContinuousTransitionMessagePassingRules, LinearAlgebra

@model function schedule_check(z, ports)
    local x, y
    a ~ MvNormalMeanCovariance(ones(4), Matrix(0.1I, 4, 4))
    for i in 1:9
        x[i] ~ NormalMeanVariance(i / 10, 1.0)
    end
    y[1:4] ~ Conv2dCT(x = x[1:9], a = a, W = [2.0;;]) where {
        algorithm = Conv2dVMP((3, 3, 1), 1, (2, 2); ports = ports)
    }
    for i in 1:4
        z[i] ~ NormalMeanVariance(y[i], 0.1)
    end
end

separate = ARGS[1] == "separate"
ports = Symbol(ARGS[2])
constraints = separate ? (@constraints begin
    q(y, x, a) = q(y)q(x)q(a)
end) : (@constraints begin
    q(y, x, a) = q(y, x)q(a)
end)
initialization = separate ? (@initialization begin
    q(a) = MvNormalMeanCovariance(ones(4), Matrix(0.1I, 4, 4))
    q(x) = MvNormalMeanCovariance(collect(0.1:0.1:0.9), Matrix(1.0I, 9, 9))
    q(y) = MvNormalMeanCovariance(zeros(4), Matrix(1.0I, 4, 4))
end) : (@initialization begin
    q(a) = MvNormalMeanCovariance(ones(4), Matrix(0.1I, 4, 4))
end)
result = infer(
    model = schedule_check(ports = ports), data = (z = [0.8, 1.2, 1.6, 2.0],),
    constraints = constraints, initialization = initialization, iterations = 3,
    free_energy = true, returnvars = (a = KeepLast(),),
)
println(mean(result.posteriors[:a]))
println(result.free_energy)
```

Verified on Julia 1.13.4 with RxInfer 6.1.3: `separate separate` and `separate default`
both fail with `Variables [ a ] have not been updated`, including when both state
marginals are initialized. The control
configurations are `joint joint` and `joint default`. In this reproducer both return
the same kernel mean `[0.9201703, 0.9209482, 0.9207089, 0.9218424]`. Their reported
free-energy histories differ: approximately `[7.2564, 7.1930, 7.1853]` for `:joint` and
`[7.4259, 7.2063, 7.1869]` for `:default`. Energy evaluation order needs a separate
engine-side investigation; matching final parameter means does not establish matching
intermediate state marginals or energy snapshots.

## Expected dependencies and investigation

For `:separate`, each y port reads `q(y)` and each x port reads `q(x)`. The local marginal
update for `q(y)` consumes incoming y messages plus `q(x)`, `q(a)`, and `q(W)`; the local
update for `q(x)` consumes incoming x messages plus `q(y)`, `q(a)`, and `q(W)`. The message
towards `a` consumes both group marginals and `q(W)`. With fixed W, its marginal is a
`PointMass`. For `:default`, state-port messages instead consume the other incoming
messages of their own group and the opposite group's marginal.

RxInfer's integration needs to deliver the initialized **group** marginals to these local
marginal dependencies, then propagate the updated groups to parameter rules and scalar
ports. Inspect the registered cluster keys, initialization delivery, subscription order,
and which dependency never emits for the a update. The error alone does not establish
which of these is responsible; changing numerical rules is not a justified fix.

Acceptance for an upstream scheduling fix: both separate configurations complete multiple
iterations and agree with each other and an equivalent dense separate-group calculation.
Their result need not equal the joint factorization, which retains cross-group covariance.
No RxInfer changes or integration CI gate are part of this package PR.

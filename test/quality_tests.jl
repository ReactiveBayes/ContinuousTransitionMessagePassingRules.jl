@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, ContinuousTransitionMessagePassingRules
    Aqua.test_all(ContinuousTransitionMessagePassingRules; deps_compat = (; check_extras = true))
end

@testitem "quality:linear node" tags = [:quality] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase
    using MessagePassingRulesBase: dependencies_spec, target_dependencies, extends_default_scheme, default_algorithm, Target, interfaces, ispure

    algorithm = LCTVMP(2, 3)
    @test (algorithm.n, algorithm.m) == (2, 3)
    @test interfaces(LinearContinuousTransition) == (:y, :x, :a, :W)
    @test ispure(algorithm)
    spec = dependencies_spec(LinearContinuousTransition, algorithm)
    @test all(t -> extends_default_scheme(spec, Target(t)), (:y, :x, :a, :W))
    @test all(t -> isempty(target_dependencies(spec, Target(t))), (:y, :x, :a, :W))
    @test default_algorithm(LinearContinuousTransition) === DefaultAlgorithm()
    @test_throws Exception getresult(call_message_update_rule(LinearContinuousTransition, :y; q = (x = nothing, a = nothing, W = nothing)))
    for dims in ((0, 2), (2, 0), (-1, 2), (2, -1))
        @test_throws ArgumentError LCTVMP(dims...)
    end
    @test_throws MethodError LCTVMP(2.0, 3.0)
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "ContinuousTransitionMessagePassingRules", deps))
    closure = Set{String}()
    frontier = collect(values(last(self).dependencies))
    while !isempty(frontier)
        info = get(deps, pop!(frontier), nothing)
        (info === nothing || info.name in closure) && continue
        push!(closure, info.name)
        append!(frontier, values(info.dependencies))
    end
    @test "MessagePassingRulesBase" in closure
    @test !("StandardMessagePassingRules" in closure)
    @test !("ReactiveMP" in closure)
    # The test tooling is for tests only, never a dependency of the rules.
    @test !("MessagePassingRulesTestUtils" in closure)
end

@testitem "quality:scalar linear node" tags = [:quality] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase
    using MessagePassingRulesBase: interfaces, interface_groups, dependencies_spec, target_dependencies, extends_default_scheme, Target, IndexedTarget, default_algorithm
    @test ScalarCT === ScalarLinearContinuousTransition
    @test interfaces(ScalarCT) == (:y, :x, :a, :W)
    @test interface_groups(ScalarCT) == (:y, :x)
    spec = dependencies_spec(ScalarCT, LCTVMP(2, 3))
    for edge in ((:y, 1), (:y, 2), (:x, 1), (:x, 3), :a, :W)
        target = edge isa Symbol ? Target(edge) : IndexedTarget(edge...)
        @test extends_default_scheme(spec, target)
        @test isempty(target_dependencies(spec, target))
    end
    @test default_algorithm(ScalarCT) === DefaultAlgorithm()
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, ContinuousTransitionMessagePassingRules
    DocMeta.setdocmeta!(ContinuousTransitionMessagePassingRules, :DocTestSetup, :(using ContinuousTransitionMessagePassingRules); recursive = true)
    doctest(ContinuousTransitionMessagePassingRules; manual = false)
end

@testitem "quality:rules" tags = [:quality] begin
    using ContinuousTransitionMessagePassingRules
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities

    # Every rule agrees with its node's declaration and with its algorithm's declared
    # dependencies, and no call could match two rules equally well.
    @test isempty(check_rules(ContinuousTransitionMessagePassingRules))
    @test isempty(check_rule_ambiguities(ContinuousTransitionMessagePassingRules))
end

@testitem "quality:nodes" tags = [:quality] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase
    using MessagePassingRulesBase: dependencies_spec, target_dependencies, extends_default_scheme, default_algorithm, Target

    @test CTransition === ContinuousTransition
    # Every target follows the factorisation, and the one towards `a` also reads q(a).
    spec = dependencies_spec(ContinuousTransition, CTVMP(identity))
    @test all(t -> extends_default_scheme(spec, Target(t)), (:y, :x, :a, :W))
    @test map(d -> (d.container, d.key), target_dependencies(spec, Target(:a))) == ((:q, :a),)
    @test isempty(target_dependencies(spec, Target(:y)))
    # No algorithm of its own: under the default one, no rule is found.
    @test default_algorithm(ContinuousTransition) === DefaultAlgorithm()
    @test_throws Exception getresult(call_message_update_rule(ContinuousTransition, :y; q = (x = nothing, a = nothing, W = nothing)))
end

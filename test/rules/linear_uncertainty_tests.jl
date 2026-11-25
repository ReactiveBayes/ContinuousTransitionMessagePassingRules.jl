@testitem "rules:LinearContinuousTransition:diagonal parameter covariance contraction" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, BayesBase, ExponentialFamily, LinearAlgebra

    for T in (Float32, Float64), (n, m) in ((1, 1), (2, 3), (3, 2), (10, 10))
        algorithm = LCTVMP(n, m)
        variances = T.(1:(n * m)) ./ T(n * m)
        qa = MvNormalMeanCovariance(zeros(T, n * m), Diagonal(variances))
        W = Matrix{T}(I, n, n) + fill(T(0.1), n, n)
        # Independent entries of A contribute only to diagonal entries of E[A' W A].
        expected = Diagonal(reshape(variances, n, m)' * diag(W))
        scratch = fill(T(NaN), m, m)
        actual = ContinuousTransitionMessagePassingRules.lct_uncertainty!(scratch, algorithm, qa, W)
        @test actual === scratch
        @test actual ≈ expected
        @test_throws DimensionMismatch ContinuousTransitionMessagePassingRules.lct_uncertainty!(zeros(T, m + 1, m), algorithm, qa, W)
    end
end

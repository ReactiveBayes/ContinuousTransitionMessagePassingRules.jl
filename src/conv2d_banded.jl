# A symmetric positive definite banded matrix, its Cholesky factor and the entries of its
# inverse inside the band. Column `j` of a band array holds rows `j:(j + b)` of the lower
# triangle: `band[i - j + 1, j]` is entry `(i, j)`.

#     banded_cholesky!(band) -> band
#
# Replace the lower band of a symmetric positive definite matrix by its Cholesky factor, in
# `O(N b²)` for `N` columns and bandwidth `b = size(band, 1) - 1`. Entries below the last row
# of the matrix are not read.
#
# # Throws
# `LinearAlgebra.PosDefException` when a pivot is not positive.
function banded_cholesky!(band::AbstractMatrix{T}) where {T}
    Base.require_one_based_indexing(band)
    b, N = size(band, 1) - 1, size(band, 2)
    @inbounds for j in 1:N
        pivot = band[1, j]
        (pivot > 0 && isfinite(pivot)) || throw(LinearAlgebra.PosDefException(j))
        pivot = sqrt(pivot)
        band[1, j] = pivot
        last = min(b, N - j)
        for d in 1:last
            band[d + 1, j] /= pivot
        end
        # Right-looking update of the columns this one reaches: contiguous in memory.
        for d in 1:last
            l = band[d + 1, j]
            iszero(l) && continue
            @simd for e in d:last
                band[e - d + 1, j + d] -= band[e + 1, j] * l
            end
        end
    end
    return band
end

#     banded_solve!(x, factor) -> x
#
# Solve `A x = x` in place, with `factor` from [`banded_cholesky!`](@ref).
function banded_solve!(x::AbstractVector, factor::AbstractMatrix)
    b, N = size(factor, 1) - 1, size(factor, 2)
    @inbounds for j in 1:N
        v = x[j] / factor[1, j]
        x[j] = v
        @simd for d in 1:min(b, N - j)
            x[j + d] -= factor[d + 1, j] * v
        end
    end
    @inbounds for j in N:-1:1
        v = x[j]
        @simd for d in 1:min(b, N - j)
            v -= factor[d + 1, j] * x[j + d]
        end
        x[j] = v / factor[1, j]
    end
    return x
end

banded_logdet(factor::AbstractMatrix{T}) where {T} = 2 * sum(j -> log(factor[1, j]), axes(factor, 2); init = zero(T))

#     banded_selected_inverse!(inverse, factor) -> inverse
#
# The entries of `A⁻¹` inside the band of `A`, from the Cholesky `factor` of `A`, by the
# Takahashi recursion: `O(N b²)`, never forming the dense inverse. `inverse` has the layout of
# `factor`: `inverse[i - j + 1, j]` is entry `(i, j)` of `A⁻¹` for `j <= i <= j + b`.
function banded_selected_inverse!(inverse::AbstractMatrix{T}, factor::AbstractMatrix{T}) where {T}
    size(inverse) == size(factor) || throw(DimensionMismatch("the inverse has size $(size(inverse)), the factor $(size(factor))"))
    b, N = size(factor, 1) - 1, size(factor, 2)
    @inbounds for j in N:-1:1
        last = min(b, N - j)
        ljj = factor[1, j]
        # Z[i, j] for i > j: -(Σ_{k > j} L[k, j] Z[max(k, i), min(k, i)]) / L[j, j].
        for d in last:-1:1
            i = j + d
            total = zero(T)
            for e in 1:last
                k = j + e
                z = k >= i ? inverse[k - i + 1, i] : inverse[i - k + 1, k]
                total += factor[e + 1, j] * z
            end
            inverse[d + 1, j] = -total / ljj
        end
        total = zero(T)
        for e in 1:last
            total += factor[e + 1, j] * inverse[e + 1, j]
        end
        inverse[1, j] = (inv(ljj) - total) / ljj
    end
    return inverse
end

# Entry (i, j) of a symmetric matrix held as a lower band; both inside the band.
Base.@propagate_inbounds band_entry(band, i, j) = i >= j ? band[i - j + 1, j] : band[j - i + 1, i]

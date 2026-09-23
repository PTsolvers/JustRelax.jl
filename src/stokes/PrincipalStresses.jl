"""
    compute_principal_stresses(backend, stokes::StokesArrays)

Compute the principal deviatoric stresses (eigenvalues and eigenvectors of the stress
tensor) at cell centers from `stokes`, returning a new `PrincipalStress`.
"""
function compute_principal_stresses(backend, stokes::JustRelax.StokesArrays)
    ni = size(stokes.P)
    σ = PrincipalStress(backend, ni)
    @parallel (@idx ni) principal_stresses_eigen!(σ, @stress_center(stokes)...)
    return σ
end

"""
    compute_principal_stresses!(stokes, σ::PrincipalStress)

In-place version of [`compute_principal_stresses`](@ref), writing into a pre-allocated `σ`.
"""
function compute_principal_stresses!(stokes, σ::JustRelax.PrincipalStress)
    ni = size(stokes.P)
    @parallel (@idx ni) principal_stresses_eigen!(σ, @stress_center(stokes)...)
    return nothing
end

@parallel_indices (I...) function principal_stresses_eigen!(σ::JustRelax.PrincipalStress, τ_xx, τ_yy, τ_xy)

    # Construct the stress tensor
    τ_11 = @inbounds τ_xx[I...]
    τ_22 = @inbounds τ_yy[I...]
    τ_12 = @inbounds τ_xy[I...]

    a = (τ_11 + τ_22) / 2
    b = √((τ_11 - τ_22)^2 / 4 + τ_12^2)
    # eigenvalues
    σ1 = a + b
    σ2 = a - b
    # angle of principal stress
    θ = atan(2 * τ_12, τ_11 - τ_22) / 2
    # eigenvectors
    sinθ, cosθ = sin(θ), cos(θ)
    e1 = SA[cosθ, sinθ]
    e2 = SA[-sinθ, cosθ]

    Base.@nexprs 2 i -> begin
        @inbounds σ.σ1[i, I...] = σ1 * e1[i]
        @inbounds σ.σ2[i, I...] = σ2 * e2[i]
    end

    return nothing
end

@parallel_indices (I...) function principal_stresses_eigen!(σ::JustRelax.PrincipalStress, τ_xx, τ_yy, τ_zz, τ_yz, τ_xz, τ_xy)

    # Construct the stress tensor
    τ_12 = @inbounds τ_xy[I...]
    τ_13 = @inbounds τ_xz[I...]
    τ_23 = @inbounds τ_yz[I...]
    τ = @SMatrix [
        τ_xx[I...] τ_12 τ_13
        τ_12 τ_yy[I...] τ_23
        τ_13 τ_23 τ_zz[I...]
    ]

    σ1, σ2, σ3 = eigen_symmetric_3x3(τ)

    Base.@nexprs 3 i -> begin
        @inbounds σ.σ1[i, I...] = σ1[i]
        @inbounds σ.σ2[i, I...] = σ2[i]
        @inbounds σ.σ3[i, I...] = σ3[i]
    end

    return nothing
end

# Cyclic Jacobi eigensolver for symmetric 3x3 matrices. The sweeps converge
# quadratically for symmetric input, including degenerate or near-degenerate spectra.

"""
    eigen_symmetric_3x3(A::SMatrix{3,3}; tol = 1e-12, max_sweeps = 50)

Eigenvalues and eigenvectors of a symmetric 3×3 matrix `A`, found by cyclic Jacobi
rotation. Returns `(σ1, σ2, σ3)`, each an `SVector{3}` holding an eigenvalue times its
unit eigenvector, sorted by eigenvalue in descending order. Throws if the off-diagonal
Frobenius norm has not dropped below `tol` times the Frobenius norm of `A` after
`max_sweeps` sweeps.
"""
function eigen_symmetric_3x3(A::SMatrix{3, 3}; tol = 1.0e-12, max_sweeps = 50)
    V = SA[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    # Jacobi rotations preserve the Frobenius norm, so the threshold is fixed
    threshold = tol^2 * sum(abs2, A)
    for _ in 1:max_sweeps
        _offdiag_norm2(A) ≤ threshold && return _sorted_eigen_3x3(A, V)
        A, V = jacobi_rotate(A, V, Val(1), Val(2))
        A, V = jacobi_rotate(A, V, Val(1), Val(3))
        A, V = jacobi_rotate(A, V, Val(2), Val(3))
    end
    # constant message: string interpolation does not compile in GPU kernels
    _offdiag_norm2(A) ≤ threshold ||
        throw(ErrorException("eigen_symmetric_3x3 did not converge"))
    return _sorted_eigen_3x3(A, V)
end

@inline _offdiag_norm2(A) = A[1, 2]^2 + A[1, 3]^2 + A[2, 3]^2

@inline function _sorted_eigen_3x3(A::SMatrix{3, 3}, V::SMatrix{3, 3})
    λ = SA[A[1, 1], A[2, 2], A[3, 3]]
    perms = reverse(sortperm(λ))

    Base.@nexprs 3 j -> σ_j = begin
        permⱼ = perms[j]
        x = Base.@ntuple 3 i -> begin
            λ[permⱼ] * V[i, permⱼ]
        end
        SVector(x...)
    end
    return σ_1, σ_2, σ_3
end

# Givens rotation that zeroes A[p, q] (and A[q, p]) of a symmetric matrix, applied as
# A' = G' A G, V' = V G.
@inline function jacobi_rotate(A::SMatrix{3, 3}, V::SMatrix{3, 3}, vp::Val{p}, vq::Val{q}) where {p, q}
    apq = A[p, q]
    iszero(apq) && return A, V
    τ = (A[q, q] - A[p, p]) / (2apq)
    t = iszero(τ) ? one(τ) : sign(τ) / (abs(τ) + √(1 + τ^2))
    c = 1 / √(1 + t^2)
    s = t * c
    G = givens_3x3(vp, vq, c, s)
    return G' * A * G, V * G
end

@inline givens_3x3(::Val{1}, ::Val{2}, c, s) = SA[c s 0.0; -s c 0.0; 0.0 0.0 1.0]
@inline givens_3x3(::Val{1}, ::Val{3}, c, s) = SA[c 0.0 s; 0.0 1.0 0.0; -s 0.0 c]
@inline givens_3x3(::Val{2}, ::Val{3}, c, s) = SA[1.0 0.0 0.0; 0.0 c s; 0.0 -s c]

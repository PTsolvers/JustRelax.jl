"""
    compute_principal_stresses(backend, stokes::JustRelax.StokesArrays)

Allocate a [`PrincipalStress`](@ref) for `backend` and fill it from the cell-centered
deviatoric stress in `stokes`, returning it. Each entry holds the principal-stress
eigenvector scaled by its eigenvalue, obtained from the eigen-decomposition of the local
stress tensor. Use [`compute_principal_stresses!`](@ref) to write into an existing object.
"""
function compute_principal_stresses(backend, stokes::JustRelax.StokesArrays)
    ni = size(stokes.P)
    σ = PrincipalStress(backend, ni)
    @parallel (@idx ni) principal_stresses_eigen!(σ, @stress_center(stokes)...)
    return σ
end

"""
    compute_principal_stresses!(stokes, σ::JustRelax.PrincipalStress)

Fill the principal-stress fields of `σ` in place from the cell-centered deviatoric stress in
`stokes`. In-place counterpart of [`compute_principal_stresses`](@ref).
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
    b = √((τ_11 - τ_22)^2 / 2 + τ_12^2)
    # eigenvalues
    σ1 = a + b
    σ2 = a - b
    # angle of principal stress
    θ = atan(2 * τ_12 / (τ_11 - τ_22)) / 2
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

    σ1, σ2, σ3 = hessenberg_eigen_3x3(τ)

    Base.@nexprs 3 i -> begin
        @inbounds σ.σ1[i, I...] = σ1[i]
        @inbounds σ.σ2[i, I...] = σ2[i]
        @inbounds σ.σ3[i, I...] = σ3[i]
    end

    return nothing
end

# Hessenberg method with spectral shift for 3x3 matrices

function hessenberg_eigen_3x3(A; tol = 1.0e-10, max_iter = 50)
    H, Q_hess = hessenberg_3x3(A)
    I_SA = SA[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    V = I_SA
    for _ in 1:max_iter
        λ = H[end, end] * I_SA
        Q, R = qr(H - λ)
        H = R * Q + λ
        V = V * Q
        check_eigen_convergence(H; tol = tol) && break
    end
    # eigenvectors
    eᵢ = Q_hess * V
    # eigenvalues
    σ = diag(H)
    perms = reverse(sortperm(σ))
    σ = σ[perms]

    Base.@nexprs 3 j -> σ_j = begin
        permⱼ = perms[j]
        x = Base.@ntuple 3 i -> begin
            σ[j] * eᵢ[i, permⱼ]
        end
        SVector(x...)
    end
    return σ_1, σ_2, σ_3
end

function check_eigen_convergence(H; tol = 1.0e-10)
    # If not converged, check if all off-diagonal elements are small enough
    converged = false
    for i in 1:3, j in 1:3
        i == j && continue
        converged = abs(H[i, j]) < tol
        converged || return false
    end
    return true
end

function hessenberg_3x3(A)

    # Extract vector to zero out a31
    x = SA[
        A[2, 1]
        A[3, 1]
    ]

    # Compute Householder vector
    α = norm(x)
    I2 = SA[
        1.0 0.0
        0.0 1.0
    ]
    Q_sub = if iszero(α)
        I2
    else
        e1 = SA[1.0, 0.0]
        v = x + sign(x[1]) * α * e1
        v = v / norm(v)

        # Householder matrix for 2x2 block
        I2 .- 2 * (v * v')
    end

    # Build full Q (3x3)
    Q = SA[
        1.0 0.0 0.0
        0.0 Q_sub[1, 1] Q_sub[1, 2]
        0.0 Q_sub[2, 1] Q_sub[2, 2]
    ]

    # Compute Hessenberg form
    H = (Q' * A) * Q

    return H, Q
end

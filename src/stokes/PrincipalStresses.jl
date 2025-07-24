function PrincipalStress(::B, ni::NTuple{2}) where {B}
    σ1 = fill(SVector(0.0, 0.0), ni...)
    σ2 = fill(SVector(0.0, 0.0), ni...)
    σ3 = fill(SVector(0.0, 0.0), 1, 1)
    return JustRelax.PrincipalStress{typeof(σ1)}(σ1, σ2, σ3)
end

function PrincipalStress(::B, ni::NTuple{3}) where {B}
    σ1 = fill(SVector(0.0, 0.0, 0.0), ni...)
    σ2 = fill(SVector(0.0, 0.0, 0.0), ni...)
    σ3 = fill(SVector(0.0, 0.0, 0.0), ni...)
    return JustRelax.PrincipalStress{typeof(σ1)}(σ1, σ2, σ3)
end

function compute_principal_stresses(backend, stokes::JustRelax.StokesArrays)
    ni = size(stokes.P)
    σ = PrincipalStress(backend, ni)
    @parallel (@idx ni) principal_stresses_eigen!(σ, @stress_center(stokes)...)
    return σ
end

function compute_principal_stresses!(stokes, σ::JustRelax.PrincipalStress)
    ni = size(stokes.P)
    @parallel (@idx ni) principal_stresses_eigen!(σ, @stress_center(stokes)...)
    return nothing
end

@parallel_indices (I...) function principal_stresses_eigen!(σ::JustRelax.PrincipalStress, τ_xx, τ_yy, τ_xy)

    # Construct the stress tensor
    τ_12 = τ_xy[I...]
    τ = @SMatrix [
        τ_xx[I...] τ_12
        τ_12 τ_yy[I...]
    ]

    # Compute the eigenvalues (principal stresses)
    vals, vecs = eigen(τ)
    # Compute the principal directions (eigenvectors)
    i2, i1 = sortperm(vals)
    σ.σ1[I...] = SA[vals[i1] * vecs[1, i1], vals[i1] * vecs[2, i1]]
    σ.σ2[I...] = SA[vals[i2] * vecs[1, i2], vals[i2] * vecs[2, i2]]

    return nothing
end

@parallel_indices (I...) function principal_stresses_eigen!(σ::JustRelax.PrincipalStress, τ_xx, τ_yy, τ_zz, τ_yz, τ_xz, τ_xy)

    # Construct the stress tensor
    τ_12 = τ_xy[I...]
    τ_13 = τ_xz[I...]
    τ_23 = τ_yz[I...]
    τ = @SMatrix [
        τ_xx[I...] τ_12 τ_13
        τ_12 τ_yy[I...] τ_23
        τ_13 τ_23 τ_zz[I...]
    ]

    # Compute the eigenvalues (principal stresses)
    vals, vecs = eigen(τ)
    # Compute the principal directions (eigenvectors)
    i3, i2, i1 = sortperm(vals)
    σ.σ1[I...] = SA[vals[i1] * vecs[1, i1], vals[i1] * vecs[2, i1], vals[i1] * vecs[3, i1]]
    σ.σ2[I...] = SA[vals[i2] * vecs[1, i2], vals[i2] * vecs[2, i2], vals[i2] * vecs[3, i2]]
    σ.σ3[I...] = SA[vals[i3] * vecs[1, i3], vals[i3] * vecs[2, i3], vals[i3] * vecs[3, i3]]

    return nothing
end

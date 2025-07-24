function PrincipalStress2D(::B, nx, ny) where B
    σ1 = @fill(SVector(0.0, 0.0), nx, ny)
    σ2 = @fill(SVector(0.0, 0.0), nx, ny)
    σ3 = @fill(SVector(0.0, 0.0), 1, 1)
    return PrincipalStress(σ1, σ2, σ3)
end

function PrincipalStress3D(::B, nx, ny, nz) where B
    σ1 = @fill(SVector(0.0, 0.0, 0.0), nx, ny, nz)
    σ2 = @fill(SVector(0.0, 0.0, 0.0), nx, ny, nz)
    σ3 = @fill(SVector(0.0, 0.0, 0.0), nx, ny, nz)
    return PrincipalStress(σ1, σ2, σ3)
end

function compute_principal_stresses(backend, stokes{AbstractArray{T, 2}}) where T
    ni = size(stokes.P)
    σ = PrincipalStress2D(backend, ni...)
    @parallel (@idx ni) principal_stresses_eigen!(σ, stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.xy_c[I...])
    return σ
end

function compute_principal_stresses!(stokes{AbstractArray{T, 3}}) where T
    ni = size(stokes.P)
    σ = PrincipalStress3D(backend, ni...)
    @parallel (@idx ni) principal_stresses_eigen!(
        σ, stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.zz[I...], stokes.τ.yz[I...], stokes.τ.xz[I...], stokes.τ.xy[I...]
    )
    return σ
end

function compute_principal_stresses!(stokes, σ::PrincipalStress{AbstractArray{T, 2}}) where T
    ni = size(stokes.P)
    @parallel (@idx ni) principal_stresses_eigen!(σ, stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.xy_c[I...])
    return nothing
end

function compute_principal_stresses!(stokes, σ::PrincipalStress{AbstractArray{T, 3}}) where T
    ni = size(stokes.P)
    @parallel (@idx ni) principal_stresses_eigen!(
        σ, stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.zz[I...], stokes.τ.yz[I...], stokes.τ.xz[I...], stokes.τ.xy[I...]
    )
    return nothing
end

@parallel_indices (I...) function principal_stresses_eigen!(σ::AbstractArray{T, 2}, τ_xx, τ_yy, τ_xy) where T

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

@parallel_indices (I...) function principal_stresses_eigen!(σ::AbstractArray{T, 3}, τ_xx, τ_yy, τ_zz, τ_yz, τ_xz, τ_xy) where T

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
## GeoParams

@inline function compute_phase(fn::F, rheology, phase::Int, args) where {F}
    return fn(rheology, phase, args)
end
@inline compute_phase(fn::F, rheology, ::Nothing, args) where {F} = fn(rheology, args)

@inline Base.@propagate_inbounds function getindex_phase(
    phase::T, I::Vararg{Int,N}
) where {N,T<:AbstractArray}
    return phase[I...]
end
@inline getindex_phase(::Nothing, I::Vararg{Int,N}) where {N} = nothing

## 3D KERNELS 

@parallel_indices (i, j, k) function compute_flux!(
    qTx::AbstractArray{_T,3}, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, _dx, _dy, _dz
) where {_T}
    d_xi(A) = _d_xi(A, i, j, k, _dx)
    d_yi(A) = _d_yi(A, i, j, k, _dy)
    d_zi(A) = _d_zi(A, i, j, k, _dz)
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    I = i, j, k

    if all(I .≤ size(qTx))
        qx = qTx2[I...] = -av_yz(K) * d_xi(T)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    if all(I .≤ size(qTy))
        qy = qTy2[I...] = -av_xz(K) * d_yi(T)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    if all(I .≤ size(qTz))
        qz = qTz2[I...] = -av_xy(K) * d_zi(T)
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
    qTx::AbstractArray{_T,3},
    qTy,
    qTz,
    qTx2,
    qTy2,
    qTz2,
    T,
    rheology,
    phase,
    θr_dτ,
    _dx,
    _dy,
    _dz,
    args,
) where {_T}
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    I = i, j, k

    @inbounds if all(I .≤ size(qTx))
        T_ijk = (T[(I .+ 1)...] + T[i, j + 1, k + 1]) * 0.5
        args_ijk = (; T=T_ijk, P=av_yz(args.P))
        K =
            (
                get_K(rheology, getindex_phase(phase, i, j + 1, k), args_ijk) +
                get_K(rheology, getindex_phase(phase, i, j, k + 1), args_ijk) +
                get_K(rheology, getindex_phase(phase, i, j + 1, k + 1), args_ijk) +
                get_K(rheology, getindex_phase(phase, i, j, k), args_ijk)
            ) * 0.25

        qx = qTx2[I...] = -K * d_xi(T)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTy))
        T_ijk = (T[(I .+ 1)...] + T[i + 1, j, k + 1]) * 0.5
        args_ijk = (; T=T_ijk, P=av_xz(args.P))
        K =
            (
                get_K(rheology, getindex_phase(phase, i + 1, j, k), args_ijk) +
                get_K(rheology, getindex_phase(phase, i, j, k + 1), args_ijk) +
                get_K(rheology, getindex_phase(phase, i + 1, j, k + 1), args_ijk) +
                get_K(rheology, getindex_phase(phase, i, j, k), args_ijk)
            ) * 0.25

        qy = qTy2[I...] = -K * d_yi(T)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTz))
        T_ijk = (T[(I .+ 1)...] + T[i + 1, j + 1, k]) * 0.5
        args_ijk = (; T=T_ijk, P=av_xy(args.P))
        K =
            (
                get_K(grheology, etindex_phase(phase, i + 1, j + 1, k), args_ijk) +
                get_K(grheology, etindex_phase(phase, i, j + 1, k), args_ijk) +
                get_K(grheology, etindex_phase(phase, i + 1, j, k), args_ijk) +
                get_K(grheology, etindex_phase(phase, i, j, k), args_ijk)
            ) * 0.25

        qz = qTz2[I...] = -K * d_zi(T)
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j, k) function update_T!(
    T::AbstractArray{_T,3}, Told, qTx, qTy, qTz, ρCp, dτ_ρ, _dt, _dx, _dy, _dz
) where {_T}
    av(A) = _av(A, i, j, k)
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)

    I = i + 1, j + 1, k + 1
    T[I...] +=
        av(dτ_ρ) *
        ((-(d_xa(qTx) + d_ya(qTy) + d_za(qTz))) - av(ρCp) * (T[I...] - Told[I...]) * _dt)

    return nothing
end

@parallel_indices (i, j, k) function update_T!(
    T::AbstractArray{_T,3},
    Told,
    qTx,
    qTy,
    qTz,
    rheology,
    phase,
    dτ_ρ,
    _dt,
    _dx,
    _dy,
    _dz,
    args,
) where {_T}
    av(A) = _av(A, i, j, k)
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)

    I = i + 1, j + 1, k + 1

    T_ijk = T[I...]
    args_ijk = (; T=T_ijk, P=av(args.P))
    phase_ijk = getindex_phase(phase, i, j, k)

    T[I...] =
        T_ijk +
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy) + d_za(qTz))) -
            compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[I...]) * _dt
        )
    return nothing
end

@parallel_indices (i, j, k) function check_res!(
    ResT::AbstractArray{_T,3}, T, Told, qTx2, qTy2, qTz2, ρCp, _dt, _dx, _dy, _dz
) where {_T}
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av(A) = _av(A, i, j, k)

    I = i + 1, j + 1, k + 1

    ResT[i, j, k] =
        -av(ρCp) * (T[I...] - Told[I...]) * _dt - (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2))

    return nothing
end

@parallel_indices (i, j, k) function check_res!(
    ResT::AbstractArray{_T,3},
    T,
    Told,
    qTx2,
    qTy2,
    qTz2,
    rheology,
    phase,
    _dt,
    _dx,
    _dy,
    _dz,
    args,
) where {_T}
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av(A) = _av(A, i, j, k)

    I = i + 1, j + 1, k + 1
    T_ijk = T[I...]
    args_ijk = (; T=T_ijk, P=av(args.P))
    phase_ijk = getindex_phase(phase, i, j, k)

    ResT[i, j, k] =
        -compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[I...]) * _dt -
        (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2))

    return nothing
end

## 2D KERNELS

@parallel_indices (i, j) function compute_flux!(
    qTx::AbstractArray{_T,2}, qTy, qTx2, qTy2, T, K, θr_dτ, _dx, _dy
) where {_T}
    nx = size(θr_dτ, 1)

    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5

    @inbounds if all((i, j) .≤ size(qTx))
        qx = qTx2[i, j] = -av_xa(K) * d_xi(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    @inbounds if all((i, j) .≤ size(qTy))
        qy = qTy2[i, j] = -av_ya(K) * d_yi(T)
        qTy[i, j] = (qTy[i, j] * av_ya(θr_dτ) + qy) / (1.0 + av_ya(θr_dτ))
    end
    return nothing
end

@parallel_indices (i, j) function compute_flux!(
    qTx::AbstractArray{_T,2}, qTy, qTx2, qTy2, T, rheology, phase, θr_dτ, _dx, _dy, args
) where {_T}
    nx = size(θr_dτ, 1)

    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5

    @inbounds if all((i, j) .≤ size(qTx))
        phase_ij = getindex_phase(phase, clamp(i - 1, 1, nx), j + 1)
        K1 = compute_phase(compute_conductivity, rheology, phase_ij, args)
        phase_ij = getindex_phase(phase, clamp(i - 1, 1, nx), j)
        K2 = compute_phase(compute_conductivity, rheology, phase_ij, args)
        K = (K1 + K2) * 0.5

        qx = qTx2[i, j] = -K * d_xi(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    @inbounds if all((i, j) .≤ size(qTy))
        phase_ij = getindex_phase(phase, i, j)
        K1 = compute_phase(compute_conductivity, rheology, phase_ij, args)
        phase_ij = getindex_phase(phase, clamp(i - 1, 1, nx), j)
        K2 = compute_phase(compute_conductivity, rheology, phase_ij, args)
        K = (K1 + K2) * 0.5

        qy = qTy2[i, j] = -K * d_yi(T)
        qTy[i, j] = (qTy[i, j] * av_ya(θr_dτ) + qy) / (1.0 + av_ya(θr_dτ))
    end

    return nothing
end


@parallel_indices (i, j) function compute_flux!(
    qTx::AbstractArray{_T,2},
    qTy,
    T,
    rheology::NTuple{N,AbstractMaterialParamsStruct},
    phase_ratios,
    args,
    _dx,
    _dy,
) where {_T,N}
    nx = size(θr_dτ, 1)

    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    compute_K(phase) = fn_ratio(compute_conductivity, rheology, phase, args)

    @inbounds if all((i, j) .≤ size(qTx))
        phase_ij = getindex_phase(phase, clamp(i - 1, 1, nx), j + 1)
        K1 = compute_K(phase_ij)
        phase_ij = getindex_phase(phase, clamp(i - 1, 1, nx), j)
        K2 = compute_K(phase_ij)
        K = (K1 + K2) * 0.5

        qx = qTx2[i, j] = -K * d_xi(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    @inbounds if all((i, j) .≤ size(qTy))
        phase_ij = getindex_phase(phase, i, j)
        K1 = compute_K(phase_ij)
        phase_ij = getindex_phase(phase, clamp(i - 1, 1, nx), j)
        K1 = compute_K(phase_ij)
        K = (K1 + K2) * 0.5

        qy = qTy2[i, j] = -K * d_yi(T)
        qTy[i, j] = (qTy[i, j] * av_ya(θr_dτ) + qy) / (1.0 + av_ya(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j) function update_T!(
    T::AbstractArray{_T,2}, Told, qTx, qTy, ρCp, dτ_ρ, _dt, _dx, _dy
) where {_T}
    nx, ny = size(ρCp)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    #! format: off
    function av(A)
        (
            A[clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)] +
            A[clamp(i - 1, 1, nx), j] +
            A[i, clamp(j - 1, 1, ny)] +
            A[i, j]
        ) * 0.25
    end
    #! format: on

    T[i + 1, j + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy))) -
            av(ρCp) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j) function update_T!(
    T::AbstractArray{_T,2},
    Told,
    qTx,
    qTy,
    rheology,
    phase,
    dτ_ρ,
    _dt,
    _dx,
    _dy,
    args::NamedTuple,
) where {_T}
    nx, ny = size(args.P)

    i0 = clamp(i - 1, 1, nx)
    i1 = clamp(i, 1, nx)
    j0 = clamp(j - 1, 1, ny)
    j1 = clamp(j, 1, ny)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av(A) = (A[i0, j0] + A[i0, j1] + A[i1, j0] + A[i1, j1]) * 0.25

    T_ij = T[i + 1, j + 1]
    args_ij = (; T=T_ij, P=av(args.P))

    ρCp =
        (
            compute_ρCp(rheology, getindex_phase(phase, i0, j0), args_ij) +
            compute_ρCp(rheology, getindex_phase(phase, i0, j1), args_ij) +
            compute_ρCp(rheology, getindex_phase(phase, i1, j0), args_ij) +
            compute_ρCp(rheology, getindex_phase(phase, i1, j1), args_ij)
        ) * 0.25

    T[i + 1, j + 1] +=
        av(dτ_ρ) * ((-(d_xa(qTx) + d_ya(qTy))) - ρCp * (T_ij - Told[i + 1, j + 1]) * _dt)
    return nothing
end

@parallel_indices (i, j) function check_res!(
    ResT::AbstractArray{_T,2}, T, Told, qTx2, qTy2, ρCp, _dt, _dx, _dy
) where {_T}
    nx, ny = size(ρCp)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    #! format: off
    function av(A)
        (
            A[clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)] +
            A[clamp(i - 1, 1, nx), clamp(j, 1, ny)] +
            A[clamp(i, 1, nx), clamp(j - 1, 1, ny)] +
            A[clamp(i, 1, nx), clamp(j, 1, ny)]
        ) * 0.25
    end
    #! format: on

    ResT[i, j] =
        -av(ρCp) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2))
    return nothing
end

@parallel_indices (i, j) function check_res!(
    ResT::AbstractArray{_T,2}, T, Told, qTx2, qTy2, rheology, phase, _dt, _dx, _dy, args
) where {_T}
    nx, ny = size(args.P)

    i0 = clamp(i - 1, 1, nx)
    i1 = clamp(i, 1, nx)
    j0 = clamp(j - 1, 1, ny)
    j1 = clamp(j, 1, ny)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av(A) = (A[i0, j0] + A[i0, j1] + A[i1, j0] + A[i1, j1]) * 0.25

    T_ij = T[i + 1, j + 1]
    args_ij = (; T=T_ij, P=av(args.P))

    ρCp =
        (
            compute_ρCp(rheology, getindex_phase(phase, i0, j0), args_ij) +
            compute_ρCp(rheology, getindex_phase(phase, i0, j1), args_ij) +
            compute_ρCp(rheology, getindex_phase(phase, i1, j0), args_ij) +
            compute_ρCp(rheology, getindex_phase(phase, i1, j1), args_ij)
        ) * 0.25

    ResT[i, j] =
        -ρCp * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2))
    return nothing
end

@parallel_indices (i) function update_ΔT!(ΔT, T, Told)
    _update_ΔT!(ΔT, T, Told, i)
    return nothing
end

@parallel_indices (i, j) function update_ΔT!(ΔT, T, Told)
    _update_ΔT!(ΔT, T, Told, i, j)
    return nothing
end

@parallel_indices (i, j, k) function update_ΔT!(ΔT, T, Told)
    _update_ΔT!(ΔT, T, Told, i, j, k)
    return nothing
end

@inline function _update_ΔT!(ΔT, T, Told, I::Vararg{Int,N}) where {N}
    return ΔT[I...] = T[I...] - Told[I...]
end

""" 
Compute shear heating withe GeoParams - should be dimension agnostic
The efficiency of shear heating is set to 1.0 by default

H_s = compute_shearheating(s:<AbstractShearheating, τ, ε, ε_el)

Computes the shear heating source term

```math  
H_s = \\Chi \\cdot \\tau_{ij} ( \\dot{\\varepsilon}_{ij} - \\dot{\\varepsilon}^{el}_{ij})
```

# Parameters
- ``\\Chi`` : The efficiency of shear heating (between 0-1)
- ``\\tau_{ij}`` : The full deviatoric stress tensor [4 components in 2D; 9 in 3D]
- ``\\dot{\\varepsilon}_{ij}`` : The full deviatoric strainrate tensor
- ``\\dot{\\varepsilon}^{el}_{ij}`` : The full elastic deviatoric strainrate tensor


"""
@parallel_indices (i,j) function compute_SH!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_old,
    τyy_old,
    τxyv_old,
    τII_old,
    εxx,
    εyy,
    εxyv,
    shearheating,
    phase_center,
    rheology, 
    dt,
)

    idx = i, j 

    phase = @inbounds phase_center[i, j]
    X = ConstantShearheating(Χ=1.0NoUnits)
    _Gdt = inv(get_G(rheology, phase) *dt)

    τ = τxx, τyy, τxy
    τ_old = τxx_old, τyy_old, τxyv_old
    ε = εxx, εyy, εxyv
  
    τij, τij_p_o, εij_p = cache_tensors(τ, τ_old, ε, idx...)
    τII[idx...] = τII_ij = second_invariant(τij...)
    τII_old[idx...] = τII_ij_old = second_invariant(τij_p_o...)

    εij_el = if !isinf(_Gdt)
        0.5 * ((τII_ij - τII_ij_old) * _Gdt)
      else
         zero(eltype(τxx))
      end
    


    shearheating = GeoParams.compute_shearheating(X, τij, εij_p, εij_el)
    return nothing
end

### SOLVERS COLLECTION BELOW - THEY SHOULD BE DIMENSION AGNOSTIC

@inline flux_range(nx, ny) = @idx (nx + 3, ny + 1)
@inline flux_range(nx, ny, nz) = @idx (nx, ny, nz)

@inline update_range(nx, ny) = @idx (nx + 1, ny - 1)
@inline update_range(nx, ny, nz) = residual_range(nx, ny, nz)

@inline residual_range(nx, ny) = update_range(nx, ny)
@inline residual_range(nx, ny, nz) = @idx (nx - 1, ny - 1, nz - 1)

function update_T(::Nothing, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
    @parallel update_range(ni...) update_T!(
        thermal.T, thermal.Told, @qT(thermal)..., ρCp, pt_thermal.dτ_ρ, _dt, _di...
    )
end

function update_T(
    ::Nothing, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args
)
    @parallel update_range(ni...) update_T!(
        thermal.T,
        thermal.Told,
        @qT(thermal)...,
        rheology,
        phase,
        pt_thermal.dτ_ρ,
        _dt,
        _di...,
        args,
    )
end

function update_T(igg, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
    # @hide_communication b_width begin # communication/computation overlap
    @parallel update_range(ni...) update_T!(
        thermal.T, thermal.Told, @qT(thermal)..., ρCp, pt_thermal.dτ_ρ, _dt, _di...
    )
    update_halo!(thermal.T)
    # end
    return nothing
end

function update_T(igg, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args)
    # @hide_communication b_width begin # communication/computation overlap
    @parallel update_range(ni...) update_T!(
        thermal.T,
        thermal.Told,
        @qT(thermal)...,
        rheology,
        phase,
        pt_thermal.dτ_ρ,
        _dt,
        _di...,
        args,
    )
    update_halo!(thermal.T)
    # end
    return nothing
end

"""
    heatdiffusion_PT!(thermal, pt_thermal, K, ρCp, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations. Both `K` and `ρCp` are n-dimensional arrays.
"""
function heatdiffusion_PT!(
    thermal::ThermalArrays,
    pt_thermal::PTThermalCoeffs,
    thermal_bc::TemperatureBoundaryConditions,
    stokes::StokesArrays,
    SH::AbstractArray,
    K::AbstractArray,
    ρCp::AbstractArray,
    dt,
    di;
    igg=nothing,
    b_width=(4, 4, 4),
    iterMax=50e3,
    nout=1e3,
    verbose=true,
)
    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T

    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ

    println("\n ====================================\n")
    println("Starting thermal diffusion solver...\n")

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel flux_range(ni...) compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, K, pt_thermal.θr_dτ, _di...
            )
            @parallel (@idx ni) compute_SH!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                stokes.τ_o.II,
                @strain(stokes)...,
                SH, 
                phase_c,
                tupleize(rheology), # needs to be a tuple
                dt,
            )
            update_T(igg, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
            thermal_bcs!(thermal.T, thermal_bc)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel residual_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    ρCp,
                    _dt,
                    _di...,
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    println("\n ...solver finished in $(round(wtime0, sigdigits=5)) seconds \n")
    println("====================================\n")

    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

    return nothing
end

"""
    heatdiffusion_PT!(thermal, pt_thermal, rheology, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations.
"""
function heatdiffusion_PT!(
    thermal::ThermalArrays,
    pt_thermal::PTThermalCoeffs,
    thermal_bc::TemperatureBoundaryConditions,
    stokes::StokesArrays,
    SH::AbstractArray,
    rheology,
    args::NamedTuple,
    dt,
    di;
    igg=nothing,
    phase=nothing,
    b_width=(4, 4, 4),
    iterMax=50e3,
    nout=1e3,
    verbose=true,
)

    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T

    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ

    println("\n ====================================\n")
    println("Starting thermal diffusion solver...\n")

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel flux_range(ni...) compute_flux!(
                @qT(thermal)...,
                @qT2(thermal)...,
                thermal.T,
                rheology,
                phase,
                pt_thermal.θr_dτ,
                _di...,
                args,
            )
            @parallel (@idx ni) compute_SH!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                stokes.τ_o.II,
                @strain(stokes)...,
                SH,
                phase_c,
                tupleize(rheology), # needs to be a tuple
                dt,
            )
            update_T(igg, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args)
            thermal_bcs!(thermal.T, thermal_bc)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel residual_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    rheology,
                    phase,
                    _dt,
                    _di...,
                    args,
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    println("\n ...solver finished in $(round(wtime0, sigdigits=5)) seconds \n")
    println("====================================\n")

    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
    @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

    return nothing
end

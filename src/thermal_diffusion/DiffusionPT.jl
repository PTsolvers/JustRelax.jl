## GeoParams

# include("Rheology.jl")

@inline get_phase(x::PhaseRatio) = x.center
@inline get_phase(x) = x

update_pt_thermal_arrays!(::Vararg{Any,N}) where {N} = nothing

function update_pt_thermal_arrays!(
    pt_thermal, phase_ratios::PhaseRatio, rheology, args, _dt
)
    ni = size(phase_ratios.center)

    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        phase_ratios.center,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        _dt,
    )

    return nothing
end

@inline function compute_phase(fn::F, rheology, phase::Int, args) where {F}
    return fn(rheology, phase, args)
end

@inline function compute_phase(fn::F, rheology, phase::SVector, args) where {F}
    return fn_ratio(fn, rheology, phase, args)
end

@inline compute_phase(fn::F, rheology, ::Nothing, args) where {F} = fn(rheology, args)

@inline Base.@propagate_inbounds function getindex_phase(
    phase::AbstractArray, I::Vararg{Int,N}
) where {N}
    return phase[I...]
end
@inline getindex_phase(::Nothing, I::Vararg{Int,N}) where {N} = nothing

@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

@inline function compute_ρCp(rheology, phase::Union{Nothing,Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
           compute_phase(compute_density, rheology, phase, args)
end

@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing,Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end

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
    d_xi(A) = _d_xi(A, i, j, k, _dx)
    d_yi(A) = _d_yi(A, i, j, k, _dy)
    d_zi(A) = _d_zi(A, i, j, k, _dz)
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    get_K(idx, args) = compute_phase(compute_conductivity, rheology, idx, args)

    I = i, j, k

    @inbounds if all(I .≤ size(qTx))
        T_ijk = (T[(I .+ 1)...] + T[i, j + 1, k + 1]) * 0.5
        args_ijk = (; T=T_ijk, P=av_yz(args.P))
        K =
            (
                get_K(getindex_phase(phase, i, j + 1, k), args_ijk) +
                get_K(getindex_phase(phase, i, j, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i, j + 1, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i, j, k), args_ijk)
            ) * 0.25

        qx = qTx2[I...] = -K * d_xi(T)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTy))
        T_ijk = (T[(I .+ 1)...] + T[i + 1, j, k + 1]) * 0.5
        args_ijk = (; T=T_ijk, P=av_xz(args.P))
        K =
            (
                get_K(getindex_phase(phase, i + 1, j, k), args_ijk) +
                get_K(getindex_phase(phase, i, j, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i + 1, j, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i, j, k), args_ijk)
            ) * 0.25

        qy = qTy2[I...] = -K * d_yi(T)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTz))
        T_ijk = (T[(I .+ 1)...] + T[i + 1, j + 1, k]) * 0.5
        args_ijk = (; T=T_ijk, P=av_xy(args.P))
        K =
            (
                get_K(getindex_phase(phase, i + 1, j + 1, k), args_ijk) +
                get_K(getindex_phase(phase, i, j + 1, k), args_ijk) +
                get_K(getindex_phase(phase, i + 1, j, k), args_ijk) +
                get_K(getindex_phase(phase, i, j, k), args_ijk)
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

    if all((i, j) .≤ size(qTx))
        ii, jj = clamp(i - 1, 1, nx), j + 1
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K1 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)

        ii, jj = clamp(i - 1, 1, nx), j
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K2 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)
        K = (K1 + K2) * 0.5

        qx = qTx2[i, j] = -K * d_xi(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    if all((i, j) .≤ size(qTy))
        ii, jj = min(i, nx), j
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K1 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)

        ii, jj = max(i - 1, 1), j
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K2 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)
        K = (K1 + K2) * 0.5

        qy = qTy2[i, j] = -K * d_yi(T)
        qTy[i, j] = (qTy[i, j] * av_ya(θr_dτ) + qy) / (1.0 + av_ya(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j) function compute_flux!(
    qTx::AbstractArray{_T,2},
    qTy,
    qTx2,
    qTy2,
    T,
    rheology::NTuple{N,AbstractMaterialParamsStruct},
    phase_ratios::CellArray{C1,C2,C3,C4},
    θr_dτ,
    _dx,
    _dy,
    args,
) where {_T,N,C1,C2,C3,C4}
    nx = size(θr_dτ, 1)

    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    compute_K(phase, args) = fn_ratio(compute_conductivity, rheology, phase, args)

    @inbounds if all((i, j) .≤ size(qTx))
        ii, jj = clamp(i - 1, 1, nx), j + 1
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K1 = compute_K(phase_ij, args_ij)

        ii, jj = clamp(i - 1, 1, nx), j
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        K2 = compute_K(phase_ij, args_ij)
        K = (K1 + K2) * 0.5

        qx = qTx2[i, j] = -K * d_xi(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    @inbounds if all((i, j) .≤ size(qTy))
        ii, jj = min(i, nx), j
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K1 = compute_K(phase_ij, args_ij)

        ii, jj = clamp(i - 1, 1, nx), j
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = ntuple_idx(args, ii, jj)
        K2 = compute_K(phase_ij, args_ij)
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
    phases = get_phase(phase)

    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T
    update_pt_thermal_arrays!(pt_thermal, phase, rheology, args, _dt)

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
                phases,
                pt_thermal.θr_dτ,
                _di...,
                args,
            )
            update_T(
                igg, b_width, thermal, rheology, phases, pt_thermal, _dt, _di, ni, args
            )
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
                    phases,
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

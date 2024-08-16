
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

@parallel_indices (I...) function update_T!(
    T::AbstractArray{_T,3},
    Told,
    qTx,
    qTy,
    qTz,
    H,
    shear_heating,
    ρCp,
    dτ_ρ,
    _dt,
    _dx,
    _dy,
    _dz,
) where {_T}
    av(A) = _av(A, I...)
    d_xa(A) = _d_xa(A, I..., _dx)
    d_ya(A) = _d_ya(A, I..., _dy)
    d_za(A) = _d_za(A, I..., _dz)

    I1 = I .+ 1
    T[I1...] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy) + d_za(qTz))) -
            av(ρCp) * (T[I1...] - Told[I1...]) * _dt
        ) +
        av(H) +
        av(shear_heating)

    return nothing
end

@parallel_indices (i, j, k) function update_T!(
    T::AbstractArray{_T,3},
    Told,
    qTx,
    qTy,
    qTz,
    H,
    shear_heating,
    adiabatic,
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
            compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[I...]) * _dt +
            av(H) +
            av(shear_heating) +
            adiabatic[i, j, k] * T_ijk
        )
    return nothing
end

@parallel_indices (i, j, k) function check_res!(
    ResT::AbstractArray{_T,3},
    T,
    Told,
    qTx2,
    qTy2,
    qTz2,
    H,
    shear_heating,
    ρCp,
    _dt,
    _dx,
    _dy,
    _dz,
) where {_T}
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av(A) = _av(A, i, j, k)

    I = i + 1, j + 1, k + 1

    ResT[i, j, k] =
        -av(ρCp) * (T[I...] - Told[I...]) * _dt - (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2)) +
        av(H) +
        av(shear_heating)

    return nothing
end

@parallel_indices (i, j, k) function check_res!(
    ResT::AbstractArray{_T,3},
    T,
    Told,
    qTx2,
    qTy2,
    qTz2,
    H,
    shear_heating,
    adiabatic,
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
        (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2)) +
        av(H) +
        av(shear_heating) +
        adiabatic[i, j, k] * T_ijk

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

# @parallel_indices (i, j) function update_T!(
#     T::AbstractArray{_T,2}, Told, qTx, qTy, H, shear_heating, ρCp, dτ_ρ, _dt, _dx, _dy
# ) where {_T}
#     nx, ny = size(ρCp)

#     d_xa(A) = _d_xa(A, i, j, _dx)
#     d_ya(A) = _d_ya(A, i, j, _dy)
#     #! format: off
#     function av(A)
#         (
#             A[clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)] +
#             A[clamp(i - 1, 1, nx), j] +
#             A[i, clamp(j - 1, 1, ny)] +
#             A[i, j]
#         ) * 0.25
#     end
#     #! format: on

#     T[i + 1, j + 1] +=
#         av(dτ_ρ) * (
#             (-(d_xa(qTx) + d_ya(qTy))) -
#             av(ρCp) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt +
#             av(H) +
#             av(shear_heating)
#         )
#     return nothing
# end


# @parallel_indices (i, j) function update_T!(
#     T::AbstractArray{_T,2},
#     Told,
#     qTx,
#     qTy,
#     H,
#     shear_heating,
#     adiabatic,
#     rheology,
#     phase,
#     dτ_ρ,
#     _dt,
#     _dx,
#     _dy,
#     args::NamedTuple,
# ) where {_T}
#     nx, ny = size(args.P)

#     i0 = clamp(i - 1, 1, nx)
#     i1 = clamp(i, 1, nx)
#     j0 = clamp(j - 1, 1, ny)
#     j1 = clamp(j, 1, ny)

#     d_xa(A) = _d_xa(A, i, j, _dx)
#     d_ya(A) = _d_ya(A, i, j, _dy)
#     av(A) = (A[i0, j0] + A[i0, j1] + A[i1, j0] + A[i1, j1]) * 0.25

#     T_ij = T[i + 1, j + 1]
#     args_ij = (; T=T_ij, P=av(args.P))

#     ρCp =
#         (
#             compute_ρCp(rheology, getindex_phase(phase, i0, j0), args_ij) +
#             compute_ρCp(rheology, getindex_phase(phase, i0, j1), args_ij) +
#             compute_ρCp(rheology, getindex_phase(phase, i1, j0), args_ij) +
#             compute_ρCp(rheology, getindex_phase(phase, i1, j1), args_ij)
#         ) * 0.25

#     T[i + 1, j + 1] +=
#         av(dτ_ρ) * (
#             (-(d_xa(qTx) + d_ya(qTy))) - ρCp * (T_ij - Told[i + 1, j + 1]) * _dt +
#             av(H) +
#             av(shear_heating) +
#             adiabatic[i, j] * T[i + 1, j + 1]
#         )
#     return nothing
# end

@parallel_indices (i, j) function update_T!(
    T::AbstractArray{_T,2}, Told, qTx, qTy, H, shear_heating, ρCp, dτ_ρ, _dt, _dx, _dy
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

    T[i + 1, j + 1] =
        (
            av(dτ_ρ) * (
                -(d_xa(qTx) + d_ya(qTy)) +
                Told[i + 1, j + 1] * av(ρCp) * _dt +
                av(H) +
                av(shear_heating)
            ) + T[i + 1, j + 1]
        ) / (one(_T) + av(dτ_ρ) * av(ρCp) * _dt)

    return nothing
end

@parallel_indices (i, j) function update_T!(
    T::AbstractArray{_T,2},
    Told,
    qTx,
    qTy,
    H,
    shear_heating,
    adiabatic,
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

    T[i + 1, j + 1] =
        (
            av(dτ_ρ) * (
                -(d_xa(qTx) + d_ya(qTy)) +
                Told[i + 1, j + 1] * ρCp * _dt +
                av(H) +
                av(shear_heating) +
                adiabatic[i, j] * T[i + 1, j + 1]
            ) + T[i + 1, j + 1]
        ) / (one(_T) + av(dτ_ρ) * ρCp * _dt)

    return nothing
end

@parallel_indices (i, j) function check_res!(
    ResT::AbstractArray{_T,2}, T, Told, qTx2, qTy2, H, shear_heating, ρCp, _dt, _dx, _dy
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
        -av(ρCp) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt -
        (d_xa(qTx2) + d_ya(qTy2)) +
        av(H) +
        av(shear_heating)
    return nothing
end

@parallel_indices (i, j) function check_res!(
    ResT::AbstractArray{_T,2},
    T,
    Told,
    qTx2,
    qTy2,
    H,
    shear_heating,
    adiabatic,
    rheology,
    phase,
    _dt,
    _dx,
    _dy,
    args,
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
        -ρCp * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2)) +
        av(H) +
        av(shear_heating) +
        adiabatic[i, j] * T[i + 1, j + 1]

    return nothing
end

@parallel_indices (I...) function update_ΔT!(ΔT, T, Told)
    ΔT[I...] = T[I...] - Told[I...]
    return nothing
end

function update_T(::Nothing, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
    @parallel update_range(ni...) update_T!(
        thermal.T,
        thermal.Told,
        @qT(thermal)...,
        thermal.H,
        thermal.shear_heating,
        ρCp,
        pt_thermal.dτ_ρ,
        _dt,
        _di...,
    )
end

function update_T(
    ::Nothing, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args
)
    @parallel update_range(ni...) update_T!(
        thermal.T,
        thermal.Told,
        @qT(thermal)...,
        thermal.H,
        thermal.shear_heating,
        thermal.adiabatic,
        rheology,
        phase,
        pt_thermal.dτ_ρ,
        _dt,
        _di...,
        args,
    )
end

@parallel_indices (i, j, k) function adiabatic_heating(
    A, Vx, Vy, Vz, P, rheology, phases, _dx, _dy, _dz
)
    I = i, j, k
    I1 = i1, j1, k1 = I .+ 1
    @inbounds begin
        α =
            (
                compute_α(rheology, getindex_phase(phases, I...)) +
                compute_α(rheology, getindex_phase(phases, i, j1, k)) +
                compute_α(rheology, getindex_phase(phases, i1, j, k)) +
                compute_α(rheology, getindex_phase(phases, i1, j1, k)) +
                compute_α(rheology, getindex_phase(phases, i, j1, k1)) +
                compute_α(rheology, getindex_phase(phases, i1, j, k1)) +
                compute_α(rheology, getindex_phase(phases, i1, j1, k1)) +
                compute_α(rheology, getindex_phase(phases, I1...))
            ) * 0.125
        # cache P around T node
        P111 = P[I...]
        P112 = P[i, j, k1]
        P121 = P[i, j1, k]
        P122 = P[i, j1, k1]
        P211 = P[i1, j, k]
        P212 = P[i1, j, k1]
        P221 = P[i1, j1, k]
        P222 = P[i1, j1, k1]
        # P averages
        Px_L = (P111 + P121 + P112 + P122) * 0.25
        Px_R = (P211 + P221 + P212 + P222) * 0.25
        Py_F = (P111 + P211 + P112 + P212) * 0.25
        Py_B = (P121 + P221 + P122 + P222) * 0.25
        Pz_B = (P111 + P211 + P121 + P221) * 0.25
        Pz_T = (P112 + P212 + P122 + P222) * 0.25
        # Vx average
        Vx_av =
            0.25 *
            (Vx[I1...] + Vx[i1, j1, k1 + 1] + Vx[i1, j1 + 1, k1] + Vx[i1, j1 + 1, k1 + 1])
        # Vy average
        Vy_av =
            0.25 *
            (Vy[I1...] + Vy[i1 + 1, j1, k1] + Vy[i1, j1, k1 + 1] + Vy[i1 + 1, j1, k1 + 1])
        # Vz average
        Vz_av =
            0.25 *
            (Vz[I1...] + Vz[i1 + 1, j1, k1] + Vz[i1, j1 + 1, k1] + Vz[i1 + 1, j1 + 1, k1])
        dPdx = Vx_av * (Px_R - Px_L) * _dx
        dPdy = Vy_av * (Py_B - Py_F) * _dy
        dPdz = Vz_av * (Pz_T - Pz_B) * _dz
        A[I...] = (dPdx + dPdy + dPdz) * α
    end
    return nothing
end

@parallel_indices (i, j) function adiabatic_heating(
    A, Vx, Vy, P, rheology, phases, _dx, _dy
)
    I = i, j
    I1 = i1, j1 = I .+ 1
    @inbounds begin
        α =
            (
                compute_α(rheology, getindex_phase(phases, I...)) +
                compute_α(rheology, getindex_phase(phases, i, j1)) +
                compute_α(rheology, getindex_phase(phases, i1, j)) +
                compute_α(rheology, getindex_phase(phases, I1...))
            ) * 0.25
        # cache P around T node
        P11 = P[I...]
        P12 = P[i, j1]
        P21 = P[i1, j]
        P22 = P[i1, j1]
        # P averages
        Px_L = (P11 + P12) * 0.5
        Px_R = (P21 + P22) * 0.5
        Py_T = (P12 + P22) * 0.5
        Py_B = (P11 + P21) * 0.5
        # Vx average
        Vx_av = (Vx[I1...] + Vx[i1, j1 + 1]) * 0.5
        # Vy average
        Vy_av = (Vy[I1...] + Vy[i1 + 1, j1]) * 0.5
        dPdx = (Px_R - Px_L) * _dx
        dPdy = (Py_T - Py_B) * _dy
        A[i1, j] = (Vx_av * dPdx + Vy_av * dPdy) * α
    end
    return nothing
end

function adiabatic_heating!(thermal, stokes, rheology, phases, di)
    idx = @idx (size(stokes.P) .- 1)
    _di = inv.(di)
    @parallel idx adiabatic_heating(
        thermal.adiabatic, @velocity(stokes)..., stokes.P, rheology, phases, _di...
    )
end

@inline adiabatic_heating!(thermal, ::Nothing, ::Vararg{Any,N}) where {N} = nothing

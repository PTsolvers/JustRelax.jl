isNotDirichlet(m, inds::Vararg{Int, N}) where {N} = iszero(m[inds...])
isNotDirichlet(::Nothing, ::Vararg{Int, N}) where {N} = true

## 3D KERNELS

@parallel_indices (I...) function compute_flux!(
        qTx::AbstractArray{_T, 3}, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, _di
    ) where {_T}
    d_xi(A, _dx) = _d_xi(A, _dx, I...)
    d_yi(A, _dy) = _d_yi(A, _dy, I...)
    d_zi(A, _dz) = _d_zi(A, _dz, I...)
    av_xy(A) = _av_xy(A, I...)
    av_xz(A) = _av_xz(A, I...)
    av_yz(A) = _av_yz(A, I...)

    if all(I .≤ size(qTx))
        _dx = @dx(_di, I[1])
        qx = qTx2[I...] = -av_yz(K) * d_xi(T, _dx)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    if all(I .≤ size(qTy))
        _dy = @dy(_di, I[2])
        qy = qTy2[I...] = -av_xz(K) * d_yi(T, _dy)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    if all(I .≤ size(qTz))
        _dz = @dz(_di, I[3])
        qz = qTz2[I...] = -av_xy(K) * d_zi(T, _dz)
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
        qTx::AbstractArray{_T, 3},
        qTy,
        qTz,
        qTx2,
        qTy2,
        qTz2,
        T,
        rheology,
        phase,
        θr_dτ,
        _di,
        args,
    ) where {_T}
    I = i, j, k

    @inline d_xi(A, _dx) = _d_xi(A, _dx, I...)
    @inline d_yi(A, _dy) = _d_yi(A, _dy, I...)
    @inline d_zi(A, _dz) = _d_zi(A, _dz, I...)
    @inline av_xy(A) = _av_xy(A, I...)
    @inline av_xz(A) = _av_xz(A, I...)
    @inline av_yz(A) = _av_yz(A, I...)

    get_K(idx, args) = compute_phase(compute_conductivity, rheology, idx, args)

    @inbounds if all(I .≤ size(qTx))
        T_ijk = (T[(I .+ 1)...] + T[i, j + 1, k + 1]) * 0.5
        args_ijk = (; T = T_ijk, P = av_yz(args.P))
        K =
            (
            get_K(getindex_phase(phase, i, j + 1, k), args_ijk) +
                get_K(getindex_phase(phase, i, j, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i, j + 1, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i, j, k), args_ijk)
        ) * 0.25

        _dx = @dx(_di, i)
        qx = qTx2[I...] = -K * d_xi(T, _dx)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTy))
        T_ijk = (T[(I .+ 1)...] + T[i + 1, j, k + 1]) * 0.5
        args_ijk = (; T = T_ijk, P = av_xz(args.P))
        K =
            (
            get_K(getindex_phase(phase, i + 1, j, k), args_ijk) +
                get_K(getindex_phase(phase, i, j, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i + 1, j, k + 1), args_ijk) +
                get_K(getindex_phase(phase, i, j, k), args_ijk)
        ) * 0.25

        _dy = @dy(_di, j)
        qy = qTy2[I...] = -K * d_yi(T, _dy)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTz))
        T_ijk = (T[(I .+ 1)...] + T[i + 1, j + 1, k]) * 0.5
        args_ijk = (; T = T_ijk, P = av_xy(args.P))
        K =
            (
            get_K(getindex_phase(phase, i + 1, j + 1, k), args_ijk) +
                get_K(getindex_phase(phase, i, j + 1, k), args_ijk) +
                get_K(getindex_phase(phase, i + 1, j, k), args_ijk) +
                get_K(getindex_phase(phase, i, j, k), args_ijk)
        ) * 0.25

        _dz = @dz(_di, k)
        qz = qTz2[I...] = -K * d_zi(T, _dz)
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (I...) function update_T!(
        T::AbstractArray{_T, 3},
        Told,
        qTx,
        qTy,
        qTz,
        H,
        shear_heating,
        ρCp,
        dτ_ρ,
        dirichlet,
        _dt,
        _di,
    ) where {_T}
    _dx, _dy, _dz = @dxi(_di, I...)
    av(A) = _av(A, I...)
    d_xa(A) = _d_xa(A, _dx, I...)
    d_ya(A) = _d_ya(A, _dy, I...)
    d_za(A) = _d_za(A, _dz, I...)

    I1 = I .+ 1

    if isdirichlet(dirichlet, I1...)
        apply_dirichlet!(T, dirichlet, I1...)

    else
        T[I1...] =
            (
            av(dτ_ρ) * (
                -(d_xa(qTx) + d_ya(qTy) + d_za(qTz)) +
                    Told[I1...] * av(ρCp) * _dt +
                    av(H) +
                    av(shear_heating)
            ) + T[I1...]
        ) / (one(_T) + av(dτ_ρ) * av(ρCp) * _dt)
    end

    return nothing
end

@parallel_indices (i, j, k) function update_T!(
        T::AbstractArray{_T, 3},
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
        dirichlet,
        _dt,
        _di,
        args,
    ) where {_T}
    nx, ny, nz = size(args.P)
    _dx = @dx(_di, min(i, nx))
    _dy = @dy(_di, min(j, ny))
    _dz = @dz(_di, min(k, nz))

    av(A) = _av(A, i, j, k)
    d_xa(A) = _d_xa(A, _dx, i, j, k)
    d_ya(A) = _d_ya(A, _dy, i, j, k)
    d_za(A) = _d_za(A, _dz, i, j, k)

    I1 = i + 1, j + 1, k + 1

    if isdirichlet(dirichlet, I1...)
        apply_dirichlet!(T, dirichlet, I1...)

    else
        T_ijk = T[I1...]
        args_ijk = (; T = T_ijk, P = av(args.P))
        phase_ijk = getindex_phase(phase, i, j, k)
        ρCp = compute_ρCp(rheology, phase_ijk, args_ijk)

        T[I1...] =
            (
            av(dτ_ρ) * (
                -(d_xa(qTx) + d_ya(qTy) + d_za(qTz)) +
                    Told[I1...] * ρCp * _dt +
                    av(H) +
                    av(shear_heating) +
                    adiabatic[i, j, k] * T_ijk
            ) + T_ijk
        ) / (one(_T) + av(dτ_ρ) * ρCp * _dt)
    end
    return nothing
end

@parallel_indices (i, j, k) function check_res!(
        ResT::AbstractArray{_T, 3},
        T,
        Told,
        qTx2,
        qTy2,
        qTz2,
        H,
        shear_heating,
        ρCp,
        dirichlet,
        _dt,
        _di,
    ) where {_T}
    _dx, _dy, _dz = @dxi(_di, i, j, k)
    d_xa(A) = _d_xa(A, _dx, i, j, k)
    d_ya(A) = _d_ya(A, _dy, i, j, k)
    d_za(A) = _d_za(A, _dz, i, j, k)
    av(A) = _av(A, i, j, k)

    I1 = i + 1, j + 1, k + 1

    ResT[i, j, k] = if isNotDirichlet(dirichlet.mask, I1...)
        -av(ρCp) * (T[I1...] - Told[I1...]) * _dt - (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2)) +
            av(H) +
            av(shear_heating)
    else
        zero(_T)
    end

    return nothing
end

@parallel_indices (i, j, k) function check_res!(
        ResT::AbstractArray{_T, 3},
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
        dirichlet,
        _dt,
        _di,
        args,
    ) where {_T}
    nx, ny, nz = size(args.P)
    _dx = @dx(_di, min(i, nx))
    _dy = @dy(_di, min(j, ny))
    _dz = @dz(_di, min(k, nz))
    d_xa(A) = _d_xa(A, _dx, i, j, k)
    d_ya(A) = _d_ya(A, _dy, i, j, k)
    d_za(A) = _d_za(A, _dz, i, j, k)
    av(A) = _av(A, i, j, k)

    I = i + 1, j + 1, k + 1
    T_ijk = T[I...]
    args_ijk = (; T = T_ijk, P = av(args.P))
    phase_ijk = getindex_phase(phase, i, j, k)

    ResT[i, j, k] = if isNotDirichlet(dirichlet.mask, I...)
        -compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[I...]) * _dt -
            (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2)) +
            av(H) +
            av(shear_heating) +
            adiabatic[i, j, k] * T_ijk
    else
        zero(_T)
    end
    return nothing
end

## 2D KERNELS

@parallel_indices (i, j) function compute_flux!(
        qTx::AbstractArray{_T, 2}, qTy, qTx2, qTy2, T, K, θr_dτ, _di_center
    ) where {_T}
    nx, ny = size(θr_dτ)

    @inbounds if all((i, j) .≤ size(qTx))
        _dx = @dx(_di_center, clamp(i, 1, nx))
        iL = clamp(i - 1, 1, nx)
        iR = clamp(i, 1, nx)
        Kx = (K[iL, j] + K[iR, j]) * 0.5
        θx = (θr_dτ[iL, j] + θr_dτ[iR, j]) * 0.5
        qx = qTx2[i, j] = -Kx * (T[i + 1, j + 1] - T[i, j + 1]) * _dx
        qTx[i, j] = (qTx[i, j] * θx + qx) / (1.0 + θx)
    end

    @inbounds if all((i, j) .≤ size(qTy))
        _dy = @dy(_di_center, clamp(j, 1, ny))
        jB = clamp(j - 1, 1, ny)
        jT = clamp(j, 1, ny)
        Ky = (K[i, jB] + K[i, jT]) * 0.5
        θy = (θr_dτ[i, jB] + θr_dτ[i, jT]) * 0.5
        qy = qTy2[i, j] = -Ky * (T[i + 1, j + 1] - T[i + 1, j]) * _dy
        qTy[i, j] = (qTy[i, j] * θy + qy) / (1.0 + θy)
    end
    return nothing
end

@parallel_indices (i, j) function compute_flux!(
        qTx::AbstractArray{_T, 2}, qTy, qTx2, qTy2, T, rheology, phase, θr_dτ, _di_center, args
    ) where {_T}

    nx, ny = size(θr_dτ)

    if all((i, j) .≤ size(qTx))
        iL = clamp(i - 1, 1, nx)
        iR = clamp(i, 1, nx)
        T_ij = (T[i, j + 1] + T[i + 1, j + 1]) * 0.5

        ii, jj = iL, j
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K1 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)

        ii, jj = iR, j
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K2 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)
        K = (K1 + K2) * 0.5
        θx = (θr_dτ[iL, j] + θr_dτ[iR, j]) * 0.5

        _dx = @dx(_di_center, clamp(i, 1, nx))
        qx = qTx2[i, j] = -K * (T[i + 1, j + 1] - T[i, j + 1]) * _dx
        qTx[i, j] = (qTx[i, j] * θx + qx) / (1.0 + θx)
    end

    if all((i, j) .≤ size(qTy))
        jB = clamp(j - 1, 1, ny)
        jT = clamp(j, 1, ny)
        T_ij = (T[i + 1, j] + T[i + 1, j + 1]) * 0.5

        ii, jj = i, jB
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K1 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)

        ii, jj = i, jT
        phase_ij = getindex_phase(phase, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K2 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)
        K = (K1 + K2) * 0.5
        θy = (θr_dτ[i, jB] + θr_dτ[i, jT]) * 0.5

        _dy = @dy(_di_center, clamp(j, 1, ny))
        qy = qTy2[i, j] = -K * (T[i + 1, j + 1] - T[i + 1, j]) * _dy
        qTy[i, j] = (qTy[i, j] * θy + qy) / (1.0 + θy)
    end

    return nothing
end

@parallel_indices (i, j) function compute_flux!(
        qTx::AbstractArray{_T, 2},
        qTy,
        qTx2,
        qTy2,
        T,
        rheology::NTuple{N, AbstractMaterialParamsStruct},
        phase_ratios::CellArray{C1, C2, C3, C4},
        θr_dτ,
        _di_center,
        args,
    ) where {_T, N, C1, C2, C3, C4}
    nx, ny = size(θr_dτ)

    compute_K(phase, args) = fn_ratio(compute_conductivity, rheology, phase, args)

    @inbounds if all((i, j) .≤ size(qTx))
        iL = clamp(i - 1, 1, nx)
        iR = clamp(i, 1, nx)
        T_ij = (T[i, j + 1] + T[i + 1, j + 1]) * 0.5

        ii, jj = iL, j
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K1 = compute_K(phase_ij, args_ij)

        ii, jj = iR, j
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K2 = compute_K(phase_ij, args_ij)
        K = (K1 + K2) * 0.5
        θx = (θr_dτ[iL, j] + θr_dτ[iR, j]) * 0.5

        _dx = @dx(_di_center, clamp(i, 1, nx))
        qx = qTx2[i, j] = -K * (T[i + 1, j + 1] - T[i, j + 1]) * _dx
        qTx[i, j] = (qTx[i, j] * θx + qx) / (1.0 + θx)
    end

    @inbounds if all((i, j) .≤ size(qTy))
        jB = clamp(j - 1, 1, ny)
        jT = clamp(j, 1, ny)
        T_ij = (T[i + 1, j] + T[i + 1, j + 1]) * 0.5

        ii, jj = i, jB
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K1 = compute_K(phase_ij, args_ij)

        ii, jj = i, jT
        phase_ij = getindex_phase(phase_ratios, ii, jj)
        args_ij = (; ntuple_idx(args, ii, jj)..., T = T_ij)
        K2 = compute_K(phase_ij, args_ij)
        K = (K1 + K2) * 0.5
        θy = (θr_dτ[i, jB] + θr_dτ[i, jT]) * 0.5

        _dy = @dy(_di_center, clamp(j, 1, ny))
        qy = qTy2[i, j] = -K * (T[i + 1, j + 1] - T[i + 1, j]) * _dy
        qTy[i, j] = (qTy[i, j] * θy + qy) / (1.0 + θy)
    end

    return nothing
end

@parallel_indices (i, j) function update_T!(
        T::AbstractArray{_T, 2},
        Told,
        qTx,
        qTy,
        H,
        shear_heating,
        ρCp,
        dτ_ρ,
        dirichlet,
        _dt,
        _di_center,
    ) where {_T}
    _dx, _dy = @dxi(_di_center, i, j)
    I1 = i + 1, j + 1

    if isdirichlet(dirichlet, I1...)
        apply_dirichlet!(T, dirichlet, I1...)

    else
        T[I1...] =
            (
            dτ_ρ[i, j] * (
                -((qTx[i + 1, j] - qTx[i, j]) * _dx + (qTy[i, j + 1] - qTy[i, j]) * _dy) +
                    Told[I1...] * ρCp[i, j] * _dt +
                    H[i, j] +
                    shear_heating[i, j]
            ) + T[I1...]
        ) / (one(_T) + dτ_ρ[i, j] * ρCp[i, j] * _dt)
    end

    return nothing
end

@parallel_indices (i, j) function update_T!(
        T::AbstractArray{_T, 2},
        Told,
        qTx,
        qTy,
        H,
        shear_heating,
        adiabatic,
        rheology,
        phase,
        dτ_ρ,
        dirichlet,
        _dt,
        _di_vertex,
        args::NamedTuple,
    ) where {_T}

    I1 = i + 1, j + 1

    if isdirichlet(dirichlet, I1...)
        apply_dirichlet!(T, dirichlet, I1...)

    else

        nx, ny = size(args.P)

        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_vertex, j)

        T_ij = T[i + 1, j + 1]
        args_ij = (; ntuple_idx(args, i, j)..., T = T_ij)
        ρCp = compute_ρCp(rheology, getindex_phase(phase, i, j), args_ij)

        T[I1...] =
            (
            dτ_ρ[i, j] * (
                -((qTx[i + 1, j] - qTx[i, j]) * _dx + (qTy[i, j + 1] - qTy[i, j]) * _dy) +
                    Told[I1...] * ρCp * _dt +
                    H[i, j] +
                    shear_heating[i, j] +
                    adiabatic[i, j] * T[I1...]
            ) + T[I1...]
        ) / (one(_T) + dτ_ρ[i, j] * ρCp * _dt)
    end

    return nothing
end

@parallel_indices (i, j) function check_res!(
        ResT::AbstractArray{_T, 2},
        T,
        Told,
        qTx2,
        qTy2,
        H,
        shear_heating,
        ρCp,
        dirichlet,
        _dt,
        _di_vertex,
    ) where {_T}
    _dx, _dy = @dxi(_di_vertex, i, j)

    I1 = i + 1, j + 1
    ResT[i, j] = if isNotDirichlet(dirichlet.mask, I1...)
        -ρCp[i, j] * (T[I1...] - Told[I1...]) * _dt -
            ((qTx2[i + 1, j] - qTx2[i, j]) * _dx + (qTy2[i, j + 1] - qTy2[i, j]) * _dy) +
            H[i, j] +
            shear_heating[i, j]
    else
        zero(_T)
    end

    return nothing
end

@parallel_indices (i, j) function check_res!(
        ResT::AbstractArray{_T, 2},
        T,
        Told,
        qTx2,
        qTy2,
        H,
        shear_heating,
        adiabatic,
        rheology,
        phase,
        dirichlet,
        _dt,
        _di_vertex,
        args,
    ) where {_T}
    _dx = @dx(_di_vertex, i)
    _dy = @dy(_di_vertex, j)

    T_ij = T[i + 1, j + 1]
    args_ij = (; ntuple_idx(args, i, j)..., T = T_ij)
    ρCp = compute_ρCp(rheology, getindex_phase(phase, i, j), args_ij)

    I1 = i + 1, j + 1
    ResT[i, j] = if isNotDirichlet(dirichlet.mask, I1...)
        -ρCp * (T[I1...] - Told[I1...]) * _dt -
            ((qTx2[i + 1, j] - qTx2[i, j]) * _dx + (qTy2[i, j + 1] - qTy2[i, j]) * _dy) +
            H[i, j] +
            shear_heating[i, j] +
            adiabatic[i, j] * T[I1...]
    else
        zero(_T)
    end

    return nothing
end

@parallel_indices (I...) function update_ΔT!(ΔT, T, Told)
    ΔT[I...] = T[I...] - Told[I...]
    return nothing
end

function update_T(::Nothing, b_width, thermal, ρCp, pt_thermal, dirichlet, _dt, _di, ni)
    return @parallel update_range(ni...) update_T!(
        thermal.T,
        thermal.Told,
        @qT(thermal)...,
        thermal.H,
        thermal.shear_heating,
        ρCp,
        pt_thermal.dτ_ρ,
        dirichlet,
        _dt,
        _di,
    )
end

function update_T(
        ::Nothing, b_width, thermal, rheology, phase, pt_thermal, dirichlet, _dt, _di, ni, args
    )
    return @parallel update_range(ni...) update_T!(
        thermal.T,
        thermal.Told,
        @qT(thermal)...,
        thermal.H,
        thermal.shear_heating,
        thermal.adiabatic,
        rheology,
        phase,
        pt_thermal.dτ_ρ,
        dirichlet,
        _dt,
        _di,
        args,
    )
end

@parallel_indices (i, j, k) function adiabatic_heating(
        A, Vx, Vy, Vz, P, rheology, phases, _di
    )
    _dx, _dy, _dz = @dxi(_di, i, j, k)
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
        A, Vx, Vy, P, rheology, phases, _di_center
    )
    _dx, _dy = @dxi(_di_center, i, j)
    nx, ny = size(P)
    @inbounds begin
        α = compute_α(rheology, getindex_phase(phases, i, j))
        iW = max(i - 1, 1)
        iE = min(i + 1, nx)
        jS = max(j - 1, 1)
        jN = min(j + 1, ny)
        dPdx = (P[iE, j] - P[iW, j]) * _dx / (iE - iW + (iE == iW))
        dPdy = (P[i, jN] - P[i, jS]) * _dy / (jN - jS + (jN == jS))
        Vx_av = (Vx[i, j + 1] + Vx[i + 1, j + 1]) * 0.5
        Vy_av = (Vy[i + 1, j] + Vy[i + 1, j + 1]) * 0.5
        A[i, j] = (Vx_av * dPdx + Vy_av * dPdy) * α
    end
    return nothing
end

function adiabatic_heating!(thermal, stokes, rheology, phases, grid::Geometry{2})
    idx = @idx size(stokes.P)
    return @parallel idx adiabatic_heating(
        thermal.adiabatic, @velocity(stokes)..., stokes.P, rheology, phases, grid._di.center
    )
end

function adiabatic_heating!(thermal, stokes, rheology, phases, grid::Geometry{3})
    idx = @idx (size(stokes.P) .- 1)
    return @parallel idx adiabatic_heating(
        thermal.adiabatic, @velocity(stokes)..., stokes.P, rheology, phases, grid._di.center
    )
end

@inline adiabatic_heating!(thermal, ::Nothing, rheology, phases, ::Geometry{2}) = nothing
@inline adiabatic_heating!(thermal, ::Nothing, rheology, phases, ::Geometry{3}) = nothing
@inline adiabatic_heating!(thermal, ::Nothing, ::Vararg{Any, N}) where {N} = nothing

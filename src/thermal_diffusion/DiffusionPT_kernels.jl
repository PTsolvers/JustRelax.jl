isNotDirichlet(m, inds::Vararg{Int, N}) where {N} = iszero(m[inds...])
isNotDirichlet(::Nothing, ::Vararg{Int, N}) where {N} = true

## 3D KERNELS

@parallel_indices (I...) function compute_flux!(
        qTx::AbstractArray{_T, 3}, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, _di, bc_flux
    ) where {_T}
    i, j, k = I
    nx, ny, nz = size(θr_dτ)

    if all(I .≤ size(qTx))
        if i == 1 && !isa(bc_flux.left, Bool)
            qTx[I...] = bc_flux.left
        elseif i == size(qTx, 1) && !isa(bc_flux.right, Bool)
            qTx[I...] = bc_flux.right
        else
            iL = clamp(i - 1, 1, nx)
            iR = clamp(i, 1, nx)
            Kx = (K[iL, j, k] + K[iR, j, k]) * 0.5
            θx = (θr_dτ[iL, j, k] + θr_dτ[iR, j, k]) * 0.5
            _dx = @dx(_di, clamp(i, 1, nx))
            qx = qTx2[I...] = -Kx * (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) * _dx
            qTx[I...] = (qTx[I...] * θx + qx) / (1.0 + θx)
        end
    end

    if all(I .≤ size(qTy))
        if j == 1 && !isa(bc_flux.front, Bool)
            qTy[I...] = bc_flux.front
        elseif j == size(qTy, 2) && !isa(bc_flux.back, Bool)
            qTy[I...] = bc_flux.back
        else
            jF = clamp(j - 1, 1, ny)
            jB = clamp(j, 1, ny)
            Ky = (K[i, jF, k] + K[i, jB, k]) * 0.5
            θy = (θr_dτ[i, jF, k] + θr_dτ[i, jB, k]) * 0.5
            _dy = @dy(_di, clamp(j, 1, ny))
            qy = qTy2[I...] = -Ky * (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) * _dy
            qTy[I...] = (qTy[I...] * θy + qy) / (1.0 + θy)
        end
    end

    if all(I .≤ size(qTz))
        if k == 1 && !isa(bc_flux.bot, Bool)
            qTz[I...] = bc_flux.bot
        elseif k == size(qTz, 3) && !isa(bc_flux.top, Bool)
            qTz[I...] = bc_flux.top
        else
            kB = clamp(k - 1, 1, nz)
            kT = clamp(k, 1, nz)
            Kz = (K[i, j, kB] + K[i, j, kT]) * 0.5
            θz = (θr_dτ[i, j, kB] + θr_dτ[i, j, kT]) * 0.5
            _dz = @dz(_di, clamp(k, 1, nz))
            qz = qTz2[I...] = -Kz * (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) * _dz
            qTz[I...] = (qTz[I...] * θz + qz) / (1.0 + θz)
        end
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
        bc_flux,
    ) where {_T}
    I = i, j, k

    get_K(idx, args) = compute_phase(compute_conductivity, rheology, idx, args)
    nx, ny, nz = size(θr_dτ)

    @inbounds if all(I .≤ size(qTx))
        if i == 1 && !isa(bc_flux.left, Bool)
            qTx[I...] = bc_flux.left
        elseif i == size(qTx, 1) && !isa(bc_flux.right, Bool)
            qTx[I...] = bc_flux.right
        else
            iL = clamp(i - 1, 1, nx)
            iR = clamp(i, 1, nx)
            T_ijk = (T[i, j + 1, k + 1] + T[i + 1, j + 1, k + 1]) * 0.5

            args_L = (; getindex_NamedTuple(args, iL, j, k)..., T = T_ijk)
            args_R = (; getindex_NamedTuple(args, iR, j, k)..., T = T_ijk)
            K = (
                get_K(getindex_phase(phase, iL, j, k), args_L) +
                    get_K(getindex_phase(phase, iR, j, k), args_R)
            ) * 0.5
            θx = (θr_dτ[iL, j, k] + θr_dτ[iR, j, k]) * 0.5

            _dx = @dx(_di, clamp(i, 1, nx))
            qx = qTx2[I...] = -K * (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) * _dx
            qTx[I...] = (qTx[I...] * θx + qx) / (1.0 + θx)
        end
    end

    @inbounds if all(I .≤ size(qTy))
        if j == 1 && !isa(bc_flux.front, Bool)
            qTy[I...] = bc_flux.front
        elseif j == size(qTy, 2) && !isa(bc_flux.back, Bool)
            qTy[I...] = bc_flux.back
        else
            jF = clamp(j - 1, 1, ny)
            jB = clamp(j, 1, ny)
            T_ijk = (T[i + 1, j, k + 1] + T[i + 1, j + 1, k + 1]) * 0.5

            args_F = (; getindex_NamedTuple(args, i, jF, k)..., T = T_ijk)
            args_B = (; getindex_NamedTuple(args, i, jB, k)..., T = T_ijk)
            K = (
                get_K(getindex_phase(phase, i, jF, k), args_F) +
                    get_K(getindex_phase(phase, i, jB, k), args_B)
            ) * 0.5
            θy = (θr_dτ[i, jF, k] + θr_dτ[i, jB, k]) * 0.5

            _dy = @dy(_di, clamp(j, 1, ny))
            qy = qTy2[I...] = -K * (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) * _dy
            qTy[I...] = (qTy[I...] * θy + qy) / (1.0 + θy)
        end
    end

    @inbounds if all(I .≤ size(qTz))
        if k == 1 && !isa(bc_flux.bot, Bool)
            qTz[I...] = bc_flux.bot
        elseif k == size(qTz, 3) && !isa(bc_flux.top, Bool)
            qTz[I...] = bc_flux.top
        else
            kB = clamp(k - 1, 1, nz)
            kT = clamp(k, 1, nz)
            T_ijk = (T[i + 1, j + 1, k] + T[i + 1, j + 1, k + 1]) * 0.5

            args_B = (; getindex_NamedTuple(args, i, j, kB)..., T = T_ijk)
            args_T = (; getindex_NamedTuple(args, i, j, kT)..., T = T_ijk)
            K = (
                get_K(getindex_phase(phase, i, j, kB), args_B) +
                    get_K(getindex_phase(phase, i, j, kT), args_T)
            ) * 0.5
            θz = (θr_dτ[i, j, kB] + θr_dτ[i, j, kT]) * 0.5

            _dz = @dz(_di, clamp(k, 1, nz))
            qz = qTz2[I...] = -K * (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) * _dz
            qTz[I...] = (qTz[I...] * θz + qz) / (1.0 + θz)
        end
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
    i, j, k = I
    _dx, _dy, _dz = @dxi(_di, I...)

    I1 = I .+ 1

    if isdirichlet(dirichlet, I1...)
        apply_dirichlet!(T, dirichlet, I1...)

    else
        T[I1...] =
            (
            dτ_ρ[I...] * (
                -(
                    (qTx[i + 1, j, k] - qTx[I...]) * _dx +
                        (qTy[i, j + 1, k] - qTy[I...]) * _dy +
                        (qTz[i, j, k + 1] - qTz[I...]) * _dz
                ) +
                    Told[I1...] * ρCp[I...] * _dt +
                    H[I...] +
                    shear_heating[I...]
            ) + T[I1...]
        ) / (one(_T) + dτ_ρ[I...] * ρCp[I...] * _dt)
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
    _dx, _dy, _dz = @dxi(_di, i, j, k)

    I1 = i + 1, j + 1, k + 1

    if isdirichlet(dirichlet, I1...)
        apply_dirichlet!(T, dirichlet, I1...)

    else
        T_ijk = T[I1...]
        args_ijk = (; getindex_NamedTuple(args, i, j, k)..., T = T_ijk)
        phase_ijk = getindex_phase(phase, i, j, k)
        ρCp = compute_ρCp(rheology, phase_ijk, args_ijk)

        T[I1...] =
            (
            dτ_ρ[i, j, k] * (
                -(
                    (qTx[i + 1, j, k] - qTx[i, j, k]) * _dx +
                        (qTy[i, j + 1, k] - qTy[i, j, k]) * _dy +
                        (qTz[i, j, k + 1] - qTz[i, j, k]) * _dz
                ) +
                    Told[I1...] * ρCp * _dt +
                    compute_density_ratio(phase_ijk, rheology, args_ijk) * fn_ratio(compute_radioactive_heat, rheology, phase_ijk, args_ijk) +
                    H[i, j, k] +
                    shear_heating[i, j, k] +
                    adiabatic[i, j, k] * T_ijk
            ) + T_ijk
        ) / (one(_T) + dτ_ρ[i, j, k] * ρCp * _dt)
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

    I1 = i + 1, j + 1, k + 1

    ResT[i, j, k] = if isNotDirichlet(dirichlet.mask, I1...)
        -ρCp[i, j, k] * (T[I1...] - Told[I1...]) * _dt -
            (
            (qTx2[i + 1, j, k] - qTx2[i, j, k]) * _dx +
                (qTy2[i, j + 1, k] - qTy2[i, j, k]) * _dy +
                (qTz2[i, j, k + 1] - qTz2[i, j, k]) * _dz
        ) +
            H[i, j, k] +
            shear_heating[i, j, k]
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
    _dx, _dy, _dz = @dxi(_di, i, j, k)

    I = i + 1, j + 1, k + 1
    T_ijk = T[I...]
    args_ijk = (; getindex_NamedTuple(args, i, j, k)..., T = T_ijk)
    phase_ijk = getindex_phase(phase, i, j, k)

    ResT[i, j, k] = if isNotDirichlet(dirichlet.mask, I...)
        -compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[I...]) * _dt -
            (
            (qTx2[i + 1, j, k] - qTx2[i, j, k]) * _dx +
                (qTy2[i, j + 1, k] - qTy2[i, j, k]) * _dy +
                (qTz2[i, j, k + 1] - qTz2[i, j, k]) * _dz
        ) +
            compute_density_ratio(phase_ijk, rheology, args_ijk) * fn_ratio(compute_radioactive_heat, rheology, phase_ijk, args_ijk) +
            H[i, j, k] +
            shear_heating[i, j, k] +
            adiabatic[i, j, k] * T_ijk
    else
        zero(_T)
    end
    return nothing
end

## 2D KERNELS

@parallel_indices (i, j) function compute_flux!(
        qTx::AbstractArray{_T, 2}, qTy, qTx2, qTy2, T, K, θr_dτ, _di_center, bc_flux
    ) where {_T}
    nx, ny = size(θr_dτ)

    @inbounds if all((i, j) .≤ size(qTx))
        if i == 1 && !isa(bc_flux.left, Bool)
            qTx[i, j] = bc_flux.left
        elseif i == size(qTx, 1) && !isa(bc_flux.right, Bool)
            qTx[i, j] = bc_flux.right
        else
            _dx = @dx(_di_center, clamp(i, 1, nx))
            iL = clamp(i - 1, 1, nx)
            iR = clamp(i, 1, nx)
            Kx = (K[iL, j] + K[iR, j]) * 0.5
            θx = (θr_dτ[iL, j] + θr_dτ[iR, j]) * 0.5
            qx = qTx2[i, j] = -Kx * (T[i + 1, j + 1] - T[i, j + 1]) * _dx
            qTx[i, j] = (qTx[i, j] * θx + qx) / (1.0 + θx)
        end
    end

    @inbounds if all((i, j) .≤ size(qTy))
        if j == 1 && !isa(bc_flux.bot, Bool)
            qTy[i, j] = bc_flux.bot
        elseif j == size(qTy, 2) && !isa(bc_flux.top, Bool)
            qTy[i, j] = bc_flux.top
        else
            _dy = @dy(_di_center, clamp(j, 1, ny))
            jB = clamp(j - 1, 1, ny)
            jT = clamp(j, 1, ny)
            Ky = (K[i, jB] + K[i, jT]) * 0.5
            θy = (θr_dτ[i, jB] + θr_dτ[i, jT]) * 0.5
            qy = qTy2[i, j] = -Ky * (T[i + 1, j + 1] - T[i + 1, j]) * _dy
            qTy[i, j] = (qTy[i, j] * θy + qy) / (1.0 + θy)
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_flux!(
        qTx::AbstractArray{_T, 2},
        qTy,
        qTx2,
        qTy2,
        T,
        rheology,
        phase,
        θr_dτ,
        _di_center,
        args,
        bc_flux,
    ) where {_T}

    nx, ny = size(θr_dτ)

    if all((i, j) .≤ size(qTx))
        if i == 1 && !isa(bc_flux.left, Bool)
            qTx[i, j] = bc_flux.left
        elseif i == size(qTx, 1) && !isa(bc_flux.right, Bool)
            qTx[i, j] = bc_flux.right
        else
            iL = clamp(i - 1, 1, nx)
            iR = clamp(i, 1, nx)
            T_ij = (T[i, j + 1] + T[i + 1, j + 1]) * 0.5

            ii, jj = iL, j
            phase_ij = getindex_phase(phase, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K1 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)

            ii, jj = iR, j
            phase_ij = getindex_phase(phase, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K2 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)
            K = (K1 + K2) * 0.5
            θx = (θr_dτ[iL, j] + θr_dτ[iR, j]) * 0.5

            _dx = @dx(_di_center, clamp(i, 1, nx))
            qx = qTx2[i, j] = -K * (T[i + 1, j + 1] - T[i, j + 1]) * _dx
            qTx[i, j] = (qTx[i, j] * θx + qx) / (1.0 + θx)
        end
    end

    if all((i, j) .≤ size(qTy))
        if j == 1 && !isa(bc_flux.bot, Bool)
            qTy[i, j] = bc_flux.bot
        elseif j == size(qTy, 2) && !isa(bc_flux.top, Bool)
            qTy[i, j] = bc_flux.top
        else
            jB = clamp(j - 1, 1, ny)
            jT = clamp(j, 1, ny)
            T_ij = (T[i + 1, j] + T[i + 1, j + 1]) * 0.5

            ii, jj = i, jB
            phase_ij = getindex_phase(phase, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K1 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)

            ii, jj = i, jT
            phase_ij = getindex_phase(phase, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K2 = compute_phase(compute_conductivity, rheology, phase_ij, args_ij)
            K = (K1 + K2) * 0.5
            θy = (θr_dτ[i, jB] + θr_dτ[i, jT]) * 0.5

            _dy = @dy(_di_center, clamp(j, 1, ny))
            qy = qTy2[i, j] = -K * (T[i + 1, j + 1] - T[i + 1, j]) * _dy
            qTy[i, j] = (qTy[i, j] * θy + qy) / (1.0 + θy)
        end
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
        bc_flux,
    ) where {_T, N, C1, C2, C3, C4}
    nx, ny = size(θr_dτ)

    compute_K(phase, args) = fn_ratio(compute_conductivity, rheology, phase, args)

    @inbounds if all((i, j) .≤ size(qTx))
        if i == 1 && !isa(bc_flux.left, Bool)
            qTx[i, j] = bc_flux.left
        elseif i == size(qTx, 1) && !isa(bc_flux.right, Bool)
            qTx[i, j] = bc_flux.right
        else
            iL = clamp(i - 1, 1, nx)
            iR = clamp(i, 1, nx)
            T_ij = (T[i, j + 1] + T[i + 1, j + 1]) * 0.5

            ii, jj = iL, j
            phase_ij = getindex_phase(phase_ratios, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K1 = compute_K(phase_ij, args_ij)

            ii, jj = iR, j
            phase_ij = getindex_phase(phase_ratios, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K2 = compute_K(phase_ij, args_ij)
            K = (K1 + K2) * 0.5
            θx = (θr_dτ[iL, j] + θr_dτ[iR, j]) * 0.5

            _dx = @dx(_di_center, clamp(i, 1, nx))
            qx = qTx2[i, j] = -K * (T[i + 1, j + 1] - T[i, j + 1]) * _dx
            qTx[i, j] = (qTx[i, j] * θx + qx) / (1.0 + θx)
        end
    end

    @inbounds if all((i, j) .≤ size(qTy))
        if j == 1 && !isa(bc_flux.bot, Bool)
            qTy[i, j] = bc_flux.bot
        elseif j == size(qTy, 2) && !isa(bc_flux.top, Bool)
            qTy[i, j] = bc_flux.top
        else
            jB = clamp(j - 1, 1, ny)
            jT = clamp(j, 1, ny)
            T_ij = (T[i + 1, j] + T[i + 1, j + 1]) * 0.5

            ii, jj = i, jB
            phase_ij = getindex_phase(phase_ratios, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K1 = compute_K(phase_ij, args_ij)

            ii, jj = i, jT
            phase_ij = getindex_phase(phase_ratios, ii, jj)
            args_ij = (; getindex_NamedTuple(args, ii, jj)..., T = T_ij)
            K2 = compute_K(phase_ij, args_ij)
            K = (K1 + K2) * 0.5
            θy = (θr_dτ[i, jB] + θr_dτ[i, jT]) * 0.5

            _dy = @dy(_di_center, clamp(j, 1, ny))
            qy = qTy2[i, j] = -K * (T[i + 1, j + 1] - T[i + 1, j]) * _dy
            qTy[i, j] = (qTy[i, j] * θy + qy) / (1.0 + θy)
        end
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
        args_ij = (; getindex_NamedTuple(args, i, j)..., T = T_ij)
        phase_ij = getindex_phase(phase, i, j)
        ρCp = compute_ρCp(rheology, phase_ij, args_ij)

        T[I1...] =
            (
            dτ_ρ[i, j] * (
                -((qTx[i + 1, j] - qTx[i, j]) * _dx + (qTy[i, j + 1] - qTy[i, j]) * _dy) +
                    Told[I1...] * ρCp * _dt +
                    compute_density_ratio(phase_ij, rheology, args_ij) * fn_ratio(compute_radioactive_heat, rheology, phase_ij, args_ij) +
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
    args_ij = (; getindex_NamedTuple(args, i, j)..., T = T_ij)
    phase_ij = getindex_phase(phase, i, j)
    ρCp = compute_ρCp(rheology, phase_ij, args_ij)

    I1 = i + 1, j + 1
    ResT[i, j] = if isNotDirichlet(dirichlet.mask, I1...)
        -ρCp * (T[I1...] - Told[I1...]) * _dt -
            ((qTx2[i + 1, j] - qTx2[i, j]) * _dx + (qTy2[i, j + 1] - qTy2[i, j]) * _dy) +
            compute_density_ratio(phase_ij, rheology, args_ij) * fn_ratio(compute_radioactive_heat, rheology, phase_ij, args_ij) +
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

"""
    update_T(::Nothing, b_width, thermal, ρCp, pt_thermal, dirichlet, _dt, _di, ni)
    update_T(::Nothing, b_width, thermal, rheology, phase, pt_thermal, dirichlet, _dt, _di, ni, args)

Launch the pseudo-transient temperature update kernel over the active thermal
domain.

These wrappers select the appropriate kernel overload depending on whether the
solver works with precomputed `ρCp` fields or rheology-derived properties.
"""
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
        A, Vx, Vy, Vz, P, P0, rheology, phases, _di, _dt
    )
    _dx, _dy, _dz = @dxi(_di, i, j, k)
    nx, ny, nz = size(P)
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
        # average P and P0 @ T node
        Pv = (P[I...] + P[i, j, k1] + P[i, j1, k] + P[i, j1, k1] + P[i1, j, k] + P[i1, j, k1] + P[i1, j1, k] + P[i1, j1, k1]) / 8
        P0v = (P0[I...] + P0[i, j, k1] + P0[i, j1, k] + P0[i, j1, k1] + P0[i1, j, k] + P0[i1, j, k1] + P0[i1, j1, k] + P0[i1, j1, k1]) / 8
        # Adiabtic heating term
        A[I...] = (Pv - P0v) * α * _dt
    end
    return nothing
end

@parallel_indices (i, j) function adiabatic_heating(
        A, Vx, Vy, P, P0, rheology, phases, _di, _dt
    )
    _dx, _dy = @dxi(_di, i, j)
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
        # average P and P0 @ T node
        Pv = (P[I...] + P[i, j1] + P[i1, j] + P[i1, j1]) * 0.25
        P0v = (P0[I...] + P0[i, j1] + P0[i1, j] + P0[i1, j1]) * 0.25
        # Adiabtic heating term
        A[i1, j] = (Pv - P0v) * α * _dt
    end
    return nothing
end

"""
    adiabatic_heating!(thermal, stokes, rheology, phases, _dt, grid)

Fill `thermal.adiabatic` with the adiabatic heating term inferred from the
pressure change between `stokes.P0` and `stokes.P`.

The kernels average the local thermal expansivity over the temperature nodes and
scale the pressure increment by `inv(dt)`, passed here as `_dt`. When `stokes`
is `nothing`, the no-op overloads leave the field unchanged.
"""
function adiabatic_heating!(thermal, stokes, rheology, phases, _dt, grid::Geometry{2})
    idx = @idx (size(stokes.P) .- 1)
    return @parallel idx adiabatic_heating(
        thermal.adiabatic, @velocity(stokes)..., stokes.P, stokes.P0, rheology, phases, grid._di.center, _dt
    )
end

function adiabatic_heating!(thermal, stokes, rheology, phases, _dt, grid::Geometry{3})
    idx = @idx (size(stokes.P) .- 1)
    return @parallel idx adiabatic_heating(
        thermal.adiabatic, @velocity(stokes)..., stokes.P, stokes.P0, rheology, phases, grid._di.center, _dt
    )
end

@inline adiabatic_heating!(thermal, ::Nothing, rheology, phases, _dt, ::Geometry{2}) = nothing
@inline adiabatic_heating!(thermal, ::Nothing, rheology, phases, _dt, ::Geometry{3}) = nothing
@inline adiabatic_heating!(thermal, ::Nothing, ::Vararg{Any, N}) where {N} = nothing

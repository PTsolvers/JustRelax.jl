export ThermalParameters

struct ThermalParameters{T}
    κ::T
    _ρCp::T # 1/ρ/Cp
end

## 3D ELASTICITY MODULE

module ThermalDiffusion3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using MPI
using Printf

import JustRelax: IGG, ThermalParameters, solve!, assign!, norm_mpi, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs 

export solve!

macro innT3(ix, iy, iz)
    return esc(
        :(
            T[$ix + 1, $iy + 1, $iz + 1] *
            T[$ix + 1, $iy + 1, $iz + 1] *
            T[$ix + 1, $iy + 1, $iz + 1]
        ),
    )
end

macro av_xi_T3(ix, iy, iz)
    return esc(
        :(
            0.5 *
            (T[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (T[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (T[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_yi_T3(ix, iy, iz)
    return esc(
        :(
            0.5 *
            (T[$ix + 1, $iy, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (T[$ix + 1, $iy, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (T[$ix + 1, $iy, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_zi_T3(ix, iy, iz)
    return esc(
        :(
            0.5 *
            (T[$ix + 1, $iy + 1, $iz] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (T[$ix + 1, $iy + 1, $iz] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (T[$ix + 1, $iy + 1, $iz] + T[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_xi_3(κ, ix, iy, iz)
    return esc(
        :(
            0.5 *
            (κ[$ix, $iy + 1, $iz + 1] + κ[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (κ[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (κ[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_yi_3(κ, ix, iy, iz)
    return esc(
        :(
            0.5 *
            (κ[$ix + 1, $iy, $iz + 1] + κ[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (κ[$ix + 1, $iy, $iz + 1] + κ[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (κ[$ix + 1, $iy, $iz + 1] + κ[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_zi_3(κ, ix, iy, iz)
    return esc(
        :(
            0.5 *
            (κ[$ix + 1, $iy + 1, $iz] + κ[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (κ[$ix + 1, $iy + 1, $iz] + κ[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            (κ[$ix + 1, $iy + 1, $iz] + κ[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_xi_Re(ix, iy, iz)
    return esc(:(π + sqrt(π * π + max_lxyz2 / @av_xi_T3($ix, $iy, $iz) * _dt)))
end

macro av_yi_Re(ix, iy, iz)
    return esc(:(π + sqrt(π * π + max_lxyz2 / @av_yi_T3($ix, $iy, $iz) * _dt)))
end

macro av_zi_Re(ix, iy, iz)
    return esc(:(π + sqrt(π * π + max_lxyz2 / @av_zi_T3($ix, $iy, $iz) * _dt)))
end

macro Re(ix, iy, iz)
    return esc(:(π + sqrt(π * π + max_lxyz2 / @innT3($ix, $iy, $iz) * _dt)))
end

macro av_xi_θr_dτ(ix, iy, iz)
    return esc(:(max_lxyz / Vpdτ / @av_xi_Re($ix, $iy, $iz) * Resc))
end

macro av_yi_θr_dτ(ix, iy, iz)
    return esc(:(max_lxyz / Vpdτ / @av_yi_Re($ix, $iy, $iz) * Resc))
end

macro av_zi_θr_dτ(ix, iy, iz)
    return esc(:(max_lxyz / Vpdτ / @av_zi_Re($ix, $iy, $iz) * Resc))
end

macro dτ_ρ(ix, iy, iz)
    return esc(:(Vpdτ * max_lxyz / @innT3($ix, $iy, $iz) / @Re($ix, $iy, $iz) * Resc))
end

@parallel_indices (ix, iy, iz) function compute_flux!(
    qTx, qTy, qTz, T, κ, Vpdτ, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz
)
    if (ix ≤ size(qTx, 1) && iy ≤ size(qTx, 2) && iz ≤ size(qTx, 3))
        qTx[ix, iy, iz] =
            (
                qTx[ix, iy, iz]*@av_xi_θr_dτ(ix, iy, iz) -
                @av_xi_3(κ, ix, iy, iz)*@av_xi_T3(ix, iy, iz) *
                _dx *
                (T[ix + 1, iy + 1, iz + 1] - T[ix, iy + 1, iz + 1])
            ) / (1.0 + @av_xi_θr_dτ(ix, iy, iz))
    end

    if (ix ≤ size(qTy, 1) && iy ≤ size(qTy, 2) && iz ≤ size(qTy, 3))
        qTy[ix, iy, iz] =
            (
                qTy[ix, iy, iz] * @av_yi_θr_dτ(ix, iy, iz) -
                @av_yi_3(κ, ix, iy, iz)*@av_yi_T3(ix, iy, iz) *
                _dy *
                (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy, iz + 1])
            ) / (1.0 + @av_yi_θr_dτ(ix, iy, iz))
    end

    if (ix ≤ size(qTz, 1) && iy ≤ size(qTz, 2) && iz ≤ size(qTz, 3))
        qTz[ix, iy, iz] =
            (
                qTz[ix, iy, iz] * @av_zi_θr_dτ(ix, iy, iz) -
                @av_zi_3(κ, ix, iy, iz)* @av_zi_T3(ix, iy, iz) *
                _dz *
                (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz])
            ) / (1.0 + @av_zi_θr_dτ(ix, iy, iz))
    end

    return nothing
end

@parallel_indices (ix, iy, iz) function compute_update!(
    T,
    Told,
    qTx,
    qTy,
    qTz,
    _ρCp,
    Vpdτ,
    Resc,
    _dt,
    max_lxyz,
    max_lxyz2,
    _dx,
    _dy,
    _dz,
    size_innT_1,
    size_innT_2,
    size_innT_3,
)
    if (ix ≤ size_innT_1 && iy ≤ size_innT_2 && iz ≤ size_innT_3)
        T[ix + 1, iy + 1, iz + 1] =
            (
                T[ix + 1, iy + 1, iz + 1] +
                @dτ_ρ(ix, iy, iz) * (
                    _dt * Told[ix + 1, iy + 1, iz + 1] - _ρCp[ix + 1, iy + 1, iz + 1]*(
                        _dx * (qTx[ix + 1, iy, iz] - qTx[ix, iy, iz]) +
                        _dy * (qTy[ix, iy + 1, iz] - qTy[ix, iy, iz]) +
                        _dz * (qTz[ix, iy, iz + 1] - qTz[ix, iy, iz])
                    )
                )
            ) / (1.0 + _dt * @dτ_ρ(ix, iy, iz))
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function compute_flux_res!(
    qTx2, qTy2, qTz2, T, κ, _dx, _dy, _dz
)
    if (ix ≤ size(qTx2, 1) && iy ≤ size(qTx2, 2) && iz ≤ size(qTx2, 3))
        qTx2[ix, iy, iz] =
            -@av_xi_3(κ, ix, iy, iz)*@av_xi_T3(ix, iy, iz) *
            _dx *
            (T[ix + 1, iy + 1, iz + 1] - T[ix, iy + 1, iz + 1])
    end
    if (ix ≤ size(qTy2, 1) && iy ≤ size(qTy2, 2) && iz ≤ size(qTy2, 3))
        qTy2[ix, iy, iz] =
            -@av_yi_T3(ix, iy, iz) *
            _dy *
            (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy, iz + 1])
    end
    if (ix ≤ size(qTz2, 1) && iy ≤ size(qTz2, 2) && iz ≤ size(qTz2, 3))
        qTz2[ix, iy, iz] =
            -@av_zi_3(κ, ix, iy, iz)*@av_zi_T3(ix, iy, iz) *
            _dz *
            (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz])
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function check_res!(
    ResT, T, Told, qTx, qTy, qTz, _ρCp, _dt, _dx, _dy, _dz
)
    if (ix ≤ size(ResT, 1) && iy ≤ size(ResT, 2) && iz ≤ size(ResT, 3))
        ResT[ix, iy, iz] =
            -_dt * (T[ix + 1, iy + 1, iz + 1] - Told[ix + 1, iy + 1, iz + 1]) - _ρCp[ix + 1, iy + 1, iz + 1]*(
                _dx * (qTx[ix + 1, iy, iz] - qTx[ix, iy, iz]) +
                _dy * (qTy[ix, iy + 1, iz] - qTy[ix, iy, iz]) +
                _dz * (qTz[ix, iy, iz + 1] - qTz[ix, iy, iz])
            )
    end
    return nothing
end

## SOLVER

function solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs, 
    thermal_parameters::ThermalParameters{<:AbstractArray{_T, 3}},
    thermal_bc::NamedTuple,
    ni::NTuple{3,Integer},
    di::NTuple{3,_T},
    igg::IGG,
    dt;
    first_solve=false,
    iterMax=10e3,
    nout=500,
    b_width=(1, 1, 1),
    verbose=true,
) where {_T, M<:AbstractArray{<:Any, 3}}

    ## UNPACK
    @assert size(thermal.T) == ni

    _dt = 1/dt
    _dx, _dy, _dz = @. 1/di
    nx, ny, nz = ni
    size_innT_1, size_innT_2, size_innT_3 = nx - 2, ny - 2, nz - 2
    len_ResT_g =
        ((nx - 2 - 2) * igg.dims[1] + 2) *
        ((ny - 2 - 2) * igg.dims[2] + 2) *
        ((nz - 2 - 2) * igg.dims[3] + 2)
        
    if first_solve
        @parallel assign!(thermal.Told, thermal.T)
    end

    # Pseudo-transient iteration
    iter = 0
    err = 2 * pt_thermal.ϵ
    while err >  pt_thermal.ϵ && iter < iterMax
        @parallel compute_flux!(
            thermal.qTx,
            thermal.qTy,
            thermal.qTz,
            thermal.T,
            thermal_parameters.κ,
            pt_thermal.Vpdτ,
            pt_thermal.Resc,
            _dt,
            pt_thermal.max_lxyz,
            pt_thermal.max_lxyz2,
            _dx,
            _dy,
            _dz,
        )
        @hide_communication b_width begin # communication/computation overlap
            @parallel compute_update!(
                thermal.T,
                thermal.Told,
                thermal.qTx,
                thermal.qTy,
                thermal.qTz,
                thermal_parameters._ρCp,
                pt_thermal.Vpdτ,
                pt_thermal.Resc,
                _dt,
                pt_thermal.max_lxyz,
                pt_thermal.max_lxyz2,
                _dx,
                _dy,
                _dz,
                size_innT_1,
                size_innT_2,
                size_innT_3,
            )
            thermal_boundary_conditions!(thermal_bc, thermal.T)
            update_halo!(thermal.T)
        end
        
        iter += 1

        if iter % nout == 0
            @parallel compute_flux_res!(
                thermal.qTx2, 
                thermal.qTy2,
                thermal.qTz2, 
                thermal.T,
                thermal_parameters.κ,
                _dx, 
                _dy,
                _dz
            )

            @parallel check_res!(
                thermal.ResT,
                thermal.T,
                thermal.Told,
                thermal.qTx2,
                thermal.qTy2,
                thermal.qTz2,
                thermal_parameters._ρCp,
                _dt,
                _dx,
                _dy,
                _dz,
            )
            err = norm_mpi(thermal.ResT) / sqrt(len_ResT_g)

            if (igg.me == 0 && (verbose || err < ϵ || iter == iterMax))
                @printf(
                    "iter = %d, err = %1.3e \n",
                    iter,
                    err
                )
            end
        end
    end

    @parallel assign!(thermal.Told, thermal.T)
    
    if isnan(err)
        error("NaN")
    end

    return iter
end

end

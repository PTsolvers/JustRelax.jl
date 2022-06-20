struct ThermalParameters{T}
    K::T # thermal conductivity
    ρCp::T # density * heat capacity
end

# 1D THERMAL DIFFUSION MODULE

module ThermalDiffusion1D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
using JustRelax
using LinearAlgebra
using MPI
using Printf

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTx2, T, K, θr_dτ, _dx)
    @all(qTx) = (@all(qTx) * @all(θr_dτ) - @all(K) * @d(T) * _dx) / (1.0 + @all(θr_dτ))
    @all(qTx2) = -@all(K) * @d(T) * _dx
    return nothing
end

@parallel function compute_update!(T, Told, qTx, ρCp, dτ_ρ, _dt, _dx)
    @inn(T) =
        @inn(T) +
        @all(dτ_ρ) * ((-(@d(qTx) * _dx)) - @all(ρCp) * (@inn(T) - @inn(Told)) * _dt)
    return nothing
end

@parallel function check_res!(ResT, T, Told, qTx2, ρCp, _dt, _dx)
    @all(ResT) = -@all(ρCp) * (@inn(T) - @inn(Told)) * _dt - @d(qTx2) * _dx
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs,
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,1}},
    thermal_bc::NamedTuple,
    ni::NTuple{1,Integer},
    di::NTuple{1,_T},
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {_T,M<:AbstractArray{<:Any,1}}
    @assert size(thermal.T) == ni
    ## Unpack
    _dt = 1 / dt
    _dx, _dy = @. 1 / di

    _sq_len_RT = 1.0 / sqrt(length(thermal.ResT))

    # Pseudo-transient iteration
    iter = 0
    err = 2 * pt_thermal.ϵ
    while err > pt_thermal.ϵ && iter < iterMax
        @parallel compute_flux!(
            thermal.qTx,
            thermal.qTx2,
            thermal.T,
            thermal_parameters.K,
            pt_thermal.θr_dτ,
            _dx,
        )

        @parallel compute_update!(
            thermal.T,
            thermal.Told,
            thermal.qTx,
            thermal_parameters.ρCp,
            pt_thermal.dτ_ρ,
            _dt,
            _dx,
        )

        thermal_boundary_conditions!(thermal_bc, thermal.T)

        iter += 1

        if iter % nout == 0
            @parallel check_res!(
                thermal.ResT,
                thermal.T,
                thermal.Told,
                thermal.qTx2,
                thermal_parameters.ρCp,
                _dt,
                _dx,
            )

            err = norm(thermal.ResT) * _sq_len_RT

            if verbose && (err < pt_thermal.ϵ) || (iter === iterMax)
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    if iter < iterMax
        @printf("Converged in %d iterations witn err = %1.3e \n", iter, err)
    end

    @parallel assign!(thermal.Told, thermal.T)

    if isnan(err)
        error("NaN")
    end

    return iter
end

end

# 2D THERMAL DIFFUSION MODULE

module ThermalDiffusion2D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using MPI
using Printf

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTy, qTx2, qTy2, T, K, θr_dτ, _dx, _dy)
    @all(qTx) = (@all(qTx) * @all(θr_dτ) - @all(K) * @d_xi(T) * _dx) / (1.0 + @all(θr_dτ))
    @all(qTy) = (@all(qTy) * @all(θr_dτ) - @all(K) * @d_yi(T) * _dy) / (1.0 + @all(θr_dτ))
    @all(qTx2) = -@all(K) * @d_xi(T) * _dx
    @all(qTy2) = -@all(K) * @d_yi(T) * _dy
    return nothing
end

@parallel function compute_update!(T, Told, qTx, qTy, ρCp, dτ_ρ, _dt, _dx, _dy)
    @inn(T) =
        @inn(T) +
        @all(dτ_ρ) * (
            (-(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy)) -
            @all(ρCp) * (@inn(T) - @inn(Told)) * _dt
        )
    return nothing
end

@parallel function check_res!(ResT, T, Told, qTx2, qTy2, ρCp, _dt, _dx, _dy)
    @all(ResT) =
        -@all(ρCp) * (@inn(T) - @inn(Told)) * _dt - (@d_xa(qTx2) * _dx + @d_ya(qTy2) * _dy)
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs,
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,2}},
    thermal_bc::NamedTuple,
    ni::NTuple{2,Integer},
    di::NTuple{2,_T},
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {_T,M<:AbstractArray{<:Any,2}}
    @assert size(thermal.T) == ni
    ## Unpack
    _dt = 1 / dt
    _dx, _dy = @. 1 / di

    _sq_len_RT = 1.0 / sqrt(length(thermal.ResT))

    # Pseudo-transient iteration
    iter = 0
    err = 2 * pt_thermal.ϵ
    while err > pt_thermal.ϵ && iter < iterMax
        @parallel compute_flux!(
            thermal.qTx,
            thermal.qTy,
            thermal.qTx2,
            thermal.qTy2,
            thermal.T,
            thermal_parameters.K,
            pt_thermal.θr_dτ,
            _dx,
            _dy,
        )

        @parallel compute_update!(
            thermal.T,
            thermal.Told,
            thermal.qTx,
            thermal.qTy,
            thermal_parameters.ρCp,
            pt_thermal.dτ_ρ,
            _dt,
            _dx,
            _dy,
        )

        thermal_boundary_conditions!(thermal_bc, thermal.T)

        iter += 1

        if iter % nout == 0
            @parallel check_res!(
                thermal.ResT,
                thermal.T,
                thermal.Told,
                thermal.qTx2,
                thermal.qTy2,
                thermal_parameters.ρCp,
                _dt,
                _dx,
                _dy,
            )

            err = norm(thermal.ResT) * _sq_len_RT

            if verbose && (err < pt_thermal.ϵ) || (iter === iterMax)
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    if iter < iterMax
        @printf("Converged in %d iterations witn err = %1.3e \n", iter, err)
    end

    @parallel assign!(thermal.Told, thermal.T)

    if isnan(err)
        error("NaN")
    end

    return iter
end

end

# 3D THERMAL DIFFUSION MODULE

module ThermalDiffusion3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using MPI
using Printf

import JustRelax:
    IGG, ThermalParameters, solve!, assign!, norm_mpi, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!

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

macro inn3(A, ix, iy, iz)
    return esc(
        :(
            $A[$ix + 1, $iy + 1, $iz + 1] *
            $A[$ix + 1, $iy + 1, $iz + 1] *
            $A[$ix + 1, $iy + 1, $iz + 1]
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

macro av_xi_3(K, ix, iy, iz)
    return esc(
        :(
            0.5 *
            ($K[$ix, $iy + 1, $iz + 1] + K[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            ($K[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            ($K[$ix, $iy + 1, $iz + 1] + T[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_yi_3(K, ix, iy, iz)
    return esc(
        :(
            0.5 *
            ($K[$ix + 1, $iy, $iz + 1] + K[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            ($K[$ix + 1, $iy, $iz + 1] + K[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            ($K[$ix + 1, $iy, $iz + 1] + K[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

macro av_zi_3(K, ix, iy, iz)
    return esc(
        :(
            0.5 *
            ($K[$ix + 1, $iy + 1, $iz] + K[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            ($K[$ix + 1, $iy + 1, $iz] + K[$ix + 1, $iy + 1, $iz + 1]) *
            0.5 *
            ($K[$ix + 1, $iy + 1, $iz] + K[$ix + 1, $iy + 1, $iz + 1])
        ),
    )
end

# TODO: @av_xi_T3  SHOULD BE @av_xi_3(K,...)
macro av_xi_Re(ρCp, K, ix, iy, iz)
    return esc(
        :(
            π + sqrt(
                π * π +
                max_lxyz2 * @av_xi_3($ρCp, $ix, $iy, $iz) / @av_xi_3($K, $ix, $iy, $iz) *
                _dt,
            )
        ),
    )
end

macro av_yi_Re(ρCp, K, ix, iy, iz)
    return esc(
        :(
            π + sqrt(
                π * π +
                max_lxyz2 * @av_yi_3($ρCp, $ix, $iy, $iz) / @av_yi_3($K, $ix, $iy, $iz) *
                _dt,
            )
        ),
    )
end

macro av_zi_Re(ρCp, K, ix, iy, iz)
    return esc(
        :(
            π + sqrt(
                π * π +
                max_lxyz2 * @av_zi_3($ρCp, $ix, $iy, $iz) / @av_zi_3($K, $ix, $iy, $iz) *
                _dt,
            )
        ),
    )
end

macro Re(ρCp, K, ix, iy, iz)
    return esc(
        :(
            π + sqrt(
                π * π +
                max_lxyz2 * @inn3($ρCp, $ix, $iy, $iz) / @inn3($K, $ix, $iy, $iz) * _dt,
            )
        ),
    )
end

macro av_xi_θr_dτ(ρCp, K, ix, iy, iz)
    return esc(:(max_lxyz / Vpdτ / @av_xi_Re($ρCp, $K, $ix, $iy, $iz) * Resc))
end

macro av_yi_θr_dτ(ρCp, K, ix, iy, iz)
    return esc(:(max_lxyz / Vpdτ / @av_yi_Re($ρCp, $K, $ix, $iy, $iz) * Resc))
end

macro av_zi_θr_dτ(ρCp, K, ix, iy, iz)
    return esc(:(max_lxyz / Vpdτ / @av_zi_Re($ρCp, $K, $ix, $iy, $iz) * Resc))
end

macro dτ_ρ(ρCp, K, ix, iy, iz)
    return esc(
        :(Vpdτ * max_lxyz / @inn3($K, $ix, $iy, $iz) / @Re($ρCp, $K, $ix, $iy, $iz) * Resc)
    )
end

@parallel_indices (ix, iy, iz) function compute_flux!(
    qTx, qTy, qTz, T, K, ρCp, Vpdτ, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz
)
    if (ix ≤ size(qTx, 1) && iy ≤ size(qTx, 2) && iz ≤ size(qTx, 3))
        qTx[ix, iy, iz] =
            (
                qTx[ix, iy, iz] * @av_xi_θr_dτ(ρCp, K, ix, iy, iz) -
                @av_xi_3(K, ix, iy, iz) *
                _dx *
                (T[ix + 1, iy + 1, iz + 1] - T[ix, iy + 1, iz + 1])
            ) / (1.0 + @av_xi_θr_dτ(ρCp, K, ix, iy, iz))
    end

    if (ix ≤ size(qTy, 1) && iy ≤ size(qTy, 2) && iz ≤ size(qTy, 3))
        qTy[ix, iy, iz] =
            (
                qTy[ix, iy, iz] * @av_yi_θr_dτ(ρCp, K, ix, iy, iz) -
                @av_yi_3(K, ix, iy, iz) *
                _dy *
                (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy, iz + 1])
            ) / (1.0 + @av_yi_θr_dτ(ρCp, K, ix, iy, iz))
    end

    if (ix ≤ size(qTz, 1) && iy ≤ size(qTz, 2) && iz ≤ size(qTz, 3))
        qTz[ix, iy, iz] =
            (
                qTz[ix, iy, iz] * @av_zi_θr_dτ(ρCp, K, ix, iy, iz) -
                @av_zi_3(K, ix, iy, iz) *
                _dz *
                (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz])
            ) / (1.0 + @av_zi_θr_dτ(ρCp, K, ix, iy, iz))
    end

    return nothing
end

@parallel_indices (ix, iy, iz) function compute_update!(
    T,
    Told,
    qTx,
    qTy,
    qTz,
    ρCp,
    K,
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
        # T[ix + 1, iy + 1, iz + 1] =
        #     (
        #         T[ix + 1, iy + 1, iz + 1] +
        #         @dτ_ρ(ix, iy, iz) * (
        #             ρCp[ix + 1, iy + 1, iz + 1] * _dt * Told[ix + 1, iy + 1, iz + 1] -
        #             (
        #                 _dx * (qTx[ix + 1, iy, iz] - qTx[ix, iy, iz]) +
        #                 _dy * (qTy[ix, iy + 1, iz] - qTy[ix, iy, iz]) +
        #                 _dz * (qTz[ix, iy, iz + 1] - qTz[ix, iy, iz])
        #             )
        #         )
        #     ) / (1.0 + _dt * ρCp[ix + 1, iy + 1, iz + 1] *  @dτ_ρ(ix, iy, iz))

        T[ix + 1, iy + 1, iz + 1] =
            T[ix + 1, iy + 1, iz + 1] +
            @dτ_ρ(ρCp, K, ix, iy, iz) * (
                -(
                    _dx * (qTx[ix + 1, iy, iz] - qTx[ix, iy, iz]) +
                    _dy * (qTy[ix, iy + 1, iz] - qTy[ix, iy, iz]) +
                    _dz * (qTz[ix, iy, iz + 1] - qTz[ix, iy, iz])
                ) -
                ρCp[ix + 1, iy + 1, iz + 1] *
                _dt *
                (T[ix + 1, iy + 1, iz + 1] - Told[ix + 1, iy + 1, iz + 1])
            )
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function compute_flux_res!(
    qTx2, qTy2, qTz2, T, K, _dx, _dy, _dz
)
    if (ix ≤ size(qTx2, 1) && iy ≤ size(qTx2, 2) && iz ≤ size(qTx2, 3))
        qTx2[ix, iy, iz] =
            -@av_xi_3(K, ix, iy, iz) *
            _dx *
            (T[ix + 1, iy + 1, iz + 1] - T[ix, iy + 1, iz + 1])
    end
    if (ix ≤ size(qTy2, 1) && iy ≤ size(qTy2, 2) && iz ≤ size(qTy2, 3))
        qTy2[ix, iy, iz] =
            -@av_yi_3(K, ix, iy, iz) *
            _dy *
            (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy, iz + 1])
    end
    if (ix ≤ size(qTz2, 1) && iy ≤ size(qTz2, 2) && iz ≤ size(qTz2, 3))
        qTz2[ix, iy, iz] =
            -@av_zi_3(K, ix, iy, iz) *
            _dz *
            (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz])
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function check_res!(
    ResT, T, Told, qTx, qTy, qTz, ρCp, _dt, _dx, _dy, _dz
)
    if (ix ≤ size(ResT, 1) && iy ≤ size(ResT, 2) && iz ≤ size(ResT, 3))
        ResT[ix, iy, iz] =
            -ρCp[ix + 1, iy + 1, iz + 1] *
            _dt *
            (T[ix + 1, iy + 1, iz + 1] - Told[ix + 1, iy + 1, iz + 1]) - (
                _dx * (qTx[ix + 1, iy, iz] - qTx[ix, iy, iz]) +
                _dy * (qTy[ix, iy + 1, iz] - qTy[ix, iy, iz]) +
                _dz * (qTz[ix, iy, iz + 1] - qTz[ix, iy, iz])
            )
    end
    return nothing
end

# @parallel function compute_flux!(
#     qTx, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, dx, dy, dz
# )
#     # @all(qTx) = (@all(qTx) * @all(θr_dτ) - @all(K) * @d_xi(T) / dx) / (1.0 + @all(θr_dτ))
#     # @all(qTy) = (@all(qTy) * @all(θr_dτ) - @all(K) * @d_yi(T) / dy) / (1.0 + @all(θr_dτ))
#     # @all(qTz) = (@all(qTz) * @all(θr_dτ) - @all(K) * @d_zi(T) / dz) / (1.0 + @all(θr_dτ))

#     # @all(qTx) = (- @all(K) * @d_xi(T) / dx) 
#     # @all(qTy) = (- @all(K) * @d_yi(T) / dy) 
#     # @all(qTz) = (- @all(K) * @d_zi(T) / dz) 

#     @all(qTx) = @all(qTx) * (1 - 1 / @all(θr_dτ))  - @all(K) * @d_xi(T) / dx / @all(θr_dτ)
#     @all(qTy) = @all(qTy) * (1 - 1 / @all(θr_dτ))  - @all(K) * @d_yi(T) / dy / @all(θr_dτ)
#     @all(qTz) = @all(qTz) * (1 - 1 / @all(θr_dτ))  - @all(K) * @d_zi(T) / dz / @all(θr_dτ)

#     @all(qTx2) = - @d_xi(T) / dx
#     @all(qTy2) = - @d_yi(T) / dy
#     @all(qTz2) = - @d_zi(T) / dz

#     return nothing
# end

# @parallel function compute_update!(T, Told, qTx, qTy, qTz, ρCp, dτ_ρ, dt, dx, dy, dz)
#     # @inn(T) =
#     #     (
#     #         @inn(T) +
#     #         @all(dτ_ρ) *
#     #         (@all(ρCp) * @inn(Told) / dt - (@d_xa(qTx) / dx + @d_ya(qTy) / dy + @d_za(qTz) / dz))
#     #     ) / (1.0 + @all(ρCp) * @all(dτ_ρ) / dt)

#     @inn(T) =
#             @inn(T) -
#             @all(ρCp) * @all(dτ_ρ) * (@inn(T) - @inn(Told)) / dt +
#             @all(dτ_ρ) * ( - (@d_xa(qTx) / dx + @d_ya(qTy) / dy + @d_za(qTz) / dz))
#     return nothing
# end

# @parallel function check_res!(ResT, T, Told, qTx2, qTy2, qTz2, ρCp, dt, dx, dy, dz)
#     @all(ResT) =
#         -@all(ρCp)*(@inn(T) - @inn(Told)) / dt -
#         (@d_xa(qTx2) / dx + @d_ya(qTy2) / dy + @d_za(qTz2) / dz)
#     return nothing
# end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs,
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,3}},
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
) where {_T,M<:AbstractArray{<:Any,3}}

    ## UNPACK
    @assert size(thermal.T) == ni

    _dt = 1 / dt
    _dx, _dy, _dz = @. 1 / di
    dx, dy, dz = di
    nx, ny, nz = ni
    size_innT_1, size_innT_2, size_innT_3 = nx - 2, ny - 2, nz - 2
    len_ResT_g =
        ((nx - 2 - 2) * igg.dims[1] + 2) *
        ((ny - 2 - 2) * igg.dims[2] + 2) *
        ((nz - 2 - 2) * igg.dims[3] + 2)

    # if first_solve
    #     @parallel assign!(thermal.Told, thermal.T)
    # end

    thermal_boundary_conditions!(thermal_bc, thermal.T)

    # Pseudo-transient iteration
    iter = 0
    err = 2 * pt_thermal.ϵ
    while err > pt_thermal.ϵ && iter < iterMax

        # @parallel compute_flux!(
        #     thermal.qTx,
        #     thermal.qTy,
        #     thermal.qTz,
        #     thermal.qTx2,
        #     thermal.qTy2,
        #     thermal.qTz2,
        #     thermal.T,
        #     thermal_parameters.K,
        #     θr_dτ,
        #     dx,
        #     dy,
        #     dz,
        # )

        # @hide_communication b_width begin # communication/computation overlap
        #     @parallel compute_update!(
        #         thermal.T,
        #         thermal.Told,
        #         thermal.qTx,
        #         thermal.qTy,
        #         thermal.qTz,
        #         thermal_parameters._ρCp,
        #         dτ_ρ,
        #         dt,
        #         dx,
        #         dy,
        #         dz,
        #     )
        #     update_halo!(thermal.T)
        # end

        # iter += 1
        # if iter % nout == 0
        #     @parallel check_res!(
        #         thermal.ResT,
        #         thermal.T,
        #         thermal.Told,
        #         thermal.qTx2,
        #         thermal.qTy2,
        #         thermal.qTz2,
        #         thermal_parameters._ρCp,
        #         dt,
        #         dx,
        #         dy,
        #         dz,
        #     )
        #     err = norm_mpi(thermal.ResT) / sqrt(len_ResT_g)

        #     if (igg.me == 0 && (verbose || err < ϵ || iter == iterMax))
        #         @printf("iter = %d, err = %1.3e \n", iter, err)
        #     end
        # end
        @hide_communication b_width begin
            @parallel (1:nx, 1:ny, 1:nz) compute_flux!(
                thermal.qTx,
                thermal.qTy,
                thermal.qTz,
                thermal.T,
                thermal_parameters.K,
                thermal_parameters._ρCp,
                pt_thermal.Vpdτ,
                pt_thermal.Resc,
                _dt,
                pt_thermal.max_lxyz,
                pt_thermal.max_lxyz2,
                _dx,
                _dy,
                _dz,
            )
        end

        @hide_communication b_width begin
            @parallel (1:nx, 1:ny, 1:nz) compute_flux_res!(
                thermal.qTx2,
                thermal.qTy2,
                thermal.qTz2,
                thermal.T,
                thermal_parameters.K,
                _dx,
                _dy,
                _dz,
            )
        end

        @hide_communication b_width begin # communication/computation overlap
            @parallel (1:nx, 1:ny, 1:nz) compute_update!(
                thermal.T,
                thermal.Told,
                thermal.qTx,
                thermal.qTy,
                thermal.qTz,
                thermal_parameters._ρCp,
                thermal_parameters.K,
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
            update_halo!(thermal.T)
        end
        thermal_boundary_conditions!(thermal_bc, thermal.T)

        iter += 1

        if iter % nout == 0
            @parallel (1:nx, 1:ny, 1:nz) check_res!(
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
                @printf("iter = %d, err = %1.3e \n", iter, err)
                # @printf("max Δt %1.3e \n", maximum(thermal.T.-thermal.Told))
                # @printf("min Δt %1.3e \n", minimum(thermal.T.-thermal.Told))
            end
        end
    end

    @show extrema(thermal.Told .- thermal.T)
    @parallel assign!(thermal.Told, thermal.T)

    if isnan(err)
        error("NaN")
    end

    return iter
end

end

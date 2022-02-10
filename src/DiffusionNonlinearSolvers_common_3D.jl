# This file is to be included after
# ParallelStencil is initialized with
# (hard-coded) parameters for the backend and numeric type,
# inside of a module with the appropriate name including
# those hard-coded parameters

# TODO BIG PROBLEM: @hide_communication is commentd out below because it caused combination to fail with an error that looks like it might relate to the gymnastics that parallelStencil is doing with its parse-time manipulation of modules.

export DiffusionNonlinearSolver
export solve!

using ImplicitGlobalGrid

# Relaxation scheme details, particular to this solver:
norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

macro innH3(ix,iy,iz)       esc(:( H[$ix+1,$iy+1,$iz+1]*H[$ix+1,$iy+1,$iz+1]*H[$ix+1,$iy+1,$iz+1] )) end
macro av_xi_H3(ix,iy,iz)    esc(:( 0.5*(H[$ix,$iy+1,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix,$iy+1,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix,$iy+1,$iz+1]+H[$ix+1,$iy+1,$iz+1]) )) end
macro av_yi_H3(ix,iy,iz)    esc(:( 0.5*(H[$ix+1,$iy,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy,$iz+1]+H[$ix+1,$iy+1,$iz+1]) )) end
macro av_zi_H3(ix,iy,iz)    esc(:( 0.5*(H[$ix+1,$iy+1,$iz]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy+1,$iz]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy+1,$iz]+H[$ix+1,$iy+1,$iz+1]) )) end
macro av_xi_Re(ix,iy,iz)    esc(:( π + sqrt(π*π + max_lxyz2 / @av_xi_H3($ix,$iy,$iz) * _dt) )) end
macro av_yi_Re(ix,iy,iz)    esc(:( π + sqrt(π*π + max_lxyz2 / @av_yi_H3($ix,$iy,$iz) * _dt) )) end
macro av_zi_Re(ix,iy,iz)    esc(:( π + sqrt(π*π + max_lxyz2 / @av_zi_H3($ix,$iy,$iz) * _dt) )) end
macro Re(ix,iy,iz)          esc(:( π + sqrt(π*π + max_lxyz2 / @innH3($ix,$iy,$iz)    * _dt) )) end
macro av_xi_θr_dτ(ix,iy,iz) esc(:( max_lxyz / Vpdτ / @av_xi_Re($ix,$iy,$iz) * Resc )) end
macro av_yi_θr_dτ(ix,iy,iz) esc(:( max_lxyz / Vpdτ / @av_yi_Re($ix,$iy,$iz) * Resc )) end
macro av_zi_θr_dτ(ix,iy,iz) esc(:( max_lxyz / Vpdτ / @av_zi_Re($ix,$iy,$iz) * Resc )) end
macro dτ_ρ(ix,iy,iz)        esc(:( Vpdτ * max_lxyz / @innH3($ix,$iy,$iz) / @Re($ix,$iy,$iz) * Resc )) end

@parallel_indices (ix,iy,iz) function compute_flux!(qHx, qHy, qHz, H, Vpdτ, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz)
    if (ix<=size(qHx,1) && iy<=size(qHx,2) && iz<=size(qHx,3)) qHx[ix,iy,iz]  = (qHx[ix,iy,iz] * @av_xi_θr_dτ(ix,iy,iz) - @av_xi_H3(ix,iy,iz) * _dx * (H[ix+1,iy+1,iz+1] - H[ix,iy+1,iz+1]) ) / (1.0 + @av_xi_θr_dτ(ix,iy,iz))  end
    if (ix<=size(qHy,1) && iy<=size(qHy,2) && iz<=size(qHy,3)) qHy[ix,iy,iz]  = (qHy[ix,iy,iz] * @av_yi_θr_dτ(ix,iy,iz) - @av_yi_H3(ix,iy,iz) * _dy * (H[ix+1,iy+1,iz+1] - H[ix+1,iy,iz+1]) ) / (1.0 + @av_yi_θr_dτ(ix,iy,iz))  end
    if (ix<=size(qHz,1) && iy<=size(qHz,2) && iz<=size(qHz,3)) qHz[ix,iy,iz]  = (qHz[ix,iy,iz] * @av_zi_θr_dτ(ix,iy,iz) - @av_zi_H3(ix,iy,iz) * _dz * (H[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz]) ) / (1.0 + @av_zi_θr_dτ(ix,iy,iz))  end
    return
end

@parallel_indices (ix,iy,iz) function compute_update!(H, Hold, qHx, qHy, qHz, Vpdτ, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz, size_innH_1, size_innH_2, size_innH_3)
    if (ix<=size_innH_1 && iy<=size_innH_2 && iz<=size_innH_3)  H[ix+1,iy+1,iz+1] = (H[ix+1,iy+1,iz+1] + @dτ_ρ(ix,iy,iz) * (_dt * Hold[ix+1,iy+1,iz+1] - (_dx * (qHx[ix+1,iy,iz] - qHx[ix,iy,iz]) + _dy * (qHy[ix,iy+1,iz] - qHy[ix,iy,iz])  + _dz * (qHz[ix,iy,iz+1] - qHz[ix,iy,iz])) )) / (1.0 + _dt * @dτ_ρ(ix,iy,iz))  end
    return
end

@parallel_indices (ix,iy,iz) function compute_flux_res!(qHx2, qHy2, qHz2, H, _dx, _dy, _dz)
    if (ix<=size(qHx2,1) && iy<=size(qHx2,2) && iz<=size(qHx2,3))  qHx2[ix,iy,iz] = -@av_xi_H3(ix,iy,iz) * _dx * (H[ix+1,iy+1,iz+1] - H[ix,iy+1,iz+1])  end
    if (ix<=size(qHy2,1) && iy<=size(qHy2,2) && iz<=size(qHy2,3))  qHy2[ix,iy,iz] = -@av_yi_H3(ix,iy,iz) * _dy * (H[ix+1,iy+1,iz+1] - H[ix+1,iy,iz+1])  end
    if (ix<=size(qHz2,1) && iy<=size(qHz2,2) && iz<=size(qHz2,3))  qHz2[ix,iy,iz] = -@av_zi_H3(ix,iy,iz) * _dz * (H[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz])  end
    return
end

@parallel_indices (ix,iy,iz) function check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, _dt, _dx, _dy, _dz)
    if (ix<=size(ResH,1) && iy<=size(ResH,2) && iz<=size(ResH,3))  ResH[ix,iy,iz] = -_dt * (H[ix+1,iy+1,iz+1] - Hold[ix+1,iy+1,iz+1]) - (_dx * (qHx[ix+1,iy,iz] - qHx[ix,iy,iz]) + _dy * (qHy[ix,iy+1,iz] - qHy[ix,iy,iz])  + _dz * (qHz[ix,iy,iz+1] - qHz[ix,iy,iz]))  end
    return
end

@parallel_indices (ix,iy,iz) function assign!(Hold, H)
    if (ix<=size(H,1) && iy<=size(H,2) && iz<=size(H,3))  Hold[ix,iy,iz] = H[ix,iy,iz]  end
    return
end


# This (immutable) struct collects data for a solve,
# so that the user can repeatedly call it to compute time steps. Later,
# this could also allow the user to perform an incomplete solve and
# and insert their own custom monitoring.
struct DiffusionNonlinearSolver
    nx
    ny
    nz
    CFL
    Resc
    tol     # tolerance
    itMax   # max number of iterations
    nout    # tol check
    me
    dims
    nprocs
    Vpdτ
    max_lxyz
    max_lxyz2
    _dx
    _dy
    _dz
    _dt
    qHx
    qHy
    qHz
    qHx2
    qHy2
    qHz2
    ResH
    Hold
    size_innH_1
    size_innH_2
    size_innH_3
    len_ResH_g
end

function DiffusionNonlinearSolver(
    nx, ny, nz,                # 1
    me, dims, nprocs,          # 2
    lx, ly, lz, dx, dy, dz, dt # 3
)
    # Derived numerics
    CFL = 1 / sqrt(3)
    Resc = 1 / 1.2
    tol = 1e-8
    itMax = 1e3
    nout = 2000
    Vpdτ = CFL * min(dx, dy, dz)
    max_lxyz = max(lx, ly, lz)
    max_lxyz2 = max_lxyz^2
    _dx, _dy, _dz, _dt = 1.0 / dx, 1.0 / dy, 1.0 / dz, 1.0 / dt
    # Array allocation
    qHx = @zeros(nx - 1, ny - 2, nz - 2)
    qHy = @zeros(nx - 2, ny - 1, nz - 2)
    qHz = @zeros(nx - 2, ny - 2, nz - 1)
    qHx2 = @zeros(nx - 1, ny - 2, nz - 2)
    qHy2 = @zeros(nx - 2, ny - 1, nz - 2)
    qHz2 = @zeros(nx - 2, ny - 2, nz - 1)

    size_innH_1, size_innH_2, size_innH_3 = nx - 2, ny - 2, nz - 2
    ResH = @zeros(size_innH_1, size_innH_2, size_innH_3)
    Hold = @zeros(nx, ny, nz)
    len_ResH_g = ((nx - 2 - 2) * dims[1] + 2) * ((ny - 2 - 2) * dims[2] + 2) * ((nz - 2 - 2) * dims[3] + 2)
    return DiffusionNonlinearSolver(
        nx, ny, nz,
        CFL,
        Resc,
        tol,
        itMax,
        nout,
        me,
        dims,
        nprocs,
        Vpdτ,
        max_lxyz,
        max_lxyz2,
        _dx, _dy, _dz, _dt,
        qHx, qHy, qHz, qHx2, qHy2, qHz2,
        ResH,
        Hold,
        size_innH_1, size_innH_2, size_innH_3,
        len_ResH_g
    )
end


function solve!(solver::DiffusionNonlinearSolver, H::Data.Array, first_solve = false)
    @assert size(H) == (solver.nx, solver.ny, solver.nz)

    if first_solve
        solver.Hold .= @ones(size(H)...) .* H
    end

    # Pseudo-transient iteration
    iter = 0;
    err = 2*solver.tol
    b_width = (16, 4, 4)   # boundary width for comm/comp overlap
    while err>solver.tol && iter<solver.itMax
        @parallel compute_flux!(solver.qHx, solver.qHy, solver.qHz, H, solver.Vpdτ, solver.Resc, solver._dt, solver.max_lxyz, solver.max_lxyz2, solver._dx, solver._dy, solver._dz)
        #@hide_communication b_width begin # communication/computation overlap
            @parallel compute_update!(H, solver.Hold, solver.qHx, solver.qHy, solver.qHz, solver.Vpdτ, solver.Resc, solver._dt, solver.max_lxyz, solver.max_lxyz2, solver._dx, solver._dy, solver._dz, solver.size_innH_1, solver.size_innH_2, solver.size_innH_3)
            update_halo!(H)
        #end
        iter += 1;
        if iter % solver.nout == 0
            @parallel compute_flux_res!(solver.qHx2, solver.qHy2, solver.qHz2, H, solver._dx, solver._dy, solver._dz)
            @parallel check_res!(solver.ResH, H, solver.Hold, solver.qHx2, solver.qHy2, solver.qHz2, solver._dt, solver._dx, solver._dy, solver._dz)
            err = norm_g(solver.ResH) / sqrt(solver.len_ResH_g)
        end
    end
    @parallel assign!(solver.Hold, H)
    if isnan(err) error("NaN") end

    return iter
end

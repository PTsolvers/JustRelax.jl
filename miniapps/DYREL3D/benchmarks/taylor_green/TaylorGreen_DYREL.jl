using Statistics: mean

# FVCA8 benchmark for the Stokes and Navier-Stokes
#     equations with the TrioCFD code – benchmark session
#     P.-E. Angeli, M.-A. Puscas, G. Fauchet, A. Cartalade
# HAL Id: cea-02434556
# https://hal-cea.archives-ouvertes.fr/cea-02434556

include(joinpath(@__DIR__, "viz_TaylorGreen_DYREL.jl"))

@parallel_indices (i, j, k) function init_single_phase!(ratios)
    @index ratios[1, i, j, k] = 1.0
    return nothing
end

function body_forces(xi::NTuple{3, T}) where {T}
    xx, yy, zz = xi
    x = PTArray(backend)([x for x in xx, y in yy, z in zz])
    y = PTArray(backend)([y for x in xx, y in yy, z in zz])
    z = PTArray(backend)([z for x in xx, y in yy, z in zz])

    fz, fy = @zeros(size(x)...), @zeros(size(x)...)
    # the momentum residual is ∇·τ - ∇P - ρg, so ρg is minus the right-hand side of
    # -∇·τ + ∇P = f
    fx = @. 36 * π^2 * cos(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)

    return fx, fy, fz
end

function velocity!(stokes, xci, xvi)
    xv, yv, zv = xvi
    di = ntuple(i -> xci[i][2] - xci[i][1], Val(3))
    xc, yc, zc = ntuple(
        i -> LinRange(xci[i][1] - di[i], xci[i][end] + di[i], length(xci[i]) + 2), Val(3)
    )
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    _velocity_x(x, y, z) = -2cos(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)
    _velocity_y(x, y, z) = sin(2 * π * x) * cos(2 * π * y) * sin(2 * π * z)
    _velocity_z(x, y, z) = sin(2 * π * x) * sin(2 * π * y) * cos(2 * π * z)

    @parallel_indices (i, j, k) function _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
        T = eltype(Vx)
        if all((i, j, k) .≤ size(Vx))
            if (i == size(Vx, 1)) ||
                    (j == size(Vx, 2)) ||
                    (k == size(Vx, 3)) ||
                    (i == 1) ||
                    (j == 1) ||
                    (k == 1)
                Vx[i, j, k] = _velocity_x(xv[i], yc[j], zc[k])
            else
                Vx[i, j, k] = zero(T)
            end
        end
        if all((i, j, k) .≤ size(Vy))
            if (i == size(Vy, 1)) ||
                    (j == size(Vy, 2)) ||
                    (k == size(Vy, 3)) ||
                    (i == 1) ||
                    (j == 1) ||
                    (k == 1)
                Vy[i, j, k] = _velocity_y(xc[i], yv[j], zc[k])
            else
                Vy[i, j, k] = zero(T)
            end
        end
        if all((i, j, k) .≤ size(Vz))
            if (i == size(Vz, 1)) ||
                    (j == size(Vz, 2)) ||
                    (k == size(Vz, 3)) ||
                    (i == 1) ||
                    (j == 1) ||
                    (k == 1)
                Vz[i, j, k] = _velocity_z(xc[i], yc[j], zv[k])
            else
                Vz[i, j, k] = zero(T)
            end
        end

        return nothing
    end

    return @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
end

function taylorGreen(;
        nx = 16, ny = 16, nz = 16, init_MPI = true, finalize_MPI = false, verbose = true
    )
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    lx = ly = lz = 1.0e0
    li = (lx, ly, lz)  # domain length in x- and y-
    origin = 0.0, 0.0, 0.0
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...) # init MPI
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1    # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(backend, ni)
    ## Setup-specific parameters and fields
    (; η) = stokes.viscosity
    η .= 1.0
    ρg = body_forces(xci) # => ρ*(gx, gy, gz)
    dt = Inf
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    for ratios in (
            phase_ratios.center, phase_ratios.vertex,
            phase_ratios.yz, phase_ratios.xz, phase_ratios.xy,
        )
        @parallel (@idx size(ratios)) init_single_phase!(ratios)
    end
    args = (; T = 0.0, P = stokes.P, dt = dt)

    ## Boundary conditions
    # the velocity is prescribed on every face, so no face is free_slip nor no_slip and
    # `flow_bcs!` leaves the boundary values written below untouched
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (
            left = false, right = false,
            top = false, bot = false,
            back = false, front = false,
        ),

        no_slip = (
            left = false, right = false,
            top = false, bot = false,
            back = false, front = false,
        ),
    )
    # impose analytical velocity at the boundaries of the domain
    velocity!(stokes, xci, xvi)
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(
        backend, stokes, rheology, phase_ratios, grid.di, dt;
        ϵ = 1.0e-8, CFL = 0.99, γfact = 20.0,
    )

    # Physical time loop
    t = 0.0

    local iters
    while t < ttot
        iters = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                iterMax = 50.0e3,
                total_iterMax = 50.0e3,
                nout = 20,
                rel_drop = 0.1,
                b_width = (4, 4, 4),
                verbose_PH = verbose,
                verbose_DR = false,
            )
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

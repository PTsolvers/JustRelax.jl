# using CUDA
using JustRelax, JustRelax.JustRelax2D
const backend_JR = CPUBackend

using JustPIC, JustPIC._2D
const backend = JustPIC.CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

# Load script dependencies
using LinearAlgebra, GeoParams, CairoMakie#GLMakie

## START OF HELPER FUNCTION ----------------------------------------------------------
function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end
    @parallel f_x(A, B)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel_indices (i, j) function init_P!(P, ρg, z)
    P[i, j] = sum(abs(ρg[i, jj] * z[jj]) for jj in j:size(P, 2))
    return nothing
end

function init_phases!(phases, particles, A)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, A)

        f(x, A, λ) = A * sin(π * x / λ)

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 2.0

            if 0e0 ≤ depth ≤ 100e3
                @index phases[ip, i, j] = 1.0

            elseif depth > (-f(x, A, 500e3) + (200e3 - A))
                @index phases[ip, i, j] = 3.0

            end

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function RT_2D(igg, nx, ny)

    # Physical domain ------------------------------------
    thick_air    = 100e3             # thickness of sticky air layer
    ly           = 500e3 + thick_air # domain length in y
    lx           = 500e3             # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = rheology = (
        # Name              = "Air",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=1e0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e16),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Crust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ=3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e20),)),
            Gravity           = ConstantGravity(; g=9.81),
        )
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 10
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    A             = 5e3    # Amplitude of the anomaly
    phase_ratios  = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, A)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 1 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend_JR, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg               = @zeros(ni...), @zeros(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left =  true, right =  true, top =  true, bot = false),
        no_slip      = (left = false, right = false, top = false, bot =  true),
        free_surface = true,
    )

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    figdir = "RayleighTaylor2D"
    take(figdir)

    # Time loop
    t, it   = 0.0, 0
    dt      = 1e3 * (3600 * 24 * 365.25)
    dt_max  = 50e3 * (3600 * 24 * 365.25)
    while it < 500

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                iterMax              =  50e3,
                iterMin              =   1e3,
                viscosity_relaxation =  1e-2,
                nout                 =   5e3,
                free_surface         =  true,
                viscosity_cutoff     = (-Inf, Inf)
            )
        )
        dt = if it ≤ 10
            min(compute_dt(stokes, di),  1e3 * (3600 * 24 * 365.25))
        elseif 10 < it ≤ 20
            min(compute_dt(stokes, di), 10e3 * (3600 * 24 * 365.25))
        elseif 20 < it ≤ 30
            min(compute_dt(stokes, di), 25e3 * (3600 * 24 * 365.25))
        else
            min(compute_dt(stokes, di), dt_max)
        end
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t        += dt

        if it == 1 || rem(it, 5) == 0
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 2

            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]

            fig = Figure(size = (900, 900), title = "t = $t")
            ax  = Axis(fig[1,1], aspect = 1, title = " t=$(round.(t/(1e3 * 3600 * 24 *365.25); digits=3)) Kyrs")
            scatter!(
                ax,
                pxv, pyv,
                color=clr,
                colormap = :lajolla,
                markersize = 3
            )
            arrows!(
                ax,
                xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))...,
                lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
                color = :darkblue,
            )
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end

    end
    return
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n        = 100
nx       = n
ny       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

RT_2D(igg, nx, ny)

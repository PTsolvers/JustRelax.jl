# using CUDA
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

using JustPIC, JustPIC._2D
const backend = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays

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
        
        f(x, A, λ) = A * sin(π*x/λ)
        
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            @cell(index[ip, i, j]) == 0 && continue

            x = @cell px[ip, i, j]
            depth = -(@cell py[ip, i, j]) 
            @cell phases[ip, i, j] = 2.0
            
            if 0e0 ≤ depth ≤ 100e3
                @cell phases[ip, i, j] = 1.0

            elseif depth > (-f(x, A, 500e3) + (200e3 - A))
                @cell phases[ip, i, j] = 3.0          
            end

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
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
        backend, nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly 
    A             = 5e3    # Amplitude of the anomaly
    init_phases!(pPhases, particles, A)
    phase_ratios  = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-8,  CFL = 0.95 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    # ----------------------------------------------------
   
    # Buoyancy forces & rheology
    ρg               = @zeros(ni...), @zeros(ni...)
    η                = @ones(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    @parallel (@idx ni) compute_viscosity!(
        η, 0.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e19, 1e24)
    )
    η_vep            = copy(η)

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(; 
        free_slip    = (left =  true, right =  true, top =  true, bot = false),
        no_slip      = (left = false, right = false, top = false, bot =  true),
        free_surface = true,
    )

    # Plot initial T and η profiles
    let
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(ρg[2][:]./9.81), Y./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        fig
    end

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    figdir = "RayleighTaylor2D"
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    dt = 1e3 * (3600 * 24 * 365.25)
    dt_max = 50e3 * (3600 * 24 * 365.25)
    while it < 500 # run only for 5 Myrs

        # Stokes solver ----------------
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
        )
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            iterMax              = 150e3,
            iterMin              =   5e3,
            viscosity_relaxation =  1e-2,
            nout                 =   5e3,
            free_surface         =  true,
            viscosity_cutoff     = (-Inf, Inf)
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
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)        
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        # @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
    
        @show it += 1
        t        += dt

        if it == 1 || rem(it, 5) == 0
            JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 2

            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]

            fig = Figure(size = (900, 900), title = "t = $t")
            ax  = Axis(fig[1,1], aspect = 1, title = " t=$(t/(1e3 * 3600 * 24 *365.25)) Kyrs")
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

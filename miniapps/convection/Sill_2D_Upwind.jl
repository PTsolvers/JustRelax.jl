using JustRelax
using CUDA
# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, SpecialFunctions
using CairoMakie

# Load file with all the rheology configurations
include("./Particles2D/Sill_rheology.jl")

# --------------------------------------------------------------------------------
# Velocity helper grids for the particle advection
function velocity_grids(xci, xvi, di)
    dx, dy  = di
    yVx     = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy     = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

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

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, x)
    depth = y[j]

    # (depth - 15e3) because we have 15km of sticky air
   # if 0e0 ≤ (depth) < 0.26e3
        T[i + 1 , j] = 273e0 + 600e0
   # end

    if (-0.175e3 < depth ≤  -0.075e3)
        T[i + 1, j] = 273e0 + 1200e0
    end

    if  (-0.15e3 < depth ≤  -0.12e3 ) && (200 < x[i] ≤  250)
        T[i + 1, j] = 273e0 + 1500e0
    end


    return nothing
end

 @parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, args)
    ϕ[i, j] = compute_meltfraction(rheology, ntuple_idx(args, i, j))
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
args_ijk = ntuple_idx(args, I...)
ϕ[I...] = compute_melt_frac(rheology, args_ijk, phase_ratios[I...])
return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return GeoParams.compute_meltfraction_ratio(phase_ratios, rheology, args)
end

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
# function main2D(; ar=8, ny=16, nx=ny*8, figdir="figs2D")

    # initialize MPI
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = JustRelax.MPI.Initialized() ? false : true)...)

    # Physical domain ------------------------------------
    ly           = 0.25e3            # domain length in y
    lx           = 0.5e3             # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0,-ly             # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------
    rheology     = init_rheologies(; is_plastic = false)
    κ            = (4 / (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 / 100 # diffusive CFL timestep limiter
   # dt = dt_diff/10
    @show dt
    # phases = @zeros(ni...)
    # init_phases!(phases, xci)
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, xci)

   # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.25 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2], xvi[1])
    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    for _ in 1:2
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    # Rheology
    η               = @ones(ni...)
    depth           = PTArray([y for x in xci[1], y in xci[2]])
    args            = (; T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf)
    viscosity_cutoff = -Inf, Inf
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, viscosity_cutoff
    )
    η_vep           = deepcopy(η)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip   = (left = true, right=true, top=true, bot=true),
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy)
    # ----------------------------------------------------

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # Time loop
    t, it = 0.0, 0
    local iters
    # while it < 150

        # Update buoyancy and viscosity
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
        )
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
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
            Inf,
            igg;
            iterMax = 150e3,
            nout=1e3,
            viscosity_cutoff= viscosity_cutoff
        );
        dt = compute_dt(stokes, di, dt_diff, igg)
        # ------------------------------
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        # if it<2
        #     dt = dt/10000
        # end

        # Thermal solver ---------------
        args_T = (; P=stokes.P)
        solve!(
            thermal,
            thermal_bc,
            stokes,
            rheology,
            args_T,
            di,
            dt
        )
        # ------------------------------

        it += 1
        t += dt

        println("\n")
        println("Time step number $it")
        println("   time = $(t/(1e6 * 3600 * 24 *365.25)) Myrs, dt = $(dt/(1e6 * 3600 * 24 *365.25)) Myrs")
        println("\n")

        # Plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            fig = Figure(size = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig[4,1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.τ.II.*1e-6) , colormap=:batlow)
            h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1, height=100)
            Colorbar(fig[2,2], h2, height=100)
            Colorbar(fig[3,2], h3, height=100)
            Colorbar(fig[4,2], h4, height=100)
            save( joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

function run()
    figdir = "Sill_2D_Upwind"
    ar     = 1 # aspect ratio
    n      = 128
    nx     = n*ar - 2
    ny     = n - 2

    main2D(; figdir=figdir, ar=ar,nx=nx, ny=ny);
end

run()

using CUDA
CUDA.allowscalar(false) # for safety

using JustRelax, JustRelax.DataIO
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
using JustPIC
using JustPIC._2D

# setup ParallelStencil.jl environment
model = PS_Setup(:CUDA, Float64, 2)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, CellArrays

# Load file with all the rheology configurations
include("Layered_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...)
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))


# Velocity helper grids for the particle advection
function velocity_grids(xci, xvi, di)
    dx, dy  = di
    yVx     = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy     = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
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
@parallel_indices (i, j) function init_T!(T, z)
    depth = -z[j]

    # (depth - 15e3) because we have 15km of sticky air
    if depth < 0e0
        T[i + 1, j] = 273e0

    elseif 0e0 ≤ (depth) < 35e3
        dTdZ        = (923-273)/35e3
        offset      = 273e0
        T[i + 1, j] = (depth) * dTdZ + offset

    elseif 110e3 > (depth) ≥ 35e3
        dTdZ        = (1492-923)/75e3
        offset      = 923
        T[i + 1, j] = (depth - 35e3) * dTdZ + offset

    elseif (depth) ≥ 110e3
        dTdZ        = (1837 - 1492)/590e3
        offset      = 1492e0
        T[i + 1, j] = (depth - 110e3) * dTdZ + offset

    end

    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if ((x[i]-xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            depth       = abs(y[j])
            dTdZ        = (2047 - 2017) / 50e3
            offset      = 2017
            T[i + 1, j] = (depth - 585e3) * dTdZ + offset
        end
        return nothing
    end
    ni = length.(xvi)
    @parallel (@idx ni) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D", save_vtk =false)

    # Physical domain ------------------------------------
    ly           = 700e3            # domain length in y
    lx           = ly * ar          # domain length in x
    ni           = nx, ny           # number of cells
    li           = lx, ly           # domain length in x- and y-
    di           = @. li / ni       # grid step in x- and -y
    origin       = 0.0, -ly         # origin coordinates (15km f sticky air layer)
    xci, xvi     = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_plastic = true)
    κ            = (10 / (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Weno model -----------------------------------------
    weno = WENO5(ni=(nx,ny).+1, method=Val{2}()) # ni.+1 for Temp
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_particle_fields_cellarrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly       = lx/2   # origin of thermal anomaly
    yc_anomaly       = -610e3 # origin of thermal anomaly
    r_anomaly        = 25e3   # radius of perturbation
    init_phases!(pPhases, particles, lx; d=abs(yc_anomaly), r=r_anomaly)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.1 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2])
    thermal_bcs!(thermal.T, thermal_bc)

    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi)
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end
    # Rheology
    η                = @zeros(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e16, 1e24)
    )
    η_vep            = copy(η)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if save_vtk
        vtk_dir      = figdir*"\\vtk"
        take(vtk_dir)
    end
    take(figdir)
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

    # WENO arrays
    T_WENO  = @zeros(ni.+1)
    Vx_v    = @zeros(ni.+1...)
    Vy_v    = @zeros(ni.+1...)
    # Time loop
    t, it = 0.0, 0
    while (t/(1e6 * 3600 * 24 *365.25)) < 5 # run only for 5 Myrs

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e18, 1e24)
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
            iterMax=50e3,
            nout=1e3,
            viscosity_cutoff=(1e18, 1e24)
        )
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        dt   = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Weno advection
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            igg     = igg,
            phase   = phase_ratios,
            iterMax = 10e3,
            nout    = 1e2,
            verbose = true,
        )
        T_WENO .= thermal.T[2:end-1, :]
        JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        thermal.T[2:end-1, :] .= T_WENO
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, tuple(), tuple(), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            # Make Makie figure
            fig = Figure(size = (900, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1])
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(T_WENO) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------


# (Path)/folder where output data and figures are stored
figdir   = "Weno2D"
ar       = 1 # aspect ratio
n        = 256
nx       = n*ar - 2
ny       = n - 2
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 0; init_MPI= true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny);

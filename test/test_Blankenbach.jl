push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil,  ParallelStencil.FiniteDifferences2D

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

import JustRelax.@cell

# Load script dependencies
using Printf, LinearAlgebra, CellArrays

# Load file with all the rheology configurations
include("../miniapps/benchmarks/stokes2D/Blankenbach2D/Blankenbach_Rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y)
    depth = -y[j]

    dTdZ        = (1273-273)/1000e3
    offset      = 273e0
    T[i, j]     = (depth) * dTdZ + offset
    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)
    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if ((x[i]-xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            T[i, j] += 20.0
        end
        return nothing
    end
    ni = size(T)
    @parallel (@idx ni) _rectangular_perturbation!(T, xc, yc, r, xvi...)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=1, nx=32, ny=32, nit = 10)

    # Physical domain ------------------------------------
    ly           = 1000e3               # domain length in y
    lx           = ly                   # domain length in x
    ni           = nx, ny               # number of cells
    li           = lx, ly               # domain length in x- and y-
    di           = @. li / ni           # grid step in x- and -y
    origin       = 0.0, -ly             # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies()
    κ            = (rheology[1].Conductivity[1].k / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.9 * min(di...)^2 / κ / 4.0 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 36, 12
    particles           = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays      = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy    = velocity_grids(xci, xvi, di)
    # temperature
    pT, pT0, pPhases    = init_cell_arrays(particles, Val(3))
    particle_args       = (pT, pT0, pPhases)
    phase_ratios        = PhaseRatio(backend_JR, ni, length(rheology))
    init_phases!(pPhases, particles)
    phase_ratios_center!(phase_ratios, particles, grid, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend_JR, ni)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 1 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal         = ThermalArrays(backend_JR, ni)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])
    # Elliptical temperature anomaly
    xc_anomaly      = 0.0    # origin of thermal anomaly
    yc_anomaly      = -600e3  # origin of thermal anomaly
    r_anomaly       = 100e3    # radius of perturbation
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi)
    thermal_bcs!(thermal, thermal_bc)
    thermal.Told    .= thermal.T
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Rayleigh number
    ΔT = thermal.T[1,1] - thermal.T[1,end]
    Ra = (rheology[1].Density[1].ρ0 * rheology[1].Gravity[1].g * rheology[1].Density[1].α * ΔT * ly^3.0 ) /
       (κ * rheology[1].CompositeRheology[1].elements[1].η )
    @show Ra

    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)

    # Buoyancy forces  & viscosity ----------------------
    ρg               = @zeros(ni...), @zeros(ni...)
    η                = @ones(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )

    # PT coefficients for thermal diffusion -------------
    pt_thermal       = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL = 0.5 / √2.1
    )

    # Boundary conditions -------------------------------
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    pT0.data    .= pT.data

    local Vx_v, Vy_v, iters
    # Time loop
    t, it   =   0.0, 1
    Urms    =   Float64[]
    Nu_top  =   Float64[]
    trms    =   Float64[]

    # Buffer arrays to compute velocity rms
    Vx_v    =   @zeros(ni.+1...)
    Vy_v    =   @zeros(ni.+1...)

    while it ≤ nit
        @show it

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            Inf,
            igg;
            kwargs = (;
                iterMax          = 150e3,
                nout             = 200,
                viscosity_cutoff = (-Inf, Inf),
                verbose          = true
            )
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 1e2,
                verbose = true,
            )
        )
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:end-1, :], subgrid_arrays, particles, xvi,  di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        # update phase ratios
        phase_ratios_center!(phase_ratios, particles, grid, pPhases)

        # Nusselt number, Nu = H/ΔT/L ∫ ∂T/∂z dx ----
        Nu_it   =   (ly / (1000.0*lx)) *
            sum( ((abs.(thermal.T[2:end-1,end] - thermal.T[2:end-1,end-1])) ./ di[2]) .*di[1])
        push!(Nu_top, Nu_it)
        # -------------------------------------------

        # Compute U rms -----------------------------
        # U₍ᵣₘₛ₎ = H*ρ₀*c₍ₚ₎/k * √ 1/H/L * ∫∫ (vx²+vz²) dx dz
        Urms_it = let
            JustRelax.JustRelax2D.velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=true)
            @. Vx_v .= hypot.(Vx_v, Vy_v) # we reuse Vx_v to store the velocity magnitude
            sqrt( sum( Vx_v.^2 .* prod(di)) / lx /ly ) *
                ((ly * rheology[1].Density[1].ρ0 * rheology[1].HeatCapacity[1].Cp) / rheology[1].Conductivity[1].k )
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # -------------------------------------------

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= 273.0
        @views T_buffer[:, 1]        .= 1273.0
        @views thermal.T[2:end-1, :] .= T_buffer
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        temperature2center!(thermal)

        it      +=  1
        t       +=  dt
        # ------------------------------
    end

    # Horizontally averaged depth profile
    Tmean   =   @zeros(ny+1)
    Emean   =   @zeros(ny)

    @show Urms[Int64(nit)] Nu_top[Int64(nit)]

    return Urms, Nu_top, iters
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

@testset "Blankenbach 2D" begin
    @suppress begin

        nx, ny   = 32, 32           # number of cells
        igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
            IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
        else
            igg
        end


        Urms, Nu_top, iters = main2D(igg; nx = nx, ny = ny);
        @test Urms[end] ≈ 0.33 rtol=1e-1
        @test Nu_top[end] ≈ 1.0312 rtol=1e-2
        @test iters.err_evo1[end] < 1e-4
    end
end

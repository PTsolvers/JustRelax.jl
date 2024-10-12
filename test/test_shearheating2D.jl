push!(LOAD_PATH, "..")
using Test, Suppressor

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

# Benchmark of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

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

# Load script dependencies
using GeoParams

# Load file with all the rheology configurations
include("../miniapps/benchmarks/stokes2D/shear_heating/Shearheating_rheology.jl")

# ## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
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
    return esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function Shearheating2D(igg; nx=32, ny=32)

    # Physical domain ------------------------------------
    ly           = 40e3              # domain length in y
    lx           = 70e3           # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_TP_Conductivity=false)
    κ            = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 50, 12
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly       = lx / 2 # origin of thermal anomaly
    yc_anomaly       = 40e3   # origin of thermal anomaly
    r_anomaly        = 3e3    # radius of perturbation
    phase_ratios     = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, xc_anomaly, yc_anomaly, r_anomaly)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend_JR, ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )

    # Initialize constant temperature
    @views thermal.T .= 273.0 + 400
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])

    # Rheology
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL= 1 / √2.1
    )

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    ## Compression and not extension - fix this
    εbg              = 5e-14
    stokes.V.Vx     .= PTArray(backend_JR)([ -(x - lx/2) * εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy     .= PTArray(backend_JR)([ (ly - abs(y)) * εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    # Time loop
    t, it = 0.0, 0
    local iters, thermal
    while it < 10

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
            dt,
            igg;
            kwargs = (;
                iterMax = 75e3,
                nout=1e3,
                viscosity_cutoff=(-Inf, Inf)
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

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
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # interpolate fields from grid vertices to particles
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= 273.0 + 400
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)
          
        @show it += 1
        t        += dt

    end

    # finalize_global_grid(; finalize_MPI=true)

    return iters, thermal
end

@testset "Shearheating2D" begin
    @suppress begin
        n = 32
        nx = n
        ny = n
        igg = if !(JustRelax.MPI.Initialized())
            IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
        else
            igg
        end

        # Initialize iters and thermal to ensure they are defined
        iters = nothing
        thermal = nothing

        try
            iters, thermal = Shearheating2D(igg; nx=nx, ny=ny)
        catch e
            @warn e
            try
                iters, thermal = Shearheating2D(igg; nx=nx, ny=ny)
            catch e2
                @warn e2
            end
        end

        # Ensure iters is defined before running the test
        @test iters != nothing && iters.err_evo1[end] < 1e-4
        @test any(x -> x < 0, thermal.shear_heating) == false
    end
end

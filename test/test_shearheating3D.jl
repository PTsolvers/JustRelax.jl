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
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end


const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end


using JustPIC
using JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end
import JustRelax.@cell

# Load script dependencies
using Printf, GeoParams

# Load file with all the rheology configurations
include("../miniapps/benchmarks/stokes3D/shear_heating/Shearheating_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    esc(:($A[$idx_k]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

nx,ny,nz = 32,32,32

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function Shearheating3D(nx=16, ny=16, nz=16)

    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg      = IGG(init_global_grid(nx, ny, nz; init_MPI = init_mpi)...)

    # Physical domain ------------------------------------
    lx           = 70e3           # domain length in x
    ly           = 70e3           # domain length in y
    lz           = 40e3              # domain length in y
    ni           = nx, ny, nz            # number of cells
    li           = lx, ly, lz            # domain length in x- and y-
    di           = @. li / (nx_g(),ny_g(),nz_g())        # grid step in x- and -y
    origin       = 0.0, 0.0, -lz          # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
     # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_TP_Conductivity=false)
    κ            = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    dt = dt_diff = 0.5 * min(di...)^3 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 10
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    # velocity grids
    grid_vx, grid_vy, grid_vz   = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly       = lx/2   # origin of thermal anomaly
    yc_anomaly       = ly/2   # origin of thermal anomaly
    zc_anomaly       = 40e3  # origin of thermal anomaly
    r_anomaly        = 3e3    # radius of perturbation
    phase_ratios     = PhaseRatio(backend_JR, ni, length(rheology))
    init_phases!(pPhases, particles, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly)
    phase_ratios_center!(phase_ratios, particles, grid, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal         = ThermalArrays(backend_JR, ni)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true , right = true , top = false, bot = false, front = true , back = true),
    )

    # Initialize constant temperature
    @views thermal.T .= 273.0 + 400
    thermal_bcs!(thermal.T, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[3], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[3], xci[3])

    # Rheology
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=5e-2 / √3
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    ## Compression and not extension - fix this
    εbg              = 5e-14
    stokes.V.Vx .= PTArray(backend_JR)([ -(x - lx/2) * εbg for x in xvi[1], _ in 1:ny+2, _ in 1:nz+2])
    stokes.V.Vy .= PTArray(backend_JR)([ -(y - ly/2) * εbg for _ in 1:nx+2, y in xvi[2], _ in 1:nz+2])
    stokes.V.Vz .= PTArray(backend_JR)([  (lz - abs(z)) * εbg for _ in 1:nx+2, _ in 1:ny+2, z in xvi[3]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)

    grid2particle!(pT, xvi, thermal.T, particles)

    # Time loop
    t, it = 0.0, 0
    local iters
    while it < 5

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
                kwargs = (
                    iterMax = 100e3,
                    nout=1e3,
                    viscosity_cutoff=(-Inf, Inf),
                    verbose=false,
                )
            )
            tensor_invariant!(stokes.ε)
            dt   = compute_dt(stokes, di, dt_diff)
            # ------------------------------

            # interpolate fields from particle to grid vertices
            particle2grid!(thermal.T, pT, xvi, particles)
            temperature2center!(thermal)

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
                kwargs =(
                    igg     = igg,
                    phase   = phase_ratios,
                    iterMax = 10e3,
                    nout    = 1e2,
                    verbose = false,
                )
            )
            # ------------------------------

            # Advection --------------------
            # advect particles in space
            advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy, grid_vz), dt)
            # advect particles in memory
            move_particles!(particles, xvi, particle_args)
            # interpolate fields from grid vertices to particles
            grid2particle_flip!(pT, xvi, thermal.T, thermal.Told, particles)
            # check if we need to inject particles
            inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)
            # update phase ratios
            phase_ratios_center!(phase_ratios, particles, grid, pPhases)

            @show it += 1
            t        += dt
            # ------------------------------
      end

    finalize_global_grid(; finalize_MPI = true)

    return iters, thermal
end

@testset "Shearheating3D" begin
    @suppress begin
        iters, thermal = Shearheating3D()
        @test passed = iters.err_evo1[end] < 1e-4
    # @test maximum.(thermal.shear_heating)
    end
end

push!(LOAD_PATH, "..")
using Test, Suppressor

# Benchmark of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
using JustRelax, JustRelax.DataIO
import JustRelax.@cell
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)  #or (CUDA, Float64,  3) or (AMDGPU, Float64, 3)

using JustPIC
using JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)  #or (:CUDA, Float64, 3) or (:AMDGPU, Float64, 3)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CellArrays

# Load file with all the rheology configurations
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

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function Shearheating3D(nx=16, ny=16, nz=16)

    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg    = IGG(init_global_grid(nx, ny, nz; init_MPI = init_mpi)...)
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
    nxcell, max_xcell, min_xcell = 20, 40, 1
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
    init_phases!(pPhases, particles, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal         = ThermalArrays(ni)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true , right = true , top = false, bot = false, front = true , back = true),
    )

    # Initialize constant temperature
    @views thermal.T .= 273.0 + 400
    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(3))

    @parallel (JustRelax.@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[3], xci[3])

    # Rheology
    η                = @ones(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )
    η_vep            = deepcopy(η)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √3
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    ## Compression and not extension - fix this
    εbg              = 5e-14
    stokes.V.Vx .= PTArray([ -(x - lx/2) * εbg for x in xvi[1], _ in 1:ny+2, _ in 1:nz+2])
    stokes.V.Vy .= PTArray([ -(y - ly/2) * εbg for _ in 1:nx+2, y in xvi[2], _ in 1:nz+2])
    stokes.V.Vz .= PTArray([  (lz - abs(z)) * εbg for _ in 1:nx+2, _ in 1:ny+2, z in xvi[3]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)

    grid2particle!(pT, xvi, thermal.T, particles)

    # Time loop
    t, it = 0.0, 0
    local iters
    while it < 5
            # Update buoyancy and viscosity -
            args = (; T = thermal.Tc, P = stokes.P,  dt = Inf)
            @parallel (@idx ni) compute_viscosity!(
                η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
            )
            @parallel (JustRelax.@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, args)
            # ------------------------------

            # Stokes solver ----------------
            iters = solve!(
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
                iterMax = 100e3,
                nout=1e3,
                viscosity_cutoff=(-Inf, Inf),
                verbose=false,
            )
            @parallel (JustRelax.@idx ni) JustRelax.Stokes3D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
            dt   = compute_dt(stokes, di, dt_diff)
            # ------------------------------

            # interpolate fields from particle to grid vertices
            particle2grid!(thermal.T, pT, xvi, particles)
            temperature2center!(thermal)

            @parallel (@idx ni) compute_shear_heating!(
                thermal.shear_heating,
                @tensor_center(stokes.τ),
                @tensor_center(stokes.τ_o),
                @strain(stokes),
                phase_ratios.center,
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
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 1e2,
                verbose = false,
            )
            # ------------------------------

            # Advection --------------------
            # advect particles in space
            advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, grid_vz, dt, 2 / 3)
            # advect particles in memory
            move_particles!(particles, xvi, particle_args)
            # interpolate fields from grid vertices to particles
            grid2particle_flip!(pT, xvi, thermal.T, thermal.Told, particles)
            # check if we need to inject particles
            inject = check_injection(particles)
            inject && inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)
            # update phase ratios
            @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)

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

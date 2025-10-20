push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
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

const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end
# Load script dependencies
using GeoParams, CellArrays, Statistics

# Load file with all the rheology configurations
include("../miniapps/benchmarks/stokes2D/Volcano2D/Caldera_setup.jl")
include("../miniapps/benchmarks/stokes2D/Volcano2D/Caldera_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

function apply_pure_shear(Vx, Vy, εbg, xvi, lx, ly)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Vx, εbg, lx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5)
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy, εbg, ly)
        yi = yv[j]
        Vy[i + 1, j] = abs(yi) * εbg
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx, εbg, lx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy, εbg, ly)

    return nothing
end

function extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    topo_idx = [findfirst(x -> x == air_phase, row) - 1 for row in eachrow(phases_GMG)]
    yv = xvi[2]
    topo_y = yv[topo_idx]
    return topo_y
end

function thermal_anomaly!(Temp, Ω_T, phase_ratios, T_chamber, T_air, conduit_phase, magma_phase, air_phase)

    @parallel_indices (i, j) function _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, vertex_ratio, conduit_phase, magma_phase, air_phase)
        # quick escape
        conduit_ratio_ij = @index vertex_ratio[conduit_phase, i, j]
        magma_ratio_ij = @index vertex_ratio[magma_phase, i, j]
        air_ratio_ij = @index vertex_ratio[air_phase, i, j]

        if conduit_ratio_ij > 0.5 || magma_ratio_ij > 0.5
            # if isone(conduit_ratio_ij) || isone(magma_ratio_ij)
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber

        elseif air_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_air
        end

        return nothing
    end

    ni = size(phase_ratios.vertex)

    @parallel (@idx ni) _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, phase_ratios.vertex, conduit_phase, magma_phase, air_phase)

    @views Ω_T[1, :] .= Ω_T[2, :]
    @views Ω_T[end, :] .= Ω_T[end - 1, :]
    @views Temp[1, :] .= Temp[2, :]
    @views Temp[end, :] .= Temp[end - 1, :]

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false, extension = 1.0e-15 * 0)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid             # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = 5.0e2 * 3600 * 24 * 365
    # dt                  = Inf # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 100
    max_xcell = 150
    min_xcell = 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)

    # Initialize marker chain
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    air_phase = 6
    topo_y = extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    for _ in 1:3
        @views hn = 0.5 .* (topo_y[1:(end - 1)] .+ topo_y[2:end])
        @views topo_y[2:(end - 1)] .= 0.5 .* (hn[1:(end - 1)] .+ hn[2:end])
        fill_chain_from_vertices!(chain, PTArray(backend)(topo_y))
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    end
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # rock ratios for variational stokes
    # RockRatios
    ϕ = RockRatio(backend, ni)
    # update_rock_ratio!(ϕ, phase_ratios, air_phase)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di;  ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-10, Re = π / 2, r = 0.7, CFL = 0.98 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)

    # Add thermal anomaly BC's
    T_chamber = 1223.0e0
    T_air = 273.0e0
    Ω_T = @zeros(size(thermal.T)...)
    thermal_anomaly!(thermal.T, Ω_T, phase_ratios, T_chamber, T_air, 5, 3, air_phase)
    JustRelax.DirichletBoundaryCondition(Ω_T)

    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = true, right = true, top = false, bot = false),
        dirichlet = (; mask = Ω_T)
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    Ttop = thermal.T[2:(end - 1), end]
    Tbot = thermal.T[2:(end - 1), 1]
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))

    # Melt fraction
    ϕ_m = @zeros(ni...)
    compute_melt_fraction!(
        ϕ_m, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
    )
    # Rheology
    args0 = (T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    # flow_bcs         = DisplacementBoundaryConditions(;
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )

    # U            = 0.02
    # stokes.U.Ux .= PTArray(backend)([(x - li[1] * 0.5) * U / dt for x in xvi[1], _ in 1:ny+2])
    # stokes.U.Uy .= PTArray(backend)([-y * U / dt for _ in 1:nx+2, y in xvi[2]])
    # flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    # displacement2velocity!(stokes, dt)

    εbg = extension
    apply_pure_shear(@velocity(stokes)..., εbg, xvi, li...)

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # ----------------------------------------------------

    local iters


    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0
    thermal.Told .= thermal.T

    while it < 2 # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        if mod(round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3), 1.5e3) == 0.0
            thermal_anomaly!(thermal.T, Ω_T, phase_ratios, T_chamber, T_air, 5, 3, air_phase)
        end
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # args = (; T=thermal.Tc, P=stokes.P, dt=Inf, ΔTc=thermal.ΔTc)
        args = (; ϕ = ϕ_m, T = thermal.Tc, P = stokes.P, dt = Inf)

        stress2grid!(stokes, pτ, xvi, xci, particles)

        iters = solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax = 100.0e3,
                nout = 2.0e3,
                free_surface = true,
                viscosity_cutoff = viscosity_cutoff,
            )
        )

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, xci, xvi, dt)

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dtmax = 2.0e3 * 3600 * 24 * 365.25
        dt = compute_dt(stokes, di, dtmax) * 0.5

        println("dt = $(dt / (3600 * 24 * 365.25)) years")
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
            kwargs = (
                igg = igg,
                phase = phase_ratios,
                iterMax = 100.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])

        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------

        # Advection --------------------
        copyinn_x!(T_buffer, thermal.T)
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        # inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
            xvi
        )

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        compute_melt_fraction!(
            ϕ_m, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
        )

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        # update_rock_ratio!(ϕ, phase_ratios, air_phase)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        tensor_invariant!(stokes.τ)

        @show it += 1
        t += dt

    end

    return iters
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
@testset "Volcano 2D" begin
    @suppress begin
        n = 64
        nx, ny = n, n >>> 1
        li, origin, phases_GMG, T_GMG = setup2D(
            nx + 1, ny + 1;
            sticky_air = 4,
            flat = false,
            chimney = false,
            chamber_T = 1.0e3,
            chamber_depth = 7.0e0,
            chamber_radius = 0.5,
            aspect_x = 6,
        )

        igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
            IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
        else
            igg
        end

        iters = main(li, origin, phases_GMG, T_GMG, igg; nx = nx, ny = ny)
        @test passed = iters.err_evo1[end] < 1.0e-2
    end
end

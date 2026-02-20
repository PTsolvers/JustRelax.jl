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
using Printf, Statistics, LinearAlgebra, CellArrays, StaticArrays


# -----------------------------------------------------------------------------------------
## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z, sticky_air)
    @all(P) = abs(@all(ρg) * (@all_j(z) - sticky_air)) #* <(@all_j(z), 0.0)
    return nothing
end

function init_phases!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly, sticky_air, top, bottom)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(
            phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly, sticky_air, top, bottom
        )
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            y = -(@index py[ip, i, j]) #- sticky_air
            if top ≤ y ≤ bottom
                @index phases[ip, i, j] = 1.0 # crust
            end

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y + yc_anomaly)^2 ≤ r_anomaly^2)
                @index phases[ip, i, j] = 2.0
            end

            if y < top
                @index phases[ip, i, j] = 3.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(
        phases,
        particles.coords...,
        particles.index,
        xc_anomaly,
        yc_anomaly,
        r_anomaly,
        sticky_air,
        top,
        bottom,
    )
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, sticky_air, top, bottom, dTdz, offset)
    depth = y[j]

    if depth ≥ 0.0e0
        T[i + 1, j] = offset

    else # if top ≤ (depth) < bottom
        dTdZ = dTdz
        offset = offset
        T[i + 1, j] = abs(depth) * dTdZ + offset

    end

    return nothing
end

function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi, sticky_air)
    @parallel_indices (i, j) function _circular_perturbation!(
            T, δT, xc_anomaly, yc_anomaly, r_anomaly, x, y, sticky_air
        )
        depth = -y[j] #- sticky_air
        if ((x[i] - xc_anomaly)^2 + (depth[j] + yc_anomaly)^2 ≤ r_anomaly^2)
            # T[i + 1, j] *= δT / 100 + 1
            T[i + 1, j] = δT
        end
        return nothing
    end

    nx, ny = size(T)

    return @parallel (1:(nx - 2), 1:ny) _circular_perturbation!(
        T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi..., sticky_air
    )
end

function linear_creep_models()
    creep_rock = LinearViscous(; η = 1.0e23 * Pa * s)
    creep_magma = LinearViscous(; η = 1.0e18 * Pa * s)
    creep_air = LinearViscous(; η = 1.0e18 * Pa * s)
    return creep_rock, creep_magma, creep_air
end

function nonlinear_creep_models()
    creep_rock = DislocationCreep(; A = 1.67e-24Pa^(-(35 // 10)) / s, n = 3.5, E = 1.87e5J / mol, V = 0 * 6.0e-6m^3 / mol, r = 0.0, R = 8.3145J / mol / K)
    creep_magma = DislocationCreep(; A = 1.67e-24Pa^(-(35 // 10)) / s, n = 3.5, E = 1.87e5J / mol, V = 0 * 6.0e-6m^3 / mol, r = 0.0, R = 8.3145J / mol / K)
    creep_air = LinearViscous(; η = 1.0e18 * Pa * s)
    return creep_rock, creep_magma, creep_air
end

function init_rheology(creep_rock, creep_magma, creep_air, CharDim; is_compressible = false, steady_state = true)
    # plasticity setup
    do_DP = true          # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg = 1.0e20Pa * s  # regularisation "viscosity" for Drucker-Prager
    Coh = 10.0MPa       # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30.0 * do_DP  # friction angle
    G0 = 6.0e11Pa      # elastic shear modulus
    G_magma = 6.0e11Pa      # elastic shear modulus perturbation

    soft_C = NonLinearSoftening(; ξ₀ = ustrip(Coh), Δ = ustrip(Coh) / 2) # softening law
    pl = DruckerPrager_regularised(; C = Coh, ϕ = ϕ, η_vp = η_reg, Ψ = 0.0, softening_C = soft_C)        # plasticity
    if is_compressible == true
        el = SetConstantElasticity(; G = G0, ν = 0.25)           # elastic spring
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.25) # elastic spring
        β_rock = 6.0e-11
        β_magma = 6.0e-11
    else
        el = SetConstantElasticity(; G = G0, ν = 0.5)            # elastic spring
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.5) # elastic spring
        β_rock = inv(get_Kb(el))
        β_magma = inv(get_Kb(el_magma))
    end
    g = 9.81m / s^2
    return rheology = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2650kg / m^3, α = 3.0e-5 / K, T0 = 0.0C, β = β_rock / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
            # LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3J / kg),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
            CharDim = CharDim,
        ),

        #Name="Magma"
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2650kg / m^3, T0 = 0.0C, β = β_magma / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 1.5Watt / K / m),
            # LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3J / kg),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_magma,
            CharDim = CharDim,
        ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(ρ = 1kg / m^3),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1000J / kg / K),
            Conductivity = ConstantConductivity(; k = 15Watt / K / m),
            LatentHeat = ConstantLatentHeat(; Q_L = 0.0J / kg),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_air,)),
            Gravity = ConstantGravity(; g = g),
            CharDim = CharDim,
        ),
    )

end

function main2D(igg; nx = 32, ny = 32, do_vtk = false)

    # Characteristic lengths
    CharDim = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    sticky_air = nondimensionalize(1.5km, CharDim)       # thickness of the sticky air layer
    L = nondimensionalize(12.5km, CharDim) + sticky_air
    lx = L                                                        # domain length in x-direction
    ly = L                                                        # domain length in y-direction
    li = lx, ly                                                   # domain length in x- and y-direction
    ni = nx, ny                                                   # number of grid points in x- and y-direction
    di = @. li / ni                                               # grid step in x- and y-direction
    origin = 0.0e0, -ly + sticky_air  # origin coordinates of the domain
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                                           # nodes at the center and vertices of the cells
    εbg = nondimensionalize(0.0 / s, CharDim)                     # background strain rate
    #---------------------------------------------------------------------------------------

    # Physical Parameters
    # rheology = init_rheology(CharDim; is_compressible = true, steady_state = false)
    # creep_rock, creep_magma, creep_air = linear_creep_models()
    creep_rock, creep_magma, creep_air = nonlinear_creep_models()
    rheology = init_rheology(creep_rock, creep_magma, creep_air, CharDim; is_compressible = true)
    rheology_inc = init_rheology(creep_rock, creep_magma, creep_air, CharDim; is_compressible = false)
    cutoff_visc = nondimensionalize((1.0e16Pa * s, 1.0e24Pa * s), CharDim)
    dt = dt_max = nondimensionalize(1.0e3 * yr, CharDim)         # diffusive CFL timestep limiter

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 60, 80, 30
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Circular temperature anomaly -----------------------
    x_anomaly = lx * 0.5
    y_anomaly = nondimensionalize(-5km, CharDim)          # origin of the small thermal anomaly
    r_anomaly = nondimensionalize(1.5km, CharDim)        # radius of perturbation
    anomaly = nondimensionalize((750 + 273)K, CharDim) # thermal perturbation (in K)
    init_phases!(pPhases, particles, x_anomaly, y_anomaly, r_anomaly, sticky_air, nondimensionalize(0.0km, CharDim), nondimensionalize(20km, CharDim))
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # Initialisation of thermal profile
    thermal = ThermalArrays(backend_JR, ni) # initialise thermal arrays and boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    Ttop = nondimensionalize((20 + 273)K, CharDim)
    Tbot = nondimensionalize((450 + 273)K, CharDim)
    ∇Tz = (Ttop - Tbot) / (L - sticky_air)
    # dTdz = nondimensionalize((450-20+273)K, CharDim) / (nondimensionalize(12.5km, CharDim))
    T1D = @. (∇Tz * (xvi[2]) + Ttop) * (xvi[2] < 0.0e0)
    T1D[xvi[2] .≥ 0.0e0] .= Ttop
    thermal.T .+= PTArray(backend_JR)(T1D')

    circular_perturbation!(
        thermal.T, anomaly, x_anomaly, y_anomaly, r_anomaly, xvi, sticky_air
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni) # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-3, ϵ_rel = 1.0e-2, CFL = 1 / √2.1)
    # ----------------------------------------------------

    args = (; T = thermal.Tc, P = stokes.P, dt = dt)
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 0.8 / √2.1
    )

    # Pure shear far-field boundary conditions
    stokes.V.Vx .= PTArray(backend_JR)(
        [
            εbg * (x - lx * 0.5) for x in xvi[1], _ in 1:(ny + 2)
        ]
    )
    stokes.V.Vy .= PTArray(backend_JR)(
        [
            (abs(y) - sticky_air) * εbg * (abs(y) > sticky_air) for _ in 1:(nx + 2), y in xvi[2]
        ]
    )

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = true,
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    ϕ = @zeros(ni...)
    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
    )
    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...) # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:5
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        # @parallel init_P!(stokes.P, ρg[2], xci[2])
        stokes.P .= PTArray(backend_JR)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))
    end

    # Arguments for functions
    args = (; T = thermal.Tc, P = stokes.P, dt = dt, ΔTc = thermal.ΔTc)
    @copy thermal.Told thermal.T
    stokes.ε.xx .= nondimensionalize(1.0e-20 / s, CharDim)
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    @copy stokes.P0 stokes.P
    thermal.Told .= thermal.T
    P_init = deepcopy(stokes.P)

    # Stokes solver -----------------
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf, ΔTc = thermal.ΔTc)
    solve!(
        stokes,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratios,
        rheology_inc,
        args,
        dt,
        igg;
        kwargs = (;
            iterMax = 100.0e3,
            free_surface = true,
            nout = 5.0e3,
            viscosity_cutoff = cutoff_visc,
            relaxation = 1.0e-3,
            λ_relaxation = 1.0e0,
        )
    )

    while it < 1

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf, ΔTc = thermal.ΔTc)

        # Stokes solver -----------------
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
            kwargs = (;
                iterMax = 100.0e3,
                free_surface = true,
                nout = 5.0e3,
                viscosity_cutoff = cutoff_visc,
                relaxation = 1.0e-3,
                λ_relaxation = 1.0e0,
            )
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dt = compute_dt(stokes, di, dt_max, igg)
        # # --------------------------------

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
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
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
            pT, T_buffer, thermal.ΔT[2:(end - 1), :], subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------
        compute_melt_fraction!(
            ϕ, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
        )
        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])

        @show it += 1
        t += dt
    end

    finalize_global_grid()

    return ϕ, stokes, thermal
end

@testset "thermal stresses" begin
    @suppress begin
        nx, ny = 32, 32           # number of cells
        igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
            IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
        else
            igg
        end

        ϕ, stokes, thermal = main2D(igg; nx = 32, ny = 32)

        nx_T, ny_T = size(thermal.T)
        @test  Array(thermal.T)[nx_T >>> 1 + 1, ny_T >>> 1 + 1] ≈ 1.4134 rtol = 1.0e-2
        @test  Array(ϕ)[nx_T >>> 1 + 1, ny_T >>> 1 + 1] ≈ 0.0819 rtol = 1.0e-2
    end
end

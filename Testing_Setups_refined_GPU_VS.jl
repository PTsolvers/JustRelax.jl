using PoissonGrids
# const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg; Pkg.activate("miniapps")

const backend_JR = @static if isCUDA
    CUDA.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie

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

function init_phases!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly, xc_pipe, r_pipe, b_pipe, t_pipe, xs_fault, α_fault, r_fault, maxd_fault, sticky_air, top, bottom)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(
            phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly, xc_pipe, r_pipe, b_pipe, t_pipe, xs_fault, α_fault, r_fault, maxd_fault, sticky_air, top, bottom
        )
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            y = -(@index py[ip, i, j]) #- sticky_air
            if top ≤ y ≤ bottom
                @index phases[ip, i, j] = 1.0 # crust
            end

            # fault
            if 0 <= y <= maxd_fault
                xc_fault = xs_fault + y / tand(α_fault)
                if abs(x - xc_fault) <= r_fault
                    @index phases[ip, i, j] = 3.0 # fault
                end
            end

            # feeding pipe
            if abs(x - xc_pipe) ≤ r_pipe && (-t_pipe ≤ y ≤ -b_pipe)
                @index phases[ip, i, j] = 1.0 # crust
            end

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y + yc_anomaly)^2 ≤ r_anomaly^2)
                @index phases[ip, i, j] = 2.0 # magma
            end

            if y < top
                @index phases[ip, i, j] = 4.0 # sticky air
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(
        phases,
        particles.coords...,
        particles.index,#
        xc_anomaly,
        yc_anomaly,
        r_anomaly,
        xc_pipe,
        r_pipe,
        b_pipe,
        t_pipe,
        xs_fault,
        α_fault,
        r_fault,
        maxd_fault,
        sticky_air,
        top,
        bottom,
    )
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, sticky_air, top, bottom, dTdz, offset)
    depth = y[j]

    if depth ≥ 0.0e0
        T[i + 1, j + 1] = offset

    else # if top ≤ (depth) < bottom
        dTdZ = dTdz
        offset = offset
        T[i + 1, j + 1] = abs(depth) * dTdZ + offset

    end

    return nothing
end

function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi, sticky_air)
    @parallel_indices (i, j) function _circular_perturbation!(
            T, δT, xc_anomaly, yc_anomaly, r_anomaly, x, y, sticky_air
        )
        depth = -y[j] #- sticky_air
        if ((x[i] - xc_anomaly)^2 + (depth + yc_anomaly)^2 ≤ r_anomaly^2)
            T[i + 1, j + 1] = δT
        end
        return nothing
    end

    ni = size(T) .- 2

    return @parallel (@idx ni) _circular_perturbation!(
        T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi..., sticky_air
    )
end

function linear_creep_models()
    creep_rock  = LinearViscous(; η = 1.0e23 * Pa * s)
    creep_magma = LinearViscous(; η = 1.0e18 * Pa * s)
    creep_air   = LinearViscous(; η = 1.0e18 * Pa * s)
    creep_fault = LinearViscous(; η = 1.0e20 * Pa * s)
    return creep_rock, creep_magma, creep_air, creep_fault
end

function nonlinear_creep_models()
    creep_rock  = DislocationCreep(; A = 1.67e-24Pa^(-(35 // 10)) / s, n = 3.5, E = 1.87e5J / mol, V = 0 * 6.0e-6m^3 / mol, r = 0.0, R = 8.3145J / mol / K)
    creep_magma = LinearViscous(; η = 1.0e18 * Pa * s) #DislocationCreep(; A = 1.67e-21Pa^(-(35 // 10)) / s, n = 3.5, E = 1.87e5J / mol, V = 0 * 6.0e-6m^3 / mol, r = 0.0, R = 8.3145J / mol / K)
    creep_air   = LinearViscous(; η = 1.0e18 * Pa * s)
    creep_fault = DislocationCreep(; A = 1.67e-22Pa^(-(35 // 10)) / s, n = 3.5, E = 1.87e5J / mol, V = 0 * 6.0e-6m^3 / mol, r = 0.0, R = 8.3145J / mol / K)
    return creep_rock, creep_magma, creep_air, creep_fault
end

function init_rheology(creep_rock, creep_magma, creep_fault, creep_air, CD; is_compressible = false, steady_state = true, is_dilatent = true)
    # plasticity setup
    do_DP = true          # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg = 1.0e19Pa * s  # regularisation "viscosity" for Drucker-Prager
    Coh = 10.0MPa         # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30.0 * do_DP      # friction angle
    G0 = 30GPa            # elastic shear modulus
    G_magma = 30GPa       # elastic shear modulus perturbation

    soft_C = NonLinearSoftening(; ξ₀ = Coh, Δ = Coh / 2) # softening law
    pl     = DruckerPrager_regularised(; C = Coh, ϕ = ϕ, η_vp = η_reg, Ψ = 0.0, softening_C = soft_C)             # plasticity
    pl_f   = DruckerPrager_regularised(; C = Coh / 2, ϕ = ϕ / 2, η_vp = η_reg, Ψ = 0.0, softening_C = soft_C)     # plasticity in fault
    pl_c   = DruckerPragerCap(; C = Coh / cosd(ϕ), ϕ = ϕ, η_vp = η_reg, Ψ = 10.0 * is_dilatent, pT = -10 * MPa)   # tensile plasticity
    pl_fc  = DruckerPragerCap(; C = Coh / cosd(ϕ) / 2, ϕ = ϕ / 2, η_vp = η_reg, Ψ = 10.0 * is_dilatent, pT = -10 * MPa)  # tensile plasticity
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
            CompositeRheology = CompositeRheology((creep_rock, el, pl_c)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
            CharDim = CD,
        ),

        #Name="Magma"
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2650kg / m^3, α = 3.0e-5 / K, T0 = 0.0C, β = β_magma / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 1.5Watt / K / m),
            # LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3J / kg),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_magma,
            CharDim = CD,
        ),

        #Name="Fault"
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 2650kg / m^3, α = 3.0e-5 / K, T0 = 0.0C, β = β_rock / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
            # LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3J / kg),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_fault, el, pl_fc)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
            CharDim = CD,
        ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase = 4,
            Density = ConstantDensity(ρ = 1kg / m^3),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1000J / kg / K),
            Conductivity = ConstantConductivity(; k = 15Watt / K / m),
            LatentHeat = ConstantLatentHeat(; Q_L = 0.0J / kg),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_air,)),
            Gravity = ConstantGravity(; g = g),
            CharDim = CD,
        ),
    )

end

# `Q` holds the volumetric strain (m³/m³) that each cell undergoes over one time step: `q` in
# the cells flagged by `ind`, zero elsewhere. `buffer` is host-side scratch of `size(Q)`, since
# logical indexing is unavailable on the GPU.
function set_volumetric_source!(Q, buffer, ind, q)
    buffer .= 0.0
    buffer[ind] .= q
    copyto!(Q, buffer)
    return Q
end

function main2D(igg; figdir = "Thermal_stresses", nx = 32, ny = 32, do_vtk = false)

    # Characteristic lengths
    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    sticky_air = nondimensionalize(1.5km, CD)                     # thickness of the sticky air layer
    D = nondimensionalize(12.5km, CD)                             # depth of the domain
    L = D + sticky_air
    lx = nondimensionalize(20.0km, CD)                            # domain length in x-direction
    ly = L                                                        # domain length in y-direction
    li = lx, ly                                                   # domain length in x- and y-direction
    ni = nx, ny                                                   # number of grid points in x- and y-direction
    origin = 0.0e0, -D                                            # origin coordinates of the domain
    M = window_monitor(5.0, 4.0, nondimensionalize(2.0km, CD), origin[1]) # function for refined grid
    xv_ref = solve_grid(-lx / 2, lx / 2, M, nx)
    #xv_ref = collect(LinRange(-lx/2, lx/2,    nx+1))
    yv_ref = collect(LinRange(-D, sticky_air, ny + 1))
    grid = Geometry(
        PTArray(backend_JR),
        xv_ref,
        yv_ref,
    )
    xci = Array.(grid.xci)
    xvi = Array.(grid.xvi)
    di_min = minimum.(grid.di.vertex)
    grid_vxi = grid.xi_vel
    εbg = nondimensionalize(1.0e-15 / s, CD)                      # background strain rate
    #---------------------------------------------------------------------------------------

    # Physical Parameters
    # rheology = init_rheology(CD; is_compressible = true, steady_state = false)
    # creep_rock, creep_magma, creep_air = linear_creep_models()
    creep_rock, creep_magma, creep_air, creep_fault = nonlinear_creep_models()
    rheology     = init_rheology(creep_rock, creep_magma, creep_fault, creep_air, CD; is_compressible = true, is_dilatent = true)
    rheology_inc = init_rheology(creep_rock, creep_magma, creep_fault, creep_air, CD; is_compressible = false, is_dilatent = false)
    cutoff_visc  = nondimensionalize((1.0e16Pa * s, 1.0e24Pa * s), CD)
    dt = dt_max  = nondimensionalize(1.0e3 * yr, CD)         # diffusive CFL timestep limiter
    # Q_in         = nondimensionalize(1.0e-3km^3 / yr, CD)
    Q_in         = nondimensionalize(1.0e-5km^3 / yr, CD)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 100, 180, 50
    particles      = init_particles(backend_JP, nxcell, max_xcell, min_xcell, Array.(grid.xi_vel[1]), Array.(grid.xi_vel[2]))
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    # temperature
    pT, pPhases    = init_cell_arrays(particles, Val(2))
    particle_args  = (pT, pPhases)

    # Circular temperature anomaly -----------------------
    x_anomaly = origin[1]
    y_anomaly = nondimensionalize(-5km, CD)          # origin of the small thermal anomaly
    r_anomaly = nondimensionalize(1.5km, CD)        # radius of perturbation
    anomaly   = nondimensionalize((750 + 273)K, CD) # thermal perturbation (in K)
    r_src     = r_anomaly / 2                     # radius of the injecting core of the anomaly
    V_src     = 4 / 3 * π * r_src^3
    # feeding pipe
    r_pipe    = nondimensionalize(0.5km, CD)
    xc_pipe   = origin[1]
    b_pipe    = -D
    t_pipe    = y_anomaly
    # fault
    xs_fault  = origin[1] - r_anomaly
    α_fault   = 60
    L_fault   = nondimensionalize(2km, CD)
    r_fault   = nondimensionalize(0.25km, CD)
    maxd_fault = L_fault * sind(α_fault)
    init_phases!(pPhases, particles, x_anomaly, y_anomaly, r_anomaly, xc_pipe, r_pipe, b_pipe, t_pipe, xs_fault, α_fault, r_fault, maxd_fault, sticky_air, nondimensionalize(0.0km, CD), nondimensionalize(20km, CD))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Marker chain tracking the free surface --------------
    nxcell_chain, min_xcell_chain, max_xcell_chain = 100, 75, 125
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell_chain, min_xcell_chain, max_xcell_chain, xv_ref, initial_elevation)

    # Rock fractions of the staggered control volumes, used to mask out the air
    air_phase = 4
    ϕ_R = RockRatio(backend_JR, ni)
    compute_rock_fraction!(ϕ_R, chain, grid.xvi, grid.di.vertex)
    # ----------------------------------------------------

    # find full magma cells
    ind       = zeros(Bool, nx, ny)
    #for i = 1:nx
    #    for j = 1:ny
    #        if phase_ratios.center[i,j][2] ≈ 1
    #            ind[i,j] = true
    #        end
    #    end
    #end

    # find innermost magma cells
    for i in 1:nx
        for j in 1:ny
            if ((xci[1][i] - x_anomaly)^2 + (xci[2][j] - y_anomaly)^2 ≤ r_src^2)
                ind[i, j] = true
            end
        end
    end

    # Initialisation of thermal profile
    thermal = ThermalArrays(backend_JR, ni) # initialise thermal arrays and boundary conditions
    Ttop = nondimensionalize((20 + 273)K, CD)
    Tbot = nondimensionalize(438.5625C, CD)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
    )
    ∇Tz = (Ttop - Tbot) / (L - sticky_air)
    # dTdz = nondimensionalize((450-20+273)K, CD) / (nondimensionalize(12.5km, CD))
    T1D = @. (∇Tz * (xci[2]) + Ttop) * (xci[2] < 0.0e0)
    T1D[xci[2] .≥ 0.0e0] .= Ttop
    thermal.T[:, 2:(end - 1)] .+= PTArray(backend_JR)(T1D')

    circular_perturbation!(
        thermal.T, anomaly, x_anomaly, y_anomaly, r_anomaly, grid.xvi, sticky_air
    )
    thermal_bcs!(thermal, thermal_bc)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni) # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di_min; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-3, CFL = 0.95 / √2.1)
    # ----------------------------------------------------

    args = (; T = thermal.T, P = stokes.P, dt = dt)
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di_min, li; ϵ = 1.0e-5, CFL = 0.8 / √2.1
    )

    # Volumetric source: the flagged cells dilate at the rate of a sphere of radius `r_src`
    # inflating at `Q_in`. The normalization is a volume, not a cell count, so the injected
    # volume does not change with grid resolution.
    Q_ini = zeros(ni...)
    set_volumetric_source!(stokes.Q, Q_ini, ind, Q_in * dt / V_src)

    # Pure shear far-field boundary conditions
    stokes.V.Vx .= PTArray(backend_JR)(
        [
            εbg * x for x in xvi[1], _ in 1:(ny + 2)
        ]
    )
    stokes.V.Vy .= PTArray(backend_JR)(
        [
            (abs(y) - sticky_air) * εbg for _ in 1:(nx + 2), y in xvi[2]
        ]
    )

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false, # zero stress boundary condition at the surface
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    ϕ = @zeros(ni...)
    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T = thermal.T, P = stokes.P)
    )

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...) # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:5
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P); air_phase = air_phase)
        compute_lithostatic_pressure!(stokes.P, ρg[2], grid.di.vertex[2], igg)
    end

    # Arguments for functions
    args = (; T = thermal.T, P = stokes.P, dt = dt, ΔT = thermal.ΔT)
    @copy thermal.Told thermal.T
    stokes.ε.xx .= nondimensionalize(1.0e-20 / s, CD)
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc; air_phase = air_phase)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Make initial temperature
    for i in 1:10
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e3,
                verbose = false,
            )
        )
    end

    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size = (1200, 900))
        ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
        scatter!(
            ax1,
            ustrip.(dimensionalize((Array(thermal.T[2:(end - 1), 2:(end - 1)])), C, CD))[:],
            ustrip.(dimensionalize(Y, km, CD)),
        )
        scatter!(
            ax2,
            # Array(ρg[2][:]),
            Array(ustrip.(dimensionalize(stokes.P[:], MPa, CD))),
            ustrip.(dimensionalize(Y, km, CD)),
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # Ghosted grid field used to compute and interpolate the characteristic time.
    dt₀ = similar(thermal.T)

    # Time loop
    t, it = 0.0, 0
    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    centroid2particle!(pT, thermal.T, particles)
    @copy stokes.P0 stokes.P
    thermal.Told .= thermal.T
    P_init = deepcopy(stokes.P)
    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, ϕ_R, grid.di, dt; ϵ = 1.0e-6)

    # Stokes solver -----------------
    args = (; T = thermal.T, P = stokes.P, dt = Inf, ΔT = thermal.ΔT)

    # # Stokes solver -----------------
    # solve_VariationalStokes!(
    #     stokes,
    #     pt_stokes,
    #     grid,
    #     flow_bcs,
    #     ρg,
    #     phase_ratios,
    #     ϕ_R,
    #     rheology_inc,
    #     args,
    #     dt,
    #     igg;
    #     air_phase = air_phase,
    #     kwargs = (;
    #         iterMax = 75.0e3,
    #         nout = 2.0e2,
    #         λ_relaxation = 1.0,
    #         viscosity_relaxation = 1.0e-3,
    #         viscosity_cutoff = cutoff_visc,
    #         free_surface = false,
    #     )
    # )

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)
    τxx_v_ghost = @zeros(ni .+ 3...)
    τyy_v_ghost = @zeros(ni .+ 3...)
    τxy_ghost   = @zeros(ni .+ 3...)
    ωxy_ghost   = @zeros(ni .+ 3...)

    while it < 250

        set_volumetric_source!(stokes.Q, Q_ini, ind, Q_in * dt / V_src)

        # Update buoyancy and viscosity -
        args = (; T = thermal.T, P = stokes.P, dt = Inf, ΔT = thermal.ΔT)

        # Stokes solver -----------------
        result = solve_VariationalDYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            ϕ_R,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                air_phase = air_phase,
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 100.0e3,
                total_iterMax = 100.0e3,
                nout = 50,
                rel_drop = 1e-1,
                λ_relaxation_PH = 1.0,
                λ_relaxation_DR = 1.0,
                pressure_relaxation = 0.75,
                viscosity_relaxation = 1.0e-3,
                viscosity_cutoff = cutoff_visc,
                free_surface = false,
            ),
        )
        rotate_stress!(pτ, stokes, particles, dt)

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dt = compute_dt(stokes, di_min, dt_max, igg)
        # --------------------------------

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
            grid;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes)
        # Populate the ghost cells before interpolating to particles.
        @views dt₀[1, :] .= dt₀[2, :]
        @views dt₀[end, :] .= dt₀[end - 1, :]
        @views dt₀[:, 1] .= dt₀[:, 2]
        @views dt₀[:, end] .= dt₀[:, end - 1]
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt
        )
        # ------------------------------
        compute_melt_fraction!(
            ϕ, phase_ratios, rheology, (T = thermal.T, P = stokes.P)
        )

       # Advection ------------------------------------------------------------

        # Advect particle coordinates, then move particles to their new cells.
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)

        # Prepare grid fields used when injecting replacement particles.
        for (ghost, field) in (
            (τxx_v_ghost, stokes.τ.xx_v),
            (τyy_v_ghost, stokes.τ.yy_v),
            (τxy_ghost, stokes.τ.xy),
            (ωxy_ghost, stokes.ω.xy),
        )
            @views ghost[2:(end - 1), 2:(end - 1)] .= field
            @views ghost[1, :] .= ghost[2, :]
            @views ghost[end, :] .= ghost[end - 1, :]
            @views ghost[:, 1] .= ghost[:, 2]
            @views ghost[:, end] .= ghost[:, end - 1]
        end

        # Advect the free surface using the same velocity and timestep.
        semilagrangian_advection_markerchain!(
            chain,
            RungeKutta2(),
            @velocity(stokes),
            grid_vxi,
            grid.xvi,
            dt,
        )

        # Invalidate particles and particle fields that crossed the surface.
        particle_args_to_reset = (pT, unwrap(pτ)...)

        update_phases_given_markerchain!(
            pPhases,
            chain,
            particles,
            grid.origin,
            grid.di.vertex,
            air_phase,
            particle_args_to_reset,
        )

        # Replenish only after surface filtering.
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (thermal.T, τxx_v_ghost, τyy_v_ghost, τxy_ghost, ωxy_ghost),
        )

        # Enforce the air temperature after injection, since newly injected particles
        # may have received interpolated values.
        @views pT.data[pPhases.data .== air_phase] .= Ttop

        # Now all active particles and particle fields are consistent.
        stress2grid!(stokes, pτ, particles)

        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ_R, chain, grid.xvi, grid.di.vertex)

        particle2centroid!(thermal.T, pT, particles)

        thermal_bcs!(thermal, thermal_bc)
        thermal.ΔT .= thermal.T .- thermal.Told

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 5) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)
            t_dim = dimensionalize(t, yr, CD).val
            t_Kyrs = t_dim / 1.0e3
            if igg.me == 0
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                if do_vtk
                    data_v = (;
                        τxy = Array(ustrip.(dimensionalize(stokes.τ.xy, s^-1, CD))),
                        εxy = Array(ustrip.(dimensionalize(stokes.ε.xy, s^-1, CD))),
                        Vx = Array(ustrip.(dimensionalize(Vx_v, cm / yr, CD))),
                        Vy = Array(ustrip.(dimensionalize(Vy_v, cm / yr, CD))),
                    )
                    data_c = (;
                        Ph_center = [argmax(p) for p in Array(phase_ratios.center)],
                        P = Array(ustrip.(dimensionalize(stokes.P, MPa, CD))),
                        T = Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), 2:(end - 1)], C, CD))),
                        τxx = Array(ustrip.(dimensionalize(stokes.τ.xx, MPa, CD))),
                        τyy = Array(ustrip.(dimensionalize(stokes.τ.yy, MPa, CD))),
                        τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CD))),
                        εxx = Array(ustrip.(dimensionalize(stokes.ε.xx, s^-1, CD))),
                        εyy = Array(ustrip.(dimensionalize(stokes.ε.yy, s^-1, CD))),
                        εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1, CD))),
                        εII_pl = Array(ustrip.(dimensionalize(stokes.ε_pl.II, s^-1, CD))),
                        η_vep = Array(ustrip.(dimensionalize(stokes.viscosity.η_vep, Pa * s, CD))),
                        η = Array(ustrip.(dimensionalize(stokes.viscosity.η, Pa * s, CD))),
                        Q = Array(stokes.Q),
                        ϕ_R = Array(ϕ_R.center),
                        ρg = Array(ustrip.(dimensionalize(ρg[2], kg / (m^2 * s^2), CD)))
                    )
                    velocity_v = (
                        Array(ustrip.(dimensionalize(Vx_v, cm / yr, CD))),
                        Array(ustrip.(dimensionalize(Vy_v, cm / yr, CD))),
                    )
                    save_vtk(
                        joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                        xvi,
                        xci,
                        data_v,
                        data_c,
                        velocity_v,
                        t = t_Kyrs
                    )
                    save_marker_chain(
                        joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")),
                        xvi[1],
                        Array(chain.h_vertices)
                    )
                    save_particles(
                        particles,
                        pPhases;
                        conversion = 1.0,
                        fname = joinpath(vtk_dir, "particles_" * lpad("$it", 6, "0")),
                        pvd = joinpath(vtk_dir, "particles"),
                        t = t_Kyrs,
                    )
                end

                # Make Makie figure
                fig = Figure(; size = (2000, 1800), createmissing = true)
                ar = li[1] / li[2]

                ax0 = Axis(
                    fig[1, 1:2];
                    aspect = ar,
                    title = "t = $(round(ustrip.(t_Kyrs); digits = 3)) Kyrs",
                    titlesize = 50,
                    height = 0.0,
                )
                ax0.ylabelvisible = false
                ax0.xlabelvisible = false
                ax0.xgridvisible = false
                ax0.ygridvisible = false
                ax0.xticksvisible = false
                ax0.yticksvisible = false
                ax0.yminorticksvisible = false
                ax0.xminorticksvisible = false
                ax0.xgridcolor = :white
                ax0.ygridcolor = :white
                ax0.ytickcolor = :white
                ax0.xtickcolor = :white
                ax0.yticklabelcolor = :white
                ax0.xticklabelcolor = :white
                ax0.yticklabelsize = 0
                ax0.xticklabelsize = 0
                ax0.xlabelcolor = :white
                ax0.ylabelcolor = :white

                ax1 = Axis(
                    fig[2, 1][1, 1];
                    aspect = ar,
                    title = L"T [\mathrm{C}]",
                    titlesize = 40,
                    yticklabelsize = 25,
                    xticklabelsize = 25,
                    xlabelsize = 25,
                )
                ax2 = Axis(
                    fig[2, 2][1, 1];
                    aspect = ar,
                    title = L"Viscosity [\mathrm{Pa s}]",
                    xlabel = "Width [km]",
                    titlesize = 40,
                    yticklabelsize = 25,
                    xticklabelsize = 25,
                    xlabelsize = 25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect = ar,
                    title = L"P [MPa]",
                    titlesize = 40,
                    yticklabelsize = 25,
                    xticklabelsize = 25,
                    xlabelsize = 25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect = ar,
                    title = L"ΔT [C]",
                    # title = L"P [MPa]",
                    titlesize = 40,
                    yticklabelsize = 25,
                    xticklabelsize = 25,
                    xlabelsize = 25,
                )
                ax5 = Axis(
                    fig[4, 1][1, 1];
                    aspect = ar,
                    title = L"\log_{10}(\dot{\varepsilon}_{\textrm{II}}) [\mathrm{s}^{-1}]",
                    xlabel = "Width [km]",
                    titlesize = 40,
                    yticklabelsize = 25,
                    xticklabelsize = 25,
                    xlabelsize = 25,
                )
                ax6 = Axis(
                    fig[4, 2][1, 1];
                    aspect = ar,
                    title = L"\tau_{\textrm{II}} [MPa]",
                    xlabel = "Width [km]",
                    titlesize = 40,
                    yticklabelsize = 25,
                    xticklabelsize = 25,
                    xlabelsize = 25,
                )
                # marker chain in dimensional coordinates, overlain on the fields below
                chain_x = ustrip.(dimensionalize(Array(chain.coords[1].data[:]), km, CD))
                chain_y = ustrip.(dimensionalize(Array(chain.coords[2].data[:]), km, CD))
                # Plot temperature
                p1 = heatmap!(
                    ax1,
                    ustrip.(dimensionalize(xvi[1], km, CD)),
                    ustrip.(dimensionalize(xvi[2], km, CD)),
                    ustrip.(dimensionalize((Array(thermal.T[2:(end - 1), 2:(end - 1)])), C, CD));
                    colormap = :batlow,
                )
                scatter!(ax1, chain_x, chain_y, color = :red, markersize = 3)
                # Plot effective viscosity
                p2 = heatmap!(
                    ax2,
                    ustrip.(dimensionalize(xci[1], km, CD)),
                    ustrip.(dimensionalize(xci[2], km, CD)),
                    log10.(@dimstrip Array(stokes.viscosity.η_vep) Pa * s CD);
                    colormap = :glasgow,
                    colorrange = (log10(1.0e16), log10(1.0e24)),
                )
                scatter!(ax2, chain_x, chain_y, color = :red, markersize = 3)
                arrows2d!(
                    ax2,
                    ustrip.(dimensionalize(xvi[1], km, CD))[1:5:(end - 1)],
                    ustrip.(dimensionalize(xvi[2], km, CD))[1:5:(end - 1)],
                    Array.(
                        (
                            ustrip.(dimensionalize(Vx_v, cm / yr, CD))[1:5:(end - 1), 1:5:(end - 1)],
                            ustrip.(dimensionalize(Vy_v, cm / yr, CD))[1:5:(end - 1), 1:5:(end - 1)],
                        )
                    )...,
                    lengthscale = 1 / max(
                        maximum(ustrip.(dimensionalize(Vx_v, cm / yr, CD))),
                        maximum(ustrip.(dimensionalize(Vy_v, cm / yr, CD)))
                    ),
                    color = :red,
                )
                # Plot Pressure difference
                p3 = heatmap!(
                    ax3,
                    ustrip.(dimensionalize(xci[1], km, CD)),
                    ustrip.(dimensionalize(xci[2], km, CD)),
                    ustrip.(dimensionalize((Array((stokes.P))), MPa, CD));
                    # ustrip.(dimensionalize((Array((stokes.P .- P_init))), MPa, CD));
                    colormap = :roma,
                )
                # Plot Pressure difference

                p4 = heatmap!(
                    ax4,
                    ustrip.(dimensionalize(xci[1], km, CD)),
                    ustrip.(dimensionalize(xci[2], km, CD)),
                    (@dimstrip(Array(thermal.T), C, CD) .- @dimstrip(Array(thermal.Told), C, CD))[2:(end - 1), 2:(end - 1)], colormap = :roma,
                )
                # Plot 2nd invariant of strain rate
                p5 = heatmap!(
                    ax5,
                    ustrip.(dimensionalize(xci[1], km, CD)),
                    ustrip.(dimensionalize(xci[2], km, CD)),
                    log10.(ustrip.(dimensionalize(Array((stokes.ε.II)), s^-1, CD)));
                    colormap = :roma,
                )
                # Plot 2nd invariant of stress
                p6 = heatmap!(
                    ax6,
                    ustrip.(dimensionalize(xci[1], km, CD)),
                    ustrip.(dimensionalize(xci[2], km, CD)),
                    ustrip.(dimensionalize(Array((stokes.τ.II)), MPa, CD));
                    colormap = :batlow,
                )
                hidexdecorations!(ax1)
                hidexdecorations!(ax2)
                hidexdecorations!(ax3)
                Colorbar(
                    fig[2, 1][1, 2], p1; height = Relative(0.7), ticklabelsize = 25, ticksize = 15
                )
                Colorbar(
                    fig[2, 2][1, 2], p2; height = Relative(0.7), ticklabelsize = 25, ticksize = 15
                )
                Colorbar(
                    fig[3, 1][1, 2], p3; height = Relative(0.7), ticklabelsize = 25, ticksize = 15
                )
                Colorbar(
                    fig[3, 2][1, 2], p4; height = Relative(0.7), ticklabelsize = 25, ticksize = 15
                )
                Colorbar(
                    fig[4, 1][1, 2], p5; height = Relative(0.7), ticklabelsize = 25, ticksize = 15
                )
                Colorbar(
                    fig[4, 2][1, 2], p6; height = Relative(0.7), ticklabelsize = 25, ticksize = 15
                )
                rowgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)
                fig

                let
                    Yv = [y for x in ustrip.(dimensionalize(xvi[1], km, CD)), y in ustrip.(dimensionalize(xvi[2], km, CD))][:]
                    Y = [y for x in ustrip.(dimensionalize(xci[1], km, CD)), y in ustrip.(dimensionalize(xci[2], km, CD))][:]
                    fig = Figure(; size = (1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
                    ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
                    a3 = Axis(fig[2, 1]; aspect = 2 / 3, title = "τII")

                    scatter!(
                        ax1, ustrip.(dimensionalize((Array(thermal.T[2:(end - 1), 2:(end - 1)])), C, CD))[:],
                        ustrip.(dimensionalize(Y, km, CD))
                    )
                    lines!(
                        ax2, ustrip.(dimensionalize((Array((stokes.P))), MPa, CD))[:],
                        ustrip.(dimensionalize(Y, km, CD))
                    )
                    scatter!(
                        a3, ustrip.(dimensionalize(Array((stokes.τ.II)), MPa, CD))[:],
                        ustrip.(dimensionalize(Y, km, CD))
                    )

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end
            end
        end
    end

    # finalize_global_grid()

    return nothing
end

do_vtk = true # set to true to generate VTK files for ParaView
n  = 64
ar = 1
nx = n * ar
ny = n
figdir = "Extension_VS_$n"
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);

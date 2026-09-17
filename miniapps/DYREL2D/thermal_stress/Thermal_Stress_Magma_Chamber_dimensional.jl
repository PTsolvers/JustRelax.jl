# using CUDA

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

# const backend_JR = CUDABackend
const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
# @init_parallel_stencil(CUDA, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using Printf, Statistics, LinearAlgebra, GeoParams, GLMakie

const YEAR_SECONDS = 365.25 * 24 * 3600
const KM_SCALE = 1.0e3
const MPA_SCALE = 1.0e6
const KELVIN_OFFSET = 273.15
const CM_PER_M = 100.0

# -----------------------------------------------------------------------------------------
## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

@inline function tensile_cap_params(sinϕ::T, cosϕ::T, sinψ::T, C::T, pT::T) where {T}
    ps = -pT               # tensile limit on tension-positive axis
    k = sinϕ
    kf = sinψ
    c = C * cosϕ

    a = sqrt(one(T) + k * k)
    cosa = inv(a)
    sina = k * cosa

    py = (ps + c * cosa) / (one(T) - sina)
    R = py - ps

    pd = py - R * sina
    sd = c + k * pd

    pf = pd + kf * (c + k * pd)
    b = sqrt(one(T) + kf * kf)
    Rf = pf - ps

    norm_pf = hypot(pd - pf, sd)
    pdf = pf + Rf * (pd - pf) / norm_pf
    sdf = Rf * sd / norm_pf

    return (; k, kf, c, a, b, pd, sd, py, R, pf, Rf, pdf, sdf)
end

function draw_yield_surface!(ax, cp, xmax, xc, yc)
    # Using names from tensile_cap_params
    py, pd, sd, c, k = cp.py, cp.pd, cp.sd, cp.c, cp.k
    hlines!(ax, 0; color = :black, linewidth = 1)
    vlines!(ax, 0; color = :black, linewidth = 1)
    lines!(ax, [py, pd], [0.0, sd]; color = :red, linestyle = :dash, linewidth = 1.5)
    lines!(ax, [pd, xmax], [sd, c + k * xmax]; color = :red, linewidth = 2)
    return lines!(ax, xc, yc; color = :red, linewidth = 2)
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
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
        T[i, j + 1] = offset

    else # if top ≤ (depth) < bottom
        dTdZ = dTdz
        offset = offset
        T[i, j + 1] = abs(depth) * dTdZ + offset

    end

    return nothing
end

function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xci, sticky_air)
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
        T, δT, xc_anomaly, yc_anomaly, r_anomaly, xci..., sticky_air
    )
end

function linear_creep_models()
    creep_rock = LinearViscous(; η = 1.0e23)
    creep_magma = LinearViscous(; η = 1.0e18)
    creep_air = LinearViscous(; η = 1.0e18)
    return creep_rock, creep_magma, creep_air
end

function nonlinear_creep_models()
    creep_rock = DislocationCreep(; A = 1.67e-24, n = 3.5, E = 1.87e5, V = 0.0, r = 0.0, R = 8.3145)
    creep_magma = DislocationCreep(; A = 1.67e-24, n = 3.5, E = 1.87e5, V = 0.0, r = 0.0, R = 8.3145)
    # creep_magma = LinearViscous(; η = 1.0e18)
    creep_air = LinearViscous(; η = 1.0e19)
    return creep_rock, creep_magma, creep_air
end

function init_rheology(creep_rock, creep_magma, creep_air; is_compressible = true, steady_state = true)
    # plasticity setup
    do_DP = true          # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg = 1.0e19        # regularisation "viscosity" for Drucker-Prager
    Coh = 15.0e6          # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30.0              # friction angle
    Ψ = 15.0              # dilatancy angle
    G0 = 60.0e9           # elastic shear modulus
    G_magma = 60.0e9      # elastic shear modulus perturbation

    soft_C = NonLinearSoftening(; ξ₀ = Coh, Δ = Coh / 2) # softening law
    # pl = DruckerPrager_regularised(; C = Coh, ϕ = ϕ, η_vp = η_reg, Ψ = 0.0, softening_C = soft_C)        # plasticity
    pl = DruckerPragerCap(; C = Coh, ϕ = ϕ, η_vp = η_reg, pT = -1.5e6, Ψ = Ψ, softening_C = soft_C)        # plasticity
    if is_compressible == true
        el = SetConstantElasticity(; G = G0, ν = 0.25)            # elastic spring
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.25) # elastic spring
        β_rock = 6.0e-11
        β_magma = 6.0e-11
    else
        el = SetConstantElasticity(; G = G0, ν = 0.5)            # elastic spring
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.5) # elastic spring
        β_rock = inv(get_Kb(el))
        β_magma = inv(get_Kb(el_magma))
    end
    g = 9.81
    return rheology = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2650.0, α = 3.0e-5, T0 = KELVIN_OFFSET, β = β_rock),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050.0),
            Conductivity = ConstantConductivity(; k = 3.0),
            # LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6),
            ShearHeat = ConstantShearheating(1.0),
            CompositeRheology = CompositeRheology((creep_rock, el, pl)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
        ),

        #Name="Magma"
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2650.0, T0 = KELVIN_OFFSET, β = β_magma),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050.0),
            Conductivity = ConstantConductivity(; k = 1.5),
            # LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6),
            ShearHeat = ConstantShearheating(0.0),
            CompositeRheology = CompositeRheology((creep_magma, el_magma, pl)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_magma,
        ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(ρ = 0.0),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.0e6),
            Conductivity = ConstantConductivity(; k = 15.0),
            LatentHeat = ConstantLatentHeat(; Q_L = 0.0),
            ShearHeat = ConstantShearheating(0.0),
            CompositeRheology = CompositeRheology((creep_air,)),
            Gravity = ConstantGravity(; g = g),
        ),
    )

end

function main2D(igg; figdir = "Thermal_stresses", nx = 32, ny = 32, do_vtk = false)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    sticky_air = 1.5e3                                           # thickness of the sticky air layer
    L = 12.5e3 + sticky_air
    lx = L                                                        # domain length in x-direction
    ly = L                                                        # domain length in y-direction
    li = lx, ly                                                   # domain length in x- and y-direction
    ni = nx, ny                                                   # number of grid points in x- and y-direction
    di = @. li / ni                                               # grid step in x- and y-direction
    origin = 0.0e0, -ly + sticky_air  # origin coordinates of the domain
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                                           # nodes at the center and vertices of the cells
    εbg = 0.0                                                     # background strain rate
    #---------------------------------------------------------------------------------------

    # Physical Parameters
    # rheology = init_rheology(; is_compressible = true, steady_state = false)
    # creep_rock, creep_magma, creep_air = linear_creep_models()
    creep_rock, creep_magma, creep_air = nonlinear_creep_models()
    rheology = init_rheology(creep_rock, creep_magma, creep_air; is_compressible = true)
    rheology_inc = init_rheology(creep_rock, creep_magma, creep_air; is_compressible = false)
    cutoff_visc = (1.0e16, 1.0e24)
    dt = dt_max = 1.0e2 * YEAR_SECONDS                           # diffusive CFL timestep limiter

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 15
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Circular temperature anomaly -----------------------
    x_anomaly = lx * 0.5
    y_anomaly = -5.0e3                                    # origin of the small thermal anomaly
    r_anomaly = 1.5e3                                     # radius of perturbation
    anomaly = 750.0 + KELVIN_OFFSET                       # thermal perturbation
    init_phases!(pPhases, particles, x_anomaly, y_anomaly, r_anomaly, sticky_air, 0.0, 20.0e3)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Initialisation of thermal profile
    thermal = ThermalArrays(backend_JR, ni) # initialise thermal arrays and boundary conditions
    Ttop = 20.0 + KELVIN_OFFSET
    Tbot = 438.5625 + KELVIN_OFFSET
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
    )
    ∇Tz = (Ttop - Tbot) / (L - sticky_air)
    # dTdz = ((450 - 20) / 12.5e3)
    T1D = @. (∇Tz * (xci[2]) + Ttop) * (xci[2] < 0.0e0)
    T1D[xci[2] .≥ 0.0e0] .= Ttop
    thermal.T[:, 2:(end - 1)] .+= PTArray(backend_JR)(T1D')

    circular_perturbation!(
        thermal.T, anomaly, x_anomaly, y_anomaly, r_anomaly, xci, sticky_air
    )
    thermal_bcs!(thermal, thermal_bc)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni) # initialise stokes arrays with the defined regime
    # ----------------------------------------------------

    args = (; T = thermal.T, P = stokes.P, dt = dt)
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
        ϕ, phase_ratios, rheology, (T = thermal.T, P = stokes.P)
    )

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...) # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:5
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
        compute_lithostatic_pressure!(stokes.P, ρg[2], di[2], igg)
    end

    # Arguments for functions
    args = (; T = thermal.T, P = stokes.P, dt = dt, ΔT = thermal.ΔT)
    @copy thermal.Told thermal.T
    stokes.ε.xx .= 1.0e-20
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size = (1200, 900))
        ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
        scatter!(
            ax1,
            (Array(thermal.T[2:(end - 1), 2:(end - 1)]) .- KELVIN_OFFSET)[:],
            Y ./ KM_SCALE,
        )
        scatter!(
            ax2,
            # Array(ρg[2][:]),
            Array(stokes.P[:]) ./ MPA_SCALE,
            Y ./ KM_SCALE,
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

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

    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-3)

    # Stokes solver -----------------
    args = (; T = thermal.T, P = stokes.P, dt = Inf, ΔT = thermal.ΔT)

    while it < 250

        # Update buoyancy and viscosity -
        args = (; T = thermal.T, P = stokes.P, ΔT = thermal.ΔT, dt = Inf)

        # Stokes solver -----------------
        solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 100.0e3,
                nout = 50,
                rel_drop = 1.0e-2,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1.0e-2,
                viscosity_cutoff = cutoff_visc,
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
            grid;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
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

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (thermal.T,))
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        @views pT.data[pPhases.data .== 3.0] .= Ttop # sticky air particles have the temperature of the top boundary condition
        particle2centroid!(thermal.T, pT, particles)
        thermal_bcs!(thermal, thermal_bc)
        thermal.ΔT .= thermal.T .- thermal.Told

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 5) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)
            t_dim = t / YEAR_SECONDS
            t_Kyrs = t_dim / 1.0e3
            if igg.me == 0
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                if do_vtk
                    data_v = (;
                        τxy = Array(stokes.τ.xy) ./ MPA_SCALE,
                        εxy = Array(stokes.ε.xy),
                        Vx = Array(Vx_v) .* (CM_PER_M * YEAR_SECONDS),
                        Vy = Array(Vy_v) .* (CM_PER_M * YEAR_SECONDS),
                    )
                    data_c = (;
                        P = Array(stokes.P) ./ MPA_SCALE,
                        T = Array(thermal.T[2:(end - 1), 2:(end - 1)]) .- KELVIN_OFFSET,
                        τxx = Array(stokes.τ.xx) ./ MPA_SCALE,
                        τyy = Array(stokes.τ.yy) ./ MPA_SCALE,
                        τII = Array(stokes.τ.II) ./ MPA_SCALE,
                        εxx = Array(stokes.ε.xx),
                        εyy = Array(stokes.ε.yy),
                        εII = Array(stokes.ε.II),
                        εII_pl = Array(stokes.ε_pl.II),
                        η = Array(stokes.viscosity.η_vep),
                        η_vep = Array(stokes.viscosity.η),
                    )
                    velocity_v = (
                        Array(Vx_v) .* (CM_PER_M * YEAR_SECONDS),
                        Array(Vy_v) .* (CM_PER_M * YEAR_SECONDS),
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
                    title = L"ΔP [MPa]",
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
                # Plot temperature
                p1 = heatmap!(
                    ax1,
                    xvi[1] ./ KM_SCALE,
                    xvi[2] ./ KM_SCALE,
                    Array(thermal.T[2:(end - 1), 2:(end - 1)]) .- KELVIN_OFFSET;
                    colormap = :batlow,
                )
                # Plot effective viscosity
                p2 = heatmap!(
                    ax2,
                    xci[1] ./ KM_SCALE,
                    xci[2] ./ KM_SCALE,
                    log10.(Array(stokes.viscosity.η_vep));
                    colormap = :glasgow,
                    colorrange = (log10(1.0e16), log10(1.0e24)),
                )
                arrows2d!(
                    ax2,
                    (xvi[1] ./ KM_SCALE)[1:5:(end - 1)],
                    (xvi[2] ./ KM_SCALE)[1:5:(end - 1)],
                    Array.(
                        (
                            (Array(Vx_v) .* (CM_PER_M * YEAR_SECONDS))[1:5:(end - 1), 1:5:(end - 1)],
                            (Array(Vy_v) .* (CM_PER_M * YEAR_SECONDS))[1:5:(end - 1), 1:5:(end - 1)],
                        )
                    )...,
                    lengthscale = 1 / max(
                        maximum(Array(Vx_v) .* (CM_PER_M * YEAR_SECONDS)),
                        maximum(Array(Vy_v) .* (CM_PER_M * YEAR_SECONDS))
                    ),
                    color = :red,
                )
                # Plot Pressure difference
                p3 = heatmap!(
                    ax3,
                    xci[1] ./ KM_SCALE,
                    xci[2] ./ KM_SCALE,
                    Array(stokes.P .- P_init) ./ MPA_SCALE;
                    colormap = :roma,
                )
                # Plot Pressure difference

                p4 = heatmap!(
                    ax4,
                    xci[1] ./ KM_SCALE,
                    xci[2] ./ KM_SCALE,
                    Array(thermal.T .- thermal.Told)[2:(end - 1), 2:(end - 1)],
                    colormap = :roma,
                )
                # Plot 2nd invariant of strain rate
                p5 = heatmap!(
                    ax5,
                    xci[1] ./ KM_SCALE,
                    xci[2] ./ KM_SCALE,
                    log10.(Array(stokes.ε.II));
                    colormap = :roma,
                )
                # Plot 2nd invariant of stress
                p6 = heatmap!(
                    ax6,
                    xci[1] ./ KM_SCALE,
                    xci[2] ./ KM_SCALE,
                    Array(stokes.τ.II) ./ MPA_SCALE;
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

                # ax4 plotting
                (; sinϕ, cosϕ, sinΨ) = rheology[1].CompositeRheology[1].elements[3]
                pTensile = abs(rheology[1].CompositeRheology[1].elements[3].pT.val / MPA_SCALE)
                Coh = rheology[1].CompositeRheology[1].elements[3].C.val / MPA_SCALE
                cp = tensile_cap_params(sinϕ.val, cosϕ.val, sinΨ.val, Coh, pTensile)
                xc_array = range(-pTensile, cp.pd; length = 100)
                yc_array = sqrt.(max.(0.0, cp.R^2 .- (collect(xc_array) .- cp.py) .^ 2))

                P_pts = vec(Array(stokes.P) ./ MPA_SCALE)
                τII_pts = vec(Array(stokes.τ.II) ./ MPA_SCALE)
                xmax_plot = maximum(P_pts) + 0.5

                figCap = Figure(; size = (2000, 1800), createmissing = true)
                ax = Axis(figCap[1, 1]; aspect = 1, title = "Yield Surface")
                draw_yield_surface!(ax, cp, max(xmax_plot, 2.0), xc_array, yc_array)
                scatter!(ax, P_pts, τII_pts; color = (:blue, 0.5), markersize = 10)
                figsave = joinpath(figdir, @sprintf("YieldSurface_%06d.png", it))
                ax.xlabel = L"$$ Pressure [MPa]"
                ax.ylabel = L"$\tau_II$ [MPa]"
                save(figsave, figCap)
                figCap

                let
                    Yv = [y for x in xvi[1] ./ KM_SCALE, y in xvi[2] ./ KM_SCALE][:]
                    Y = [y for x in xci[1] ./ KM_SCALE, y in xci[2] ./ KM_SCALE][:]
                    fig = Figure(; size = (1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
                    ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
                    a3 = Axis(fig[2, 1]; aspect = 2 / 3, title = "τII")

                    scatter!(
                        ax1, (Array(thermal.T[2:(end - 1), 2:(end - 1)]) .- KELVIN_OFFSET)[:],
                        Y
                    )
                    lines!(
                        ax2, (Array(stokes.P) ./ MPA_SCALE)[:],
                        Y
                    )
                    scatter!(
                        a3, (Array(stokes.τ.II) ./ MPA_SCALE)[:],
                        Y
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

figdir = "Thermal_stresses_around_cooling_magma_NonLinear_Cap"
do_vtk = true # set to true to generate VTK files for ParaView
n = 64
ar = 1
nx = n * ar
ny = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);

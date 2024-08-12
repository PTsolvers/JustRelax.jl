push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using Test, Suppressor
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

import JustRelax.@cell
using GeoParams, SpecialFunctions

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

using Printf, LinearAlgebra, GeoParams, SpecialFunctions, CellArrays

# function to compute strain rate (compulsory)
@inline function custom_εII(a::CustomRheology, TauII; args...)
    η = custom_viscosity(a; args...)
    return TauII / η * 0.5
end

# function to compute deviatoric stress (compulsory)
@inline function custom_τII(a::CustomRheology, EpsII; args...)
    η = custom_viscosity(a; args...)
    return 2.0 * η * EpsII
end

# helper function (optional)
@inline function custom_viscosity(a::CustomRheology; P=0.0, T=273.0, depth=0.0, kwargs...)
    (; η0, Ea, Va, T0, R, cutoff) = a.args
    η = η0 * exp((Ea + P * Va) / (R * T) - Ea / (R * T0))
    correction = (depth ≤ 660e3) + (2740e3 ≥ depth > 660e3) * 1e1  + (depth > 2700e3) * 1e-1
    η = clamp(η * correction, cutoff...)
end

# HELPER FUNCTIONS ---------------------------------------------------------------
import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_j(z))
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j) function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    yr      = 3600*24*365.25
    dTdz    = (Tm-Tp)/2890e3
    zᵢ      = abs(z[j])
    Tᵢ      = Tp + dTdz*(zᵢ)
    time    = 100e6 * yr
    Ths     = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5)
    T[i, j] = min(Tᵢ, Ths)
    return
end

function circular_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if (((x[i] - xc))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i, j] *= δT / 100 + 1
        end
        return nothing
    end

    @parallel _circular_perturbation!(T, δT, xc, yc, r, xvi...)
end

function random_perturbation!(T, δT, xbox, ybox, xvi)

    @parallel_indices (i, j) function _random_perturbation!(T, δT, xbox, ybox, x, y)
        @inbounds if (xbox[1] ≤ x[i] ≤ xbox[2]) && (abs(ybox[1]) ≤ abs(y[j]) ≤ abs(ybox[2]))
            δTi = δT * (rand() - 0.5) # random perturbation within ±δT [%]
            T[i, j] *= δTi / 100 + 1
        end
        return nothing
    end

    @parallel (@idx size(T)) _random_perturbation!(T, δT, xbox, ybox, xvi...)
end

## WENO% particles script

# Initial pressure profile - not accurate
@parallel function init_P1!(P, ρg, z)
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

include("../miniapps/convection/WENO5/Layered_rheology.jl")
# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function thermal_convection2D(igg; ar=8, ny=16, nx=ny*8, thermal_perturbation = :circular)

    # Physical domain ------------------------------------
    ly           = 2890e3
    lx           = ly * ar
    origin       = 0.0, -ly                         # origin coordinates
    ni           = nx, ny                           # number of cells
    li           = lx, ly                           # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Weno model -----------------------------------------
    weno = WENO5(ni = ni .+ 1, method = Val(2)) # ni.+1 for Temp
    # ----------------------------------------------------

    # create rheology struct
    v_args = (; η0=5e20, Ea=200e3, Va=2.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
    creep = CustomRheology(custom_εII, custom_τII, v_args)

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    G0        = 70e9 # shear modulus
    cohesion  = 30e6
    friction  = asind(0.01)
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.5) # elastic spring
    β         = inv(get_Kb(el))

    rheology = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.1e3, β=β, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; Cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=9.81),
    )
    rheology_plastic = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.5e3, β=β, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; Cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el, pl)),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=9.81),
    )
    # heat diffusivity
    κ            = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].Cp * rheology.Density[1].ρ0)).val
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    adiabat     = 0.3 # adiabatic gradient
    Tp          = 1900
    Tm          = Tp + adiabat * 2890
    Tmin, Tmax  = 300.0, 3.5e3
    @parallel init_T!(thermal.T, xvi[2], κ, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal, thermal_bc)
    # Temperature anomaly
    if thermal_perturbation == :random
        δT          = 5.0               # thermal perturbation (in %)
        random_perturbation!(thermal.T, δT, (lx*1/8, lx*7/8), (-2000e3, -2600e3), xvi)

    elseif thermal_perturbation == :circular
        δT          = 10.0              # thermal perturbation (in %)
        xc, yc      = 0.5*lx, -0.75*ly  # center of the thermal anomaly
        r           = 150e3             # radius of perturbation
        circular_perturbation!(thermal.T, δT, xc, yc, r, xvi)
    end
    @views thermal.T[:, 1]   .= Tmax
    @views thermal.T[:, end] .= Tmin
    update_halo!(thermal.T)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.8 / √2.1)

    # Buoyancy forces
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    ρg               = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        compute_ρg!(ρg[2], rheology, args)
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    # Rheology
    viscosity_cutoff = (1e16, 1e24)
    compute_viscosity!(stokes, args, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend_JR, rheology, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip   = (left = true , right = true , top = true , bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # ----------------------------------------------------


    # WENO arrays
    T_WENO  = @zeros(ni.+1)
    Vx_v    = @zeros(ni.+1...)
    Vy_v    = @zeros(ni.+1...)

    # Time loop
    t, it = 0.0, 0
    local iters
    while it < 5

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        compute_ρg!(ρg[end], rheology, (T=thermal.Tc, P=stokes.P))
        compute_viscosity!(
            stokes, args, rheology, viscosity_cutoff
        )
        # ------------------------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax          = 150e3,
                nout             = 1e3,
                viscosity_cutoff = viscosity_cutoff
            )
        );
        dt = compute_dt(stokes, di, dt_diff, igg)
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
                igg     = igg,
                iterMax = 10e3,
                nout    = 1e2,
                verbose = true
            ),
        )

        # Weno advection
        T_WENO .= @views thermal.T[2:end-1, :]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        @views thermal.T[2:end-1, :] .= T_WENO
        # ------------------------------

        it += 1
        t += dt

    end


    # finalize_global_grid(; finalize_MPI=true)

    return iters
end


## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=8, ny=16, nx=ny*8)

    # Physical domain ------------------------------------
    ly           = 700e3            # domain length in y
    lx           = ly * ar          # domain length in x
    ni           = nx, ny           # number of cells
    li           = lx, ly           # domain length in x- and y-
    di           = @. li / ni       # grid step in x- and -y
    origin       = 0.0, -ly         # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_plastic = true)
    κ            = (10 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Weno model -----------------------------------------
    weno = WENO5(ni=(nx,ny).+1, method=Val{2}()) # ni.+1 for Temp
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles           = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly       = lx/2   # origin of thermal anomaly
    yc_anomaly       = -610e3 # origin of thermal anomaly
    r_anomaly        = 25e3   # radius of perturbation
    init_phases!(pPhases, particles, lx; d=abs(yc_anomaly), r=r_anomaly)
    phase_ratios     = PhaseRatio(backend_JR, ni, length(rheology))
    phase_ratios_center!(phase_ratios, particles, grid, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.1 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend_JR, ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2])
    thermal_bcs!(thermal, thermal_bc)

    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P1!(stokes.P, ρg[2], xci[2])
    end

    args             = (; T = thermal.Tc, P = stokes.P, dt = dt, ΔTc = thermal.ΔTc)
    # Rheology
    viscosity_cutoff = (1e16, 1e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    (; η, η_vep) = stokes.viscosity

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # WENO arrays
    T_WENO  = @zeros(ni.+1)
    Vx_v    = @zeros(ni.+1...)
    Vy_v    = @zeros(ni.+1...)
    # Time loop
    t, it = 0.0, 0
    while it < 5 # run only for 5 Myrs

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=dt, ΔTc = thermal.ΔTc)
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        compute_viscosity!(
            stokes, phase_ratios, args, rheology, viscosity_cutoff
        )
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
                nout             = 1e3,
                viscosity_cutoff = viscosity_cutoff
            )
        )
        tensor_invariant!(stokes.ε)
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
            kwargs = (
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 1e2,
                verbose = true
            ),
        )
        T_WENO .= thermal.T[2:end-1, :]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        thermal.T[2:end-1, :] .= T_WENO
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_WENO,), xvi)
        # update phase ratios
        phase_ratios_center!(phase_ratios, particles, grid, pPhases)
        @show it += 1
        t        += dt

    end

    return iters
end


@testset "WENO5 2D" begin
    @suppress begin
        n = 32
        nx = n
        ny = n
        igg = if !(JustRelax.MPI.Initialized())
            IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
        else
            igg
        end

        iters = thermal_convection2D(igg; ar=8, ny=ny, nx=nx, thermal_perturbation = :circular)
        @test passed = iters.err_evo1[end] < 1e-4

        iters = main2D(igg; ar=8, ny=ny, nx=nx)
        @test passed = iters.err_evo1[end] < 1e-4
    end
end

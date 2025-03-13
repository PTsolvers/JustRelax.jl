using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO


const backend = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using GeoParams, GLMakie, SpecialFunctions

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
@inline function custom_viscosity(a::CustomRheology; P = 0.0, T = 273.0, depth = 0.0, kwargs...)
    (; η0, Ea, Va, T0, R, cutoff) = a.args
    η = η0 * exp((Ea + P * Va) / (R * T) - Ea / (R * T0))
    correction = (depth ≤ 660.0e3) + (2740.0e3 ≥ depth > 660.0e3) * 1.0e1 + (depth > 2700.0e3) * 1.0e-1
    return η = clamp(η * correction, cutoff...)
end

# HELPER FUNCTIONS ---------------------------------------------------------------
import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_j(z))
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j) function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    yr = 3600 * 24 * 365.25
    dTdz = (Tm - Tp) / 2890.0e3
    zᵢ = abs(z[j])
    Tᵢ = Tp + dTdz * (zᵢ)
    time = 100.0e6 * yr
    Ths = Tmin + (Tm - Tmin) * erf((zᵢ) * 0.5 / (κ * time)^0.5)
    T[i, j] = min(Tᵢ, Ths)
    return
end

function circular_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, xc, yc, r, x, y)
        if (((x[i] - xc))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i + 1, j] *= δT / 100 + 1
        end
        return nothing
    end
    nx, ny = size(T)
    return @parallel (1:(nx - 2), 1:ny) _circular_perturbation!(T, δT, xc, yc, r, xvi...)
end

function random_perturbation!(T, δT, xbox, ybox, xvi)

    @parallel_indices (i, j) function _random_perturbation!(T, δT, xbox, ybox, x, y)
        if (xbox[1] ≤ x[i] ≤ xbox[2]) && (abs(ybox[1]) ≤ abs(y[j]) ≤ abs(ybox[2]))
            δTi = δT * (rand() - 0.5) # random perturbation within ±δT [%]
            T[i + 1, j] *= δTi / 100 + 1
        end
        return nothing
    end
    nx, ny = size(T)
    return @parallel (1:(nx - 2), 1:ny) _random_perturbation!(T, δT, xbox, ybox, xvi...)
end

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function thermal_convection2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", thermal_perturbation = :circular)

    # Physical domain ------------------------------------
    ly = 2890.0e3
    lx = ly * ar
    origin = 0.0, -ly                         # origin coordinates
    ni = nx, ny                           # number of cells
    li = lx, ly                           # domain length in x- and y-
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Weno model -----------------------------------------
    weno = WENO5(backend, Val(2), ni .+ 1) # ni.+1 for Temp
    # ----------------------------------------------------

    # create rheology struct
    v_args = (; η0 = 5.0e20, Ea = 200.0e3, Va = 2.6e-6, T0 = 1.6e3, R = 8.3145, cutoff = (1.0e16, 1.0e25))
    creep = CustomRheology(custom_εII, custom_τII, v_args)

    # Physical properties using GeoParams ----------------
    η_reg = 1.0e16
    G0 = 70.0e9 # shear modulus
    cohesion = 30.0e6
    friction = asind(0.01)
    pl = DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    el = SetConstantElasticity(; G = G0, ν = 0.5) # elastic spring
    β = inv(get_Kb(el))

    rheology = SetMaterialParams(;
        Name = "Mantle",
        Phase = 1,
        Density = PT_Density(; ρ0 = 3.1e3, β = β, T0 = 0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; Cp = 1.2e3),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((creep, el)),
        Elasticity = el,
        Gravity = ConstantGravity(; g = 9.81),
    )
    rheology_plastic = SetMaterialParams(;
        Name = "Mantle",
        Phase = 1,
        Density = PT_Density(; ρ0 = 3.5e3, β = β, T0 = 0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; Cp = 1.2e3),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((creep, el, pl)),
        Elasticity = el,
        Gravity = ConstantGravity(; g = 9.81),
    )
    # heat diffusivity
    κ = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].Cp * rheology.Density[1].ρ0)).val
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    adiabat = 0.3 # adiabatic gradient
    Tp = 1900
    Tm = Tp + adiabat * 2890
    Tmin, Tmax = 300.0, 3.5e3
    @parallel init_T!(thermal.T, xvi[2], κ, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal, thermal_bc)
    # Temperature anomaly
    if thermal_perturbation == :random
        δT = 5.0               # thermal perturbation (in %)
        random_perturbation!(thermal.T, δT, (lx * 1 / 8, lx * 7 / 8), (-2000.0e3, -2600.0e3), xvi)

    elseif thermal_perturbation == :circular
        δT = 10.0              # thermal perturbation (in %)
        xc, yc = 0.5 * lx, -0.75 * ly  # center of the thermal anomaly
        r = 150.0e3             # radius of perturbation
        circular_perturbation!(thermal.T, δT, xc, yc, r, xvi)
    end
    @views thermal.T[:, 1] .= Tmax
    @views thermal.T[:, end] .= Tmin
    update_halo!(thermal.T)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, CFL = 0.8 / √2.1)

    # Buoyancy forces
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    ρg = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        compute_ρg!(ρg[2], rheology, args)
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    # Rheology
    viscosity_cutoff = (1.0e16, 1.0e24)
    compute_viscosity!(stokes, args, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 1.0e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # ----------------------------------------------------

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    take(figdir)
    # ----------------------------------------------------

    # WENO arrays
    T_WENO = @zeros(ni .+ 1)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0
    local iters
    igg.me == 0 && println("Starting model")
    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 4.5e3

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_ρg!(ρg[end], rheology, (T = thermal.Tc, P = stokes.P))
        compute_viscosity!(
            stokes, args, rheology, viscosity_cutoff
        )
        # ------------------------------

        igg.me == 0 && println("Starting stokes solver...")
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
                iterMax = 150.0e3,
                nout = 1.0e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        dt = compute_dt(stokes, di, dt_diff, igg)
        println("Rank $(igg.me) ...stokes solver finished")
        # ------------------------------

        # Thermal solver ---------------
        igg.me == 0 && println("Starting thermal solver...")
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
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            ),
        )
        igg.me == 0 && println("...thermal solver finished")
        # Weno advection
        T_WENO .= @views thermal.T[2:(end - 1), :]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        @views thermal.T[2:(end - 1), :] .= T_WENO
        # ------------------------------

        it += 1
        t += dt

        println("\n")
        println("Time step number $it")
        println("   time = $(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs, dt = $(dt / (1.0e6 * 3600 * 24 * 365.25)) Myrs")
        println("\n")

        # Plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            fig = Figure(size = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[3, 1], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig[4, 1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.T), colormap = :batlow)
            h2 = heatmap!(ax2, xci[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(stokes.V.Vy[2:(end - 1), :]), colormap = :batlow)
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.τ.II .* 1.0e-6), colormap = :batlow)
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(η_vep)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1, height = 100)
            Colorbar(fig[2, 2], h2, height = 100)
            Colorbar(fig[3, 2], h3, height = 100)
            Colorbar(fig[4, 2], h4, height = 100)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    finalize_global_grid()

    return nothing
end

figdir = "figs2D_test_weno"
ar = 8 # aspect ratio
n = 32
nx = n * ar - 2
ny = n - 2
thermal_perturbation = :circular
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...)
else
    igg
end
thermal_convection2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, thermal_perturbation = thermal_perturbation);

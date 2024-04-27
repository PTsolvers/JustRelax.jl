using JustRelax
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# setup ParallelStencil.jl environment
model = PS_Setup(:threads, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

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
    @all(P) = @all(ρg)*abs(@all_j(z))
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

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function thermal_convection2D(; ar=8, ny=16, nx=ny*8, figdir="figs2D", thermal_perturbation = :circular)

    # initialize MPI
    igg = IGG(init_global_grid(nx, ny, 0; init_MPI = JustRelax.MPI.Initialized() ? false : true)...)

    # Physical domain ------------------------------------
    ly           = 2890e3
    lx           = ly * ar
    origin       = 0.0, -ly                         # origin coordinates
    ni           = nx, ny                           # number of cells
    li           = lx, ly                           # domain length in x- and y-
    di           = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
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
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    adiabat     = 0.3 # adiabatic gradient
    Tp          = 1900
    Tm          = Tp + adiabat * 2890
    Tmin, Tmax  = 300.0, 3.5e3
    @parallel init_T!(thermal.T, xvi[2], κ, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal.T, thermal_bc)
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
    @parallel (@idx ni) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.8 / √2.1)
    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    for _ in 1:2
        @parallel (@idx ni) compute_ρg!(ρg[2], rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    # Rheology
    η               = @ones(ni...)
    depth           = PTArray([y for x in xci[1], y in xci[2]])
    args            = (; T = thermal.Tc, P = stokes.P, depth = depth, dt = dt, ΔTc = thermal.ΔTc)
    viscosity_cutoff = 1e18, 1e23
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, @strain(stokes)..., args, rheology, viscosity_cutoff
    )
    η_vep           = deepcopy(η)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip   = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy)
    # ----------------------------------------------------

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig0 = let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y =  [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        lines!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        lines!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, -2890, 0)
        ylims!(ax2, -2890, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # Time loop
    t, it = 0.0, 0
    local iters
    while (t / (1e6 * 3600 * 24 * 365.25)) < 4.5e3
        # Stokes solver ----------------
        args = (; T = thermal.Tc, P = stokes.P, depth = depth, dt=dt, ΔTc = thermal.ΔTc)
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            rheology,
            args,
            dt,
            igg;
            iterMax=50e3,
            nout=1e3,
            viscosity_cutoff = viscosity_cutoff
        );
        dt = compute_dt(stokes, di, dt_diff, igg)
        # ------------------------------

        # Thermal solver ---------------
        args_T = (; P=stokes.P)
        solve!(
            thermal,
            thermal_bc,
            stokes,
            rheology,
            args_T,
            di,
            dt
        )
        # ------------------------------

        it += 1
        t += dt

        println("\n")
        println("Time step number $it")
        println("   time = $(t/(1e6 * 3600 * 24 *365.25)) Myrs, dt = $(dt/(1e6 * 3600 * 24 *365.25)) Myrs")
        println("\n")

        # Plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            fig = Figure(size = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig[4,1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.τ.II.*1e-6) , colormap=:batlow)
            h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1, height=100)
            Colorbar(fig[2,2], h2, height=100)
            Colorbar(fig[3,2], h3, height=100)
            Colorbar(fig[4,2], h4, height=100)
            save( joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

function run()
    figdir = "figs2D_test"
    ar     = 8 # aspect ratio
    n      = 128
    nx     = n*ar - 2
    ny     = n - 2

    thermal_convection2D(; figdir=figdir, ar=ar,nx=nx, ny=ny);
end

run()

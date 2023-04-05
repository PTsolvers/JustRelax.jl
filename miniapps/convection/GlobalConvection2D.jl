using JustRelax
# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

# HELPER FUNCTIONS ---------------------------------------------------------------
@parallel function update_buoyancy!(fz, T, ρ0gα)
    @all(fz) = ρ0gα * @all(T)
    return nothing
end

@parallel_indices (i, j) function computeViscosity!(η, v, args)
    @inline av(T) = 0.25 * (T[i + 1, j] + T[i + 2, j] + T[i + 1, j + 1] + T[i + 2, j + 1])

    @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T=av(args.T)))

    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j) function init_T!(T, z, k, Tm, Tp, Tmin, Tmax)
    dTdz = Tm - Tp
    @inbounds zᵢ = z[j]
    Tᵢ = Tp + dTdz * (1 - zᵢ)
    time = 5e-4
    Ths = Tmin + (Tm - Tmin) * erf((1 - zᵢ) * 0.5 / (k * time)^0.5)
    Tᵢ = min(Tᵢ, Ths)
    time = 1e-3
    Ths = Tmax - (Tmax + Tm) * erf(zᵢ * 0.5 / (k * time * 5)^0.5)
    @inbounds T[i, j] = max(Tᵢ, Ths)
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, r, xvi)
    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if (((x[i] - xc))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i, j] *= δT / 100 + 1
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end
# --------------------------------------------------------------------------------

function thermal_convection2D(; ar=8, ny=16, nx=ny * 8, figdir="figs2D")

    # Physical domain ------------------------------------
    CharUnits = GEO_units(; viscosity=1e23, length=2900km, temperature=1000K)
    ly = 2900km
    lx = ly * ar
    lx_nd = nondimensionalize(lx, CharUnits)
    ly_nd = nondimensionalize(ly, CharUnits)
    origin = 0.0, 0.0                         # Origin coordinates
    ni = nx, ny                           # number of cells
    li = lx_nd, ly_nd                     # domain length in x- and y-
    di = @. li / ni                       # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    η_reg = 1e1
    G0 = Inf                                                             # shear modulus
    pl = DruckerPrager_regularised(; C=1e4, ϕ=90.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el = SetConstantElasticity(; G=G0, ν=0.5)                            # elastic spring
    creep = ArrheniusType()                                                 # Arrhenius-like (T-dependant) viscosity
    Ra = 1e6
    # Define rheolgy struct
    rheology = SetMaterialParams(;
        Name="Mantle",
        Phase=1,
        Density=ConstantDensity(; ρ=1),
        HeatCapacity=ConstantHeatCapacity(; cp=1),
        Conductivity=ConstantConductivity(; k=1),
        CompositeRheology=CompositeRheology(el, creep),
        Elasticity=SetConstantElasticity(; G=G0, ν=0.5),
        Gravity=ConstantGravity(; g=Ra),
    )
    rheology_depth = SetMaterialParams(;
        Name="Mantle",
        Phase=1,
        Density=ConstantDensity(; ρ=1),
        HeatCapacity=ConstantHeatCapacity(; cp=1),
        Conductivity=ConstantConductivity(; k=1),
        CompositeRheology=CompositeRheology(el, creep, pl),
        Elasticity=SetConstantElasticity(; G=G0, ν=0.5),
        Gravity=ConstantGravity(; g=Ra),
    )
    κ = 1.0                          # heat diffusivity
    dt = dt_diff = 0.5 / 2.1 * min(di...)^2 / κ # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    args_T = (; stokes.P)
    # initialize thermal profile - Half space cooling
    k = 1.0
    Tm, Tp = 1900 / 2300, 1600 / 2300
    Tmin, Tmax = 0.12, 1.0
    @parallel init_T!(thermal.T, xvi[2], k, Tm, Tp, Tmin, Tmax)
    # Elliptical temperature anomaly 
    xc, yc = 0.5 * lx_nd, 0.15 * ly_nd  # origin of thermal anomaly
    δT = 10.0                   # thermal perturbation (in %)
    r = 0.1                  # radius of perturbation
    elliptical_perturbation!(thermal.T, δT, xc, yc, r, xvi)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=1 / √2.1)
    # Rheology
    η = @ones(ni...)
    args_η = (; T=thermal.T)
    @parallel (@idx ni) computeViscosity!(η, rheology.CompositeRheology[1], args_η) # init viscosity field
    η_vep = deepcopy(η)
    dt_elasticity = Inf
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    # ----------------------------------------------------

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    nt = 500
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        @parallel (@idx ni) computeViscosity!(η, rheology.CompositeRheology[1], args_η)
        @parallel update_buoyancy!(ρg[2], thermal.T, -Ra)
        # ------------------------------

        # Stokes solver ----------------
        iters = @edit solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            it > 3 ? rheology_depth : rheology, # do a few initial time-steps without plasticity to improve convergence
            dt_elasticity,
            iterMax=25e3,
            nout=1e3,
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        solve!(thermal, thermal_bc, stokes, rheology, args_T, di, dt)
        # ------------------------------

        @show it += 1
        t += dt

        # Plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            fig = Figure(; resolution=(900, 1600), title="t = $t")
            ax1 = Axis(fig[1, 1]; aspect=ar, title="T")
            ax2 = Axis(fig[2, 1]; aspect=ar, title="Vy")
            ax3 = Axis(fig[3, 1]; aspect=ar, title="τII")
            ax4 = Axis(fig[4, 1]; aspect=ar, title="η")
            h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T); colormap=:batlow)
            h2 = heatmap!(
                ax2, xci[1], xvi[2], Array(stokes.V.Vy[2:(end - 1), :]); colormap=:batlow
            )
            h3 = heatmap!(ax3, xci[1], xci[2], Array(stokes.τ.II); colormap=:romaO)
            h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(η_vep)); colormap=:batlow)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[3, 2], h3)
            Colorbar(fig[4, 2], h4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

figdir = "figs2D"
ar = 8 # aspect ratio
n = 32
nx = n * ar - 2
ny = n - 2

thermal_convection2D(; figdir = figdir, ar = ar, nx = nx, ny = ny);

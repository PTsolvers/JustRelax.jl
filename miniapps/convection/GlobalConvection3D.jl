using JustRelax, Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions
# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 3)
environment!(model)

@parallel_indices (i, j, k) function update_buoyancy!(fz, T, ρ0gα)
    fz[i, j, k] =
        ρ0gα *
        0.125 *
        (
            T[i, j, k] +
            T[i + 1, j, k] +
            T[i, j + 1, k] +
            T[i + 1, j + 1, k] +
            T[i, j, k + 1] +
            T[i + 1, j, k + 1] +
            T[i, j + 1, k + 1] +
            T[i + 1, j + 1, k + 1]
        )

    return nothing
end

@parallel_indices (i, j, k) function computeViscosity!(η, v, args)
    Base.Base.@propagate_inbounds @inline av(T)
        0.125 * (
            T[i, j, k] +
            T[i + 1, j, k] +
            T[i, j + 1, k] +
            T[i + 1, j + 1, k] +
            T[i, j, k + 1] +
            T[i + 1, j, k + 1] +
            T[i, j + 1, k + 1] +
            T[i + 1, j + 1, k + 1]
        )
    

    @inbounds η[i, j, k] = computeViscosity_εII(v, 1.0, (; T=av(args.T)))

    return nothing
end

@parallel_indices (i, j, k) function _elliptical_perturbation!(
    T, δT, xc, yc, zc, r, x, y, z
)
    if (((x[i] - xc))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
        T[i, j, k] *= δT / 100 + 1
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)
    @parallel_indices (i, j, k) function _elliptical_perturbation!(
        T, δT, xc, yc, zc, r, x, y, z
    )
        @inbounds if (((x[i] - xc))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            T[i, j, k] *= δT / 100 + 1
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi...)
end

@parallel_indices (i, j, k) function init_T!(T, z, k, Tm, Tp, Tmin, Tmax)
    dTdz = Tm - Tp
    zᵢ = z[k]
    Tᵢ = Tp + dTdz * (1 - zᵢ)
    time = 5e-4
    Ths = Tᵢ * erf((1 - zᵢ) * 0.5 / (k * time)^0.5)
    Tᵢ = min(Tᵢ, Ths)
    time = 1e-3
    Ths = erf(zᵢ * 0.5 / (k * time * 5)^0.5)
    Ths *= (Tmax - Tᵢ)
    Ths = Tmax - Ths
    Tᵢ = max(Tᵢ, Ths)
    T[i, j, k] = max(Tᵢ, Tmin)
    return nothing
end

function thermal_convection3D(; ar=8, ny=16, nx=ny * 8, nz=ny * 8, figdir="figs3D")

    # Physical domain ------------------------------------
    CharUnits = GEO_units(; viscosity=1e23, length=2900km, temperature=1000K)
    lz = 2900km
    lx = lz * ar
    ly = lz * ar
    lx_nd = nondimensionalize(lx, CharUnits)
    ly_nd = nondimensionalize(ly, CharUnits)
    lz_nd = nondimensionalize(lz, CharUnits)
    ni = nx, ny, nz
    li = lx_nd, ly_nd, lz_nd               # domain length in x-, y-, and z-
    di = @. li / ni                        # grid step in x-, y-,y- and z-
    origin = 0.0, 0.0, 0.0                     # Origin coordinates
    xci, xvi = lazy_grid(di, li, ni, origin=origin)  # nodes at the center and vertices of the cells
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=true)...) # init MPI
    ni_v = (nx - 2) * igg.dims[1], (ny - 2) * igg.dims[2], (nz - 2) * igg.dims[3]
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    G0 = Inf
    η_reg = 0.1
    pl = DruckerPrager_regularised(; C=1e4, ϕ=90.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el = SetConstantElasticity(; G=G0, ν=0.5)                           # elastic spring
    creep = ArrheniusType()                                                # Arrhenius-like (T-dependant) viscosity
    Ra = 1e7                                                            # Rayleigh number Ra = ρ0 * g * α * ΔT * lz_nd^3 / (η0* κ)
    # Define rheolgy struct
    rheology = SetMaterialParams(;
        Name="Mantle", # optional
        Phase=1,
        Density=ConstantDensity(; ρ=1.0),
        HeatCapacity=ConstantHeatCapacity(; cp=1.0),
        Conductivity=ConstantConductivity(; k=1.0),
        CompositeRheology=CompositeRheology(creep, el),
        Elasticity=el,
        Gravity=ConstantGravity(; g=Ra),
    )
    rheology_pl = SetMaterialParams(;
        Name="Mantle", # optional
        Phase=1,
        Density=ConstantDensity(; ρ=1.0),
        HeatCapacity=ConstantHeatCapacity(; cp=1.0),
        Conductivity=ConstantConductivity(; k=1.0),
        CompositeRheology=CompositeRheology(creep, el, pl),
        Elasticity=el,
        Gravity=ConstantGravity(; g=Ra),
    )
    κ = 1.0                          # heat diffusivity
    dt = dt_diff = 0.5 / 6.1 * min(di...)^3 / κ # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(ni) # allocate thermal arrays
    args_T = (;) # arguments for GeoParams
    # initialize thermal profile - Half space cooling
    k = 1.0
    Tm, Tp = 1900 / 2300, 1600 / 2300
    Tmin, Tmax = 0.12, 1.0
    @parallel init_T!(thermal.T, xvi[3], k, Tm, Tp, Tmin, Tmax)
    # Thermal anomaly - elliptical perturbation
    xc, yc, zc = 0.5 * lx_nd, 0.5 * ly_nd, 0.25 * lz_nd # center of perturbation
    δT = 10.0                              # thermal perturbation (in %)
    r = 0.05                             # radius of perturbation
    elliptical_perturbation!(thermal.T, δT, xc, yc, zc, r, xvi)
    # Boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false, front=true, back=true),
        periodicity=(left=false, right=false, top=false, bot=false, front=false, back=false),
    )
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=1 / √3)
    # Rheology
    η = @ones(ni...)
    # v               = CompositeRheology( (ArrheniusType(),) )
    args_η = (; dt=dt, T=thermal.T)
    @parallel (@idx ni) computeViscosity!(η, rheology.CompositeRheology[1], args_η) # init viscosity field
    η_vep = deepcopy(η)
    dt_elasticity = Inf
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    # Boundary conditions
    flow_bc = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true, front=true, back=true),
        no_slip=(left=false, right=false, top=false, bot=false, front=false, back=false),
        periodicity=(left=false, right=false, top=false, bot=false, front=false, back=false),
    )
    # ----------------------------------------------------

    # PLOTTING -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # mpi arrays
    Tc = @zeros(ni)
    Tg = zeros(ni_v...)
    ηg = zeros(ni_v...)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    nt = 3000
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        @parallel (@idx ni) computeViscosity!(η, v, args_η) # update viscosity field
        @parallel (@idx ni) update_buoyancy!(ρg[3], thermal.T, Ra)
        # -------------------------------

        # Stokes solver -----------------
        iters = solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            flow_bc,
            ρg,
            η,
            η_vep,
            it > 3 ? rheology_pl : rheology,
            dt_elasticity,
            igg;
            iterMax=250e3,
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
        if igg.me == 0 && it == 1 || rem(it, 25) == 0
            @parallel (1:nx, 1:ny, 1:nz) vertex2center!(Tc, thermal.T)
            gather!(Array(Tc[2:(end - 1), 2:(end - 1), 2:(end - 1)]), Tg)
            gather!(Array(η_vep[2:(end - 1), 2:(end - 1), 2:(end - 1)]), ηg)

            fig = Figure(; resolution=(900, 1800))
            ax1 = Axis(fig[1, 1]; aspect=ar, title="T")
            ax2 = Axis(fig[2, 1]; aspect=ar, title="η")
            ax3 = Axis(fig[3, 1]; aspect=ar, title="τII")
            ax4 = Axis(fig[4, 1]; aspect=ar, title="Vz")
            h1 = heatmap!(
                ax1,
                xci[1][2:(end - 1)],
                xci[2][2:(end - 1)],
                Array(Tg[nx ÷ 2, :, :]);
                colormap=:batlow,
            )
            h2 = heatmap!(
                ax2,
                xci[1][2:(end - 1)],
                xci[2][2:(end - 1)],
                Array(log10.(ηg[nx ÷ 2, :, :]));
                colormap=:batlow,
            )
            h3 = heatmap!(
                ax3,
                xci[1][2:(end - 1)],
                xci[2][2:(end - 1)],
                Array(stokes.τ.II[2:(end - 1), ny ÷ 2, :]);
                colormap=:batlow,
            )
            h4 = heatmap!(
                ax4,
                xci[1][2:(end - 1)],
                xvi[2][2:(end - 1)],
                Array(stokes.V.Vz[2:(end - 1), ny ÷ 2, :]);
                colormap=:batlow,
            )
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[3, 2], h3)
            Colorbar(fig[4, 2], h4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    finalize_global_grid(; finalize_MPI=true)

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

figdir = "figs3D"
ar = 3
n = 32
nx = (n - 2) * ar
ny = (n - 2) * ar
nz = (n - 2)

thermal_convection3D(; figdir=figdir, ar=ar, nx=nx, ny=ny, nz=nz);

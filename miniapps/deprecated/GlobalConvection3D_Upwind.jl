using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO


const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3) #or (CUDA, Float64, 3) or (AMDGPU, Float64, 3)

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
@inline function custom_viscosity(a::CustomRheology; P = 0.0, T = 273.0, depth = 0.0, kwargs...)
    (; η0, Ea, Va, T0, R, cutoff) = a.args
    η = η0 * exp((Ea + P * Va) / (R * T) - Ea / (R * T0))
    correction = (depth ≤ 660.0e3) + (2740.0e3 ≥ depth > 660.0e3) * 1.0e1 + (depth > 2700.0e3) * 1.0e-1
    return η = clamp(η * correction, cutoff...)
end

# HELPER FUNCTIONS ---------------------------------------------------------------
const idx_k = ParallelStencil.INDICES[3]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_k(z))
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j, k) function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    yr = 3600 * 24 * 365.25
    dTdz = (Tm - Tp) / 2890.0e3
    zᵢ = abs(z[k])
    Tᵢ = Tp + dTdz * (zᵢ)
    time = 100.0e6 * yr
    Ths = Tmin + (Tm - Tmin) * erf((zᵢ) * 0.5 / (κ * time)^0.5)
    T[i, j, k] = min(Tᵢ, Ths)
    return
end


function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, δT, xc, yc, zc, r, x, y, z)
        @inbounds if ((x[i] - xc)^2 + (y[j] - yc)^2 + (z[k] - zc))^2 ≤ r^2
            T[i, j, k] *= δT / 100 + 1
        end
        return nothing
    end

    return @parallel _elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi...)
end

function random_perturbation!(T, δT, xbox, ybox, zbox, xvi)

    @parallel_indices (i, j, k) function _random_perturbation!(T, δT, xbox, ybox, zbox, x, y, z)
        inbox =
            (xbox[1] ≤ x[i] ≤ xbox[2]) &&
            (ybox[1] ≤ y[j] ≤ ybox[2]) &&
            (abs(zbox[1]) ≤ abs(z[k]) ≤ abs(zbox[2]))
        @inbounds if inbox
            δTi = δT * (rand() - 0.5) # random perturbation within ±δT [%]
            T[i, j, k] *= δTi / 100 + 1
        end
        return nothing
    end

    return @parallel (@idx size(T)) _random_perturbation!(T, δT, xbox, ybox, zbox, xvi...)
end

Rayleigh_number(ρ, α, ΔT, κ, η0) = ρ * 9.81 * α * ΔT * 2890.0e3^3 * inv(κ * η0)

function thermal_convection3D(; ar = 8, nz = 16, nx = ny * 8, ny = nx, figdir = "figs3D", thermal_perturbation = :random)

    # initialize MPI
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = JustRelax.MPI.Initialized() ? false : true)...)

    # Physical domain ------------------------------------
    lz = 2890.0e3
    lx = ly = lz * ar
    origin = 0.0, 0.0, -lz                        # origin coordinates
    ni = nx, ny, nz                           # number of cells
    li = lx, ly, lz                           # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g())     # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # create rheology struct
    v_args = (; η0 = 5.0e20, Ea = 200.0e3, Va = 2.6e-6, T0 = 1.6e3, R = 8.3145, cutoff = (1.0e16, 1.0e25))
    creep = CustomRheology(custom_εII, custom_τII, v_args)

    # Physical properties using GeoParams ----------------
    η_reg = 1.0e18
    G0 = 70.0e9    # shear modulus
    cohesion = 30.0e6
    friction = asind(0.01)
    pl = DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    el = SetConstantElasticity(; G = G0, ν = 0.5)                                     # elastic spring
    β = inv(get_Kb(el))

    # Define rheolgy struct
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
    rheology_depth = SetMaterialParams(;
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
    dt = dt_diff = min(di...)^2 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
    )
    # initialize thermal profile - Half space cooling
    adiabat = 0.3 # adiabatic gradient
    Tp = 1900
    Tm = Tp + adiabat * 2890
    Tmin, Tmax = 300.0, 3.5e3
    # thermal.T  .= 1600.0
    @parallel init_T!(thermal.T, xvi[3], κ, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal, thermal_bc)
    # Elliptical temperature anomaly
    if thermal_perturbation == :random
        δT = 5.0              # thermal perturbation (in %)
        random_perturbation!(thermal.T, δT, (lx * 1 / 8, lx * 7 / 8), (ly * 1 / 8, ly * 7 / 8), (-2000.0e3, -2600.0e3), xvi)

    elseif thermal_perturbation == :circular
        δT = 15.0                      # thermal perturbation (in %)
        xc, yc, zc = 0.5 * lx, 0.5 * ly, -0.75 * lz  # origin of thermal anomaly
        r = 150.0e3                     # radius of perturbation
        elliptical_perturbation!(thermal.T, δT, xc, yc, zc, r, xvi)

    end

    @views thermal.T[:, :, 1] .= Tmax
    @views thermal.T[:, :, end] .= Tmin
    temperature2center!(thermal)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, CFL = 1.0 / √3.1)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    for _ in 1:2
        compute_ρg!(ρg[3], rheology, (T = thermal.Tc, P = stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3])
    end
    # Rheology
    args = (; T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf)
    compute_viscosity!(stokes, args, rheology, (1.0e18, 1.0e24))

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # ----------------------------------------------------

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # creata Paraview .vtu file for time series collections
    data_series = VTKDataSeries(joinpath(figdir, "full_simulation"), xci)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig0 = let
        Zv = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[:]), Zv ./ 1.0e3)
        scatter!(ax2, Array(log10.(stokes.viscosity.η[:])), Z ./ 1.0e3)
        ylims!(ax1, minimum(xvi[3]) ./ 1.0e3, 0)
        ylims!(ax2, minimum(xvi[3]) ./ 1.0e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # Time loop
    t, it = 0.0, 0
    local iters
    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 4.5e3

        # Update arguments needed to compute several physical properties
        # e.g. density, viscosity, etc -
        args = (; T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf)

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            rheology,
            args,
            Inf,
            igg;
            kwargs = (;
                iterMax = 150.0e3,
                nout = 2.0e3,
            )
        )

        println("starting non linear iterations")
        dt = compute_dt(stokes, di, dt_diff, igg)
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args_T,
            dt,
            di;
            kwargs = (
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            ),
        )
        # ------------------------------

        it += 1
        t += dt

        println("\n")
        println("Time step number $it")
        println("   time = $(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs, dt = $(dt / (1.0e6 * 3600 * 24 * 365.25)) Myrs")
        println("\n")

        # Plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            slice_j = Int(ny ÷ 2)

            fig = Figure(size = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Vz [m/s]")
            ax3 = Axis(fig[3, 1], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig[4, 1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[3] .* 1.0e-3, Array(thermal.T[:, slice_j, :]), colormap = :batlow)
            h2 = heatmap!(ax2, xci[1] .* 1.0e-3, xvi[3] .* 1.0e-3, Array(stokes.V.Vz[:, slice_j, :]), colormap = :batlow)
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(stokes.τ.II[:, slice_j, :] .* 1.0e-6), colormap = :batlow)
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep[:, slice_j, :])), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[3, 2], h3)
            Colorbar(fig[4, 2], h4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)

            # save vtk time series
            data_c = (; Temperature = Array(thermal.Tc), TauII = Array(stokes.τ.II), Density = Array(ρg[3] ./ 9.81))
            JustRelax.DataIO.append!(data_series, data_c, it, t)
        end
        # ------------------------------

    end

    return (ni = ni, xci = xci, li = li, di = di), thermal
end

figdir = "figs3D_test"
ar = 3 # aspect ratio
n = 32
nx = n * ar - 2
ny = nx
nz = n - 2
thermal_convection3D(; figdir = figdir, ar = ar, nx = nx, ny = ny, nz = nz);

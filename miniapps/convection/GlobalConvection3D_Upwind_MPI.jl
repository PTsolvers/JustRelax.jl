using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model = PS_Setup(:threads, Float64, 3)
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

# HELPER FUNCTIONS ---------------------
const idx_k = ParallelStencil.INDICES[3]
macro all_k(A)
    esc(:($A[$idx_k]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_k(z))
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j, k) function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    yr         = 3600*24*365.25
    dTdz       = (Tm-Tp)/2890e3
    zᵢ         = abs(z[k])
    Tᵢ         = Tp + dTdz*(zᵢ)
    time       = 100e6 * yr
    Ths        = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5)
    T[i, j, k] = min(Tᵢ, Ths)
    return
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, δT, xc, yc, zc, r, x, y, z)
        @inbounds if ((x[i]-xc )^2 + (y[j] - yc)^2 + (z[k] - zc))^2 ≤ r^2
            T[i, j, k] *= δT / 100 + 1
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi...)
end

function random_perturbation!(T, δT, xbox, ybox, zbox, xvi)

    @parallel_indices (i, j, k) function _random_perturbation!(T, δT, xbox, ybox, zbox, x, y, z)
        inbox =
            (xbox[1] ≤ x[i] ≤ xbox[2]) &&
            (ybox[1] ≤ y[j] ≤ ybox[2]) &&
            (abs(zbox[1]) ≤ abs(z[k]) ≤ abs(zbox[2]))
        @inbounds if inbox
            δTi         = δT * (rand() - 0.5) # random perturbation within ±δT [%]
            T[i, j, k] *= δTi / 100 + 1
        end
        return nothing
    end

    @parallel (@idx size(T)) _random_perturbation!(T, δT, xbox, ybox, zbox, xvi...)
end

Rayleigh_number(ρ, α, ΔT, κ, η0) = ρ * 9.81 * α * ΔT * 2890e3^3 * inv(κ * η0)

function thermal_convection3D(igg; ar=8, nz=16, nx=ny*8, ny=nx, figdir="figs3D", thermal_perturbation = :circular)

    # Physical domain ------------------------------------
    lz           = 2890e3
    lx = ly      = lz * ar
    origin       = 0.0, 0.0, -lz                        # origin coordinates
    ni           = nx, ny, nz                           # number of cells
    li           = lx, ly, lz                           # domain length in x- and y-
    di           = @. li / (nx_g(), ny_g(), nz_g())     # grid step in x- and -y
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # create rheology struct
    v_args = (; η0=5e20, Ea=200e3, Va=2.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e18, 1e24))
    creep = CustomRheology(custom_εII, custom_τII, v_args)

    # Physical properties using GeoParams ----------------
    η_reg     = 1e18
    G0        = 70e9    # shear modulus
    cohesion  = 30e6
    friction  = asind(0.01)
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.5)                                     # elastic spring
    β         = inv(get_Kb(el))

    # Define rheolgy struct
    rheology = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.1e3, β=β, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=9.81),
    )
    # rheology_depth = SetMaterialParams(;
    #     Name              = "Mantle",
    #     Phase             = 1,
    #     Density           = PT_Density(; ρ0=3.5e3, β=β, T0=0.0, α = 1.5e-5),
    #     HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
    #     Conductivity      = ConstantConductivity(; k=3.0),
    #     CompositeRheology = CompositeRheology((creep, el, pl)),
    #     Elasticity        = el,
    #     Gravity           = ConstantGravity(; g=9.81),
    # )
    # heat diffusivity
    κ            = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].cp * rheology.Density[1].ρ0)).val
    dt = dt_diff = min(di...)^2 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false, front=true, back=true),
        periodicity = (left = false, right = false, top = false, bot = false, front=false, back=false),
    )
    # initialize thermal profile - Half space cooling
    adiabat     = 0.3 # adiabatic gradient
    Tp          = 1900
    Tm          = Tp + adiabat * 2890
    Tmin, Tmax  = 300.0, 3.5e3
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[3], κ, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal.T, thermal_bc)
    # Elliptical temperature anomaly
    if thermal_perturbation == :random
        δT          = 5.0              # thermal perturbation (in %)
        random_perturbation!(thermal.T, δT, (lx*1/8, lx*7/8), (ly*1/8, ly*7/8), (-2000e3, -2600e3), xvi)

    elseif thermal_perturbation == :circular
        δT          = 1.0                      # thermal perturbation (in %)
        xc, yc, zc  = 0.5*lx, 0.5*ly, -0.75*lz  # origin of thermal anomaly
        r           = 150e3                     # radius of perturbation
        elliptical_perturbation!(thermal.T, δT, xc, yc, zc, r, xvi)

    end
    # @views thermal.T[:, :, 1]   .= Tmax
    # @views thermal.T[:, :, end] .= Tmin
    update_halo!(thermal.T)
    @parallel (@idx ni) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.5 / √3.1)
    # Buoyancy forces
    ρg              = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    for _ in 1:2
        @parallel (@idx ni) compute_ρg!(ρg[3], rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3])
    end
    # Rheology
    η               = @ones(ni...)
    depth           = PTArray([abs(z) for x in xci[1], y in xci[2], z in xci[3]])
    args            = (; T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf)
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0,  @strain(stokes)..., args, rheology, (1e18, 1e24)
    )
    η_vep           = deepcopy(η)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip   = (left=true , right=true , top=true , bot=true , front=true , back=true ),
        no_slip     = (left=false, right=false, top=false, bot=false, front=false, back=false),
        periodicity = (left=false, right=false, top=false, bot=false, front=false, back=false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # ----------------------------------------------------

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    take(figdir)
    # creata Paraview .vtu file for time series collections
    # data_series = VTKDataSeries(joinpath(figdir, "full_simulation"), xci)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig0 = let
        Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[:]), Zv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Z./1e3 )
        ylims!(ax1, -lz / 1e3, 0)
        ylims!(ax2, -lz / 1e3, 0)
        hideydecorations!(ax2)
        save("initial_profile_rank$(igg.me).png", fig)
        fig
    end

    # return nothing 

    # global arrays
    ## grid center arrays
    nx_v         = (nx - 2) * igg.dims[1]
    ny_v         = (ny - 2) * igg.dims[2]
    nz_v         = (nz - 2) * igg.dims[3]
    τII_v        = zeros(nx_v, ny_v, nz_v)
    η_vep_v      = zeros(nx_v, ny_v, nz_v)
    εII_v        = zeros(nx_v, ny_v, nz_v)
    T_v          = zeros(nx_v, ny_v, nz_v)
    τII_nohalo   = zeros(nx-2, ny-2, nz-2)
    η_vep_nohalo = zeros(nx-2, ny-2, nz-2)
    εII_nohalo   = zeros(nx-2, ny-2, nz-2)
    T_nohalo     = zeros(nx-2, ny-2, nz-2)
    xci_v        = LinRange(0, 1, nx_v), LinRange(0, 1, ny_v), LinRange(0, 1, nz_v)

    # ## grid vertices arrays
    # nx_vv        = nx_v + 1
    # ny_vv        = ny_v + 1
    # nz_vv        = nz_v + 1
    # T_v          = zeros(nx_vv, ny_vv, nz_vv)
    # T_nohalo     = zeros(nx + 1 - 2, ny + 1 - 2, nz + 1 - 2)
    # xvi_v        = LinRange(0, 1, nx_v+1), LinRange(0, 1, ny_v+1), LinRange(0, 1, nz_v+1)
  
    # Time loop
    t, it = 0.0, 0
    

        # # gather MPI arrays for plotting
        # @views τII_nohalo   .= Array(stokes.τ.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        # @views η_vep_nohalo .= Array(η_vep[2:end-1, 2:end-1, 2:end-1])       # Copy data to CPU removing the halo
        # @views εII_nohalo   .= Array(stokes.ε.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        # @views T_nohalo     .= Array(thermal.Tc[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        # gather!(τII_nohalo, τII_v)
        # gather!(η_vep_nohalo, η_vep_v)
        # gather!(εII_nohalo, εII_v)
        # gather!(T_nohalo, T_v)


    # # Plotting ---------------------
    # if igg.me == 0 
    #         slice_j = ny_v >>> 1

    #         fig = Figure(size = (1000, 1000), title = "t")
    #         ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]")
    #         ax2 = Axis(fig[2,1], aspect = ar, title = "log10(εII)")
    #         ax3 = Axis(fig[1,2], aspect = ar, title = "τII [MPa]")
    #         ax4 = Axis(fig[2,2], aspect = ar, title = "log10(η)")
    #         h1 = heatmap!(ax1, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(T_v[:, slice_j, :]),         colormap=:batlow)
    #         h2 = heatmap!(ax2, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(log10.(εII_v[:, slice_j, :])),   colormap=:batlow)
    #         h3 = heatmap!(ax3, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(τII_v[:, slice_j, :].*1e-6),     colormap=:batlow)
    #         h4 = heatmap!(ax4, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(log10.(η_vep_v[:, slice_j, :])), colormap=:batlow)
    #         hidexdecorations!(ax1)
    #         hidexdecorations!(ax2)
    #         hidexdecorations!(ax3)
    #         # Colorbar(fig[1,2], h1)
    #         # Colorbar(fig[2,2], h2)
    #         # Colorbar(fig[3,2], h3)
    #         # Colorbar(fig[4,2], h4)
    #         figname = "potato.png"
    #         save(figname, fig)

    #         # # save vtk time series
    #         # data_c = (; Temperature = Array(thermal.Tc),  TauII = Array(stokes.τ.II), Density=Array(ρg[3]./9.81))
    #         # JustRelax.DataIO.append!(data_series, data_c, it, t)
    # end
    
    # return nothing

    while it < 1 #(t / (1e6 * 3600 * 24 * 365.25)) < 4.5e3

        # Update arguments needed to compute several physical properties
        # e.g. density, viscosity, etc -
        args = (; T=thermal.Tc, P=stokes.P, depth=depth, dt=Inf)
        @parallel (@idx ni) compute_ρg!(ρg[3], rheology, (T=thermal.Tc, P=stokes.P))
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0,  @strain(stokes)..., args, rheology, (1e18, 1e24)
        )
        
        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            rheology,
            args,
            Inf,
            igg;
            iterMax=50e3,
            # iterMax=150e3,
            nout=1e3,
            viscosity_cutoff = (1e18, 1e24)
        );

        # Plot initial T and η profiles
        fig0 = let
            Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
            Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
            fig = Figure(size = (1200, 900))
            ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
            ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
            scatter!(ax1, Array(thermal.T[:]), Zv./1e3)
            scatter!(ax2, Array(log10.(η[:])), Z./1e3 )
            ylims!(ax1, -lz / 1e3, 0)
            ylims!(ax2, -lz / 1e3, 0)
            hideydecorations!(ax2)
            save("profile_rank$(igg.me).png", fig)
            fig
        end
        # dt = compute_dt(stokes, di, dt_diff, igg)
        # # ------------------------------

        # # Thermal solver ---------------
        # solve!(
        #     thermal,
        #     thermal_bc,
        #     stokes,
        #     rheology,
        #     args,
        #     di,
        #     dt
        # )
        # # ------------------------------

        it += 1
        t += dt

        if igg.me == 0 
            println("\n")
            println("Time step number $it")
            println("   time = $(t/(1e6 * 3600 * 24 *365.25)) Myrs, dt = $(dt/(1e6 * 3600 * 24 *365.25)) Myrs")
            println("\n")
        end

        # slice_j = ny >>> 1

        f, = heatmap(stokes.V.Vx[:, nx >>> 1, :], colormap=:batlow)
        save("potato_$(igg.me).png", f)
        break

        # fig = Figure(size = (1000, 1000), title = "t")
        # ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]")
        # ax2 = Axis(fig[2,1], aspect = ar, title = "log10(εII)")
        # h1 = heatmap!(ax1, xci[1].*1e-3, xci[3].*1e-3, Array(thermal.T[:, slice_j, :]),   colormap=:batlow)
        # h3 = heatmap!(ax2, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(η_vep[:, slice_j, :])), colormap=:batlow)
        # for a in (ax1, ax2)
        #     xlims(a, 0, lx)
        #     ylims(a, -lz, 0)
        # end
        # hidexdecorations!(ax1)
        # hidexdecorations!(ax2)
        # figname = joinpath(figdir, "Rank_$(igg.me)_$(it).png")
        # println("Saving $figname ...")
        # save(figname, fig)

        # if (it == 1 || rem(it, 5)) == 0
        # gather MPI arrays for plotting
        @views τII_nohalo   .= Array(stokes.τ.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(η_vep[2:end-1, 2:end-1, 2:end-1])       # Copy data to CPU removing the halo
        # @views η_vep_nohalo .= Array(η[2:end-1, 2:end-1, 2:end-1])       # Copy data to CPU removing the halo
        @views εII_nohalo   .= Array(stokes.ε.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views T_nohalo     .= Array(thermal.Tc[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(τII_nohalo, τII_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(εII_nohalo, εII_v)
        gather!(T_nohalo, T_v)

       
            # Plotting ---------------------
            if igg.me == 0 
                slice_j = ny_v >>> 1

                fig = Figure(size = (1000, 1000), title = "t")
                ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]")
                ax2 = Axis(fig[2,1], aspect = ar, title = "log10(εII)")
                ax3 = Axis(fig[1,2], aspect = ar, title = "τII [MPa]")
                ax4 = Axis(fig[2,2], aspect = ar, title = "log10(η)")
                h1 = heatmap!(ax1, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(T_v[:, slice_j, :]),         colormap=:batlow)
                h2 = heatmap!(ax2, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(log10.(εII_v[:, slice_j, :])),   colormap=:batlow)
                h3 = heatmap!(ax3, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(τII_v[:, slice_j, :].*1e-6),     colormap=:batlow)
                h4 = heatmap!(ax4, xci_v[1].*1e-3, xci_v[3].*1e-3, Array(log10.(η_vep_v[:, slice_j, :])), colormap=:batlow)
                hidexdecorations!(ax1)
                hidexdecorations!(ax2)
                hidexdecorations!(ax3)
                figname = joinpath(figdir, "$(it).png")
                println("Saving $figname ...")
                save(figname, fig)

                # # save vtk time series
                # data_c = (; Temperature = Array(thermal.Tc),  TauII = Array(stokes.τ.II), Density=Array(ρg[3]./9.81))
                # JustRelax.DataIO.append!(data_series, data_c, it, t)
            end
        # end
        # ------------------------------

    end
    
    finalize_global_grid()

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

figdir = "figs3D_test"
ar     = 1 # aspect ratio
n      = 16+2
nx     = n * ar
ny     = 50#nx
nz     = n + n-2
nz     = ny 
igg    = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true, select_device = false)...)
else
    igg
end

thermal_convection3D(igg; figdir=figdir, ar=ar,nx=nx, ny=ny, nz=nz);

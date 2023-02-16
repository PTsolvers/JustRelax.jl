using JustRelax
# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

using Statistics, LinearAlgebra, Parameters
using ParallelStencil.FiniteDifferences2D
using GeophysicalModelGenerator, StencilInterpolations, StaticArrays

include("MTK/Utils.jl")
include("MTK/Advection.jl")
include("MTK/Dikes.jl")


# HELPER FUNCTIONS ---------------------------------------------------------------
@parallel function update_buoyancy!(fz, T, ρ0gα)
    @all(fz) = ρ0gα * @all(T)
    return nothing
end

@parallel function compute_maxRatio!(Musτ2::AbstractArray, Musτ::AbstractArray)
    @inn(Musτ2) = @maxloc(Musτ) / @minloc(Musτ)
    return nothing
end

@parallel_indices (i, j) function compute_ρg!(A, MatParam, Phases, args)
    A[i, j] =
        compute_density(MatParam, Phases[i, j],  ntuple_idx(args, i, j)) *
        compute_gravity(MatParam, Phases[i, j])
    return nothing
end

@parallel function computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
    # We assume that η is f(ϕ), following Deubelbeiss, Kaus, Connolly (2010) EPSL 
    # factors for hexagons
    @all(η) = min(
        η_f * (1.0 - S * (1.0 - @all(ϕ)))^mfac,
        η_s, # upper cutoff
    )
    return nothing
end

@parallel_indices (i, j) function initViscosity!(η, Phases, MatParam)
    @inbounds η[i, j] = MatParam[Phases[i, j]].CreepLaws[1].η.val
    return nothing
end

# --------------------------------------------------------------------------------

function MTK2D(; ar=2, ny=16, nx=ny*8, figdir="figs2D")

    # Physical domain ------------------------------------
    CharUnits  = GEO_units(; length=10km, viscosity=1e20Pa * s)        # Characteristic dimensions
    ly       = 40km
    lx       = ly * ar
    lx_nd    = nondimensionalize(lx, CharUnits)
    ly_nd    = nondimensionalize(ly, CharUnits)   
    origin   = (-lx_nd / 2, -ly_nd)             # Origin coordinates
    ni       = nx, ny                           # number of cells
    li       = lx_nd, ly_nd                     # domain length in x- and y-
    di       = @. li / ni                       # grid step in x- and -y
    xci, xvi = lazy_grid(di, li; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    creep = LinearViscous(; η=1e16Pa * s)
    rheology = (
        SetMaterialParams(;
            Name="Rock",
            Phase=1,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            HeatCapacity=ConstantHeatCapacity(; cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=1.5Watt / K / m),
            CreepLaws=LinearViscous(; η=1e22Pa * s),
            CompositeRheology = CompositeRheology((creep,)),
            Melting=MeltingParam_Caricchi(),
            Elasticity=ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
            CharDim=CharUnits,
        ),
        SetMaterialParams(;
            Name="Magma",
            Phase=2,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            HeatCapacity=ConstantHeatCapacity(; cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=1.5Watt / K / m),
            CompositeRheology = CompositeRheology((creep,)),
            CreepLaws=LinearViscous(; η=1e16Pa * s),
            Melting=MeltingParam_Caricchi(),
            Elasticity=ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
            CharDim=CharUnits,
        ),
    )

    ΔT = nondimensionalize(600C, CharUnits)               # initial temperature perturbation K
    cp = rheology[2].HeatCapacity[1].cp.val
    η  = rheology[2].CreepLaws[1].η.val
    ρ0 = rheology[2].Density[1].ρ0.val
    k0 = rheology[2].Conductivity[1].k.val
    κ  = k0 / (ρ0 * cp)
    g  = rheology[2].Gravity[1].g.val
    α  = rheology[2].Density[1].α.val
    Ra = ρ0 * g * α * ΔT * ly_nd^3 / (η * κ)
    dt = dt_diff = 0.5 / 6.1 * min(di...)^3 / κ # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Set the Phases distribution ------------------------
    phase_v   = ones(Int64, ni.+1...)         # constant for now
    phase_c   = ones(Int64, ni...)         # constant for now
    a_ellipse = ustrip(nondimensionalize(35km, CharUnits))
    b_ellipse = ustrip(nondimensionalize(15km, CharUnits))
    z_c       = ustrip(nondimensionalize(-20km, CharUnits))
    for i in CartesianIndices(phase_v)
        x, z = xvi[1][i[1]], xvi[2][i[2]]
        if (((x - 0) / a_ellipse)^2 + ((z - z_c) / b_ellipse)^2) < 1.0
            phase_v[i] = 2
        end
    end
    for i in CartesianIndices(phase_c)
        x, z = xci[1][i[1]], xci[2][i[2]]
        if (((x - 0) / a_ellipse)^2 + ((z - z_c) / b_ellipse)^2) < 1.0
            phase_c[i] = 2
        end
    end
    # ----------------------------------------------------
    
    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true , right = true , top = false, bot = false), 
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    # args_T = (; P=stokes.P, thermal.T)
    # initialize thermal profile
    GeoTherm = -(ΔT - nondimensionalize(0C, CharUnits)) / li[2]
    w = 1e-2 * li[2]    # initial perturbation standard deviation, m
    thermal.T .= PTArray([
        0.5 * ΔT * exp(-(xvi[1][ix] / w)^2 - ((xvi[2][iy] + 0.5 * li[2]) / w)^2) +
        xvi[2][iy] * GeoTherm +
        nondimensionalize(0C, CharUnits) for ix in 1:ni[1]+1, iy in 1:ni[2]+1
    ])
    Tnew_cpu = Matrix{Float64}(undef, (ni.+1)...)
    @views thermal.T[:, 1] .= ΔT
    @views thermal.T[:, end] .= nondimensionalize(0C, CharUnits)
    @copy  thermal.Told thermal.T
    @copy  Tnew_cpu Array(thermal.T)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL=1 / √2.1)
    # Rheology
    η               = @ones(ni...)
    args_η          = (; T=thermal.T)
    @parallel (@idx ni) initViscosity!(η, phase_c, rheology) # init viscosity field
    η_vep           = deepcopy(η)
    dt_elasticity   = Inf

    # to be added in GP...
    ϕ       = @zeros(ni...) # melt fraction
    S, mfac = 1.0, -2.8 # factors for hexagons
    η_f     = rheology[2].CreepLaws[1].η.val    # melt viscosity
    η_s     = rheology[1].CreepLaws[1].η.val    # solid viscosity

    # Buoyancy forces
    ρg              = @zeros(ni...), @zeros(ni...)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip   = (left=true, right=true, top=true, bot=true), 
    )
    # ----------------------------------------------------

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    nt    = 500
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        @parallel computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
        @parallel (@idx ni) compute_ρg!(ρg[2], rheology, phase_c, (T=thermal.T, P=stokes.P))
        # ------------------------------

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_v,
            phase_c,
            it > 3 ? rheology_depth : rheology, # do a few initial time-steps without plasticity to improve convergence
            dt,
            iterMax=150e3,
            nout=1e3,
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            stokes,
            phase_v,
            rheology,
            (; P=stokes.P, T=thermal.T),
            di,
            dt
        )
        # ------------------------------

        @show it += 1
        t += dt

        # Plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            fig = Figure(resolution = (900, 1600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII")
            ax4 = Axis(fig[4,1], aspect = ar, title = "η")
            h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1], xvi[2], Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1], xci[2], Array(stokes.τ.II) , colormap=:romaO) 
            h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(η_vep)) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            Colorbar(fig[4,2], h4)
            fig
            save( joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

figdir = "figs2D"
ar     = 2 # aspect ratio
n      = 96
nx, ny = 96 * ar - 2, 1 * 96 - 2  # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf

# MTK2D(; figdir=figdir, ar=ar,nx=nx, ny=ny);

@parallel_indices (i, j) function foo!(P, P_old, RP, ∇V, η, rheology::NTuple{N, MaterialParams}, phase, dt, r, θ_dτ) where N
    RP[i, j] = -∇V[i, j] - (P[i, j] - P_old[i, j]) / (get_Kb(rheology, phase[i,j]) * dt)
    P[i, j] = P[i, j] + RP[i, j] / (1.0 / (r / θ_dτ * η[i, j]) + 1.0 / (get_Kb(rheology, phase[i,j])  * dt))
    return nothing
end

@parallel (@idx ni) foo!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                rheology,
                phase_c,
                dt,
                r,
                θ_dτ,
            )

get_Kb(v1)

v1=rheology[1]
v=rheology
get_Kb(rheology, phase_c[1,1]) *dt
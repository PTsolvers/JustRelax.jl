using JustRelax, Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions
# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 3)
environment!(model)

@parallel_indices (i, j, k) function update_buoyancy!(fz, T, ρ0gα)
    
    fz[i, j, k] = ρ0gα * 0.125 * (
        T[i, j, k  ] + T[i+1, j, k  ] + T[i, j+1, k  ] + T[i+1, j+1, k  ] +
        T[i, j, k+1] + T[i+1, j, k+1] + T[i, j+1, k+1] + T[i+1, j+1, k+1]
    )
    
    return nothing
end

@parallel_indices (i, j, k) function computeViscosity!(η, v, args)

    Base.Base.@propagate_inbounds @inline av(T) = 0.125* (
        T[i, j, k  ] + T[i+1, j, k  ] + T[i, j+1, k  ] + T[i+1, j+1, k  ] +
        T[i, j, k+1] + T[i+1, j, k+1] + T[i, j+1, k+1] + T[i+1, j+1, k+1]
    )

    @inbounds η[i, j, k] = computeViscosity_εII(v, 1.0, (; T = av(args.T)))

    return nothing
end

@parallel_indices (i, j, k) function init_T!(T, z, time, k, Tm, Tp, Tmin, Tmax)
    dTdz       = Tm-Tp
    zᵢ         = z[k]
    Tᵢ         = Tp + dTdz*(1-zᵢ)
    time       = 5e-4
    Ths        = Tᵢ * erf((1-zᵢ)*0.5/(k*time)^0.5)
    Tᵢ         = min(Tᵢ, Ths)
    time       = 1e-3
    Ths        = erf(zᵢ*0.5/(k*time*5)^0.5)
    Ths       *= (Tmax - Tᵢ) 
    Ths        = Tmax - Ths 
    Tᵢ         = max(Tᵢ, Ths);
    T[i, j, k] = max(Tᵢ, Tmin)
    return 
end

function thermal_convection3D(; ar=8, ny=16, nx=ny*8, nz=ny*8)

    # This tests the MaterialParameters structure
    CharUnits = GEO_units(; viscosity=1e23, length=2900km, temperature=1000K)

    # Define a struct for a first phase
    G0 = Inf
    η_reg = 0.1
    pl = DruckerPrager_regularised(; C = 1e4, ϕ=0.0, η_vp=η_reg, Ψ=0.0)        # non-regularized plasticity
    rheology = SetMaterialParams(;
        Name="Rock",
        Phase=1,
        # Density=PT_Density(; ρ0=3000kg/m^3),
        Density=ConstantDensity(; ρ=4000kg/m^3),
        HeatCapacity=ConstantHeatCapacity(; cp=1250J/kg/K),
        Conductivity=ConstantConductivity(; k=5.0Watt/K/m),
        CreepLaws=LinearViscous(; η=1e23Pa*s),
        Elasticity = SetConstantElasticity(; G=G0, ν=0.5),
        CharDim=CharUnits,
    )

    # This tests the MaterialParameters structure
    CharUnits = GEO_units(; viscosity=1e23, length=2900km, temperature=1000K)

    # Define a struct for a first phase
    rheology2 = SetMaterialParams(;
        Name="Rock",
        Phase=1,
        Density=ConstantDensity(; ρ=1),
        HeatCapacity=ConstantHeatCapacity(; cp=1),
        Conductivity=ConstantConductivity(; k=1),
        CompositeRheology=CompositeRheology(
                SetConstantElasticity(; G=G0, ν=0.5), ArrheniusType()#, pl
            ),
        Elasticity = SetConstantElasticity(; G=G0, ν=0.5),
        Gravity=ConstantGravity(; g=1e6),
    )

    # Physical domain
    lz = 2900km
    lx = lz * ar
    ly = lz * ar
    lx_nd = nondimensionalize(lx, CharUnits)
    ly_nd = nondimensionalize(ly, CharUnits)
    lz_nd = nondimensionalize(lz, CharUnits)
    Xo = (0.0, 0.0, 0.0)                     # Origin coordinates
    ni = (nx, ny, nz)
    li = (lx_nd, ly_nd, lz_nd)               # domain length in x- and y-
    di = @. li / ni                          # grid step in x- and -y
    xci, xvi = lazy_grid(di, li; origin=Xo)  # nodes at the center and vertices of the cells
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=true)...) # init MPI
    ni_v = (nx-2)*igg.dims[1], (ny-2)*igg.dims[2], (nz-2)*igg.dims[3]

    # Physical parameters
    η0 = rheology.CreepLaws[1].η.val                                 # viscosity, Pa*s
    κ = NumValue(
        rheology.Conductivity[1].k /
        (rheology.Density[1].ρ * rheology.HeatCapacity[1].cp),
    )                          # heat diffusivity, m^2/s
    ΔT = nondimensionalize(1000K, CharUnits)                                 # initial temperature perturbation K
    ρ0 = rheology.Density[1].ρ.val
    g = rheology.Gravity[1].g.val
    α = 0.03
    Ra = ρ0 * g * α * ΔT * lz_nd^3 / (η0* κ)
    Ra = 1e6
    dt = dt_diff = 0.5 / 6.1 * min(di...)^3 / κ      # diffusive CFL timestep limiter
    println("\n Ra-number is $Ra")
    
    # Thermal diffusion ----------------------------------
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    k = @fill(rheology.Conductivity[1].k.val, ni...)
    thermal_bc = (flux_x=true, flux_y=false, flux_z=false)

    time= 1e-4
    k=1
    Tm=1900/2300
    Tp=1600/2300
    Tmin, Tmax = 0.12, 1.0
    @parallel init_T!(thermal.T, xvi[3], time, k, Tm, Tp, Tmin, Tmax)

    # Elliptical temperature anomaly ---------------------
    xc, yc, zc =  0.5*lx_nd, 0.5*ly_nd, 0.25*lz_nd
    ri         =  0.1
    Elli       =  [ ((x-xc ))^2 + ((y - yc))^2 + ((z - zc))^2 for x in xvi[1], y in xvi[2], z in xvi[3]]
    thermal.T[Elli .< ri.^2]  .*= 1.1
    @parallel assign!(thermal.Told, thermal.T)
    # ----------------------------------------------------

    # Stokes ---------------------------------------------
    ## Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL=1 / √3)

    ## Rheology
    η = @ones(ni...)
    v = CompositeRheology( (ArrheniusType(),) )
    args_η = (;T=thermal.T)
    @parallel (1:nx, 1:ny, 1:nz) computeViscosity!(η, v, args_η) # init viscosity field
    η_vep = deepcopy(η)
    dt_elasticity = Inf
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)

    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true, freeslip_z=true)

    # MPI Plotting
    Tc = @zeros(ni)
    Tg = zeros(ni_v...)
    # Tg_inn = zeros(ni_v.-1...)
    # Vzg = zeros(ni_v[1], ni_v[2], ni_v[3]+1)
    ηg = zeros(ni_v...)
    # ηg_inn = zeros(ni_v.-2...)

    # Physical time loop
    t   =  0.0
    it  =  0
    nt  =  350

    args = (;)
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        @parallel (1:nx, 1:ny, 1:nz) computeViscosity!(η, v, args_η) # update viscosity field
        @parallel (1:ni[1], 1:ni[2], 1:ni[3]) update_buoyancy!(ρg[3], thermal.T, Ra)
        # ------------------------------
        
        # Stokes solver ----------------
        iters = solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            freeslip,
            ρg,
            η,
            η_vep,
            rheology2,
            dt_elasticity,
            igg;
            iterMax = 15e3,
            nout=1e3,
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------
         
        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            stokes,
            rheology,
            args,
            di,
            dt
        )
        # ------------------------------

        @show it += 1
        t += dt

        if igg.me == 0 && it == 1 || rem(it, 5) == 0
            @parallel (1:nx, 1:ny, 1:nz) vertex2center!(Tc, thermal.T)
            gather!(Array(Tc[2:end-1, 2:end-1, 2:end-1]), Tg)
            gather!(Array(η[2:end-1, 2:end-1, 2:end-1]), ηg)

            fig = Figure(resolution = (900, 1200))
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "η")
            ax3 = Axis(fig[3,1], aspect = ar, title = "Vz")
            h1 = heatmap!(ax1, xci[1][2:end-1], xci[2][2:end-1], Array(Tg[nx÷2,:,:]) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1][2:end-1], xci[2][2:end-1], Array(log10.(ηg[nx÷2,:,:])) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1][2:end-1], xvi[2][2:end-1], Array(stokes.V.Vz[2:end-1,ny÷2,:]) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            fig
            save("figs3d/$(it).png", fig)
        end

    end

    finalize_global_grid(; finalize_MPI=false)

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

ar = 2
n = 32
nx = (n-2)*ar
ny = (n-2)*ar
nz = (n-2)

thermal_convection3D(;ar=ar,nx=nx, ny=ny, nz=nz);

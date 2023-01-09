using JustRelax
# setup ParallelStencil.jl environment
# model = PS_Setup(:gpu, Float64, 3)
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

function compute_dt(S::StokesArrays, di, dt_diff)
    return compute_dt(S.V, di, dt_diff)
end

function compute_dt(V::Velocity, di::NTuple{3,T}, dt_diff) where {T}
    return compute_dt(V.Vx, V.Vy, V.Vz, di[1], di[2], di[3], dt_diff)
end

function compute_dt(Vx, Vy, Vz, dx, dy, dz, dt_diff)
    dt_adv = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 3.1
    return min(dt_diff, dt_adv)
end

@parallel function update_buoyancy!(fz, T, ρ0gα)
    @all(fz) = -ρ0gα .* @all(T)
    return nothing
end

@inline function tupleindex(args::NamedTuple, I::Vararg{Int, N}) where N
    k = keys(args)
    v = getindex.(values(args), I...)
    return (; zip(k, v)...)
end

@parallel_indices (i, j, k) function computeViscosity!(η, v, args)
    η[i, j, k] = computeViscosity_εII(v, 1.0, tupleindex(args, i, j, k))
    return nothing
end

function thermal_convection2D(; ny=16, nx=ny*8, nz=ny*8)

    # This tests the MaterialParameters structure
    CharUnits = GEO_units(; viscosity=1e23, length=2900km, temperature=1000K)

    # Define a struct for a first phase
    rheology = SetMaterialParams(;
        Name="Rock",
        Phase=1,
        # Density=PT_Density(; ρ0=3000kg/m^3),
        Density=ConstantDensity(; ρ=4000kg/m^3),
        HeatCapacity=ConstantHeatCapacity(; cp=1250J/kg/K),
        Conductivity=ConstantConductivity(; k=5.0Watt/K/m),
        CreepLaws=LinearViscous(; η=1e23Pa*s),
        CharDim=CharUnits,
    )

    # Physical domain
    ar = 3
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
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=true, periodx=1, periody=1)...) # init MPI
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
    # α = rheology.Density[1].α.val
    α = 0.03
    Cp0 = rheology.HeatCapacity[1].cp.val
    Ra = ρ0 * g * α * ΔT * lz_nd^3 / (η0* κ)
    Ra = 1e4
    dt = dt_diff = 0.5 / 6.1 * min(di...)^3 / κ      # diffusive CFL timestep limiter
    println("\n Ra-number is $Ra")
    
    # Thermal diffusion ----------------------------------
    # general thermal arrays
    thermal = ThermalArrays(ni.+1)

    # physical parameters
    ρ = @fill(ρ0, ni...)
    ρCp = @fill(ρ0 * Cp0, ni...) 
    k = @fill(rheology.Conductivity[1].k.val, ni...)
    thermal_parameters = ThermalParameters(k, ρCp)
    thermal = ThermalArrays(ni.+1)
    thermal_bc = (flux_x=true, flux_y=false, flux_z=false)

    time= 1e-4
    k=1
    Tm=1900/2300
    Tp=1600/2300
    Tmin, Tmax = 0.3, 1.0
    dTdz = Tm-Tp
    thermal.T .= PTArray([Tp + dTdz*(1-z) for x in xvi[1], y in xvi[2], z in xvi[3]])
    Ths = thermal.T .* PTArray([erf((1-z)/2/(k*time)^0.5) for x in xvi[1], y in xvi[2], z in xvi[3]])
    @. thermal.T = min(thermal.T,Ths);
    time= 1e-4
    Ths .= Tmax .-(Tmax.-thermal.T) .* PTArray([erf(z/2/(k*time*5)^0.5) for x in xvi[1], y in xvi[2], z in xvi[3]])
    @. thermal.T = max(thermal.T, Ths);
    @. thermal.T = max(thermal.T, Tmin);

    # GeoTherm = -(ΔT - nondimensionalize(0C, CharUnits)) / li[2]
    # thermal.T .= ΔT
    # Tmax,Tmin = nondimensionalize(1500K, CharUnits), nondimensionalize(300K, CharUnits)

    
    # Elliptical temperature anomaly ---------------------
    xc, yc, zc      =   0.5*lx_nd, 0.5*ly_nd, 0.25*lz_nd
    ri              =   0.1
    Elli            =  [ ((x-xc ))^2 + ((y - yc))^2 + ((z - zc))^2 for x in xvi[1], y in xvi[2], z in xvi[3]]
    thermal.T[Elli .< ri.^2]  .*= 1.05
    # thermal.T .*= 1 .+ rand(ni...).*0.05
    clamp!(thermal.T, Tmin, Tmax)
    @parallel assign!(thermal.Told, thermal.T)
    # ----------------------------------------------------

    # Stokes ---------------------------------------------
    ## Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(ni, di; ϵ=1e-4,  CFL=1 / √3)

    ## Setup-specific parameters and fields
    η = @ones(ni...)
    v = CompositeRheology( (ArrheniusType(),) )
    args_η = (;T=thermal.T)
    @parallel (1:nx, 1:ny, 1:nz) computeViscosity!(η, v, args_η) # init viscosity field
    dt_elasticity = Inf
    G = @fill(1.0, ni...) 
    Kb = @fill(Inf, ni...) 
    fz = Ra .* thermal.T
    ρg = (@zeros(ni...), @zeros(ni...), fz)

    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true, freeslip_z=true)

    # MPI Plotting
    Tg = @zeros(ni_v.+1...)
    Vzg = @zeros(ni_v[1], ni_v[2], ni_v[3]+1)
    ηg = @zeros(ni_v...)

    # Physical time loop
    t   =  0.0
    it  =  0
    nt  =  150

    args = (;)
    local iters
    while it < nt

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            li,
            freeslip,
            ρg,
            η,
            Kb,
            G,
            dt_elasticity,
            igg;
            iterMax=150000,
            nout=1000,
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            rheology,
            args,
            di,
            dt
        )
        scatter!(thermal.T[:], Z)
        # ------------------------------

        # Update buoyancy and viscosity -
        @parallel (1:nx, 1:ny, 1:nz) computeViscosity!(η, v, args_η) # update viscosity field
        @parallel update_buoyancy!(fz, thermal.T, Ra)
        # ------------------------------

        @show it += 1
        t += dt

        if igg.me == 0 && it == 1 || rem(it, 10) == 0
            gather!(thermal.T[2:end-1,2:end-1,2:end-1], Tg)
            gather!(stokes.V.Vz[2:end-1,2:end-1,2:end-1], Vzg)
            gather!(η[2:end-1,2:end-1,2:end-1], ηg)
            
            fig = Figure(resolution = (900, 1200))
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
            ax3 = Axis(fig[3,1], aspect = ar, title = "η")
            h1 = GLMakie.heatmap!(ax1, xvi[1][2:end-1], xvi[2][2:end-1], Array(Tg[nx÷2,:,:]) , colormap=:batlow)
            h2 = GLMakie.heatmap!(ax2, xvi[1][2:end-1], xvi[2][2:end-1], Array(Vzg[:,ny÷2,:]) , colormap=:batlow)
            h3 = GLMakie.heatmap!(ax3, xvi[1][2:end-1], xvi[2][2:end-1], Array(log10.(ηg[nx÷2,:,:])) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            fig
            save("figs3d/$(it).png", fig)

            # fig = Figure(resolution = (900, 1200))
            # ax1 = Axis(fig[1,1], aspect = 3, title = "T")
            # ax2 = Axis(fig[2,1], aspect = 3, title = "Vy")
            # ax3 = Axis(fig[3,1], aspect = 3, title = "η")
            # h1 = GLMakie.heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T[nx÷2,:,:]) , colormap=:batlow)
            # h2 = GLMakie.heatmap!(ax2, xvi[1], xvi[2], Array(stokes.V.Vz[:,ny÷2,:]) , colormap=:batlow)
            # h3 = GLMakie.heatmap!(ax3, xvi[1], xvi[2], Array(log10.(η[nx÷2,:,:])) , colormap=:batlow)
            # Colorbar(fig[1,2], h1)
            # Colorbar(fig[2,2], h2)
            # Colorbar(fig[3,2], h3)
            # fig
            # save("figs3d/$(it).png", fig)
        end

    end

    finalize_global_grid(; finalize_MPI=true)

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

n = 16
nx = n*3
ny = n*3
nz = n

thermal_convection2D(;nx=nx, ny=ny, nz=nz);



X = [x for x in xvi[1], y in xvi[2], z in xvi[3]][:]
Y = [y for x in xvi[1], y in xvi[2], z in xvi[3]][:]
Z = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
scatter(thermal.T[:], Z)
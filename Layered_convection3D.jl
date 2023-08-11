using JustRelax, CUDA
using JustPIC
# backend = "CUDA"
# backend = "Threads"
# set_backend(backend)

# setup ParallelStencil.jl environment
# model = PS_Setup(:gpu, Float64, 2)
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, CellArrays
include("Layered_rheology.jl")

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, nx, ny, nz)
    ni = nx, ny, nz
    ncells = nx * ny * nz
    np = max_xcell * ncells
    px, py, pz = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(3))

    inject = @fill(false, ni..., eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    δ() = rand(0.05:1e-5: 0.95)

    @parallel_indices (i, j, k) function fill_coords_index()    
        # lower-left corner of the cell
        x0, y0, z0 = x[i], y[j], z[k]
        # fill index array
        for l in 1:nxcell
            @cell px[l, i, j, k] = x0 + dx * δ()
            @cell py[l, i, j, k] = y0 + dy * δ()
            @cell pz[l, i, j, k] = z0 + dz * δ()
            @cell index[l, i, j, k] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny, 1:nz) fill_coords_index()    

    return Particles(
        (px, py, pz), index, inject, nxcell, max_xcell, min_xcell, np, ni
    )
end


function velocity_grids(xci, xvi, di)
    dx, dy, dz = di
    x, y, z = xci
    x_ghost = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    y_ghost = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    z_ghost = LinRange(xci[3][1] - dz, xci[3][end] + dz, length(xci[3])+2)

    grid_vx = xvi[1], y_ghost, z_ghost
    grid_vy = x_ghost, xvi[2], z_ghost
    grid_vy = x_ghost, y_ghost, xvi[3]

    return grid_vx, grid_vy
end
#############################################################

# HELPER FUNCTIONS ---------------------------------------------------------------
# useful semi-closures
@inline Base.@propagate_inbounds gather_yz(A, i, j, k) =  A[i, j, k], A[i    , j + 1, k], A[i, j    , k + 1], A[i    , j + 1, k + 1]
@inline Base.@propagate_inbounds gather_xz(A, i, j, k) =  A[i, j, k], A[i + 1, j    , k], A[i, j    , k + 1], A[i + 1, j    , k + 1]
@inline Base.@propagate_inbounds gather_xy(A, i, j, k) =  A[i, j, k], A[i + 1, j    , k], A[i, j + 1, k    ], A[i + 1, j + 1, k    ]
@inline Base.@propagate_inbounds av(T, i, j, k) = 0.125 * (
    T[i, j, k  ] + T[i, j+1, k  ] + T[i+1, j, k  ] + T[i+1, j+1, k  ] +
    T[i, j, k+1] + T[i, j+1, k+1] + T[i+1, j, k+1] + T[i+1, j+1, k+1]
)

# viscosity with GeoParams
@parallel_indices (i, j, k) function compute_viscosity!(η, ratios_center, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, MatParam)

    # convinience closures
    av_T()        = av(args.T, i, j, k)
    _gather_yz(A) = gather_yz(A, i, j, k)
    _gather_xz(A) = gather_xz(A, i, j, k)
    _gather_xy(A) = gather_xy(A, i, j, k)
    
    εII_0 = (εxx[i, j, k] == 0 && εyy[i, j, k] == 0 && εzz[i, j, k] == 0) ? 1e-20 : 0.0
    _zeros = (0.0, 0.0, 0.0, 0.0)
    ratio_ij = ratios_center[i, j, k]
    args_ij  = (; dt = args.dt, P = args.P[i, j, k],  T=av_T(), τII_old=0.0)
    εij_p = (
        εxx[i, j, k] + εII_0, 
        εyy[i, j, k] - εII_0 * 0.5, 
        εzz[i, j, k] - εII_0 * 0.5, 
        _gather_yz(εyzv), 
        _gather_xz(εxzv), 
        _gather_xy(εxyv)
    )
    τij_p_o  = (
        0.0, 
        0.0, 
        0.0, 
        _zeros,
        _zeros,
        _zeros
    )
    # update stress and effective viscosity
    _, _, ηi   = compute_τij_ratio(MatParam, ratio_ij, εij_p, args_ij, τij_p_o)
    η[i, j, k] = clamp(ηi, 1e16, 1e24)
    
    return nothing
end

@generated function compute_phase_τij(MatParam::NTuple{N,AbstractMaterialParamsStruct}, ratio, εij_p, args_ij, τij_p_o) where N
    quote
        Base.@_inline_meta 
        empty_args = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 0.0
        Base.@nexprs $N i -> a_i = ratio[i] == 0 ? empty_args : compute_τij(MatParam[i].CompositeRheology[1], εij_p, args_ij, τij_p_o) 
        Base.@ncall $N tuple a
    end
end

function compute_τij_ratio(MatParam::NTuple{N,AbstractMaterialParamsStruct}, ratio, εij_p, args_ij, τij_p_o) where N
    data = compute_phase_τij(MatParam, ratio, εij_p, args_ij, τij_p_o)
    # average over phases
    τij =  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    τII = 0.0
    η_eff = 0.0
    for n in 1:N
        τij    = @. data[n][1] * ratio[n] + τij
        τII   +=    data[n][2] * ratio[n]
        η_eff +=    data[n][3] * ratio[n]
    end
    return τij, τII, η_eff
end

# Initial pressure guess
@parallel_indices (i,j,k) function init_P!(P, ρg, z)
    P[i,j,k] = abs(ρg[i,j,k] * z[k])
    return nothing
end

# Initial temperature profile
@parallel_indices (i, j, k) function init_T!(T, z)
    zi = z[k] #+ 45e3
    if 0 ≥ zi > -35e3
        dTdz = 600 / 35e3
        T[i, j, k] = dTdz * -zi + 273

    elseif -120e3 ≤ zi < -35e3
        dTdz = 700 / 85e3
        T[i, j, k] = dTdz * (-zi-35e3) + 600 + 273

    elseif zi < -120e3
        T[i, j, k] = 1280 * 3e-5 * 9.81 * (-zi-120e3) / 1200 + 1300 + 273

    else
        T[i, j, k] = 273.0
        
    end
    return 
end

@parallel_indices (i, j, k) function init_T2!(T, z, κ, Tm, Tp, Tmin, Tmax, time)
    yr         = 3600*24*365.25
    dTdz       = (Tm-Tp)/650e3
    zᵢ         = abs(z[k])
    Tᵢ         = Tp + dTdz*(zᵢ)
    time      *= yr
    Ths        = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5)
    T[i, j, k] = min(Tᵢ, Ths)
    return 
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, x, y, z)
        @inbounds if (((x[i]-xc))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            # T[i, j, k] = 2e3
            T[i, j, k] += δT
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, xvi...)
end

@parallel_indices (i, j, k) function compute_ρg!(ρg, rheology, args)
   
    av_T() = av(args.T, i, j, k) - 273.0

    @inbounds ρg[i, j, k] = compute_density(rheology, (; T = av_T(), P=args.P[i, j, k])) * compute_gravity(rheology.Gravity[1])

    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg, phase_ratios, rheology, args)

    av_T() = av(args.T, i, j, k) - 273.0

    ρg[i, j, k] = compute_density_ratio(phase_ratios[i, j, k], rheology, (; T = av_T(), P=args.P[i, j, k])) *
        compute_gravity(rheology[1])
    return nothing
end

@parallel_indices (i, j, k) function compute_invariant!(II, xx, yy, zz, yz, xz, xy)

    # convinience closure
    _gather_yz(A) = gather_yz(A, i, j, k)
    _gather_xz(A) = gather_xz(A, i, j, k)
    _gather_xy(A) = gather_xy(A, i, j, k)
    
    @inbounds begin
        ij = (
            xx[i, j, k]   , yy[i, j, k]   , zz[i, j, k]   ,
            _gather_yz(yz), _gather_xz(xz), _gather_xy(xy),
        )
        II[i, j, k] = second_invariant(ij...)
    end
    
    return nothing
end

function main3D(nx, ny, nz, ar; figdir="figs2D")

    # Physical domain ------------------------------------
    lz       = 650e3             # domain length in y
    lx = ly  = lz * ar
    origin   = 0.0, 0.0, -lz                        # origin coordinates
    ni       = nx, ny, nz                           # number of cells
    li       = lx, ly, lz                           # domain length in x- and y-
    di       = @. li / ni                           # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Init MPI -------------------------------------------
    igg  = IGG(
        init_global_grid(nx, ny, nz; init_MPI=JustRelax.MPI.Initialized() ? false : true)...
    )
    # ni_v = nx_g(), ny_g(), nz_g()
    # ni_v = (nx-2)*igg.dims[1], (ny-2)*igg.dims[2], (nz-2)*igg.dims[3]
    # ----------------------------------------------------
    
    # Physical properties using GeoParams ----------------
    # Define rheolgy struct
    # rheology     = init_rheologies(; is_plastic = false)
    rheology     = init_rheologies_simple(; is_plastic = false)
    # rheology_pl  = init_rheologies(; is_plastic = true)
    # rheology  = init_rheologies_isoviscous()
    κ            = (rheology[1].Conductivity[1].k / (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ0)).val
    dt = dt_diff = min(di...)^2 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # Initialize particles -------------------------------
    # nxcell, max_xcell, min_xcell = 16, 24, 8
    nxcell, max_xcell, min_xcell = 24, 48, 18
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], xvi[3], di[1], di[2], di[3], nx, ny, nz
    );
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_particle_fields_cellarrays(particles, Val(2))
    particle_args = pT, pPhases

    # Elliptical temperature anomaly 
    δT          = 150.0           # temperature perturbation    
    xc_anomaly  = 0.5*lx
    yc_anomaly  = 0.5*ly
    zc_anomaly  = -650e3  # origin of thermal anomaly
    r_anomaly   = 100e3            # radius of perturbation
    init_phases!(pPhases, particles, lx, ly; d=abs(zc_anomaly), r=r_anomaly)
    phase_ratios = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.95 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true, right = true, top = false, bot = false, front=true, back=true), 
        periodicity = (left = false, right = false, top = false, bot = false, front=false, back=false),
    )
    # initialize thermal profile - Half space cooling
    # Tmin, Tmax  = 273.0, 1500.0
    # @parallel init_T!(thermal.T, xvi[2])
    # initialize thermal profile - Half space cooling
    Tmin, Tmax  = 273.0, 2e3  # 1500.0 +273
    @views thermal.T            .= Tmax
    @views thermal.T[:, :, end] .= Tmin
   
    t = 0
    while (t/(1e6 * 3600 * 24 *365.25)) < 25
        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            rheology,
            phase_ratios,
            (; P=stokes.P),
            di,
            dt 
        )
        t+=dt
    end
    thermal_bcs!(thermal.T, thermal_bc)
   
    elliptical_perturbation!(thermal.T, δT, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, xvi)
    Tbot = thermal.T[:, :, 1]
    Tmax = 2e2
    @views thermal.T[:, :, end] .= Tmin
    # @views thermal.T[:, :, 1]   .= Tmax
    # ----------------------------------------------------
   
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, (T=thermal.T, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3])
    end
    # Rheology
    η               = @ones(ni...)
    args_ηv         = (; T = thermal.T, P = stokes.P, depth = xci[3], dt = Inf)
    @parallel (@idx ni) compute_viscosity!(η, phase_ratios.center, @strain(stokes)..., args_ηv, rheology)
 
    η_vep           = deepcopy(η)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip   = (left=true , right=true , top=true , bot=true , front=true , back=true ),
        no_slip     = (left=false, right=false, top=false, bot=false, front=false, back=false),
        periodicity = (left=false, right=false, top=false, bot=false, front=false, back=false),
        free_surface = false
    )
   
    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig0 = let
        Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(resolution = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        lines!(ax1, Array(thermal.T[:]), Zv./1e3; linewidth=5)
        lines!(ax2, Array(log10.(η[:])), Z./1e3 ; linewidth=5)
        ylims!(ax1, minimum(xvi[3])./1e3, 0)
        ylims!(ax2, minimum(xvi[3])./1e3, 0)
        hideydecorations!(ax2)
        save( joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # # Update buoyancy and viscosity
    # args_ηv = (; T = thermal.T, P = stokes.P, dt=Inf)
    # @parallel (@idx ni) compute_viscosity!(η, phase_ratios.center, @strain(stokes)..., args_ηv, rheology)
    # @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, (T=thermal.T, P=stokes.P))
    # # ------------------------------

    # Time loop
    t, it = 0.0, 0
    nt    = 250
    # ηs    = similar(η);
    # η0    = deepcopy(η);
    
    local iters

    while it < nt
    # while (t/(1e6 * 3600 * 24 *365.25)) < 100
        # Update buoyancy and viscosity
        args_ηv = (; T = thermal.T, P = stokes.P, dt=Inf)
        @parallel (@idx ni) compute_viscosity!(η, phase_ratios.center, @strain(stokes)..., args_ηv, rheology)
        @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, (T=thermal.T, P=stokes.P))
        # ------------------------------

        # for _ in 1:10
        #     @parallel JustRelax.Elasticity3D.smooth!(ηs, η, 1)
        #     η, ηs = ηs, η
        #     @views η[1,:,:]   .= η[2,:,:]
        #     @views η[:,1,:]   .= η[:,2,:]
        #     @views η[:,:,1]   .= η[:,:,2]
        #     @views η[end,:,:] .= η[end-1,:,:]
        #     @views η[:,end,:] .= η[:,end-1,:]
        #     @views η[:,:,end] .= η[:,:,end-1]
        # end
        
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
            phase_ratios,
            rheology,
            dt,
            igg,
            iterMax=150e3,
            nout=1000,
        );

        @parallel (@idx ni) compute_invariant!(stokes.ε.II, @strain(stokes)...)
        dt = compute_dt(stokes, di, dt_diff) * .75
        println("dt = $(dt/(1e6 * 3600 * 24 *365.25)) Myrs")
        # ------------------------------

        # Thermal solver ---------------
        args_T = (; P=stokes.P)
        solve!(
            thermal,
            thermal_bc,
            rheology,
            phase_ratios,
            args_T,
            di,
            dt 
        )
        # ------------------------------

        # Advection --------------------
        # interpolate fields from grid vertices to particles
        grid2particle_xvertex!(pT, xvi, thermal.T, particles.coords)
        # advect particles in space
        V = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz
        advection_RK2!(particles, V, grid_vx, grid_vy, grid_vz, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles_vertex!(particles, xvi, particle_args)
        clean_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        # interpolate fields from particle to grid vertices
        gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
        # gather_temperature_xvertex!(thermal.T, pT, ρCₚp, xvi, particles.coords)
        @views thermal.T[:, :, end] .= 273.0
        @views thermal.T[:, :, 1]   .= Tbot
        

        @show it += 1
        t += dt

        # # Plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            slice_j = Int(ny ÷ 2)
            # p = particles.coords
            # ppx, ppy, ppz = p
            # pxv = ppx.data[:]./1e3;
            # pyv = ppy.data[:]./1e3;
            # pzv = ppz.data[:]./1e3;
            # clr = pPhases.data[:];
            # ppT = pT.data[:];
            # idxv = particles.index.data[:];

            fig = Figure(resolution = (1000, 1600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vz [m/s]")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII [MPa]")
            # ax4 = Axis(fig[4,1], aspect = ar, title = "ρ [kg/m3]")
            # ax4 = Axis(fig[4,1], aspect = ar, title = "τII - τy [Mpa]")
            ax4 = Axis(fig[4,1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[3].*1e-3, Array(thermal.T[:,slice_j,:]), colormap=:batlow)
            # h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[3].*1e-3, Array(abs.(ρg[3][:,slice_j,:]./9.81)) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1].*1e-3, xvi[3].*1e-3, Array(stokes.V.Vz[:, slice_j,:]) , colormap=:batlow)
            # h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
            # h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.τ.II[:,slice_j,:].*1e-6) , colormap=:batlow) 
            h3 = heatmap!(ax3, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(stokes.ε.II[:,slice_j,:])) , colormap=:batlow) 
            # # h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(abs.(stokes.ε.xx))) , colormap=:batlow) 
            h4 = heatmap!(ax4, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(η[:,slice_j,:])) , colormap=:batlow)
            # h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(abs.(ρg[2]./9.81)) , colormap=:batlow)
            # h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(@.(stokes.P * friction  + cohesion - stokes.τ.II)/1e6) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1) #, height=100)
            Colorbar(fig[2,2], h2) #, height=100)
            Colorbar(fig[3,2], h3) #, height=100)
            Colorbar(fig[4,2], h4) #, height=100)
            fig
            save( joinpath(figdir, "$(it).png"), fig)

            # save vtk files
            data_v = (; Temperature = Array(thermal.T))
            data_c = (; TauII = Array(stokes.τ.II))
            JustRelax.DataIO.save_vtk(joinpath(figdir, "$(it)"), xvi, xci, data_v, data_c)
        end
        # ------------------------------

    end

    return nothing
end

function run()
    figdir = "Plume3D"
    ar     = 2 # aspect ratio
    n      = 32
    nx     = n*ar - 2
    ny     = n*ar - 2
    nz     = n - 2

    main3D(nx, ny, nz, ar; figdir=figdir);
end

run()
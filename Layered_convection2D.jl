using JustRelax, CUDA, JustRelax.DataIO, JustPIC
backend = "CUDA"
# backend = "Threads"
# set_backend(backend)

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
# model = PS_Setup(:cpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, CellArrays
include("Layered_rheology.jl")
# include("src/phases/phases.jl")

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> fill(NaN, max_xcell, nx, ny), Val(2))
    # min_xcell = ceil(Int, nxcell / 2)
    # min_xcell = 4

    # index = zeros(UInt32, np)
    inject = falses(nx, ny)
    index = falses(max_xcell, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        # center of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            # px[l, i, j] = x0 + dx_2 * (1.0 + 0.8 * (rand() - 0.5))
            # py[l, i, j] = y0 + dy_2 * (1.0 + 0.8 * (rand() - 0.5))
            px[l, i, j] = x0 + dx * rand(0.05:1e-5: 0.95)
            py[l, i, j] = y0 + dy * rand(0.05:1e-5: 0.95)
            index[l, i, j] = true
        end
    end
    
    # nsplits = 3
    # max_xcell = 2 * nsplits^2 * 2
    # nxcell = round(Int, max_xcell / 2)
    # ncells = nx * ny
    # np = max_xcell * ncells
    # px, py = ntuple(_ -> fill(NaN, max_xcell, nx, ny), Val(2))
    #     # min_xcell = ceil(Int, nxcell / 2)
    #     # min_xcell = 4
    
    # # index = zeros(UInt32, np)
    # inject = falses(nx, ny)
    # index = falses(max_xcell, nx, ny)
    # @inbounds for j in 1:ny, i in 1:nx
    #     # center of the cell
    #     x0, y0 = x[i], y[j]
    #     x_split = LinRange(x0, x0+dx, nsplits + 1)
    #     y_split = LinRange(y0, y0+dy, nsplits + 1)
    #     dx_split = x_split[2] - x_split[1]
    #     dy_split = y_split[2] - y_split[1]
    #     # fill index array
    #     l = 1
    #     while l ≤ nxcell
    #         for ix in 1:nsplits, iy in 1:nsplits
    #             px[l, i, j] = x_split[ix] + dx_split * rand()
    #             py[l, i, j] = y_split[iy] + dy_split * rand()
    #             index[l, i, j] = true
    #             l += 1
    #         end
    #     end
    # end

    if JustPIC.backend === "CUDA"
        pxi = CuArray.((px, py))
        return Particles(
            pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, min_xcell, np, (nx, ny)
        )

    else
        return Particles(
            (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
        )
    end
end


function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index)    
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            JustRelax.@cell px[l, i, j] = x0 + dx * rand(0.05:1e-5: 0.95)
            JustRelax.@cell py[l, i, j] = y0 + dy * rand(0.05:1e-5: 0.95)
            JustRelax.@cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index)    

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

function velocity_grids(xci, xvi, di)
    dx, dy = di
    yVx = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)

    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end
#############################################################

# HELPER FUNCTIONS ---------------------------------------------------------------
function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end


import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j) function init_T!(T, z)
    zi = z[j] #+ 45e3
    if 0 ≥ zi > -35e3
        dTdz = 600 / 35e3
        T[i, j] = dTdz * -zi + 273

    elseif -120e3 ≤ zi < -35e3
        dTdz = 700 / 85e3
        T[i, j] = dTdz * (-zi-35e3) + 600 + 273

    elseif zi < -120e3
        T[i, j] = 1280 * 3e-5 * 9.81 * (-zi-120e3) / 1200 + 1300 + 273

    else
        T[i, j] = 273.0
        
    end
    return 
end

@parallel_indices (i, j) function init_T2!(T, z, κ, Tm, Tp, Tmin, Tmax, time)
    yr      = 3600*24*365.25
    dTdz    = (Tm-Tp)/650e3
    zᵢ      = abs(z[j])
    Tᵢ      = Tp + dTdz*(zᵢ)
    time   *= yr
    Ths     = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5)
    T[i, j] = min(Tᵢ, Ths)
    return 
end


@parallel_indices (i, j) function init_T_Attila!(T, z)
    depth   = abs(z[j])

    if depth < 35e3
        dTdZ = (923-273)/35e3
        offset = 273e0
        # dTdZ, offset
        T[i, j] = depth * dTdZ + offset
    
    elseif 110e3 > depth ≥ 35e3
        dTdZ = (1492-923)/75e3
        offset = 923
        # dTdZ, offset
        T[i, j] = (depth - 35e3) * dTdZ + offset

    elseif depth ≥ 110e3 
        dTdZ = (1837 - 1492)/590e3
        offset = 1492e0
        # dTdZ, offset
        T[i, j] = (depth - 110e3) * dTdZ + offset

    end
    
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if (((x[i]-xc ))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i, j] = 2e3
            # T[i, j] += δT
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end

function rectangular_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if ((x[i]-xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            depth = abs(y[j])
            dTdZ = (2047 - 2017) / 50e3
            offset = 2017
            T[i, j] = (depth - 585e3) * dTdZ + offset
        end
        return nothing
    end

    @parallel _rectangular_perturbation!(T, δT, xc, yc, r, xvi...)
end
# --------------------------------------------------------------------------------

Rayleigh_number(ρ, α, ΔT, κ, η0) = ρ * 9.81 * α * ΔT * 2890e3^3 * inv(κ * η0) 

@parallel_indices (i, j) function compute_invariant!(II, xx, yy, xyv)

    # convinience closure
    @inline Base.@propagate_inbounds gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 
   
    @inbounds begin
        ij       = xx[i, j], yy[i, j], gather(xyv)
        II[i, j] = GeoParams.second_invariant_staggered(ij...)
    end
    
    return nothing
end

@parallel_indices (i,j) function compute_K!(K, rheology, phase, args)
    K[i, j] =  fn_ratio(compute_conductivity, rheology, phase[i, j], (; T=args.T[i, j], P=args.P[i, j]))
    return nothing
end

function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D")

    # Physical domain ------------------------------------
    ly       = 700e3 - 15e3 # domain length in y
    lx       = ly * ar      # domain length in x
    ni       = nx, ny       # number of cells
    li       = lx, ly       # domain length in x- and y-
    di       = @. li / ni   # grid step in x- and -y
    origin   = 0.0, -ly     # origin coordinates
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # Define rheolgy struct
    rheology     = init_rheologies(; is_plastic = true)
    # rheology     = init_rheologies_isoviscous()
    κ            = (10/ (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # Initialize particles -------------------------------
    # nxcell, max_xcell, min_xcell = 16, 24, 8
    nxcell, max_xcell, min_xcell = 20, 20, 10
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_particle_fields_cellarrays(particles, Val(3))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly 
    δT          = 50.0           # temperature perturbation    
    xc_anomaly  = lx/2  # origin of thermal anomaly
    yc_anomaly  = -610e3  # origin of thermal anomaly
    r_anomaly   = 25e3            # radius of perturbation
    init_phases!(pPhases, particles, lx; d=abs(yc_anomaly), r=r_anomaly)
    phase_ratios = PhaseRatio(ni, length(rheology))
    @parallel (JustRelax.@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.95 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true, right = true, top = false, bot = false), 
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel init_T_Attila!(thermal.T, xvi[2])
    Tmax =  1837e0

    # t = 0
    # while (t/(1e6 * 3600 * 24 *365.25)) < 25
    #     # Thermal solver ---------------
    #     solve!(
    #         thermal,
    #         thermal_bc,
    #         rheology,
    #         phase_ratios,
    #         (; P=0.0),
    #         di,
    #         dt 
    #     )
    #     t+=dt
    # end
    thermal_bcs!(thermal.T, thermal_bc)
   
    rectangular_perturbation!(thermal.T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi)
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------
   
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        @parallel (JustRelax.@idx ni) JustRelax.compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end
    # Rheology
    η = @ones(ni...)
    args_ηv = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (JustRelax.@idx ni) JustRelax.compute_viscosity!(
        η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args_ηv, rheology, (1e16, 1e24)
    )
    η_vep = deepcopy(η)

    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip    = (left = true, right=true, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig0 = let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y =  [y for x in xci[1], y in xci[2]][:]
        fig = Figure(resolution = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        lines!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        lines!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        save( joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # Time loop
    t, it = 0.0, 0
    nt    = 30
    T_buffer = @zeros(ni.+1) 
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles.coords)

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    local iters
    # while it < 150
    while (t/(1e6 * 3600 * 24 *365.25)) < 100
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        @parallel (JustRelax.@idx ni) JustRelax.compute_viscosity!(
            η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args, rheology, (1e16, 1e24)
        )
        @parallel (JustRelax.@idx ni) JustRelax.compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
        # ------------------------------
 
        # Stokes solver ----------------
        @time solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            iterMax=50e3,
            nout=1e3,
        );
        @parallel (JustRelax.@idx ni) compute_invariant!(stokes.ε.II, @strain(stokes)...)
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            rheology,
            phase_ratios,
            args,
            di,
            dt 
        )
        # ------------------------------

        # Advection --------------------
        # interpolate fields from grid vertices to particles
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        grid2particle!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        # inject && break
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # # update phase ratios
        @parallel (JustRelax.@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles.coords)
        @views T_buffer[:, end]      .= 273.0
        @views thermal.T[2:end-1, :] .= T_buffer
        @views thermal.T[:, 1]       .= Tmax
        temperature2center!(thermal)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)
            JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            data_v = (; 
                T = Array(thermal.T[2:end-1, :]),
                τxy = Array(stokes.τ.xy),
                εxy = Array(stokes.ε.xy),
                Vx = Array(Vx_v),
                Vy = Array(Vy_v),
            )
            data_c = (; 
                P = Array(stokes.P),
                τxx = Array(stokes.τ.xx),
                τyy = Array(stokes.τ.yy),
                εxx = Array(stokes.ε.xx),
                εyy = Array(stokes.ε.yy),
                η = Array(η),
            )
            save_vtk(
                joinpath(figdir, "vtk_" * lpad("$it", 6, "0")),
                xvi,
                xci, 
                data_v, 
                data_c
            )

            # p = particles.coords
            # ppx, ppy = p
            # pxv = ppx.data[:]./1e3
            # pyv = ppy.data[:]./1e3
            # clr = pPhases.data[:]
            # ppT = pT.data[:]
            # idxv = particles.index.data[:];

            fig = Figure(resolution = (1000, 1600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            # ax3 = Axis(fig[3,1], aspect = ar, title = "τII [MPa]")
            ax3 = Axis(fig[3,1], aspect = ar, title = "log10(εII)")
            # ax4 = Axis(fig[4,1], aspect = ar, title = "ρ [kg/m3]")
            # ax4 = Axis(fig[4,1], aspect = ar, title = "τII - τy [Mpa]")
            ax4 = Axis(fig[4,1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(abs.(ρg[2]./9.81)) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            # h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(ppT[idxv]))
            # h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.τ.II.*1e-6) , colormap=:batlow) 
            h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow) 
            # # h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(abs.(stokes.ε.xx))) , colormap=:batlow) 
            h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η)) , colormap=:batlow)
            # h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(abs.(ρg[2]./9.81)) , colormap=:batlow)
            # h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(@.(stokes.P * friction  + cohesion - stokes.τ.II)/1e6) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            Colorbar(fig[4,2], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save( joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end

# function run()
    figdir = "Plume2D"
    metadata(pwd(), "Layered_convection2D.jl", joinpath(figdir, "metadata"))
    ar     = 1 # aspect ratio
    n      = 141
    nx     = n*ar - 2
    ny     = n - 2
    igg  = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(nx, ny, 0; init_MPI= true)...)
    else
        igg
    end
    main2D(igg; figdir=figdir, ar=ar,nx=nx, ny=ny);
# end

# run()
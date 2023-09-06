using CUDA
CUDA.allowscalar(false)
using JustRelax, JustRelax.DataIO, JustPIC
# using CSV, DataFrames
backend = "CUDA_Float64_2D"
# backend = "Threads"
# set_backend(backend)

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
# model = PS_Setup(:cpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, CellArrays

const λ = 0.9142

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

distance(p1, p2) = mapreduce(x->(x[1]-x[2])^2, +, zip(p1, p2)) |> sqrt

function phase_ratios_center_closest(x::PhaseRatio, cell_center, pcoords, phases, cell::Vararg{Int,N}) where {N}
    return phase_ratios_center_closest(x.center, cell_center, pcoords, phases, cell...)
end

@inline function phase_ratios_center_closest(x::CellArray, cell_center, p, phases, cell::Vararg{Int,N}) where {N}
    # number of active particles in this cell
    dist_min = Inf
    idx_min = -1
    for i in 1:JustRelax.cellnum(phases)
        p1 = ntuple(j-> JustRelax.@cell(p[j][i, cell...]), Val(N))
        dist = distance(p1, cell_center)
        if dist < dist_min 
            dist_min = dist
            idx_min = i
        end
        if i ≤  JustRelax.cellnum(x)
            JustRelax.@cell x[i, cell...] = 0.0
        end
    end
    phase_min = Int(JustRelax.@cell((phases[idx_min, cell...])))
    JustRelax.@cell x[phase_min, cell...] = 1.0
    # for i in 1:JustRelax.cellnum(x)
    #     if i != phase_min
    #         JustRelax.@cell x[i, cell...] = 0.0
    #     else
    #         JustRelax.@cell x[i, cell...] = 1.0
    #     end
    # end
end

@parallel_indices (i, j) function phase_ratios_center_closest(x, xci, pcoords, phases)
    phase_ratios_center_closest(x, (xci[1][i], xci[2][j]), pcoords, phases, i, j)
    return nothing
end

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

    if occursin("CUDA", JustPIC.backend)
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

function init_phases!(phases, particles)
    ni = size(phases)
    
    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = JustRelax.@cell py[ip, i, j]

            # plume - rectangular
            if y > 0.2 + 0.02*cos(π*x/λ)
                JustRelax.@cell phases[ip, i, j] = 2.0
            else
                JustRelax.@cell phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index)
end

# --------------------------------------------------------------------------------


@parallel_indices (i, j) function compute_invariant!(II, xx, yy, xyv)

    # convinience closure
    @inline Base.@propagate_inbounds gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 
   
    @inbounds begin
        ij       = xx[i, j], yy[i, j], gather(xyv)
        II[i, j] = GeoParams.second_invariant_staggered(ij...)
    end
    
    return nothing
end

function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D")

    # Physical domain ------------------------------------
    ly       = 1            # domain length in y
    lx       = ly           # domain length in x
    ni       = nx, ny       # number of cells
    li       = lx, ly       # domain length in x- and y-
    di       = @. li / ni   # grid step in x- and -y
    origin   = 0.0, 0.0     # origin coordinates
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    dt       = Inf 
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=1),
            Gravity           = ConstantGravity(; g=1),
            CompositeRheology = CompositeRheology((LinearViscous(;η=1e0),)),

        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ=2),
            Gravity           = ConstantGravity(; g=1),
            CompositeRheology = CompositeRheology((LinearViscous(;η=1e0),)),
        ),
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 40, 40, 1
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pPhases, = init_particle_fields_cellarrays(particles, Val(1))
    particle_args = (pPhases, )
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) JustRelax.phase_ratios_center(phase_ratios.center, particles.coords..., xci..., di, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true, right = true, top = false, bot = false), 
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    # ----------------------------------------------------
   
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (JustRelax.@idx ni) JustRelax.compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # Rheology
    η = @ones(ni...)
    compute_viscosity!(
        η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args, rheology, (-Inf, Inf)
    )
    η_vep = deepcopy(η)

    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        # free_slip    = (left = true, right=true, top=true, bot=true),
        free_slip    = (left = true, right=true, top=false, bot=false),
        no_slip      = (left = false, right=false, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # T_buffer = @zeros(ni.+1) 
    # Told_buffer = similar(T_buffer)
    # for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
    #     copyinn_x!(dst, src)
    # end
    # grid2particle!(pT, xvi, T_buffer, particles.coords)

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    # Time loop
    t, it = 0.0, 0
    tmax = 2e3
    Urms = Float64[]
    trms = Float64[]
    sizehint!(Urms, 100000)
    sizehint!(trms, 100000)
    while t < tmax
        # while (t/(1e6 * 3600 * 24 *365.25)) < 100
        # Update buoyancy
        @parallel (JustRelax.@idx ni) JustRelax.compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
        # ------------------------------

        # particle2grid!(T_buffer, pT, xvi, particles.coords)
        # # @views T_buffer[:, end]      .= 273.0
        # @views thermal.T[2:end-1, :] .= T_buffer
        # # @views thermal.T[:, 1]       .= Tmax
        # temperature2center!(thermal)
        # @copy thermal.Told thermal.T
        # copyinn_x!(Told_buffer, thermal.Told)

        # Stokes solver ----------------
        to = solve!(
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
            Inf,
            igg;
            iterMax=10e3,
            nout=50,
            viscosity_cutoff=(-Inf, Inf)
        )
        @show to
        @parallel (JustRelax.@idx ni) compute_invariant!(stokes.ε.II, @strain(stokes)...)
        dt = compute_dt(stokes, di) / 10
        # ------------------------------

        # Compute U rms ---------------
        Urms_it = let
            JustRelax.velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx[:, 2:end-1], stokes.V.Vy[2:end-1, :]; ghost_nodes=false)
            U = hypot.(Vx_v, Vy_v)
            sum(U.^2 .* prod(di)) |> sqrt
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # ------------------------------

        # advect particles in space
        # advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 0.5)
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # copyinn_x!(T_buffer, thermal.T)
        # grid2particle!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        # # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        # inject && break
        inject && inject_particles_phase!(particles, pPhases, (), (), xvi)
        # # update phase ratios
        @parallel (@idx ni) JustRelax.phase_ratios_center2(phase_ratios.center, particles.coords..., xci..., di, pPhases)
        # @parallel (JustRelax.@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        # @parallel (JustRelax.@idx ni) phase_ratios_center_closest(phase_ratios.center, xci, particles.coords, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1000) == 0 || t >= tmax
            checkpointing(figdir, stokes, thermal.T, η, t)

            # p = particles.coords
            # ppx, ppy = p
            # pxv = ppx.data[:]./1e3
            # pyv = ppy.data[:]./1e3
            # clr = pPhases.data[:]
            # ppT = pT.data[:]
            # idxv = particles.index.data[:];

            fig = Figure(resolution = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = 1/λ, title = "t=$t")
            heatmap!(ax1, xvi[1], xvi[2], Array(ρg[2]), colormap=:oleron)
            save( joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    df = DataFrame(t=trms, Urms=Urms)
    CSV.write(joinpath(figdir, "Urms.csv"), df)

    return nothing
end

# function run()
    figdir = "VanKeken_heun"
    # metadata(pwd(), "VanKeken.jl", joinpath(figdir, "metadata"))
    ar     = 1 # aspect ratio
    n      = 128+2
    nx     = n - 2
    ny     = n - 2
    igg  = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(nx, ny, 0; init_MPI= true)...)
    else
        igg
    end
#     main2D(igg; figdir=figdir, ar=ar,nx=nx, ny=ny);
# # end


# data = CSV.read("VanKeken\\Urms.csv", DataFrame)

# trms = data.t
# urms = data.Urms
# idx = trms .> 15
# fig,ax,=lines(trms[idx], urms[idx])

# data = CSV.read("VanKeken_rk2\\Urms.csv", DataFrame)

# trms = data.t
# urms = data.Urms
# idx = trms .> 15
# lines!(ax, trms[idx], urms[idx])

# data = CSV.read("VanKeken_heun\\Urms.csv", DataFrame)

# idx = data.t .> 15
# trms = data.t[idx]
# urms = data.Urms[idx]
# urms_max = maximum(urms)
# tmax = trms[argmax(urms)]

# lines!(ax, trms, urms)
# fig

# urms_max = 0.003075
# t_urms_max = 211
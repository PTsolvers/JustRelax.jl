using CUDA
CUDA.allowscalar(false)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, CellArrays
using JustRelax, JustRelax.DataIO, JustPIC, CSV, DataFrames
backend = "CUDA_Float64_2D" # options: "CUDA_Float64_2D" "Threads_Float64_2D"
# set_backend(backend) # run this on the REPL to switch backend

# setup ParallelStencil.jl environment
@static if occursin("CUDA", JustPIC.backend) 
    model  = PS_Setup(:CUDA, Float64, 2)
    environment!(model)
else
    model  = PS_Setup(:Threads, Float64, 2)
    environment!(model)
end

# x-length of the domain
const λ = 0.9142

# HELPER FUNCTIONS ---------------------------------------------------------------

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni     = nx, ny
    ncells = nx * ny
    np     = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool) # true if injection in given cell is required
    index  = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) # array that says if there's a particle in a given memory location
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index)    
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            JustRelax.@cell px[l, i, j]    = x0 + dx * rand(0.05:1e-5: 0.95)
            JustRelax.@cell py[l, i, j]    = y0 + dy * rand(0.05:1e-5: 0.95)
            JustRelax.@cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index)    

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

# Define velocity grids with ghost nodes (for particle interpolations of the velocity field)
function velocity_grids(xci, xvi, di)
    dx, dy  = di
    yVx     = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy     = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

# Initial pressure guess
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

# Initialize phases on the particles
function init_phases!(phases, particles)
    ni = size(phases)
    
    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = JustRelax.@cell py[ip, i, j]

            # plume - rectangular
            if y > 0.2 + 0.02 * cos(π * x / λ)
                JustRelax.@cell phases[ip, i, j] = 2.0
            else
                JustRelax.@cell phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
# END OF HELPER FUNCTIONS --------------------------------------------------------


# MAIN SCRIPT --------------------------------------------------------------------
function main2D(igg; ny=16, nx=ny*8, figdir="model_figs")

    # Physical domain ------------------------------------
    ly       = 1            # domain length in y
    lx       = ly           # domain length in x
    ni       = nx, ny       # number of cells
    li       = lx, ly       # domain length in x- and y-
    di       = @. li / ni   # grid step in x- and -y
    origin   = 0.0, 0.0     # origin coordinates
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    dt       = Inf 

    # Physical properties using GeoParams ----------------
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 1),
            Gravity           = ConstantGravity(; g = 1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1e0),)),

        ),
        # High density phase
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 2),
            Gravity           = ConstantGravity(; g = 1),
            CompositeRheology = CompositeRheology((LinearViscous(;η = 1e0),)),
        ),
    )

    # Initialize particles -------------------------------
    nxcell, max_p, min_p = 40, 40, 1
    particles            = init_particles_cellarrays(
        nxcell, max_p, min_p, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy     = velocity_grids(xci, xvi, di)
    # temperature
    pPhases,             = init_particle_fields_cellarrays(particles, Val(1))
    particle_args        = (pPhases, )
    init_phases!(pPhases, particles)
    phase_ratios         = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes               = StokesArrays(ni, ViscoElastic)
    pt_stokes            = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √2.1)

    # Buoyancy forces
    ρg                   = @zeros(ni...), @zeros(ni...)
    args                 = (; T = @zeros(ni...), P = stokes.P, dt = Inf)
    @parallel (JustRelax.@idx ni) JustRelax.compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    
    # Rheology
    η                    = @ones(ni...)
    η_vep                = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs             = FlowBoundaryConditions(; 
        free_slip = (left =  true, right =  true, top = false, bot = false),
        no_slip   = (left = false, right = false, top =  true, bot =  true),
    ) 

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Buffer arrays to compute velocity rms
    Vx_v  = @zeros(ni.+1...)
    Vy_v  = @zeros(ni.+1...)

    # Time loop
    t, it = 0.0, 0
    tmax  = 2e3
    Urms  = Float64[]
    trms  = Float64[]
    sizehint!(Urms, 100000)
    sizehint!(trms, 100000)

    while t < tmax

        # Update buoyancy
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
        solve!(
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
            iterMax          = 10e3,
            nout             = 50,
            viscosity_cutoff = (-Inf, Inf)
        )
        dt = compute_dt(stokes, di) / 1
        # ------------------------------

        # Compute U rms ---------------
        Urms_it = let
            JustRelax.velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=true)
            @. Vx_v .= hypot.(Vx_v, Vy_v) # we reuse Vx_v to store the velocity magnitude
            sum(Vx_v.^2) * prod(di) |> sqrt
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # ------------------------------

        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        # inject && break
        inject && inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

        @show it += 1
        t        += dt

        # Plotting ---------------------
        if it == 1 || rem(it, 1000) == 0 || t >= tmax
            fig = Figure(resolution = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = 1/λ, title = "t=$t")
            heatmap!(ax1, xvi[1], xvi[2], Array(ρg[2]), colormap = :oleron)
            save( joinpath(figdir, "$(it).png"), fig)
            fig
        end

    end

    df = DataFrame(t=trms, Urms=Urms)
    CSV.write(joinpath(figdir, "Urms.csv"), df)

    return nothing
end

figdir = "VanKeken"
n      = 128 + 2
nx     = n - 2
ny     = n - 2
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main2D(igg; figdir = figdir, nx = nx, ny = ny);

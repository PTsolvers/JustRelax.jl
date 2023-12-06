using JustRelax, JustRelax.DataIO, JustPIC
import @cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
function plot_particles(particles, pPhases)
    p        = particles.coords
    ppx, ppy = p
    pxv      = ppx.data[:] ./ 1e3
    pyv      = ppy.data[:] ./ 1e3
    clr      = pPhases.data[:]
    idxv     = particles.index.data[:]
    f, _, h  = scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma)
    Colorbar(f[1,2], h)
    f
end

@inline init_particle_fields(particles)                              = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields)                     = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N})            where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni     = nx, ny
    ncells = nx * ny
    np     = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))
    inject = @fill(false, nx, ny, eltype=Bool)
    index  = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index)    
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            @cell px[l, i, j]    = x0 + dx * rand(0.05:1e-5:0.95)
            @cell py[l, i, j]    = y0 + dy * rand(0.05:1e-5:0.95)
            @cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index)    

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

# Velocity helper grids for the particle advection
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

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

function init_phases!(phases, particles, A)
    ni = size(phases)
    
    @parallel_indices (i, j) function init_phases!(phases, px, py, index, A)
        
        f(x, A, λ) = A * sin(π*x/λ)
        
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            @cell(index[ip, i, j]) == 0 && continue
            x                      = @cell px[ip, i, j]
            depth                  = -(@cell py[ip, i, j]) 
            @cell phases[ip, i, j] = 2.0
            
            if 0e0 ≤ depth ≤ 100e3
                @cell phases[ip, i, j] = 1.0
            elseif depth > (-f(x, A, 500e3) + (200e3 + A))
                @cell phases[ip, i, j] = 3.0          
            end

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function RT_2D(igg, nx, ny)

    # Physical domain ------------------------------------
    thick_air    = 100e3              # thickness of sticky air layer
    ly           = 500e3 + thick_air # domain length in y
    lx           = 500e3           # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    xci, xvi     = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = rheology = (
        # Name              = "Air",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=1e0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e16),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Crust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ=3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e20),)),
            Gravity           = ConstantGravity(; g=9.81),
        )
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 48, 64, 1
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_particle_fields_cellarrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly 
    A             = 5e3    # Amplitude of the anomaly
    init_phases!(pPhases, particles, A)
    phase_ratios  = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-5,  CFL = 0.95 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    # ----------------------------------------------------
   
    # Buoyancy forces & rheology
    ρg               = @zeros(ni...), @zeros(ni...)
    η                = @zeros(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )
    η_vep            = copy(η)

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(; 
        free_slip    = (left = true, right=true, top=true, bot=false),
        no_slip      = (left = false, right=false, top=false, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )

    # For plotting: velocity arrays at cell vertices
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    # Folder where to store figures
    figdir = "FreeSurface"
    take(figdir)

    # Time loop
    t, it  = 0.0, 0
    dt     = 5e3 * (3600 * 24 *365.25)
    dt_max = 25e3 * (3600 * 24 *365.25)
    while t < (6 * (1e6 * 3600 * 24 *365.25))

        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
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
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            iterMax = 75e3,
            nout=1e3,
            viscosity_cutoff=(-Inf, Inf)
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)        
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        
        @show it += 1
        t        += dt

        if it == 1 || rem(it, 1) == 0
            JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 5

            fig = Figure(resolution = (900, 900), title = "t = $t")
            ax  = Axis(fig[1,1], aspect = 6/5, title = " t=$(t/(1e3 * 3600 * 24 *365.25)) Kyrs")
            heatmap!(ax, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η)), colormap = :grayC)
            arrows!(
                ax,
                xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))..., 
                lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
                color = :red,
            )
            fig
            save(joinpath(figdir, "$(it).png"), fig)

        end

        dt = min(compute_dt(stokes, di) / 1, dt_max)
        dt = compute_dt(stokes, di)

    end
    # return fig
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n        = 128
nx       = n
ny       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 0; init_MPI= true)...)
else
    igg
end

RT_2D(igg, nx, ny)

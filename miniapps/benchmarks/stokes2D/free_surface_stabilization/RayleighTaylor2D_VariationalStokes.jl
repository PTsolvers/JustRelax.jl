const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams
using  GLMakie

# Velocity helper grids for the particle advection
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

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

function init_phases!(phases, particles, A)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, A)

        f(x, A, λ) = A * sin(π * x / λ)

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 2.0

            if 0e0 ≤ depth ≤ 100e3
                @index phases[ip, i, j] = 1.0

            elseif depth > (-f(x, A, 500e3) + (200e3 - A))
                @index phases[ip, i, j] = 3.0

            end

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n        = 101
nx       = n
ny       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny)

    # Physical domain ------------------------------------
    thick_air    = 100e3             # thickness of sticky air layer
    ly           = 500e3 + thick_air # domain length in y
    lx           = 500e3             # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
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
    nxcell, max_xcell, min_xcell = 125, 175, 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    A             = 5e3    # Amplitude of the anomaly
    phase_ratios  = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, A)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # RockRatios
    air_phase = 1
    ϕ         = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # Initialize marker chain-------------------------------
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation = -100e3
    chain             = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation);
    # ----------------------------------------------------


    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3e0, r=0.7, CFL = 0.98 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg               = @zeros(ni...), @zeros(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    # @parallel init_P!(stokes.P, ρg[2], xci[2])
    compute_viscosity!(stokes, phase_ratios, args, rheology, air_phase, (-Inf, Inf))

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left =  true, right =  true, top =  true, bot = false),
        no_slip      = (left = false, right = false, top = false, bot =  true),
        free_surface = true,
    )

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    figdir = "RayleighTaylor2D"
    take(figdir)

    # Time loop
    t, it   = 0.0, 0
    dt      = 1e3 * (3600 * 24 * 365.25)
    dt_max  = 3e3 * (3600 * 24 * 365.25)
    
    while it < 1000 #00
        
        if t / (1e3 * 3600 * 24 *365.25) > 500
            dt_max  = 25e3 * (3600 * 24 * 365.25)
        end
        ## variational solver
        # Stokes solver ----------------
        solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                iterMax              =  50e3,
                iterMin              =   1e3,
                viscosity_relaxation =  1e-2,
                nout                 =   2e3,
                free_surface         = true,
                viscosity_cutoff     = (-Inf, Inf)
            )
        )
        dt = compute_dt(stokes, di, dt_max)
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        
        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ, phase_ratios, air_phase)

        @show it += 1
        t        += dt

        if it == 1 || rem(it, 5) == 0
            px, py = particles.coords

            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 5
            fig = Figure(size = (900, 900), title = "t = $t")
            ax  = Axis(fig[1,1], aspect = 1, title = " t=$(round.(t/(1e3 * 3600 * 24 *365.25); digits=3)) Kyrs")
            # heatmap!(ax, xci[1].*1e-3, xci[2].*1e-3, Array([argmax(p) for p in phase_ratios.vertex]), colormap = :grayC)
            scatter!(ax, Array(px.data[:]).*1e-3, Array(py.data[:]).*1e-3, color =Array(pPhases.data[:]), colormap = :grayC)
            arrows!(
                ax,
                xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))...,
                lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
                color = :red,
            )
            fig
            save(joinpath(figdir, "$(it).png"), fig)

        end
    end
    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
main(igg, nx, ny)
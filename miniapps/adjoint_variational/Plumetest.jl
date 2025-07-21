#const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D_AD, JustRelax.DataIO

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
using CairoMakie

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end
function custom_argmax(p)
    if p[1] != 0.0
        return 0
    else
        return argmax(p)
    end
end

function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
    r = 100.0e3
    f(x, A, λ) = A * sin(π * x / λ)

    @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 2.0

            if -50.0e3 ≤ depth ≤ 0.0
                @index phases[ip, i, j] = 1.0

            else
                @index phases[ip, i, j] = 2.0

                if 0e0 ≤ depth ≤ 30.0e3
                    @index phases[ip, i, j] = 5.0
                end

                if 30.0e3 ≤ depth ≤ 50.0e3
                    @index phases[ip, i, j] = 6.0
                end

                if 50.0e3 ≤ depth ≤ 95.0e3
                    @index phases[ip, i, j] = 7.0
                end
#=
                if (0.0 ≤ x ≤ 50.0e3) & (depth >= 0.0)
                    @index phases[ip, i, j] = 2.0
                end

                if (1100e3 ≤ x ≤ 1200e3) & (depth >= 0.0)
                    @index phases[ip, i, j] = 2.0
                end
=#
                # plume
                center_x     = 600.0e3
                center_depth = 500.0e3
                r            = 100.0e3
                if ((x - center_x)^2 + (depth - center_depth)^2 ≤ r^2)
                    @index phases[ip, i, j] = 3.0
                end

    

                #=
                #inner plume
                center_x     = 600.0e3
                center_depth = 500.0e3
                r            = 50.0e3
                if ((x - center_x)^2 + (depth - center_depth)^2 ≤ r^2)
                    @index phases[ip, i, j] = 4.0
                end
                =#
            end

        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny,figdir)

    # Physical domain ------------------------------------
    thick_air    = 50.0e3                            # thickness of sticky air layer
    ly           = 660.0e3 + thick_air               # domain length in y
    lx           = 1200.0e3                          # domain length in x
    ni = nx, ny                # number of cells
    li = lx, ly                # domain length in x- and y-
    di = @. li / ni            # grid step in x- and -y
    origin = 0.0, -ly + thick_air         # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    el    = ConstantElasticity(; G = 40e9, ν = 0.3)
    rheology =  (
        # Name              = "Air",
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0e1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19),el)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 5.0e20),el)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Plume",
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 3.28e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),el)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Plume Inner",
        SetMaterialParams(;
            Phase = 4,
            Density = ConstantDensity(; ρ = 3.28e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),el)),
            Gravity = ConstantGravity(; g = 9.81),
            ),
        # Name              = "Crust",
        SetMaterialParams(;
            Phase = 5,
            Density = ConstantDensity(; ρ = 2.9e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 7.5e22),el)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "MantleLithosphereStrongCore",
        SetMaterialParams(;
            Phase = 6,
            Density = ConstantDensity(; ρ = 3.365e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.5e23),el)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "MantleLithosphereWeakLow",
        SetMaterialParams(;
            Phase = 7,
            Density = ConstantDensity(; ρ = 3.365e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 3e21),el)),
            Gravity = ConstantGravity(; g = 9.81),
            ),
        # Fake
        SetMaterialParams(;
            Phase = 8,
            Density = ConstantDensity(; ρ = 3.365e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 3e21),el)),
            Gravity = ConstantGravity(; g = 9.81),
            ),
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 100, 150, 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # rock ratios for variational stokes
    # RockRatios
    air_phase = 1
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # air object adjoint
    air_phase_a = 8
    ϕa = RockRatio(backend, ni)
    update_rock_ratio!(ϕa, phase_ratios, air_phase_a)
    # ----------------------------------------------------

    # Initialize marker chain-------------------------------
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation = 0.0e3
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    #pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-8, Re = 15π, r = 1e0, CFL = 0.#98 / √2.1)
    pt_stokes   = PTStokesCoeffs(li, di; ϵ_rel=1e-8, ϵ_abs=1e-12, Re=15π, r=1.0, CFL=0.95 / √2.1)
    #pt_stokesAD = PTStokesCoeffs(li, di; ϵ_rel=1e-4, ϵ_abs=1e-12, Re=3π, r=0.7, CFL=0.95 / √2.1) #V low Visc-contrast
    pt_stokesAD = PTStokesCoeffs(li, di; ϵ_rel=1e-4, ϵ_abs=1e-12, Re=12π, r=1.0, CFL=0.95 / √2.1)
    #pt_stokesAD = PTStokesCoeffs(li, di; ϵ_rel=1e-4, ϵ_abs=1e-12, Re=4π, r=0.7, CFL=0.8 / √2.1)
    # ----------------------------------------------------

    pp = 0.0
    CUDA.allowscalar() do
    pp = [custom_argmax(p) for p in phase_ratios.center]
    end
    # StokesArraysAdjoint
    stokesAD = StokesArraysAdjoint(backend, ni)    
    ind      = findall(pp .== 3.0) # Plume 
    ind      = CartesianIndex.((getindex.(ind, 1) .+ 1), getindex.(ind, 2))

    #indx     = findall((xci[1] .>= 500.0*1e3) .& (xci[1] .<= 700.0*1e3))
    #indy     = findall((xvi[2] .>= -22.0*1e3) .& (xvi[2] .<= -8.0*1e3)) .+ 1 
    #ind      = vec([CartesianIndex(i, j) for i in indx, j in indy])
    SensInd  = ind
    SensType = "Vy"

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf); air_phase = air_phase)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false
    )

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    take(figdir)

        # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored

    vtk_dir = joinpath(figdir, "vtk")
    take(vtk_dir)

    # ----------------------------------------------------

    # initilaize vtk collection]
    Vx_c  = @zeros(ni...)
    Vy_c  = @zeros(ni...)
    V_mag = @zeros(ni...)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    vtkc = VTKDataSeries(joinpath(figdir, "vtk_series"), xci)
    vtkv = VTKDataSeries(joinpath(figdir, "vtk_series"), xvi)

    ana = false


    # Time loop
    t, it = 0.0, 0
    dt     = 5e3 * (3600 * 24 * 365.25)
    dt_max = 50e3 * (3600 * 24 * 365.25)
    εbg = 2e-15
    while it < 400

        CUDA.allowscalar() do
        pp       = [custom_argmax(p) for p in phase_ratios.center]
        end
        stokesAD = StokesArraysAdjoint(backend, ni)    
        ind      = findall(pp .== 3.0) # Plume 
        ind      = CartesianIndex.((getindex.(ind, 1) .+ 1), getindex.(ind, 2))
        SensInd  = ind
    
        # Stokes -----------------------
        adjoint_solve_VariationalStokes!(
            stokes,
            stokesAD,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            dt,
            it, #Glit
            SensInd,
            SensType,
            grid,
            origin,
            li,
            ana,
            igg;
            kwargs = (;
                grid,
                origin,
                li,
                iterMax = 600.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (1e19, 1.5e23),
                free_surface = true,
                ADout=10,
                pt_stokesAD = pt_stokesAD,
                accλ = false,
                ϕa=ϕ
            )
        )
        dt = compute_dt(stokes, di, dt_max) * 0.95
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)

        # advect marker chain
#        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
#        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ, phase_ratios, air_phase)
        # ------------------------------

        @show it += 1
        t += dt

         # Data I/O and plotting ---------------------
         if it == 1 || rem(it, 10) == 0
            (; η_vep, η) = stokes.viscosity
           # checkpointing(figdir, stokes, thermal.T, η, t)
           (; η_vep, η) = stokes.viscosity

           velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
           Vx_c .= 0.0
           Vy_c .= 0.0
           vertex2center!(Vx_c, Vx_v)
           vertex2center!(Vy_c, Vy_v)
           V_mag .= sqrt.((Vx_c.^2 .+ Vy_c.^2))

           # Make particles plottable
           p        = particles.coords
           ppx, ppy = p
           pxv      = ppx.data[:]./1e3
           pyv      = ppy.data[:]./1e3
           clr      = pPhases.data[:]
           idxv     = particles.index.data[:];
           CUDA.allowscalar() do
           pp = [custom_argmax(p) for p in phase_ratios.center]
           end

            DataIO.append!(vtkc, (Phase=pp,T = thermal.T[2:end-1,:],P=Array(stokes.P), Vx=Array(Vx_c), Vy=Array(Vy_c), ρ=Array((ρg[2]./9.81)), η=Array(η), η_vep=Array(η_vep),  τII=Array(stokes.τ.II), εII=Array(stokes.ε.II), EII_pl=Array(stokes.EII_pl), η_sens=Array(stokesAD.η), ρ_sens=Array(stokesAD.ρ),G_sens=Array(stokesAD.G)), it, (t/(1e6 * 3600 * 24 *365.25)))
            end
    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n  = 128
nx = n * 2
ny = n
figdir = "RisingSphereSA_Litho_VE_frel8_arel4"
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(igg, nx, ny,figdir)
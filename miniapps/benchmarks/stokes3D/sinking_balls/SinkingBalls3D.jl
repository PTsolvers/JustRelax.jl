# Benchmark of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = CPUBackend

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)  #or (CUDA, Float64,  3) or (AMDGPU, Float64, 3)

using JustPIC
using JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CellArrays
using GLMakie

# Load file with all the rheology configurations
include("SinkingBalls_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    esc(:($A[$idx_k]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main3D(igg; ar=8, ny=16, nx=ny*8, nz=ny*8, figdir="figs3D", do_vtk =false)

    # Physical domain ------------------------------------
    lx           = 1e0                             # domain length in x
    ly           = 1e0                             # domain length in y
    lz           = 1e0                             # domain length in y
    ni           = nx, ny, nz                       # number of cells
    li           = lx, ly, lz                       # domain length in x- and y-
    di           = @. li / (nx_g(),ny_g(),nz_g())   # grid step in x- and -y
    origin       = 0.0, 0.0, 0.0                    # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                             # nodes at the center and vertices of the cells
     # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies()
    dt = dt_diff = 10 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 125, 175, 1
        particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    # velocity grids
    grid_vx, grid_vy, grid_vz   = velocity_grids(xci, xvi, di)
    # temperature
    pPhases,      = init_cell_arrays(particles, Val(2))
    particle_args = (pPhases, )

    # Elliptical temperature anomaly
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal     = ThermalArrays(backend_JR, ni)
       # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[3], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[3], xci[3])

    # Rheology
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
   
    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # # Plot initial T and η profiles
    # let # let block to avoid polluting the global namespace
    #     Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
    #     Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
    #     fig = Figure(size = (1200, 900))
    #     ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
    #     ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
    #     scatter!(ax1, Array(thermal.T[:]), Zv./1e3)
    #     scatter!(ax2, Array(log10.(η[:])), Z./1e3 )
    #     ylims!(ax1, minimum(xvi[3])./1e3, 0)
    #     ylims!(ax2, minimum(xvi[3])./1e3, 0)
    #     hideydecorations!(ax2)
    #     save(joinpath(figdir, "initial_profile.png"), fig)
    # end


    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Vz_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0
    while it < 1000
            # Update buoyancy and viscosity -
            args = (; T = thermal.Tc, P = stokes.P,  dt = Inf)

            # Stokes solver ----------------
            solve!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                rheology,
                args,
                dt,
                igg;
                kwargs = (;
                    iterMax = 100e3,
                    nout=1e3,
                    viscosity_cutoff=(-Inf, Inf)
                )
            )
            tensor_invariant!(stokes.ε)
            dt   = compute_dt(stokes, di, dt_diff)
            # ------------------------------

            # Advection --------------------
            # advect particles in space
            advection!(particles, RungeKutta2(), @velocity(stokes), ( grid_vx, grid_vy, grid_vz), dt)
            # advect particles in memory
            move_particles!(particles, xvi, particle_args)
            # interpolate fields from grid vertices to particles
            # check if we need to inject particles
            inject_particles_phase!(particles, pPhases, (), (), xvi)
            # update phase ratios
            update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

            @show it += 1
            t        += dt

            # Data I/O and plotting ---------------------
            if it == 1 || rem(it, 50) == 0
                checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

                if do_vtk
                    velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                    data_v = (;
                        τII = Array(stokes.τ.II),
                        εII = Array(stokes.ε.II),
                    )
                    data_c = (;
                        P   = Array(stokes.P),
                        η   = Array(stokes.viscosity.η),
                    )
                    velocity_v = (
                        Array(Vx_v),
                        Array(Vy_v),
                        Array(Vz_v),
                    )
                    save_vtk(
                        joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                        xvi,
                        xci,
                        data_v,
                        data_c,
                        velocity_v,
                        t=t
                    )
                end

                # # Make Makie figure
                # slice_j = ny >>> 1
                # fig     = Figure(size = (1200, 1200), title = "t = $t")
                # ar      = li[1] / li[3]
                # ax1     = Axis(fig[1,1], aspect = ar, title = "T [C]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
                # ax2     = Axis(fig[2,1], aspect = ar, title = "Shear heating [W/m3]")
                # ax3     = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
                # ax4     = Axis(fig[2,3], aspect = ar, title = "log10(η)")
                # # Plot temperature
                # h1      = heatmap!(ax1, xvi[1].*1e-3, xvi[3].*1e-3, Array(thermal.T[:, slice_j, :].-273.0) , colormap=:batlow)
                # # Plot particles phase
                # h2      = heatmap!(ax2, xvi[1].*1e-3, xvi[3].*1e-3, Array(thermal.shear_heating[:, slice_j, :]) , colormap=:batlow)
                # # Plot 2nd invariant of strain rate
                # h3      = heatmap!(ax3, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(stokes.ε.II[:, slice_j, :])) , colormap=:batlow)
                # # Plot effective viscosity
                # h4      = heatmap!(ax4, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(stokes.viscosity.η_vep[:, slice_j, :])) , colormap=:batlow)
                # hidexdecorations!(ax1)
                # hidexdecorations!(ax2)
                # hidexdecorations!(ax3)
                # Colorbar(fig[1,2], h1)
                # Colorbar(fig[2,2], h2)
                # Colorbar(fig[1,4], h3)
                # Colorbar(fig[2,4], h4)
                # linkaxes!(ax1, ax2, ax3, ax4)
                # save(joinpath(figdir, "$(it).png"), fig)
                # fig
            end
            # ------------------------------

        end

    return nothing
end

figdir   = "SinkingBalls3D"
do_vtk   = false # set to true to generate VTK files for ParaView
n        = 32
nx       = n
ny       = n
nz       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
else
    igg
end

main3D(igg; ar=ar, ny=ny, nx=nx, nz=nz,figdir=figdir, do_vtk=do_vtk)


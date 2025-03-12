# const isCUDA = false
const isCUDA = true

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
using GeoParams, GLMakie, CellArrays

# Load file with all the rheology configurations
include("Subduction2D_setup.jl")
include("VariationalSubduction2D_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
using StaticArrays
function correct_phase_ratio(air_phase, ratio::SVector{N, T}) where {N, T}
    if iszero(air_phase)
        return ratio
    elseif ratio[air_phase] ≈ 1
        return SVector{N, T}(zero(T) for _ in 1:N)
    else
        mask = ntuple(i -> (i !== air_phase), Val(N))
        # set air phase ratio to zero
        corrected_ratio = ratio .* mask
        # normalize phase ratios without air
        # return corrected_ratio ./ sum(corrected_ratio)
    end
end

@parallel_indices (I...) function renormalize_phase_ratios(ratios_center, ratios_vertex, air_phase)
    # renormalize centers
    if all(I .≤ size(ratios_center))
        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        @cell ratios_center[I...] = correct_phase_ratio(air_phase, ratio_ij)
    end

    # renormalize centers
    # local phase ratio
    ratio_ij = @cell ratios_vertex[I...]
    # remove phase ratio of the air if necessary & normalize ratios
    @cell ratios_vertex[I...] = correct_phase_ratio(air_phase, ratio_ij)
    return nothing
end

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = 10.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 50
    max_xcell = 75
    min_xcell = 30
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    phase_ratio_vx = @fill(0.0, nx + 1, ny, celldims = length(rheology))
    phase_ratio_vy = @fill(0.0, nx, ny + 1, celldims = length(rheology))

    phase_ratios_midpoint!(phase_ratio_vx, particles, xci, pPhases, :x)
    phase_ratios_midpoint!(phase_ratio_vy, particles, xci, pPhases, :y)
    # ----------------------------------------------------

    # RockRatios
    air_phase = 3
    ϕ_R = RockRatio(backend, ni)
    update_rock_ratio!(ϕ_R, phase_ratios, (phase_ratio_vx, phase_ratio_vy), air_phase)
    # @parallel (@idx ni.+1) renormalize_phase_ratios(phase_ratios.center, phase_ratios.vertex, air_phase)

    # marker chain
    nxcell, min_xcell, max_xcell = 12, 6, 24
    initial_elevation = 0.0e0
    chain = init_markerchain(JustPIC.CPUBackend, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, Re = 3.0e0, r = 0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))

    # Rheology
    args0 = (T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, air_phase, viscosity_cutoff)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0

    while it < 100 # run only for 5 Myrs

        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)

        # Stokes solver ----------------
        t_stokes = @elapsed begin
            # out = solve!(
            #     stokes,
            #     pt_stokes,
            #     di,
            #     flow_bcs,
            #     ρg,
            #     phase_ratios,
            #     rheology,
            #     args,
            #     dt,
            #     igg;
            #     kwargs = (
            #         iterMax          = 100e3,
            #         nout             = 2e3,
            #         viscosity_cutoff = viscosity_cutoff,
            #         free_surface     = false,
            #         viscosity_relaxation = 1e-2
            #     )
            # );
            solve_VariationalStokes!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                ϕ_R,
                rheology,
                args,
                dt,
                igg;
                kwargs = (;
                    iterMax = 50.0e3, #250e3,
                    # free_surface     = false,
                    nout = 2.0e3, #5e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di) * 0.8
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        phase_ratios_midpoint!(phase_ratio_vx, particles, xci, pPhases, :x)
        phase_ratios_midpoint!(phase_ratio_vy, particles, xci, pPhases, :y)
        update_rock_ratio!(ϕ_R, phase_ratios, (phase_ratio_vx, phase_ratio_vy), air_phase)
        @parallel (@idx ni .+ 1) renormalize_phase_ratios(phase_ratios.center, phase_ratios.vertex, air_phase)

        advect_markerchain!(chain, method, V, grid_vxi, dt)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                )
                data_c = (;
                    P = Array(stokes.P),
                    η = Array(η_vep),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            # Make particles plottable
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:] ./ 1.0e3
            pyv = ppy.data[:] ./ 1.0e3
            clr = pPhases.data[:]
            # clr      = pT.data[:]
            idxv = particles.index.data[:]

            chain_x = chain.coords[1].data[:] ./ 1.0e3
            chain_y = chain.coords[2].data[:] ./ 1.0e3

            # Make Makie figure
            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "log10(εII)  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "τII")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.ε.II)), colormap = :batlow)
            scatter!(ax1, chain_x, chain_y, markersize = 3)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 1)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array((stokes.τ.II)), colormap = :batlow)
            scatter!(ax3, chain_x, chain_y, markersize = 3)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            display(fig)
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return stokes
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView
figdir = "Subduction2D_MQS_variational"
nx, ny = 250, 100
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx + 1, ny + 1)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

stokes = main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);

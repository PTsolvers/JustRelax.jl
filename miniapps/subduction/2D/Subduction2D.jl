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
using GeoParams, GLMakie, CellArrays

# Load file with all the rheology configurations
include("Subduction2D_setup.jl")
include("Subduction2D_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, igg; nx=16, ny=16, figdir="figs2D", do_vtk =false)

    # Physical domain ------------------------------------
    ni                  = nx, ny           # number of cells
    di                  = @. li / ni       # grid steps
    grid                = Geometry(ni, li; origin = origin)
    (; xci, xvi)        = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # rheology            = init_rheology_linear()
    # rheology            = init_rheology_nonNewtonian()
    rheology            = init_rheology_nonNewtonian_plastic()
    dt                  = 10e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell              = 40
    max_xcell           = 60
    min_xcell           = 20
    particles           = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays      = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi            = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT         = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ  = pτxx, pτyy, pτxy        = init_cell_arrays(particles, Val(3)) # stress
    pτ_o = pτxx_o, pτyy_o, pτxy_o = init_cell_arrays(particles, Val(3)) # old stress
    pω   = pωxy,                  = init_cell_arrays(particles, Val(1)) # vorticity
    particle_args                 = (pT, pPhases, pτ..., pω...)
    particle_args_reduced         = (pT, pτ..., pω...)

    # Assign particles phases anomaly
    phases_device    = PTArray(backend)(phases_GMG)
    phase_ratios     = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni);
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3π, r=0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop             = 20 + 273
    Tbot             = maximum(T_GMG)
    thermal          = ThermalArrays(backend, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1]   .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Rheology
    args0            = (T=thermal.Tc, P=stokes.P, dt = Inf)
    viscosity_cutoff = (1e18, 1e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ=1e-5, CFL=0.95 / √3
    )

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    # Time loop
    t, it = 0.0, 0

    while it < 1000 # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= Ttop
        @views T_buffer[:, 1]        .= Tbot
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)

        # Stokes solver ----------------
        particle2centroid!(stokes.τ.xx, pτxx, xci, particles)
        particle2centroid!(stokes.τ.yy, pτyy, xci, particles)
        particle2grid!(stokes.τ.xy, pτxy, xvi, particles)


        a = copy(stokes.τ.xx)
        b = copy(stokes.τ.yy)
        c = copy(stokes.τ.xy)

        particle2centroid!(a, pτxx, xci, particles)
        particle2centroid!(b, pτyy, xci, particles)
        particle2grid!(c, pτxy, xvi, particles)

        t_stokes = @elapsed begin
            out = solve!(
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
                kwargs = (
                    iterMax          = 50e3,
                    nout             = 2e3,
                    viscosity_cutoff = viscosity_cutoff,
                    free_surface     = false,
                    viscosity_relaxation = 1e-2
                )
            );
        end

        centroid2particle!(pτxx , xci, stokes.τ.xx, particles)
        centroid2particle!(pτyy , xci, stokes.τ.yy, particles)
        grid2particle!(pτxy, xvi, stokes.τ.xy, particles)
        rotate_stress_particles!(pτ, pω, particles, dt)

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")
        tensor_invariant!(stokes.ε)
        dt   = compute_dt(stokes, di) * 0.8
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 50e3,
                nout    = 1e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)

        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(T_buffer),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (;
                    P   = Array(stokes.P),
                    η   = Array(η_vep),
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
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]
            # clr      = pT.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            ar  = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # Plot particles phase
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            # Plot 2nd invariant of strain rate
            h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            # Plot effective viscosity
            h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[1,4], h3)
            Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "Subduction2D"
n        = 128
nx, ny   = 128, 64
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx+1, ny+1)
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

# main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);

# @parallel_indices (I...) function rotate_stress_particles_rotation_matrix!(
#     xx, yy, xy, ω, index, dt
# )

#     for ip in cellaxes(index)
#         @index(index[ip, I...]) || continue # no particle in this location

#         θ = dt * @index ω[ip, I...]
#         sinθ, cosθ = sincos(θ)

#         τ_xx = @index xx[ip, I...]
#         τ_yy = @index yy[ip, I...]
#         τ_xy = @index xy[ip, I...]

#         R = @SMatrix [
#             cosθ -sinθ
#             sinθ cosθ
#         ]

#         τ = @SMatrix [
#             τ_xx τ_xy
#             τ_xy τ_yy
#         ]

#         # this could be fully unrolled in 2D
#         τr = R * τ * R'

#         @index xx[ip, I...] = τr[1, 1]
#         @index yy[ip, I...] = τr[2, 2]
#         @index xy[ip, I...] = τr[1, 2]
#     end

#     return nothing
# end

# @parallel (@idx ni) rotate_stress_particles_rotation_matrix!(
#     pτxx, pτyy, pτxy, pωxy, particles.index, dt
# )


# function rotate_stress_particles!(τ::NTuple, ω::NTuple, particles, dt; method::Symbol = :matrix)
#     fn = if method === :matrix
#         rotate_stress_particles_rotation_matrix!

#     elseif method === :jaumann
#         rotate_stress_particles_jaumann!

#     else
#         error("Unknown method: $method. Valid methods are :matrix and :jaumann")
#     end
#     @parallel (@idx ni) fn(τ..., ω..., particles.index, dt)
    
#     return nothing 
# end

# rotate_stress_particles!(
#     (pτxx, pτyy, pτxy), (pωxy,), particles, dt
# )


# const isCUDA = false
const isCUDA = false

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
using GeoParams, CairoMakie, CellArrays

# Load file with all the rheology configurations
include("Subduction2D_setup_MPI.jl")
include("Subduction2D_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

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
function main(x_global, z_global, li, origin, phases_GMG, T_GMG, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / (nx_g(), ny_g())       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # rheology            = init_rheology_linear()
    # rheology            = init_rheology_nonNewtonian()
    rheology = init_rheology_nonNewtonian_plastic()
    dt = 10.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 40
    max_xcell = 60
    min_xcell = 20
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ = pτxx, pτyy, pτxy = init_cell_arrays(particles, Val(3)) # stress
    # pτ_o = pτxx_o, pτyy_o, pτxy_o = init_cell_arrays(particles, Val(3)) # old stress
    pω = pωxy, = init_cell_arrays(particles, Val(1)) # vorticity
    particle_args = (pT, pPhases, pτ..., pω...)
    particle_args_reduced = (pT, pτ..., pω...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    update_cell_halo!(particles.coords..., particle_args...)
    update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, Re = 3.0e0, r = 0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7
    # pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-5, Re = 2π√2, r=0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop = 20 + 273
    Tbot = maximum(T_GMG)
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1] .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))

    # Rheology
    args0 = (T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, 0, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

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
        Vx = @zeros(ni...)
        Vy = @zeros(ni...)
    end

    #MPI
    # global array
    nx_v = (nx - 2) * igg.dims[1]
    ny_v = (ny - 2) * igg.dims[2]
    # center
    P_v = zeros(nx_v, ny_v)
    τII_v = zeros(nx_v, ny_v)
    η_vep_v = zeros(nx_v, ny_v)
    εII_v = zeros(nx_v, ny_v)
    phases_c_v = zeros(nx_v, ny_v)
    #center nohalo
    P_nohalo = zeros(nx - 2, ny - 2)
    τII_nohalo = zeros(nx - 2, ny - 2)
    η_vep_nohalo = zeros(nx - 2, ny - 2)
    εII_nohalo = zeros(nx - 2, ny - 2)
    phases_c_nohalo = zeros(nx - 2, ny - 2)
    #vertex
    Vxv_v = zeros(nx_v, ny_v)
    Vyv_v = zeros(nx_v, ny_v)
    T_v = zeros(nx_v, ny_v)
    #vertex nohalo
    Vxv_nohalo = zeros(nx - 2, ny - 2)
    Vyv_nohalo = zeros(nx - 2, ny - 2)
    T_nohalo = zeros(nx - 2, ny - 2)

    xci_v = LinRange(minimum(x_global) .* 1.0e3, maximum(x_global) .* 1.0e3, nx_v),
        LinRange(minimum(z_global) .* 1.0e3, maximum(z_global) .* 1.0e3, ny_v)


    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0

    fig_iters = Figure(size = (1200, 800))
    ax_iters1 = Axis(fig_iters[1, 1], aspect = 1, title = "error")
    ax_iters2 = Axis(fig_iters[1, 2], aspect = 1, title = "num iters / ny")

    while it < 1000 # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)

        particle2centroid!(stokes.τ.xx, pτxx, xci, particles)
        particle2centroid!(stokes.τ.yy, pτyy, xci, particles)
        particle2grid!(stokes.τ.xy, pτxy, xvi, particles)

        # Stokes solver ----------------
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
                    iterMax = 100.0e3,
                    nout = 2.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                    free_surface = false,
                    viscosity_relaxation = 1.0e-2,
                )
            )

            scatter!(ax_iters1, [it], [log10(out.err_evo1[end])], markersize = 10, color = :blue)
            scatter!(ax_iters2, [it], [out.iter / ny], markersize = 10, color = :blue)
            fig_iters

            if it == 1 || rem(it, 10) == 0
                save(joinpath(figdir, "errors.png"), fig_iters)
            end
        end

        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        centroid2particle!(pτxx, xci, stokes.τ.xx, particles)
        centroid2particle!(pτyy, xci, stokes.τ.yy, particles)
        grid2particle!(pτxy, xvi, stokes.τ.xy, particles)
        rotate_stress_particles!(pτ, pω, particles, dt)

        if igg.me == 0
            println("Stokes solver time             ")
            println("   Total time:      $t_stokes s")
            println("   Time/iteration:  $(t_stokes / out.iter) s")
        end
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dt = compute_dt(stokes, di) * 0.8
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
                igg = igg,
                phase = phase_ratios,
                iterMax = 50.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)

        update_cell_halo!(particles.coords..., particle_args...)
        update_cell_halo!(particles.index)

        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        # inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
            xvi
        )

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        if igg.me == 0
            @show it += 1
            t += dt
        end

        #MPI gathering
        phase_center = [argmax(p) for p in Array(phase_ratios.center)]
        #centers
        @views P_nohalo .= Array(stokes.P[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views τII_nohalo .= Array(stokes.τ.II[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1), 2:(end - 1)])       # Copy data to CPU removing the halo
        @views εII_nohalo .= Array(stokes.ε.II[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views phases_c_nohalo .= Array(phase_center[2:(end - 1), 2:(end - 1), 2:(end - 1)])
        @async gather!(P_nohalo, P_v)
        gather!(τII_nohalo, τII_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(εII_nohalo, εII_v)
        gather!(phases_c_nohalo, phases_c_v)
        #vertices
        if do_vtk
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            vertex2center!(Vx, Vx_v)
            vertex2center!(Vy, Vy_v)
            @views Vxv_nohalo .= Array(Vx[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            @views Vyv_nohalo .= Array(Vy[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            gather!(Vxv_nohalo, Vxv_v)
            gather!(Vyv_nohalo, Vyv_v)
        end
        @views T_nohalo .= Array(thermal.Tc[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)
            if do_vtk

                data_c = (;
                    T = T_v,
                    P = P_v,
                    τII = τII_v,
                    εII = εII_v,
                    η = η_vep_v,
                    phases = phases_c_v,


                )
                velocity_v = (
                    Array(Vxv_v),
                    Array(Vyv_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$(it)", 6, "0")),
                    xci_v ./ 1.0e3,
                    data_c,
                    velocity_v;
                    t = t
                )
            end

            # Make Makie figure
            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xci_v[1] .* 1.0e-3, xci_v[2] .* 1.0e-3, Array(T_v), colormap = :batlow)
            # Plot particles phase
            # h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            h2 = heatmap!(ax2, xci_v[1] .* 1.0e-3, xci_v[2] .* 1.0e-3, Array(phases_c_v), colormap = :batlow)
            # Plot 2nd invariant of strain rate
            # h3  = heatmap!(ax3, xci_v[1].*1e-3, xci_v[2].*1e-3, Array(log10.(εII_v)) , colormap=:batlow)
            h3 = heatmap!(ax3, xci_v[1] .* 1.0e-3, xci_v[2] .* 1.0e-3, Array(τII_v), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci_v[1] .* 1.0e-3, xci_v[2] .* 1.0e-3, Array(log10.(η_vep_v)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView
nx, ny = 128, 128


igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
extents = Array{NTuple}[]

# GLOBAL Physical domain ------------------------------------
model_depth = 660
x_global = range(0, 3000, nx_g());
air_thickness = 15.0
z_global = range(-model_depth, air_thickness, ny_g());
origin = (x_global[1], z_global[1])
li = (abs(last(x_global) - first(x_global)), abs(last(z_global) - first(z_global)))

ni = nx, ny           # number of cells
di = @. li / (nx_g(), ny_g())           # grid steps
grid_global = Geometry(ni, li; origin = origin)

figdir = "Subduction2D_$(nx_g())x$(ny_g())"

li_GMG, origin_GMG, phases_GMG, T_GMG = GMG_subduction_2D(model_depth, grid_global.xvi, nx + 1, ny + 1)


function generate_extents(nx, ny, num_processes)
    extents = Array{NTuple{2, UnitRange{Int}}}(undef, num_processes)
    for i in 1:num_processes
        x_start = (i - 1) * div(nx, num_processes[1]) + 1
        x_end = i * div(nx, num_processes)
        y_start = (i - 1) * div(ny, num_processes[2]) + 1
        y_end = ny
        extents[i] = (x_start:x_end, y_start:y_end)
    end
    return extents
end


ni = nx, ny           # number of cells
di = @. li_GMG / (nx_g(), ny_g())       # grid steps
grid = Geometry(ni, li; origin = origin_GMG)
(; xci, xvi) = grid

# extents = generate_extents(nx(), ny(), igg.dims)
# push!(extents, xci...)
# println("extents: ",extents)
println("nx()", @nx())
println("ny()", @ny())

#  main(x_global, z_global,li_GMG, origin_GMG, phases_GMG, T_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);

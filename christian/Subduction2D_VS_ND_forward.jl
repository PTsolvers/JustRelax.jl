# Load script dependencies
using GeoParams, GLMakie

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

# Load file with all the rheology configurations
include("Subduction2D_setup.jl")
include("VariationalSubduction2D_rheology.jl")

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

# @parallel_indices (i, j) function update_Dirichlet_mask!(mask, phase_ratio_vertex, air_phase)
#     @inbounds mask[i + 1, j] = @index(phase_ratio_vertex[air_phase, i, j]) ≈ 1
#     nothing
# end

@parallel_indices (i, j) function update_Dirichlet_mask!(mask, y, h_vertices)
    @inbounds mask[i + 1, j] = y[j] .≥ h_vertices[i]
    nothing
end
    

@parallel_indices (I...) function fix_air_particle_T!(pPhases, pT, Tair, air_phase)

    for ip in cellaxes(pT)
        if @index(pPhases[ip, I...]) == air_phase
            @inbounds @index pT[ip, I...] = Tair
        end
    end

    nothing
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false)
    
    thickness = 750e3 * m
    η0        = 1.0e22 * Pa * s
    CharDim   = GEO_units(;
        length = thickness, viscosity = η0, temperature = maximum(T_GMG) * K
    )
    li_nd     = nondimensionalize(li .* m, CharDim)
    origin_nd = nondimensionalize(origin .* m, CharDim)
   
    # Physical domain ------------------------------------
    ni    = nx, ny           # number of cells
    di    = @. li_nd / ni       # grid steps
    grid  = Geometry(ni, li_nd; origin = origin_nd)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim)
    # rheology = init_rheologies(CharDim)
    # rheology = init_rheology_linear(CharDim)
    dt       = nondimensionalize(1e3 * yr, CharDim)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell    = 40
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
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # marker chain
    nxcell, min_xcell, max_xcell = 100, 75, 125
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)

    # RockRatios
    air_phase = 11
    ϕ_R = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ_R, chain, xvi, di)
    air_phase_real = 11
    update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, di, air_phase_real, particle_args_reduced)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(
        li_nd, di;
        ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, Re = 17π, r = 0.7, CFL = 0.9 / √2.1
        # ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, Re = 25.0e0, r = 0.7, CFL = 0.9 / √2.1
    ) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    # Ttop     = nondimensionalize((20 + 273e0)K, CharDim)
    T_GMG_nd   = nondimensionalize(T_GMG * K, CharDim)
    Ttop, Tbot = extrema(T_GMG_nd)
    thermal    = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG_nd)

    # Add thermal anomaly BC's
    T_air = Ttop
    Ω_T   = @zeros(size(thermal.T)...)
    
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (; left = true, right = true, top = false, bot = false),
        # dirichlet   = (; constant = T_air, mask = Ω_T)
    )
    # @parallel (@idx ni .+ 1) update_Dirichlet_mask!(thermal_bc.dirichlet.mask, xvi[2], chain.h_vertices)
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
    viscosity_cutoff = nondimensionalize((1.0e18, 1.0e23) .* (Pa * s), CharDim)
    # compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase_real)
    # compute_viscosity_εII!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase_real)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li_nd; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
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

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    # for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
    #     copyinn_x!(dst, src)
    # end

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0
    dt_max = nondimensionalize(10e3 * yr, CharDim)

    for _ = 1:(10 * 5) 
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            nondimensionalize(100e3 * yr, CharDim),
            di;
            kwargs = (
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 10.0e3,
                nout    = 1.0e2,
                verbose = false,
            )
        )
    end
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase_real)

    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    while it < 500 # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        @parallel (@idx ni) fix_air_particle_T!(pPhases, pT, Ttop, air_phase_real)
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]          .= Ttop
        @views T_buffer[:, 1]            .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # Tmin, Tmax = extrema(thermal.T)
        # @show dimensionalize_and_strip(minimum(Tmin), C, CharDim)
        # @show dimensionalize_and_strip(minimum(Tmax), C, CharDim)
        # if dimensionalize_and_strip(minimum(Tmin), C, CharDim) < 15
        #     println("pew pew 1")
        #     break
        # end
        
        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, xvi, xci, particles)

        # Stokes solver ----------------
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        t_stokes = @elapsed begin
            out = solve_VariationalStokes!(
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
                    iterMax = 100.0e3,
                    free_surface = true,
                    nout = 5.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                    viscosity_relaxation = 1.0e-3,
                    λ_relaxation = 1e-2,
                )
            )
        end

        # print some stuff
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")

        # compute time step
        dt_max = nondimensionalize(
            25e3 * yr, 
            # (it < 5 ? 1e3 : 5e3) * yr, 
            CharDim
        )

        dt     = compute_dt(stokes, di, dt_max) * 0.95
        dt_dim = dimensionalize_and_strip(dt, yr, CharDim) / 1e3
        println("   dt:              $dt_dim kyrs")

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, xci, xvi, dt)
        # compute strain rate 2nd invartian - for plotting
        tensor_invariant!(stokes.ε)
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

        # Tmin, Tmax = extrema(thermal.T)
        # @show dimensionalize_and_strip(minimum(Tmin), C, CharDim)
        # @show dimensionalize_and_strip(minimum(Tmax), C, CharDim)
        # if dimensionalize_and_strip(minimum(Tmin), C, CharDim) < 15
        #     println("pew pew 1")
        #     break
        # end

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
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, di, air_phase_real, particle_args_reduced)

        # check if we need to inject particles
        # need stresses on the vertices for injection purposes
        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
            xvi
        )

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        compute_rock_fraction!(ϕ_R, chain, xvi, di)

        # @parallel (@idx ni .+ 1) update_Dirichlet_mask!(thermal_bc.dirichlet.mask, phase_ratios.vertex, air_phase_real)
        # @parallel (@idx ni .+ 1) update_Dirichlet_mask!(thermal_bc.dirichlet.mask, xvi[2], chain.h_vertices)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            xvi_dim = ntuple(i -> dimensionalize_and_strip(xvi[i], km, CharDim), Val(2))
            xci_dim = ntuple(i -> dimensionalize_and_strip(xci[i], km, CharDim), Val(2))
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = dimensionalize_and_strip(Array(T_buffer), C, CharDim),
                    density = dimensionalize_and_strip(Array(ρg[2]),m/s^2 *kg/m^3, CharDim)./9.81,
                    τII = dimensionalize_and_strip(Array(stokes.τ.II), Pa, CharDim),
                    εII = dimensionalize_and_strip(Array(stokes.ε.II), s^-1, CharDim),
                )
                data_c = (;
                    P       = dimensionalize_and_strip(Array(stokes.P), Pa, CharDim),
                    density = dimensionalize_and_strip(Array(η_vep), Pa*s, CharDim),
                    η_vep   = dimensionalize_and_strip(Array(η_vep), Pa*s, CharDim),
                    η       = dimensionalize_and_strip(Array(η), Pa*s, CharDim),
                )
                velocity_v = (
                    dimensionalize_and_strip(Array(Vx_v), m/s, CharDim),
                    dimensionalize_and_strip(Array(Vy_v), m/s, CharDim),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi_dim,
                    xci_dim,
                    data_v,
                    data_c,
                    velocity_v;
                    t = t
                )
                save_marker_chain(
                        joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                        LinRange(xvi_dim[1][1], xvi_dim[1][end], length(xvi_dim[1])),
                        Array(ustrip(dimensionalize(chain.h_vertices, km, CharDim))),
                )
            end

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = dimensionalize_and_strip(Array(ppx.data[:]), km, CharDim)
            pyv      = dimensionalize_and_strip(Array(ppy.data[:]), km, CharDim)
            clr      = Array(pPhases.data[:])
            idxv     = Array(particles.index.data[:])

            # Make Makie figure
            t_Myrs = dimensionalize_and_strip(t, Myr, CharDim)
            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$t_Myrs Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            # ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "τII")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(
                ax1, xvi_dim..., dimensionalize_and_strip(Array(thermal.T[2:(end - 1), :]), K, CharDim), colormap = :batlow
            )
            # Plot particles phase
            h2 = scatter!(ax2, pxv[idxv], pyv[idxv], color = clr[idxv], markersize = 1)
            # Plot 2nd invariant of strain rate
            # h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            h3 = heatmap!(
                ax3, xci_dim..., dimensionalize_and_strip(Array((stokes.τ.II)), MPa, CharDim), colormap = :batlow
            )
            # Plot effective viscosity
            h4 = heatmap!(
                ax4, xci_dim..., 
                log10.(dimensionalize_and_strip(Array((stokes.viscosity.η_vep)), Pa*s, CharDim)), 
                colormap = :batlow
            )
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
figdir = "Subduction2D_VS"
# n = 128
n = 254
nx, ny = n * 2, n
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx + 1, ny + 1)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(li, origin, phases_GMG, T_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
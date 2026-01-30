# const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
import JustRelax.@cell

const backend_JR = @static if isCUDA
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
# Threads is the default backend_JR,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, GLMakie
using JLD2

# Load file with all the rheology configurations
include("Rift2DSetupFaultInclusion.jl")
include("RiftRheology.jl")

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
    @all(P) = abs(@all(ρg) * @all_k(z)) #* <(@all_k(z), 0.0)
    return nothing
end


# Initial thermal profile
function thermal_anomaly!(T, xvi, anomaly_center, anomaly_radius, ΔT)

    @parallel_indices (i, j) function f_anomaly!(T, xci, anomaly_center, anomaly_radius, ΔT)
        r = sqrt(( xci[1][i] - anomaly_center[1])^2 + ( xci[2][j] - anomaly_center[2])^2)
        if r ≤ anomaly_radius
            T[i+1, j] +=  ΔT
        end
        return nothing
    end
    ni = length.(xvi)
    @parallel (@idx ni) f_anomaly!(T, xvi, anomaly_center, anomaly_radius, ΔT)

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, T_GMG, phases_GMG, igg; nx=16, ny=16, figdir="figs2D", do_vtk =false)

    thickness = li[1] * m
    η0        = 1.0e22 * Pa * s
    CharDim   = GEO_units(;
        length = thickness, viscosity = η0, temperature = maximum(T_GMG)*K
    )
    # Physical domain ------------------------------------
    li_nd        = nondimensionalize(li .* m, CharDim)
    origin_nd    = nondimensionalize(origin .* m, CharDim)
    ni           = nx, ny           # number of cells
    di           = @. li_nd / ni       # grid steps
    grid         = Geometry(ni, li_nd; origin = origin_nd)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim; is_plastic = true)
    dt       = nondimensionalize(10e3 * yr, CharDim) # initial timestep in seconds
    dtmax    = nondimensionalize(50e3 * yr, CharDim) # initial timestep in seconds
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell    = 75
    max_xcell = 100
    min_xcell = 50
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi...
    )
    # velocity grids
    grid_vxi       = velocity_grids(xci, xvi, di)
    # material phase & temperature
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    pPhases, pT    = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend_JR)(phases_GMG)
    phase_ratios  = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    # init_phases!(pPhases, phases_device, particles, xvi)
    init_phases!(pPhases, particles, CharDim)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------
 
    # RockRatios
    air_phase = 5
    ϕ_R = RockRatio(backend_JR, ni)
 
    # marker chain
    nxcell, min_xcell, max_xcell = 100, 75, 125
    initial_elevation = nondimensionalize(0e3 * m, CharDim)
    # initial_elevation = -0.005
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, di, air_phase)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    compute_rock_fraction!(ϕ_R, chain, xvi, di)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li_nd, di;  ϵ_rel=1e-4, ϵ_abs = 1e-3, Re=3*√10*π, r=1e0, CFL = 0.98 / √2.1) # Re=3π, r=0.7
    # pt_stokes        = PTStokesCoeffs(li_nd, di;  ϵ_rel=1e-5,ϵ_abs=1e-4, Re=3*√10*π/2, r=0.5, CFL = 0.85 / √2.1) # Re=3π, r=0.7
    σ = PrincipalStress(backend_JR, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    T_GMG_nd       = nondimensionalize(T_GMG .* K, CharDim)
    Ttop           = nondimensionalize(0C, CharDim) #minimum(T_GMG_nd)
    Tbot           = nondimensionalize(1330C, CharDim) #maximum(T_GMG_nd)
    thermal        = ThermalArrays(backend_JR, ni)

    anomaly_center = nondimensionalize( (0e0, -25e3) .* m, CharDim)
    anomaly_radius = nondimensionalize( 5e3 * m, CharDim)
    ΔT             = nondimensionalize( 100 * K, CharDim)
    @views thermal.T[2:end-1, :] .= PTArray(backend_JR)(T_GMG_nd)
    # thermal_anomaly!(thermal.T, xvi, anomaly_center, anomaly_radius, ΔT)

    # Add thermal anomaly BC's
    # T_air = nondimensionalize(273.0e0 .* K, CharDim)
    # Ω_T = @zeros(size(thermal.T)...)
    
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (; left = true, right = true, top = false, bot = false),
        # dirichlet   = (; constant = T_air, mask = Ω_T)
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1]   .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    for _ in 1:10
        compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        stokes.P        .= PTArray(backend_JR)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))
    end
    # PT coefficients for thermal diffusion
    args0            = (T=thermal.Tc, P=stokes.P, dt = Inf)

    pt_thermal       = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args0, dt, ni, di, li_nd; ϵ=1e-5, CFL=0.98 / √2.1
    )

    # Thermal solver ---------------
    thermal.T    .= nondimensionalize(1330C, CharDim)
    thermal.Told .= thermal.T

    dt_thermal = nondimensionalize(1e6 * yr, CharDim)
    tmax_thermal = nondimensionalize(5e6 * yr, CharDim)
    t_thermal = 0e0
    while t_thermal < tmax_thermal
        thermal.T[:, xvi[2].>0] .= nondimensionalize(0C, CharDim)
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args0,
            dt_thermal,
            di;
            kwargs = (
                igg     = nothing,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 2e2,
                verbose = true,
            )
        )
        t_thermal += dt_thermal
    end
    thermal_anomaly!(thermal.T, xvi, anomaly_center, anomaly_radius, ΔT)
    temperature2center!(thermal)
    thermal_bcs!(thermal, thermal_bc)
    
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
    )

    εbg = nondimensionalize(1e-14 / s, CharDim)  # background strain rate 
    air_thickness = nondimensionalize(10e3 * m, CharDim)
    @views stokes.V.Vx[:, 2:(end - 1)] .= PTArray(backend_JR)([ εbg * x for x in xvi[1], y in xci[2]])
    @views stokes.V.Vy[2:(end - 1), :] .= PTArray(backend_JR)([-εbg * y for x in xci[1], y in xvi[2]])

    # Rheology
    stokes.ε.xx     .= εbg # to get a good initial guess
    viscosity_cutoff = nondimensionalize((1e17, 1e23) .* (Pa * s), CharDim)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    update_halo!(@velocity(stokes)...)
    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
        fig_dir      = joinpath(figdir, "fig")
        take(vtk_dir)
        take(fig_dir)
    end
    # ----------------------------------------------------

    local Vx_v, Vy_v,Ux_v, Uy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Ux_v = @zeros(ni.+1...)
        Uy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    P_lith = @zeros(ni...)
    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)
    # Time loop
    t, it = 0.0, 0

    tmax = nondimensionalize(50.0e6 * yr, CharDim) # max time in seconds

    # while it < 2 # run only for 5 Myrs
    while t < tmax # run only for 5 Myrs
        
        if it == 1
            P_lith .= stokes.P
        end
        
        # flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        # update_halo!(@velocity(stokes)...)      

        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        @show it += 1
        t        += dt

        # Stokes solver ----------------
        result, t_stokes,_,_,_ = @timed begin
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
                    iterMax              = 150e3,
                    free_surface         = true,
                    nout                 = 2e3,
                    viscosity_relaxation = 1e-4,
                    λ_relaxation         = 1e-3,
                    viscosity_cutoff     = viscosity_cutoff,
                )
            );
        end;
        dt     = compute_dt(stokes, di, dtmax)
        dt_dim = dimensionalize(dt, yr, CharDim)

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("           Δt:      $(ustrip(dt_dim/1e3)) kyrs")
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        # ------------------------------

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, xci, xvi, dt)

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

        # Thermal solver ---------------
        # update mask of Dirichlet BC
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (
                igg     = nothing,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 2e2,
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
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
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

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        compute_rock_fraction!(ϕ_R, chain, xvi, di)

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= Ttop
        @views T_buffer[:, 1]        .= Tbot
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, xvi, xci, particles)

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            xvi_dim = ntuple(i->ustrip(dimensionalize(xvi[i], km, CharDim)), Val(2))
            xci_dim = ntuple(i->ustrip(dimensionalize(xci[i], km, CharDim)), Val(2))
                    
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
            if do_vtk

                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                velocity2vertex!(Ux_v, Uy_v, @displacement(stokes)...)
                data_v = (;
                    T  = Array(ustrip(dimensionalize(T_buffer, C, CharDim))),
                    Vx = Array(ustrip(dimensionalize(Vx_v, m/s, CharDim))),
                    Vy = Array(ustrip(dimensionalize(Vy_v, m/s, CharDim))),
                    # Ux = Array(dimensionalize(Ux_v, , CharDim)),
                    # Uy = Array(dimensionalize(Uy_v, , CharDim)),
                )
                data_c = (;
                    Pdyno  = ustrip(dimensionalize(Array(stokes.P) .- Array(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2)), Pa, CharDim)),
                    P      = Array(ustrip(dimensionalize(stokes.P, Pa, CharDim))),
                    η      = Array(ustrip(dimensionalize(η, Pa*s, CharDim))),
                    η_vep  = Array(ustrip(dimensionalize(η_vep, Pa*s, CharDim))),
                    τII    = Array(ustrip(dimensionalize(stokes.τ.II, Pa, CharDim))),
                    τxx    = Array(ustrip(dimensionalize(stokes.τ.xx, Pa, CharDim))),
                    τyy    = Array(ustrip(dimensionalize(stokes.τ.yy, Pa, CharDim))),
                    εII    = Array(ustrip(dimensionalize(stokes.ε.II, s^-1, CharDim))),
                    εII_pl = Array(ustrip(dimensionalize(stokes.ε_pl.II, s^-1, CharDim))),
                    ρ      = Array(ustrip(dimensionalize(ρg[2], m/s^2 * kg / m^3, CharDim)), ) ./ 9.81,
                )
                velocity_v = (
                    Array(ustrip(dimensionalize(Vx_v, m / s, CharDim))),
                    Array(ustrip(dimensionalize(Vy_v, m / s, CharDim))),
                )
                save_marker_chain(
                    joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                    LinRange(xvi_dim[1][1], xvi_dim[1][end], length(xvi_dim[1])),
                    Array(ustrip(dimensionalize(chain.h_vertices, km, CharDim))),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi_dim,
                    xci_dim,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            checkpointing_jld2(
                figdir,
                stokes,
                thermal,
                t,
                dt,
            )

            checkpointing_particles(
                figdir,
                particles,
                phases = pPhases,
                phase_ratios = phase_ratios,
                chain = chain,
                t = t,
                dt = dt,
                particle_args = particle_args,
            )

            fig = Figure(size = (800, 400))

            # Add an axis
            ax = Axis(fig[1, 1],  title = "Err vs PT-Iteration", xlabel = "Pseudo-Iteration", ylabel = "log10(Error Abs)")
            lines!(ax, 1:length(result.err_evo1), log10.(result.err_evo1), color = :blue)
            scatter!(ax, 1:length(result.err_evo1), log10.(result.err_evo1), color = :red)
            save(joinpath(fig_dir, "err_" * lpad("$it", 6, "0") * ".png"),fig)

            # Make particles plottable
            tensor_invariant!(stokes.ε)
            tensor_invariant!(stokes.ε_pl)

            stokes.viscosity.η_vep[stokes.viscosity.η_vep.==0] .= NaN

            t_dim = dimensionalize(t, yr, CharDim)
            chain_x = ustrip(dimensionalize(Array(chain.coords[1].data[:]), km, CharDim))
            chain_y = ustrip(dimensionalize(Array(chain.coords[2].data[:]), km, CharDim))
            ar = DataAspect()
            fig2 = Figure(size = (1200, 900))
            ax1 = Axis(fig2[1, 1], aspect = ar, title = "τxx [MPa]")
            ax2 = Axis(fig2[2, 1], aspect = ar, title = "T [C]")
            ax3 = Axis(fig2[1, 3], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig2[2, 3], aspect = ar, title = "log10(η)")
            ax5 = Axis(fig2[3, 1], aspect = ar, title = "EII_pl")
            ax6 = Axis(fig2[3, 3], aspect = ar, title = "ΔP from P_lith")
            supertitle = Label(fig2[0, :], "t = $(dimensionalize(t, yr, CharDim)/ 1e3) kyr", fontsize = 30)
            # Plot temperature
            h1 = heatmap!(ax1, xvi_dim..., Array(ustrip(dimensionalize(stokes.τ.xx, MPa, CharDim))), colormap = :batlow)
            # Plot velocity
            Vx_dim = Array(ustrip(dimensionalize(thermal.T, C, CharDim)))
            V_range = maximum(abs.(Vx_dim))
            h2 = heatmap!(ax2, xvi_dim..., Vx_dim, colormap = :romaO, colorrange= (0, @dimstrip(Tbot, C, CharDim)))
            scatter!(ax2, Array(chain_x), chain_y, color = :red, markersize = 3)
            # Plot 2nd invariant of stress
            h3 = heatmap!(ax3, xci_dim..., ustrip(dimensionalize(Array(stokes.τ.II), MPa, CharDim)), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci_dim..., log10.(ustrip(dimensionalize(Array(stokes.viscosity.η_vep), Pa*s, CharDim))), colormap = :lipari, colorrange= (18, 23))
            h5 = heatmap!(ax5, xci_dim..., Array(stokes.EII_pl), colormap = :batlow)
            # contour!(ax5, xci_dim..., Array(ϕ_m), levels = [0.5, 0.75, 1.0], color = :white, linewidth = 1.5, labels=true)
            # h6  = heatmap!(ax6, xci[1].*1e-3, xci[2].*1e-3, Array(ϕ_m) , colormap=:lipari, colorrange=(0.0,1.0))
            h6 = heatmap!(ax6, xci_dim..., ustrip(dimensionalize(Array(stokes.P .- P_lith), MPa, CharDim)), colormap = :roma)

            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            hidexdecorations!(ax4)
            hideydecorations!(ax3)
            hideydecorations!(ax4)
            hideydecorations!(ax6)

            Colorbar(fig2[1, 2], h1)
            Colorbar(fig2[2, 2], h2)
            Colorbar(fig2[1, 4], h3)
            Colorbar(fig2[2, 4], h4)
            Colorbar(fig2[3, 2], h5)
            Colorbar(fig2[3, 4], h6)
            linkaxes!(ax1, ax2, ax3, ax4, ax5)
            fig2
            save(joinpath(fig_dir, "sol_" * lpad("$it", 6, "0") * ".png"), fig2)

            # allocate CellArrays to store the velocity field of the marker chain
            chain_V = similar(chain.coords[1]), similar(chain.coords[1])
            chain_V[1].data .*= 0
            chain_V[2].data .*= 0

            interpolate_velocity_to_markerchain!(chain, chain_V, (stokes.V.Vx, stokes.V.Vy), grid_vxi)
            args = Dict(
                :Vx_chain => Array(chain_V[1]),
                :Vy_chain => Array(chain_V[2]),
                :x => xvi[1],
                :y => Array(chain.h_vertices)
            )
            jldsave(joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0") * ".jld2"); args...)

        end
        # ------------------------------


    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "output2_slow_softening/Rift2D_velocity"
# n        = 160
# nx, ny   = n, n÷2
    # Lx = 500
    # Ly = 110
nx, ny   = (250, 125) #.* 2
# nx, ny   = (100, 20) #.* 2
# li, origin, phases_GMG, T_GMG = Setup_Topo(nx+1, ny+1)
li, origin, phases_GMG, T_GMG = flat_setup(nx+1, ny+1)

nx, ny = size(T_GMG).-1

igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

main(li, origin, T_GMG, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);

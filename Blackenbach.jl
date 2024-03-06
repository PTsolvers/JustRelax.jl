# using CUDA
using JustRelax, JustRelax.DataIO
import JustRelax.@cell
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2) #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays
# using Statistics.mean

# Load file with all the rheology configurations
include("Blackenbach_Rheology.jl")
include("asd.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

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

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, thick_air)
    depth = -y[j] - thick_air
    
    if depth < 0e0
        T[i + 1, j] = 273e0

    elseif 0e0 ≤ (depth) < 1000e3
        dTdZ        = (1273-273)/1000e3
        offset      = 273e0
        T[i + 1, j] = (depth) * dTdZ + offset

    elseif (depth) ≥ 1000e3
        offset      = 273e0
        T[i + 1, j] = 1000.0 + offset

    end

    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi, thick_air)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if ((x[i]-xc)^2 ≤ r^2) && ((y[j] - yc - thick_air)^2 ≤ r^2)            
            T[i + 1, j] += 100.0
        end
        return nothing
    end
    ni = length.(xvi)
    @parallel (@idx ni) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D", save_vtk =false)
    # Physical domain ------------------------------------
    thick_air    = 0                    # thickness of sticky air layer
    ly           = 1000e3 + thick_air   # domain length in y
    lx           = ly * ar              # domain length in x
    ni           = nx, ny               # number of cells
    li           = lx, ly               # domain length in x- and y-
    di           = @. li / ni           # grid step in x- and -y
    origin       = 0.0, -ly             # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies()
    κ            = (rheology[1].Conductivity[1].k / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 4.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 30, 8
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pT0, pPhases      = init_cell_arrays(particles, Val(3))
    particle_args    = (pT, pT0, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly       = 0.0    # origin of thermal anomaly
    yc_anomaly       = -600e3  # origin of thermal anomaly
    r_anomaly        = 100e3    # radius of perturbation
    init_phases!(pPhases, particles, lx, yc_anomaly, r_anomaly, thick_air)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.99 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2], thick_air)
    thermal_bcs!(thermal.T, thermal_bc)

    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi, thick_air)
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # Rayleigh number
    ΔT           = thermal.T[1,1] - thermal.T[1,end]
    Ra           = (rheology[1].Density[1].ρ0 * rheology[1].Gravity[1].g * rheology[1].Density[1].α * ΔT * ly^3.0 ) / 
                        (κ * rheology[1].CompositeRheology[1].elements[1].η )
    @show Ra

    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end
    # Rheology
    η                = @ones(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e19, 1e25)
    )
    η_vep            = copy(η)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL= 1e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if save_vtk
        vtk_dir      = figdir*"\\vtk"
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    pT0.data .= pT.data

    local Vx_v, Vy_v
    if save_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0
    nit = 5e3
    Urms = Float64[]
    Nu_top = Float64[]
    trms = Float64[]

    # Buffer arrays to compute velocity rms
    Vx_v  = @zeros(ni.+1...)
    Vy_v  = @zeros(ni.+1...)

    while it <= nit
        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= 273.0        
        @views T_buffer[:, 1]        .= 1273.0
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)
 
        # Update buoyancy and viscosity 
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
            iterMax          = 150e3,
            nout             = 50,
            viscosity_cutoff = (1e18, 1e24)
            verbose          = false,
        )
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        dt = compute_dt(stokes, di, dt_diff)
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
            igg     = igg,
            phase   = phase_ratios,
            iterMax = 10e3,
            nout    = 50,
            verbose = false,
        )
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:end-1, :], subgrid_arrays, particles, xvi,  di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # clean_particles!(particles, xvi, particle_args)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        clean_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject = check_injection(particles)
        # inject && break
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)        
      
        Nu_it = -ly / (1273.0*lx) * mean(( thermal.T[2:end-1,end] - thermal.T[2:end-1,end-1]) / di[2])
        push!(Nu_top, Nu_it)

        # Compute U rms ---------------
        Urms_it = let
            JustRelax.velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=true)
            @. Vx_v .= hypot.(Vx_v, Vy_v) # we reuse Vx_v to store the velocity magnitude
            sqrt(sum(Vx_v.^2) * prod(di)) * ((-ly * rheology[1].Density[1].ρ0 * rheology[1].HeatCapacity[1].Cp) / rheology[1].Conductivity[1].k )
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # ------------------------------

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 100) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)

            if save_vtk
                JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(thermal.T[2:end-1, :]),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (;
                    P   = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    η   = Array(η),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c
                )
            end

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pT.data[:] #pPhases.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            fig = Figure(size = (900, 900), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[1,3], aspect = ar, title = "Vx [m/s]")
            ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            #
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:lajolla, colorrange=(273,1273) )
            # 
            h2  = heatmap!(ax2, xvi[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vy) , colormap=:batlow)
            # 
            h3  = heatmap!(ax3, xvi[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vx) , colormap=:batlow)
            # 
            h4  = scatter!(ax4, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:lajolla, colorrange=(273,1273), markersize=5)    
            #h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[1,4], h3)
            Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            # fig
            fig2 = Figure(size = (900, 900), title = "V_RMS")
            ax21 = Axis(fig2[1,1], aspect = ar, title = "V_{RMS}")
            lines!(ax21,trms,Urms)
            save(joinpath(figdir, "V_RMS.png"), fig2)
            # fig2     
        end

        @show it += 1
        t        += dt
        # ------------------------------
    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
figdir   = "Blankenbach_subgrid"
save_vtk = false # set to true to generate VTK files for ParaView
ar       = 1 # aspect ratio
n        = 64
nx       = n
ny       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, save_vtk = save_vtk);
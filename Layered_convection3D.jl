# using CUDA
# CUDA.allowscalar(false) # for safety

# NOTE: For now, need to use a specific brand of JustPIC.jl
# type `] add https://github.com/JuliaGeodynamics/JustPIC.jl.git#adm-rework` 
# in the REPL to install it
using JustRelax, JustRelax.DataIO, JustPIC
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays

# Load file with all the rheology configurations
include("Layered_rheology.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
@inline create_dir(dir) = !isdir(dir) && mkpath(dir)

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, ni::NTuple{3, Int})
    ncells     = prod(ni)
    np         = max_xcell * ncells
    px, py, pz = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(3))
    inject     = @fill(false, ni..., eltype=Bool)
    index      = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j, k) function fill_coords_index(px, py, pz, index)
        @inline r()= rand(0.05:1e-5:0.95)
        I          = i, j, k
        # lower-left corner of the cell
        x0, y0, z0 = x[i], y[j], z[k]
        # fill index array
        for l in 1:nxcell
            JustRelax.@cell px[l, I...]    = x0 + dx * r()
            JustRelax.@cell py[l, I...]    = y0 + dy * r()
            JustRelax.@cell pz[l, I...]    = z0 + dz * r()
            JustRelax.@cell index[l, I...] = true
        end
        return nothing
    end

    @parallel (@idx ni) fill_coords_index(px, py, pz, index)    

    return Particles(
        (px, py, pz), index, inject, nxcell, max_xcell, min_xcell, np, ni
    )
end

# Velocity helper grids for the particle advection
function velocity_grids(xci, xvi, di)
    xghost     = ntuple(Val(3)) do i
        LinRange(xci[i][1] - di[i], xci[i][end] + di[i], length(xci[i])+2)
    end
    grid_vx = xvi[1]   , xghost[2], xghost[3]
    grid_vy = xghost[1], xvi[2]   , xghost[3]
    grid_vz = xghost[1], xghost[2], xvi[3]

    return grid_vx, grid_vy, grid_vz
end

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

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

# Initial thermal profile
@parallel_indices (I...) function init_T!(T, z)
    depth = -z[I[3]]

    # (depth - 15e3) because we have 15km of sticky air
    if depth < 0e0
        T[I...]  = 0.0

    elseif 0e0 ≤ (depth - 15e3) < 35e3
        dTdZ    = (923-273)/35e3
        offset  = 273e0
        T[I...] = (depth - 15e3) * dTdZ + offset
    
    elseif 110e3 > (depth - 15e3) ≥ 35e3
        dTdZ    = (1492-923)/75e3
        offset  = 923
        T[I...] = (depth - 15e3 - 35e3) * dTdZ + offset

    elseif (depth - 15e3) ≥ 110e3 
        dTdZ    = (1837 - 1492)/590e3
        offset  = 1492e0
        T[I...] = (depth - 15e3 - 110e3) * dTdZ + offset

    end
    
    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if (abs(x[i]-xc) ≤ r) && (abs(y[j] - yc) ≤ r) && (abs(z[k] - zc) ≤ r)
            depth      = abs(z[k])
            dTdZ       = (2047 - 2017) / 50e3
            offset     = 2017
            T[i, j, k] = (depth - 585e3) * dTdZ + offset
        end
        return nothing
    end

    @parallel _rectangular_perturbation!(T, xc, yc, r, xvi...)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D", save_vtk =false)

    # Physical domain ------------------------------------
    lz       = 700e3                # domain length in z
    lx = ly  = lz * ar              # domain length in x and y
    ni       = nx, ny, nz           # number of cells
    li       = lx, ly, lz           # domain length
    di       = @. li / ni           # grid steps
    origin   = 0.0, 0.0, -lz + 15e3 # origin coordinates (15km of sticky air layer)
    xci, xvi = lazy_grid(
        di, 
        li, 
        ni; 
        origin = origin
    ) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_plastic = true)
    κ            = (10 / (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi..., di..., ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_particle_fields_cellarrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly 
    xc_anomaly       = lx/2   # origin of thermal anomaly
    yc_anomaly       = ly/2   # origin of thermal anomaly
    zc_anomaly       = -610e3 # origin of thermal anomaly
    r_anomaly        = 25e3   # radius of perturbation
    init_phases!(pPhases, particles, lx, ly; d=abs(zc_anomaly), r=r_anomaly)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords..., xci..., di, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 1 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(; 
        no_flux      = (left = true, right = true, top = false, bot = false), 
    )
    # initialize thermal profile - Half space cooling
    @parallel init_T!(thermal.T, xvi[3])
    thermal_bcs!(thermal.T, thermal_bc)
   
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, xvi)
    @parallel (@idx ni) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------
   
    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(3))
    for _ in 1:1
        @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3])
    end
    # Rheology
    η                = @ones(ni...)
    η_vep            = similar(η)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e16, 1e24)
    )

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √3
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(; 
        free_slip    = (left = true, right=true, top=true, bot=true),
    )

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if save_vtk
        vtk_dir      = figdir*"\\vtk"
        create_dir(vtk_dir)
    end
    create_dir(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(resolution = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        lines!(ax1, Array(thermal.T[:, 2:end-1,:][:]), Yv./1e3)
        lines!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[3])./1e3, 0)
        ylims!(ax2, minimum(xvi[3])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    grid2particle!(pT, xvi, thermal.T, particles.coords)

    local Vx_v, Vy_v, Vz_v
    if save_vtk 
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Vz_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0
    while (t/(1e6 * 3600 * 24 *365.25)) < 5 # run only for 5 Myrs

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e18, 1e24)
        )
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, args)
        # ------------------------------
 
        # Stokes solver ----------------
        to   = solve!(
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
            iterMax=75e3,
            nout=1e3,
            viscosity_cutoff=(1e18, 1e24)
        )
        @show to
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        dt   = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles.coords)
        @views T_buffer[:, end]      .= 273.0
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)

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
            nout    = 1e2,
            verbose = true,
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)        
        # interpolate fields from grid vertices to particles
        grid2particle!(pT, xvi, thermal.T, thermal.Told, particles.coords)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords..., xci..., di, pPhases)
        
        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
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
            clr      = pPhases.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            fig = Figure(resolution = (1000, 1600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[3,1], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[4,1], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # Plot particles phase
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
            # Plot 2nd invariant of strain rate
            h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow) 
            # Plot effective viscosity
            h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            Colorbar(fig[4,2], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------


# (Path)/folder where output data and figures are stored
figdir   = "Plume2D"
save_vtk = false # set to true to generate VTK files for ParaView
ar       = 1 # aspect ratio
n        = 32
nx       = n - 2
ny       = n - 2
nz       = n - 2
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 0; init_MPI= true)...)
else
    igg
end
# save script metada
# @edit metadata(@__DIR__, "Layered_convection2D.jl", joinpath(figdir, "metadata"))

# # run main script
# main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, save_vtk = save_vtk);
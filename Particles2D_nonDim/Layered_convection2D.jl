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
using GeoParams, GLMakie

# Load file with all the rheology configurations
include("Layered_rheology.jl")
include("Layered_setup.jl")

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
# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if ((x[i]-xc)^2 ≤ r^2) && ((y[j] - yc )^2 ≤ r^2)
            T[i + 1, j] *= 1.05
        end
        return nothing
    end

    ni = length.(xvi)
    @parallel (@idx ni) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li_dim, phases_GMG, T_GMG, igg; nx=16, ny=16, figdir="figs2D", do_vtk =false)

    η0           = 1e20
    CharDim      = GEO_units(;
        length = li_dim[end]m, viscosity = η0, temperature = T_GMG[1]K
    )
    # Physical domain ------------------------------------
    thick_air    = nondimensionalize(0e0km, CharDim)     # thickness of sticky air layer
    lx           = nondimensionalize(li_dim[1]m, CharDim) # domain length in x
    ly           = nondimensionalize(li_dim[2]m, CharDim) # domain length in y
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(CharDim; is_plastic = true)
    κ            = (4 / (rheology[4].HeatCapacity[1].Cp * rheology[4].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell           = 40
    max_xcell        = 60
    min_xcell        = 20
    particles        = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    );
    subgrid_arrays   = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Assign particles phases anomaly
    phases_device    = PTArray(backend)(phases_GMG)
    phase_ratios     = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni);
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-3, Re =1e0, r=0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend, ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )
    T_GMG_nd         = nondimensionalize(T_GMG*K, CharDim)
    Tbot             = T_GMG_nd[1,1]
    Ttop             = T_GMG_nd[end,end]
    @views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG_nd)

    xc_anomaly       = lx/2    # origin of thermal anomaly
    yc_anomaly       = nondimensionalize(-2610km, CharDim) # origin of thermal anomaly
    r_anomaly        = nondimensionalize(100km, CharDim) # radius of perturbation
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi)
  
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg, phase_ratios, rheology, args)
    stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Rheology
    viscosity_cutoff = nondimensionalize((1e18Pa*s, 1e23Pa*s), CharDim)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-8, CFL= 0.98 / √2.1
    )

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
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
        scatter!(
            ax1, 
            Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim)))[:],
            # Array(thermal.T[2:end-1,:][:]),
            Array(ustrip.(dimensionalize(Yv, km, CharDim)))
        )
        # scatter!(ax1, Array(ρg[2][:]), Y)
        scatter!(
            ax2, 
            Array(log10.(ustrip.(dimensionalize(stokes.viscosity.η_vep,Pa*s,CharDim))))[:],
            # Array(log10.(stokes.viscosity.η[:])), 
            Array(ustrip.(dimensionalize(Y, C, CharDim)))
        )
        # ylims!(ax1, minimum(xvi[2]), 0)
        # ylims!(ax2, minimum(xvi[2]), 0)
        hideydecorations!(ax2)
        # save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    T_buffer    = @zeros(ni.+1)
    dt₀         = similar(stokes.P)
    copyinn_x!(T_buffer, thermal.T)
    grid2particle!(pT, xvi, T_buffer, particles)

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0
    while t < nondimensionalize(5e6yr, CharDim) # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views thermal.T[2:end-1, :] .= T_buffer
        @views thermal.T[:, end]     .= Ttop
        @views thermal.T[:, 1]       .= Tbot
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        # ------------------------------

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
            Inf,
            igg;
            kwargs = (;
                iterMax          = 150e3,
                nout             = 1e3,
                viscosity_cutoff = viscosity_cutoff
            )
        )
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
                iterMax = 10e3,
                nout    = 1e2,
                verbose = true
            ),
        )
        copyinn_x!(T_buffer, thermal.T)
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
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim))),
                    τxy = Array(ustrip.(dimensionalize(stokes.τ.xy, s^-1, CharDim))),
                    εxy = Array(ustrip.(dimensionalize(stokes.ε.xy, s^-1, CharDim))),
                    Vx  = Array(ustrip.(dimensionalize(Vx_v,cm/yr,CharDim))),
                    Vy  = Array(ustrip.(dimensionalize(Vy_v, cm/yr, CharDim))),
                )
                data_c = (;
                    P   = Array(ustrip.(dimensionalize(stokes.P,MPa,CharDim))),
                    τxx = Array(ustrip.(dimensionalize(stokes.τ.xx, MPa,CharDim))),
                    τyy = Array(ustrip.(dimensionalize(stokes.τ.yy,MPa,CharDim))),
                    τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CharDim))),
                    εxx = Array(ustrip.(dimensionalize(stokes.ε.xx, s^-1,CharDim))),
                    εyy = Array(ustrip.(dimensionalize(stokes.ε.yy, s^-1,CharDim))),
                    εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1,CharDim))),
                    η   = Array(ustrip.(dimensionalize(stokes.viscosity.η_vep,Pa*s,CharDim))),
                )
                velocity_v = (
                    Array(ustrip.(dimensionalize(Vx_v,cm/yr,CharDim))),
                    Array(ustrip.(dimensionalize(Vy_v, cm/yr, CharDim))),
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

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]
            pyv      = ppy.data[:]
            clr      = pPhases.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            ar = lx/ly
            t_dim = Float16(dimensionalize(t, yr, CharDim).val / 1e3)
            fig = Figure(size = (900, 900), title = "t = $t_dim [kyr]")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K] ; t=$t_dim [kyrs]")
            ax2 = Axis(fig[2,1], aspect = ar, title = "phase")
            # ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1], xvi[2], Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim))) , colormap=:batlow)
            # Plot particles phase
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:grayC)
            # Plot 2nd invariant of strain rate
            h3  = heatmap!(ax3, xci[1], xci[2], Array(log10.(ustrip.(dimensionalize(stokes.ε.II, s^-1,CharDim)))) , colormap=:batlow)
            # Plot effective viscosity
            h4  = heatmap!(ax4, xci[1], xci[2], Array(log10.(ustrip.(dimensionalize(stokes.viscosity.η_vep,Pa*s,CharDim)))) , colormap=:batlow)
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

# (Path)/folder where output data and figures are stored
figdir   = "GlobalConvection2D"
do_vtk   = true # set to true to generate VTK files for ParaView
ar       = 14 # aspect ratio
n        = 64
nx       = n * 2
ny       = n

li_dim, origin, phases_GMG, T_GMG = setup(nx+1, ny+1)

igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

# run main script
# main(li_dim, phases_GMG, T_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
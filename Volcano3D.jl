const isCUDA = false
# const isCUDA = true

@static if isCUDA 
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO
import JustRelax.@cell

const backend_JR = @static if isCUDA 
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences3D

@static if isCUDA 
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC, JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if isCUDA 
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using GeoParams, GLMakie, CellArrays

include("VolcanoSetup.jl")
include("VolcanoRheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_j = INDICES[3]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z, sticky_air)
    @all(P) = @all(ρg)
    # @all(P) = abs(@all(ρg) * (@all_j(z) + sticky_air)) * <((@all_j(z) + sticky_air), 0.0)
    return nothing
end

function main3D(
    igg, li_GMG, origin_GMG, phases_GMG, T_GMG; 
    figdir = "output", nx = 64, ny = 64, nz = 64, do_vtk = false
)

    # Characteristic lengths for non dimensionalization
    CharDim      = GEO_units(;length=50km, viscosity=1e21, temperature = 1e3C)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    # sticky_air   = nondimensionalize(0km, CharDim)                 # thickness of the sticky air layer
    sticky_air   = nondimensionalize(10km, CharDim)                 # thickness of the sticky air layer
    lx           = nondimensionalize(li_GMG[1]*km, CharDim)             # domain length in x-direction
    ly           = nondimensionalize(li_GMG[2]*km, CharDim)             # domain length in x-direction
    lz           = nondimensionalize(li_GMG[3]*km,CharDim) - sticky_air # domain length in y-direction
    li           = lx, ly, lz                                       # domain length in x- and y-direction
    ni           = nx, ny, nz                                       # number of grid points in x- and y-direction
    di           = @. li / ni                                       # grid step in x- and y-direction
    origin       = ntuple(Val(3)) do i 
        nondimensionalize(origin_GMG[i] * km,CharDim) # origin coordinates of the domain
    end
    grid         = Geometry(ni, li; origin=origin)
    (; xci, xvi) = grid                         # nodes at the center and vertices of the cells
    εbg          = nondimensionalize(0.0 / s, CharDim)    # background strain rate
    #---------------------------------------------------------------------------------------
    
    # Physical Parameters
    rheology     = init_rheology(CharDim; is_compressible=true)
    cutoff_visc  = nondimensionalize((1e16Pa*s, 1e24Pa*s),CharDim)
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    κ            = (4 / (rheology[2].HeatCapacity[1].Cp * rheology[2].Density[1].ρ0))
    dt = dt_diff = (0.5 * min(di...)^3 / κ / 3.01)         # diffusive CFL timestep limiter
    dt = nondimensionalize(10e3*yr, CharDim)

    # Initialize particles -------------------------------
    nxcell           = 20
    max_xcell        = 40
    min_xcell        = 15
    particles        = init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni);

    subgrid_arrays   = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi         = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases);

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend_JR)(phases_GMG)
    phase_ratios = PhaseRatio(backend_JR, ni, length(rheology));
    init_phases!(pPhases, phases_dev, particles, xvi)
    phase_ratios_center!(phase_ratios, particles, grid, pPhases)

    # Initialisation of thermal profile
    thermal     = ThermalArrays(backend_JR, ni) # initialise thermal arrays and boundary conditions
    thermal.T  .= PTArray(backend_JR)(nondimensionalize(T_GMG.*C, CharDim))
    thermal_bc  = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, front = true, back = true, top = false, bot = false),
    )

    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend_JR, ni) # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=0.9 / √3.1)
    # ----------------------------------------------------

    args = (; T=thermal.Tc, P=stokes.P, dt=dt)
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL = 1e-2 / √3.1
    )
    εbg = nondimensionalize(1e-15 / s, CharDim)
    pureshear_bc!(
        stokes, xci, xvi, εbg, backend_JR
    )

    flow_bcs = VelocityBoundaryConditions(;
        free_slip    = (left=true, right=true, front=true, back=true, top=true, bot=true),
        no_slip      = (left=false, right=false, front=false, back=false, top=false, bot=false),
        free_surface = true,
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # # Thermal solver ---------------
    # for _ in 1:5
    #     heatdiffusion_PT!(
    #         thermal,
    #         pt_thermal,
    #         thermal_bc,
    #         rheology,
    #         args,
    #         dt,
    #         di;
    #         kwargs =(;
    #             igg     = igg,
    #             phase   = phase_ratios,
    #             iterMax = 5e3,
    #             nout    = 5e1,
    #             verbose = false,
    #         )
    #     )
    # end
    
    # Buoyancy force & viscosity
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...) # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3], sticky_air)
    end
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    # Arguments for functions
    @copy thermal.Told thermal.T

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------


    # Time loop
    t, it = 0.0, 0
    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
    end

    dt₀ = similar(stokes.P)
    grid2particle!(pT, xvi, thermal.T, particles)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T
    Tsurf, Tbot  = extrema(thermal.T)

    while it < 25

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

        # Stokes solver -----------------
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
                iterMax          = 150e3,#50e3,
                free_surface     = false,
                nout             = 2e3,#5e3,
                viscosity_cutoff = cutoff_visc,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff, igg) * 0.75
        # --------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs =(;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 150e3,
                nout    = 1e3,
                verbose = true,
            )
        )
        # subgrid diffusion
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
        inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)
        # update phase ratios
        phase_ratios_center!(phase_ratios, particles, grid, pPhases)

        particle2grid!(thermal.T, pT, xvi, particles)
        # @views thermal.T[:, :, end] .= Tsurf
        # @views thermal.T[:, :, 1] .= Tbot
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        # thermal.ΔT .= thermal.T .- thermal.Told
        # vertex2center!(thermal.ΔTc, thermal.ΔT)

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            (; η) = stokes.viscosity
            # checkpointing(figdir, stokes, thermal.T, η, t)

            if igg.me == 0
                if do_vtk
                    velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                    data_v = (;
                        T  = Array(ustrip.(dimensionalize(thermal.T, C, CharDim))),
                        τxy= Array(ustrip.(dimensionalize(stokes.τ.xy, s^-1, CharDim))),
                        εxy= Array(ustrip.(dimensionalize(stokes.ε.xy, s^-1, CharDim))),
                    )
                    data_c = (;
                        P   = Array(ustrip.(dimensionalize(stokes.P,MPa,CharDim))),
                        τxx = Array(ustrip.(dimensionalize(stokes.τ.xx, MPa,CharDim))),
                        τyy = Array(ustrip.(dimensionalize(stokes.τ.yy,MPa,CharDim))),
                        τzz = Array(ustrip.(dimensionalize(stokes.τ.zz,MPa,CharDim))),
                        τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CharDim))),
                        εxx = Array(ustrip.(dimensionalize(stokes.ε.xx, s^-1,CharDim))),
                        εyy = Array(ustrip.(dimensionalize(stokes.ε.yy, s^-1,CharDim))),
                        εzz = Array(ustrip.(dimensionalize(stokes.ε.zz, s^-1,CharDim))),
                        εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1,CharDim))),
                        ρ   = Array(ustrip.(dimensionalize(ρg[end] ./ rheology[1].Gravity[1].g.val,kg/m^3,CharDim))),
                        η   = Array(ustrip.(dimensionalize(stokes.viscosity.η,Pa*s,CharDim))),
                        η_vep = Array(ustrip.(dimensionalize(stokes.viscosity.η_vep,Pa*s,CharDim))),
                    )
                    velocity_v = (
                        Array(ustrip.(dimensionalize(Vx_v,cm/yr,CharDim))),
                        Array(ustrip.(dimensionalize(Vy_v, cm/yr, CharDim))),
                        Array(ustrip.(dimensionalize(Vz_v, cm/yr, CharDim))),
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

                let
                    Zv  = [z for _ in 1:nx+1, _ in 1:ny+1, z in ustrip.(dimensionalize(xvi[3],km,CharDim))][:]
                    Z   = [z for _ in 1:nx  , _ in 1:ny  , z in ustrip.(dimensionalize(xci[3],km,CharDim))][:]
                    fig = Figure(; size=(1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect = 1, title="T")
                    ax2 = Axis(fig[1, 2]; aspect = 1, title="Pressure")
                    ax3 = Axis(fig[2, 1]; aspect = 1, title="τII")
                    ax4 = Axis(fig[2, 2]; aspect = 1, title="Vz")
                    ns  = ny >>> 1
                    heatmap!(ax1, ustrip.(dimensionalize((Array(thermal.T)), C, CharDim))[:,ns,:]  )
                    heatmap!(ax2, ustrip.(dimensionalize((Array(stokes.P)), MPa, CharDim))[:,ns,:] )
                    heatmap!(ax3,  ustrip.(dimensionalize(Array(stokes.τ.II), MPa, CharDim))[:,ns,:])
                    heatmap!(ax4,  ustrip.(dimensionalize(Array(stokes.V.Vz), m/s, CharDim))[:,ns,:])

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end
            end
        end
    end

    # finalize_global_grid()
    return nothing
end

figdir = "Volcano3D"
do_vtk = true # set to true to generate VTK files for ParaView
n      = 64
nx     = n
ny     = n
nz     = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI=true)...)
else
    igg
end

li_GMG, origin_GMG, phases_GMG, T_GMG = volcano_setup(n+1)

# run main script
main3D(igg, li_GMG, origin_GMG, phases_GMG, T_GMG; figdir=figdir, nx=nx, ny=ny, nz=nz, do_vtk = do_vtk);
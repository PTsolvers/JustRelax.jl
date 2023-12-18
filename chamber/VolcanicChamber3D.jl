using JustRelax, JustRelax.DataIO, JustPIC
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

# setup ParallelStencil.jl environment
model = PS_Setup(:CUDA, Float64, 3) # or (:Threads, Float64, 3) or (:AMDGPU, Float64, 3)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays

# Load file with all the rheology configurations
include("RheologyChamber3D.jl")

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function Chamber3D(igg, nx, ny, nz; figdir="figs3D", do_vtk =false)

    # Physical domain ------------------------------------
    ar            = 2              # aspect ratio   
    lz            = 35e3           # domain length in z
    lx = ly       = lz*ar             # domain length in x and y
    ni            = nx, ny, nz     # number of cells
    li            = lx, ly, lz     # domain length
    di            = @. li / ni     # grid steps
    origin        = 0.0, 0.0, -lz  # origin coordinates (15km of sticky air layer)
    xci, xvi      = lazy_grid(
        di, 
        li, 
        ni; 
        origin = origin
    ) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies()
    κ            = (10 / (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^3 / κ / 3.01 # diffusive CFL timestep limiter
    # dt = dt_diff = 0.5 * min(di...)^3 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 20, 1
    particles                    = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., di..., ni
    )
    # velocity grids
    grid_vx, grid_vy, grid_vz   = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases                 = init_particle_fields_cellarrays(particles, Val(2))
    particle_args               = (pT, pPhases)

    # Elliptical temperature anomaly 
    z_anomaly        = -15e3 # origin of thermal anomaly
    a_anomaly        =  15e3 # radius of perturbation
    b_anomaly        =  15e3 # radius of perturbation
    c_anomaly        =   5e3 # radius of perturbation
    init_phases!(pPhases, particles, lx, ly, z_anomaly; a = a_anomaly, b = b_anomaly, c = c_anomaly)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    # @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.75 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(; 
        no_flux     = (left = true , right = true , top = false, bot = false, front = true , back = true), 
        periodicity = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    # initialize thermal profile
    dTdZ = 30 # C/Km
    @parallel init_T!(thermal.T, xvi..., dTdZ, lx, ly, z_anomaly, a_anomaly, b_anomaly, c_anomaly)
    thermal_bcs!(thermal.T, thermal_bc)
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
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e16, 1e24)
    )
    η_vep            = deepcopy(η)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-2 / √3
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(; 
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
        periodicity  = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = figdir*"\\vtk"
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig = let
        Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(resolution = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[:]), Zv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Z./1e3)
        ylims!(ax1, minimum(xvi[3])./1e3, 0)
        ylims!(ax2, minimum(xvi[3])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    grid2particle!(pT, xvi, thermal.T, particles.coords)

    local Vx_v, Vy_v, Vz_v
    if do_vtk 
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Vz_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0
    while (t/(1e6 * 3600 * 24 *365.25)) < 5 # run only for 5 Myrs
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e16, 1e24)
        )
        @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, args)
 
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
            iterMax          = 75e3,
            nout             = 1e3,
            viscosity_cutoff = (1e16, 1e24),
            do_viscosity     = Val(false)
        );
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        dt   = compute_dt(stokes, di, dt_diff) / 2
        dt   = min(dt, 100e3 * 3600 * 24 *365.25)
        # ------------------------------

        # interpolate fields from particle to grid vertices
        particle2grid!(thermal.T, pT, xvi, particles.coords)
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
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, grid_vz, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)        
        # interpolate fields from grid vertices to particles
        grid2particle_flip!(pT, xvi, thermal.T, thermal.Told, particles.coords)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
        
        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)

            if do_vtk 
                JustRelax.velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (; 
                    T   = Array(thermal.T),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (; 
                    Tc  = Array(thermal.Tc),
                    P   = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    η   = Array(log10.(η)),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci, 
                    data_v, 
                    data_c
                )
            end

            xz_slice = ny >>> 1
            # Make Makie figure
            fig = Figure(resolution = (1400, 1800), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "τII [MPa]")
            ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = ar, title = "Vz")
            # ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[3].*1e-3, Array(thermal.T[:, xz_slice, :]) , colormap=:batlow)
            # Plot particles phase
            h2  = heatmap!(ax2, xci[1].*1e-3, xci[3].*1e-3, Array(stokes.τ.II[:, xz_slice, :]./1e6) , colormap=:batlow) 
            # Plot 2nd invariant of strain rate
            h3  = heatmap!(ax3, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(stokes.ε.II[:, xz_slice, :])) , colormap=:batlow) 
            # Plot effective viscosity
            h4  = heatmap!(ax4, xci[1].*1e-3, xci[3].*1e-3, Array(stokes.V.Vz[:, xz_slice, :]) , colormap=:batlow)
            # h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep[:, xz_slice, :])) , colormap=:batlow)
            hideydecorations!(ax3)
            hideydecorations!(ax4)
            # Colorbar(fig[1,2], h1)
            # Colorbar(fig[2,2], h2)
            # Colorbar(fig[1,4], h3)
            # Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

do_vtk   = true # set to true to generate VTK files for ParaView
ar       = 1 # aspect ratio
n        = 64
nx       = n
ny       = n
nz       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
else
    igg
end

# # (Path)/folder where output data and figures are stored
figdir   = "Chamber3D_$n"
Chamber3D(igg, nx, ny, nz; figdir = figdir, do_vtk = do_vtk);

# # Make particles plottable
# ppx, ppy, ppz = particles.coords;
# pxv      = Array(ppx.data[:]./1e3);
# pyv      = Array(ppy.data[:]./1e3);
# pzv      = Array(ppz.data[:]./1e3);
# clr      = Array(pPhases.data[:]);
# clrT     = Array(pT.data[:]);
# idxv     = Array(particles.index.data[:]);

# xz_slice = ny >>> 1

# X   = [x for x in xvi[1], z in xvi[3]][:]
# Z   = [z for x in xvi[1], z in xvi[3]][:]
# Tv  = Array(thermal.T[:, xz_slice, :])[:]
# Tv[Tv .< 900] .= NaN
# # heatmap(xvi[1].*1e-3, xvi[3].*1e-3, Array(thermal.T[:, xz_slice, :]))

# scatter(pxv[idxv], pzv[idxv], color=clr[idxv])
# scatter!(X.*1e-3, Z.*1e-3, color=Tv, marker =:diamond)


# scatter!(Array(thermal.T[:]), Zv./1e3)

#  # Plot initial T and η profiles
#  fig = let
#     Zv  = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
#     Z   = [z for x in xci[1], y in xci[2], z in xci[3]][:]
#     fig = Figure(resolution = (1200, 900))
#     ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
#     ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
#     scatter!(ax1, Array(thermal.T[:]), Zv./1e3)
#     scatter!(ax2, Array(log10.(η[:])), Z./1e3)
#     ylims!(ax1, minimum(xvi[3])./1e3, 0)
#     ylims!(ax2, minimum(xvi[3])./1e3, 0)
#     hideydecorations!(ax2)
#     save(joinpath(figdir, "initial_profile.png"), fig)
#     fig
# end
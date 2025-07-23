const isCUDA = false
# const isCUDA = true
using Smoothing

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

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

using Statistics#, Interpolations

function apply_sealevel_change!(h_selevel_new, pPhases, chain, particles, origin, di, air_phase)
    dy          = di[2]
    h_selevel0  = mean(chain.h_vertices)
    dh          = (h_selevel0 - h_selevel_new)
    steps       = Int(dh ÷ dy) + 1
    for _ in steps
        chain.coords[2].data  .= h_selevel_new
        chain.h_vertices      .= h_selevel_new
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    end
    return nothing
end

include("IceLayers_setup-2.jl")
include("IceLayers_rheology.jl")

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny, li, origin, phases_GMG, T_GMG, figdir; do_vtk = true)

    # # Sea level change info ------------------------------
    # sea_level_curve = SeaLevel(:Bintanja_3Ma)
    # sea_level_interpolant = linear_interpolation(sea_level_curve.age, sea_level_curve.elevation)
    # # ----------------------------------------------------

    # Physical domain ------------------------------------
    ni = nx, ny            # number of cells
    di = @. li / ni        # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # rheology = init_rheologies()
    Cp = 1200
    el = ConstantElasticity(; G = 25.0e9, ν = 0.45) #: ConstantElasticity(; G = G0, ν = 0.25)
    β = inv(get_Kb(el))
    η_reg = 1.0e15
    C = 20.0e6
    ϕ = 15
    Ψ = 0.0
    soft_C = NonLinearSoftening(; ξ₀ = C, Δ = C / 1.0e5)       # nonlinear softening law
    pl = DruckerPrager_regularised(; C = C, ϕ = ϕ, η_vp = η_reg, Ψ = Ψ)
    # el_magma = ConstantElasticity(; G = G_magma, ν = 0.45)
  
    disl_crust  = SetDislocationCreep(GeoParams.Dislocation.dry_anorthite_Rybacki_2000)
    # disl_crust  = SetDislocationCreep(GeoParams.Dislocation.wet_quartzite_Hirth_2001)
    disl_litho  = SetDislocationCreep(GeoParams.Dislocation.dry_olivine_Karato_2003)
    # diffusion laws
    diff_litho  = SetDiffusionCreep(GeoParams.Diffusion.dry_olivine_Hirth_2003)
    ρ_ice = if isCUDA
        cudaconvert(Ref(0.917e3))
    else
        Ref(0.917e3)
    end
    # rheology = rheology = (
    #     # Name              = "Upper Crust",
    #     SetMaterialParams(;
    #         Phase = 1,
    #         Density = PT_Density(; ρ0 = 2.7e3, β = β, T0 = 273, α = 3.5e-5),
    #         HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
    #         Conductivity = ConstantConductivity(; k = 2.5),
    #         CompositeRheology = CompositeRheology((disl_crust, el, pl)),
    #         Gravity = ConstantGravity(; g = 9.81),
    #     ),
    #     # Name              = "Lower Crust",
    #     SetMaterialParams(;
    #         Phase = 2,
    #         Density = PT_Density(; ρ0 = 2.75e3, β = β, T0 = 273, α = 3.5e-5),
    #         # Density = T_Density(; ρ0 = 2.70e3, T0 = 273.15),
    #         HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
    #         Conductivity = ConstantConductivity(; k = 2.5),
    #         CompositeRheology = CompositeRheology((disl_crust, el, pl)),
    #         Gravity = ConstantGravity(; g = 9.81),
    #     ),
    #     # Name              = "Lithosphere Mantle",
    #     SetMaterialParams(;
    #         Phase = 3,
    #         Density = PT_Density(; ρ0 = 3.3e3, β = β, T0 = 273, α = 3e-5),
    #         HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
    #         Conductivity = ConstantConductivity(; k = 2.5),
    #         CompositeRheology = CompositeRheology((disl_litho, diff_litho,el,pl)),
    #         Gravity = ConstantGravity(; g = 9.81),
    #     ),
    #     # Name              = "ice",
    #     SetMaterialParams(;
    #         Phase = 4,
    #         Density = ConstantMutableDensity(ρ_ice),
    #         HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
    #         Conductivity = ConstantConductivity(; k = 1.5),
    #         # CompositeRheology = CompositeRheology((disl_crust, el, pl)),
    #         CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23), el)),
    #         Gravity = ConstantGravity(; g = 9.81),
    #     ),
    #     # Name              = "air",
    #     SetMaterialParams(;
    #         Phase = 5,
    #         Density = ConstantDensity(; ρ = 0),
    #         HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
    #         Conductivity = ConstantConductivity(; k = 2.5),
    #         CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19),)),
    #         Gravity = ConstantGravity(; g = 9.81),
    #     ),
    # )

    rheology = (
        # Name = "Upper crust",
        SetMaterialParams(;
            Phase = 1,
            # Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            # CompositeRheology = CompositeRheology((disl_top, el, pl)),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e22), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 517.9, b = -1619.0, c = 1699.0, d = -597.4), #mafic melting curve
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name = "Lower crust",
        SetMaterialParams(;
            Phase = 2,
            # Density = PT_Density(; ρ0 = 2.75e3, T0 = 273.15, β = β),
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            # CompositeRheology = CompositeRheology((disl_bot, el, pl)),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 517.9, b = -1619.0, c = 1699.0, d = -597.4), #mafic melting curve
            Gravity = ConstantGravity(; g = 9.81),
        ),
         # Name              = "Lithosphere Mantle",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.3e3, β = β, T0 = 273, α = 3e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η=5e20),el,pl)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "ice",
        SetMaterialParams(;
            Phase = 4,
            Density = ConstantMutableDensity(ρ_ice),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 1.5),
            # CompositeRheology = CompositeRheology((disl_crust, el, pl)),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 5,
            Density = ConstantDensity(; ρ = 0.0e0),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e22), el, pl)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )

    # ----------------------------------------------------
    dt = 1e0 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 60, 80, 40
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))
    particle_args = (pPhases, pT)

    # Elliptical temperature anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    
    # Initialize marker chain
    nxcell, max_xcell, min_xcell = 10, 15, 7
    initial_elevation = 0e0
    chain  = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    topo_y = extract_topo(xvi..., phases_GMG, 5)
    air_phase = 5

    # for _ in 1:3
        # topo_y .= Smoothing.binomial(topo_y, 1)
        fill_chain_from_vertices!(chain, PTArray(backend)(topo_y))
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    # end
    # ----------------------------------------------------

    # RockRatios
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-3, ϵ_rel = 1.0e-1, Re = 3π, r = 0.7, CFL = 0.98 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)

    # Add thermal anomaly BC's
    # T_chamber = 1223.0e0
    # T_air = 273.0e0
    # Ω_T = @zeros(size(thermal.T)...)
    # thermal_anomaly!(thermal.T, Ω_T, phase_ratios, T_chamber, T_air, 5, 3, 4, air_phase)
    # JustRelax.DirichletBoundaryCondition(Ω_T)

    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = true, right = true, top = false, bot = false),
        # dirichlet    = (; mask = Ω_T)
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    Ttop = thermal.T[2:(end - 1), end]
    Tbot = thermal.T[2:(end - 1), 1]
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg = @zeros(ni...), @zeros(ni...)
    args0 = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args0)
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot = false),
        free_surface = false,
    )

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it  = 0.0, 0
    year   = 3600 * 24 * 365.25
    dt     = 5e2 * year
    dt_max = 1e3 * year
    justdoit = true

    while it < 1000

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        # clamp!(T_buffer, 273e0, 1223e0)
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        ## variational solver
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        # Stokes solver ----------------
        solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                iterMax = 100e3,
                iterMin = 5e3,
                viscosity_relaxation = 1.0e-2,
                free_surface = true,
                nout = 2.0e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_max)
        @show dt / year
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
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        )

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        # inject_particles_phase!(particles, pPhases, (), (), xvi)
       
        # advect marker chain
        semilagrangian_advection_markerchain!(chain, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), xvi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        @show it += 1
        t += dt

        # h_selevel_new = sea_level_interpolant(t / year)
        # apply_sealevel_change!(h_selevel_new, pPhases, chain, particles, origin, di, air_phase)
        # compute_rock_fraction!(ϕ, chain, xvi, di)

        # if t / year > 300e3
        #     # ρ_ice_new = GeoUnit(min(rheology[4].Density[1].ρ[] + 25, ρ_ice / 2))
        #     ρ_ice_new = max(rheology[4].Density[1].ρ[] * 0.95, 10)
        #     println("
        #         Current ice density: $ρ_ice_new
        #     ")
        #     rheology[4].Density[1].ρ[] = ρ_ice_new
        # end

        
        # if t / year > 300e3
        #     # ρ_ice_new = GeoUnit(min(rheology[4].Density[1].ρ[] + 25, ρ_ice / 2))
        #     # ρ_ice_new = max(rheology[4].Density[1].ρ[] * 0.95, 10)
        #     # println("
        #     #     Current ice density: $ρ_ice_new
        #     # ")
        #     # rheology[4].Density[1].ρ[] = ρ_ice_new
        #     ice_phase, air_phase = 4.0, 5.0
        #     remove_ice!(chain, particles, pPhases, ice_phase, air_phase)
        #     force_air_above_chain!(chain, particles, pPhases, air_phase)
        # end

        if justdoit && t / year > 15e3
            ice_phase, air_phase = 4.0, 5.0
            remove_ice!(chain, particles, pPhases, ice_phase, air_phase)
            force_air_above_chain!(chain, particles, pPhases, air_phase)

            stokes.τ.xx .= 0e0
            stokes.τ.yy .= 0e0
            stokes.τ.xy .= 0e0
            stokes.τ.II .= 0e0
            stokes.ε.xx .= 0e0
            stokes.ε.yy .= 0e0
            stokes.ε.xy .= 0e0
            stokes.ε.II .= 0e0
            stokes.viscosity.η_vep .= 0e0

            update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
            compute_rock_fraction!(ϕ, chain, xvi, di)

            dt_max = 1e3 * year
            justdoit = false
        end

        if t / year > 500e3
            dt_max = 5e3 * year
        end

        t / year > 1.5e6 && break

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            (; η_vep, η) = stokes.viscosity
            
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    τII            = Array(stokes.τ.II),
                    εII            = Array(stokes.ε.II),
                    stress_xy      = Array(stokes.τ.xy),
                    strain_rate_xy = Array(stokes.ε.xy),
                )
                data_c = (;
                    P                      = Array(stokes.P),
                    η                      = Array(η_vep),
                    phases                 = [argmax(p) for p in Array(phase_ratios.center)],
                    EII_pl                 = Array(stokes.EII_pl),
                    stress_II              = Array(stokes.τ.II),
                    strain_rate_II         = Array(stokes.ε.II),
                    plastic_strain_rate_II = Array(stokes.ε_pl.II),
                    density                = Array(ρg[2] ./ 9.81),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
              
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi ./ 1.0e3,
                    xci ./ 1.0e3,
                    data_v,
                    data_c,
                    velocity_v;
                    t   = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3),
                )
            end

            println(xci[1])
            println(xci)
            
            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:] ./ 1.0e3
            pyv      = ppy.data[:] ./ 1.0e3
            clr      = pPhases.data[:]
            # clr    = pT.data[:]
            idxv     = particles.index.data[:]

            # Make Makie figure
            ar  = li[1] / li[2]
            fig = Figure(size = (1300, 1000), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "log10(εII)  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "τII")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            
            # Plot strain rate
            h1 = heatmap!(ax1, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.ε.II)), colormap = :batlow)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 1)
            lines!(ax2, xvi[1] .* 1e-3, Array(chain.h_vertices) .* 1.0e-3, color = :red, linewidth = 3)
            # Plot 2nd invariant of stress
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.τ.II), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow, colorrange = log10.(viscosity_cutoff))
            
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            h = 250
            Colorbar(fig[1, 2], h1, height = h)
            Colorbar(fig[2, 2], h2, height = h)
            Colorbar(fig[1, 4], h3, height = h)
            Colorbar(fig[2, 4], h4, height = h)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)

            # # Checkpointing
            # checkpoint = joinpath(figdir, "checkpoint")
            # if igg.me == 0 && it == 1
            #     metadata(pwd(), checkpoint, basename(@__FILE__), joinpath(@__DIR__, "Salt_rawi_2D_v2.jl"), joinpath(@__DIR__, "Salt_rawi_2D_v2.jl"))
            # end
           
            # checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
            # checkpointing_particles(checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, particle_args = particle_args, t = t, dt = dt)
        end
    end
    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
do_vtk = true # set to true to generate VTK files for ParaView
figdir = "IceLayers2D"
n = 150
nx, ny = 2*n, n

li, origin, phases_GMG, T_GMG = setup2D_Jacob(
    nx + 1, ny + 1;
    sticky_air    = 10,
    ice_thickness = 5,
    dimensions    = (250.0e0, 125.0e0),
)

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(igg, nx, ny, li, origin, phases_GMG, T_GMG, figdir; do_vtk = do_vtk)

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

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2) #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, CellArrays
using StaticArrays
using ImplicitGlobalGrid
using MPI: MPI
using WriteVTK

# -----------------------------------------------------------------------------------------
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
    return esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z, sticky_air)
    @all(P) = (abs(@all(ρg) * (@all_j(z))) - (@all(ρg) * sticky_air)) * <((@all_j(z)), 0.0)
    return nothing
end

function init_phases!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly, sticky_air)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(
        phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly, sticky_air
    )
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = -(JustRelax.@cell py[ip, i, j]) - sticky_air
            if 0e0 ≤ y ≤ 20e3
                @cell phases[ip, i, j] = 1.0 # crust
            end

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y + yc_anomaly)^2 ≤ r_anomaly^2)
                JustRelax.@cell phases[ip, i, j] = 2.0
            end

            if y < 0.0
                @cell phases[ip, i, j] = 3.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(
        phases,
        particles.coords...,
        particles.index,
        xc_anomaly,
        yc_anomaly,
        r_anomaly,
        sticky_air,
    )
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, sticky_air)
    depth = -y[j] - sticky_air

    if depth < 0e0
        T[i + 1, j] = 273e0

    elseif 0e0 ≤ (depth) < 15e3
        dTdZ = (723 - 273) / 15e3
        offset = 273e0
        T[i + 1, j] = (depth) * dTdZ + offset

    end

    return nothing
end

function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi, sticky_air)
    @parallel_indices (i, j) function _circular_perturbation!(
        T, δT, xc_anomaly, yc_anomaly, r_anomaly, x, y, sticky_air
    )
        depth = -y[j] - sticky_air
        @inbounds if ((x[i] - xc_anomaly)^2 + (depth[j] + yc_anomaly)^2 ≤ r_anomaly^2)
            # T[i + 1, j] *= δT / 100 + 1
            T[i + 1, j] = δT
        end
        return nothing
    end

    nx, ny = size(T)

    @parallel (1:(nx - 2), 1:ny) _circular_perturbation!(
        T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi..., sticky_air
    )
end

@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, args)
    ϕ[i, j] = compute_meltfraction(rheology, ntuple_idx(args, i, j))
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    ϕ[I...] = compute_melt_frac(rheology, args_ijk, phase_ratios[I...])
    return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return GeoParams.compute_meltfraction_ratio(phase_ratios, rheology, args)
end

function phase_change!(phases, particles)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, px, py, index)
        @inbounds for ip in JustRelax.cellaxes(phases)
            #quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip, I...]
            y = (JustRelax.@cell py[ip, I...])
            phase_ij = @cell phases[ip, I...]
            if y > 0.0 && (phase_ij == 2.0 || phase_ij == 3.0)
                @cell phases[ip, I...] = 4.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) _phase_change!(
        phases, particles.coords..., particles.index
    )
end

function init_rheology(; is_compressible = false, steady_state=true)
    # plasticity setup
    do_DP = true                        # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg = 1.0e16                      # regularisation "viscosity" for Drucker-Prager
    Coh = 10.0MPa                       # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30.0 * do_DP                    # friction angle
    G0 = 6.0e11Pa                         # elastic shear modulus
    G_magma = 6.0e11Pa                    # elastic shear modulus perturbation

    # soft_C = LinearSoftening((ustrip(Coh)/2, ustrip(Coh)), (0e0, 1e-1)) # softening law
    soft_C = NonLinearSoftening(; ξ₀=ustrip(Coh), Δ=ustrip(Coh) / 2) # softening law
    pl = DruckerPrager_regularised(; C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0.0, softening_C = soft_C)        # plasticity
    if is_compressible == true
        el = SetConstantElasticity(; G=G0, ν=0.25)           # elastic spring
        el_magma = SetConstantElasticity(; G=G_magma, ν=0.25)# elastic spring
        β_rock = 6.0e-11
        β_magma = 6.0e-11
    else
        el = SetConstantElasticity(; G=G0, ν=0.5)            # elastic spring
        el_magma = SetConstantElasticity(; G=G_magma, ν=0.5) # elastic spring
        β_rock = inv(get_Kb(el))
        β_magma = inv(get_Kb(el_magma))
    end
    if steady_state == true
        creep_rock = LinearViscous(; η=1e23 * Pa * s)
        creep_magma = LinearViscous(; η=1e18 * Pa * s)
        creep_air = LinearViscous(; η=1e18 * Pa * s)
    else
        creep_rock = DislocationCreep(; A=1.67e-24, n=3.5, E=1.87e5, V=6e-6, r=0.0, R=8.3145)
        creep_magma = DislocationCreep(; A=1.67e-24, n=3.5, E=1.87e5, V=6e-6, r=0.0, R=8.3145)
        creep_air = LinearViscous(; η=1e18 * Pa * s)
        β_rock = 6.0e-11
        β_magma = 6.0e-11
    end

    rheology = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase=1,
            Density=PT_Density(; ρ0=2650kg / m^3, α=3e-5 / K, T0=273.0, β=β_rock / Pa),
            HeatCapacity=ConstantHeatCapacity(; Cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=3.0Watt / K / m),
            LatentHeat=ConstantLatentHeat(; Q_L=350e3J / kg),
            ShearHeat=ConstantShearheating(1.0NoUnits),
            CompositeRheology=CompositeRheology((creep_rock, el, pl)),
            Melting=MeltingParam_Caricchi(),
            Elasticity=el,
        ),

        #Name="Magma"
        SetMaterialParams(;
            Phase=2,
            Density=PT_Density(; ρ0=2650kg / m^3, T0=273.0, β=β_magma / Pa),
            HeatCapacity=ConstantHeatCapacity(; Cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=1.5Watt / K / m),
            LatentHeat=ConstantLatentHeat(; Q_L=350e3J / kg),
            ShearHeat=ConstantShearheating(0.0NoUnits),
            CompositeRheology=CompositeRheology((creep_magma, el_magma)),
            Melting=MeltingParam_Caricchi(),
            Elasticity=el_magma,
        ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase=3,
            Density   = ConstantDensity(ρ=10kg/m^3,),
            HeatCapacity=ConstantHeatCapacity(; Cp=1000J / kg / K),
            Conductivity=ConstantConductivity(; k=15Watt / K / m),
            LatentHeat=ConstantLatentHeat(; Q_L=0.0J / kg),
            ShearHeat=ConstantShearheating(0.0NoUnits),
            CompositeRheology=CompositeRheology((creep_air,)),
        ),
    )

end


function main2D(igg; figdir=figdir, nx=nx, ny=ny, do_vtk=false)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    sticky_air = 1.5e3                      # thickness of the sticky air layer
    ly = 12.5e3 + sticky_air                # domain length in y-direction
    lx = 15.5e3                             # domain length in x-direction
    li = lx, ly                             # domain length in x- and y-direction
    ni = nx, ny                             # number of grid points in x- and y-direction
    di = @. li / ni                         # grid step in x- and y-direction
    origin = 0.0, -ly                       # origin coordinates of the domain
    grid = Geometry(ni, li; origin=origin)
    (; xci, xvi) = grid                     # nodes at the center and vertices of the cells
    εbg = 0.0                               # background strain rate
    #---------------------------------------------------------------------------------------

    # Physical Parameters
    rheology = init_rheology(; is_compressible=false, steady_state=true)
    cutoff_visc = (1e16, 1e24)
    κ = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = (0.5 * min(di...)^2 / κ / 2.01)         # diffusive CFL timestep limiter

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 15
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...)

    subgrid_arrays   = SubgridDiffusionCellArrays(particles)

    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(3))
    particle_args = (pT, pPhases)

    # Circular temperature anomaly--------------------------
    x_anomaly = lx * 0.5
    y_anomaly = -5e3  # origin of the small thermal anomaly
    r_anomaly = 1.5e3             # radius of perturbation
    anomaly = 750 + 273              # thermal perturbation (in K)
    init_phases!(pPhases, particles, x_anomaly, y_anomaly, r_anomaly, sticky_air)
    phase_ratios = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

    # Initialisation of thermal profile
    thermal = ThermalArrays(ni)                                # initialise thermal arrays and boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false),
    )
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2], sticky_air)
    circular_perturbation!(
        thermal.T, anomaly, x_anomaly, y_anomaly, r_anomaly, xvi, sticky_air
    )
    thermal_bcs!(thermal.T, thermal_bc)
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )
    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(ni, ViscoElastic)                         # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=0.99 / √2.1)
    # ----------------------------------------------------

    args = (; T=thermal.Tc, P=stokes.P, dt=dt)

    pt_thermal = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=5e-2 / √2.1
    )
    # Boundary conditions of the flow
    stokes.V.Vx .= PTArray([
        εbg * (x - lx * 0.5) / (lx / 2) / 2 for x in xvi[1], _ in 1:(ny + 2)
    ])
    stokes.V.Vy .= PTArray([
        (abs(y) - sticky_air) * εbg * (abs(y) > sticky_air) for _ in 1:(nx + 2), y in xvi[2]
    ])

    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true),
        free_surface =true,
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    η = @ones(ni...)                                     # initialise viscosity
    ϕ = similar(η)                                       # melt fraction center

    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, cutoff_visc
    )
    η_vep = copy(η)

    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
    )

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:2
        @parallel (JustRelax.@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
        )
        @parallel init_P!(stokes.P, ρg[2], xci[2], sticky_air)
    end

    # Arguments for functions
    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)
    @copy thermal.Told thermal.T

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size=(1200, 900))
        ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T [C]")
        ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
        scatter!(ax1, Array(thermal.T[2:(end - 1), :][:] .- 273), Yv)
        lines!(ax2, Array(stokes.P[:]./1e6), Y)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # Time loop
    t, it = 0.0, 0
    local iters
    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    @copy stokes.P0 stokes.P
    P_init = deepcopy(stokes.P)


    while it < 25 #nt

        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= 293.0
        @views thermal.T[2:end - 1, :] .= T_buffer
        thermal_bcs!(thermal.T, thermal_bc)
        temperature2center!(thermal)
        thermal.ΔT .= thermal.T .- thermal.Told
        @parallel (@idx size(thermal.ΔTc)...) temperature2center!(thermal.ΔTc, thermal.ΔT)

        # Update buoyancy and viscosity -
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, cutoff_visc
        )
        @parallel (@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
        )
        # Stokes solver -----------------
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
            dt,
            igg;
            iterMax=250e3,
            free_surface=true,
            nout=5e3,
            viscosity_cutoff=cutoff_visc,
        )
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(
            stokes.ε.II, @strain(stokes)...
        )
        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(
            @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
        )
        dt = compute_dt(stokes, di, dt_diff, igg)
        # --------------------------------

        @parallel (@idx ni) compute_shear_heating!(
            thermal.shear_heating,
            @tensor_center(stokes.τ),
            @tensor_center(stokes.τ_o),
            @strain(stokes),
            phase_ratios.center,
            rheology,
            dt,
        )
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            igg=igg,
            phase=phase_ratios,
            iterMax=150e3,
            nout=1e3,
            verbose=true,
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
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
        )

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)

            if igg.me == 0
                if do_vtk
                    JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                    data_v = (;
                        T=Array(thermal.T[2:(end - 1), :]),
                        τxy=Array(stokes.τ.xy),
                        εxy=Array(stokes.ε.xy),
                        Vx=Array(Vx_v),
                        Vy=Array(Vy_v),
                    )
                    data_c = (;
                        P=Array(stokes.P),
                        τxx=Array(stokes.τ.xx),
                        τyy=Array(stokes.τ.yy),
                        εxx=Array(stokes.ε.xx),
                        εyy=Array(stokes.ε.yy),
                        η=Array(η),
                    )
                    save_vtk(
                        joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                        xvi,
                        xci,
                        data_v,
                        data_c,
                    )
                end

                # Make particles plottable
                p = particles.coords
                ppx, ppy = p
                pxv = ppx.data[:] ./ 1e3
                pyv = ppy.data[:] ./ 1e3
                clr = pPhases.data[:]
                idxv = particles.index.data[:]

                # Make Makie figure
                fig = Figure(; size=(2000, 1800), createmissing=true)
                ar = li[1] / li[2]

                ax0 = Axis(
                    fig[1, 1:2];
                    aspect=ar,
                    title="t = $(t./(3600*24*365*1e3)) Kyrs",
                    titlesize=50,
                    height=0.0,
                )
                ax0.ylabelvisible = false
                ax0.xlabelvisible = false
                ax0.xgridvisible = false
                ax0.ygridvisible = false
                ax0.xticksvisible = false
                ax0.yticksvisible = false
                ax0.yminorticksvisible = false
                ax0.xminorticksvisible = false
                ax0.xgridcolor = :white
                ax0.ygridcolor = :white
                ax0.ytickcolor = :white
                ax0.xtickcolor = :white
                ax0.yticklabelcolor = :white
                ax0.xticklabelcolor = :white
                ax0.yticklabelsize = 0
                ax0.xticklabelsize = 0
                ax0.xlabelcolor = :white
                ax0.ylabelcolor = :white

                ax1 = Axis(
                    fig[2, 1][1, 1];
                    aspect=ar,
                    title=L"T [\mathrm{C}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax2 = Axis(
                    fig[2, 2][1, 1];
                    aspect=ar,
                    title=L"Viscosity [\mathrm{Pa s}]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect=ar,
                    title=L"ΔP [MPa]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect=ar,
                    title=L"P [MPa]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax5 = Axis(
                    fig[4, 1][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\dot{\varepsilon}_{\textrm{II}}) [\mathrm{s}^{-1}]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax6 = Axis(
                    fig[4, 2][1, 1];
                    aspect=ar,
                    title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # Plot temperature
                p1 = heatmap!(
                    ax1,
                    xvi[1] .* 1e-3,
                    xvi[2] .* 1e-3,
                    Array(thermal.T[2:(end - 1), :] .- 273);
                    colormap=:batlow,
                )
                # Plot effective viscosity
                p2 = heatmap!(
                    ax2,
                    xci[1] .* 1e-3,
                    xci[2] .* 1e-3,
                    Array(log10.(η_vep));
                    colormap=:glasgow,
                    colorrange=(log10(1e16), log10(1e24)),
                )
                arrows!(
                    ax2,
                    xvi[1][1:5:end-1]./1e3, xvi[2][1:5:end-1]./1e3,
                    Array.((Vx_v[1:5:end-1, 1:5:end-1],
                    Vy_v[1:5:end-1, 1:5:end-1]))...,
                    lengthscale = 2 / max(maximum(Vx_v),  maximum(Vy_v)),
                    color = :darkblue,
                )
                # Plot Pressure difference
                p3 = heatmap!(
                    ax3,
                    xci[1] .* 1e-3,
                    xci[2] .* 1e-3,
                    Array((stokes.P .- P_init) ./ 1e6);
                    colormap=:roma,
                )
                # Plot Pressure difference
                p4 = heatmap!(
                    ax4,
                    xci[1] .* 1e-3,
                    xci[2] .* 1e-3,
                    Array((stokes.P) ./ 1e6);
                    colormap=:roma,
                )
                    # Plot 2nd invariant of strain rate
                p5 = heatmap!(
                    ax5,
                    xci[1] .* 1e-3,
                    xci[2] .* 1e-3,
                    Array(log10.(stokes.ε.II));
                    colormap=:roma,
                )
                # Plot 2nd invariant of stress
                p6 = heatmap!(
                    ax6,
                    xci[1] .* 1e-3,
                    xci[2] .* 1e-3,
                    Array((stokes.τ.II)./1e6);
                    colormap=:batlow,
                )
                hidexdecorations!(ax1)
                hidexdecorations!(ax2)
                hidexdecorations!(ax3)
                Colorbar(
                    fig[2, 1][1, 2], p1; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[2, 2][1, 2], p2; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 1][1, 2], p3; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 2][1, 2], p4; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 1][1, 2], p5; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 2][1, 2], p6; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                rowgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)
                fig

                let
                    Yv = [y for x in xvi[1], y in xvi[2]][:]
                    Y = [y for x in xci[1], y in xci[2]][:]
                    fig = Figure(; size=(1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
                    ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
                    a3 = Axis(fig[2, 1]; aspect=2 / 3, title="τII")

                    scatter!(ax1, Array(thermal.T[2:(end - 1), :][:] .- 273.0), Yv)
                    lines!(ax2, Array(stokes.P[:]./1e6), Y)
                    scatter!(a3, Array((stokes.τ.II[:])./1e6), Y)

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end
            end
        end
    end
    # finalize_global_grid()

end

figdir = "Thermal_stresses_around_cooling_magma"
do_vtk = true # set to true to generate VTK files for ParaView
ar = 1 # aspect ratio
n = 128
nx = n * ar - 2
ny = n - 2
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
else
    igg
end

# run main script
main2D(igg; figdir=figdir, nx=nx, ny=ny, do_vtk=do_vtk);

function plot_particles(particles, pPhases)
    p = particles.coords
    ppx, ppy = p
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    idxv = particles.index.data[:]
    f, ax, h = scatter(
        Array(pxv[idxv]), Array(pyv[idxv]); color=Array(clr[idxv]), colormap=:roma
    )
    Colorbar(f[1, 2], h)
    return f
end

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
model = PS_Setup(:cpu, Float64, 2) #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

using Printf, Statistics, LinearAlgebra, GeoParams, GLMakie, CellArrays
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
    esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z, sticky_air)
    @all(P) = (abs(@all(ρg) * (@all_j(z))) - (@all(ρg)*sticky_air)) * <((@all_j(z)), 0.0)
    return nothing
end

function init_phases!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly, sticky_air)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(
        phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly, sticky_air)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = -(JustRelax.@cell py[ip, i, j]) - sticky_air
            if 0e0 ≤ y ≤ 20e3
                @cell phases[ip, i, j] = 1.0 # crust
            end

            # # # chamber - elliptical
            # if (((x - xc)^2 / ((a)^2)) + ((y + yc)^2 / ((b)^2)) ≤ r^2)
            #     JustRelax.@cell phases[ip, i, j] = 2.0
            # end

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y + yc_anomaly)^2 ≤ r_anomaly^2)
                JustRelax.@cell phases[ip, i, j] = 2.0
            end

            if y < 0.0
                @cell phases[ip, i, j] = 4.0
            end

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(
        phases, particles.coords..., particles.index, xc_anomaly, yc_anomaly, r_anomaly, sticky_air)
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, sticky_air)
    depth = -y[j] - sticky_air

    # (depth - 15e3) because we have 15km of sticky air
    if depth < 0e0
        T[i + 1, j] = 273e0


    elseif 0e0 ≤ (depth) < 25e3
        dTdZ        = (923-273)/35e3
        offset      = 273e0
        T[i + 1, j] = (depth) * dTdZ + offset

    elseif 110e3 > (depth) ≥ 35e3
        dTdZ        = (1492-923)/75e3
        offset      = 923
        T[i + 1, j] = (depth - 35e3) * dTdZ + offset

    elseif (depth) ≥ 110e3
        dTdZ        = (1837 - 1492)/590e3
        offset      = 1492e0
        T[i + 1, j] = (depth - 110e3) * dTdZ + offset

    end

    return nothing
end


function circular_anomaly!(T, anomaly, xc, yc, r, xvi, sticky_air)
    @parallel_indices (i, j) function _circular_anomaly!(T, anomaly, xc, yc, r, x, y, sticky_air)
        depth = -y[j] - sticky_air
        @inbounds if (((x[i] - xc)^2 + (depth[j] + yc)^2) ≤ r^2)
            T[i + 1, j] = anomaly
        end
        return nothing
    end

    ni = length.(xvi)
    @parallel (@idx ni) _circular_anomaly!(T, anomaly, xc, yc, r, xvi..., sticky_air)
    return nothing
end

function elliptical_anomaly!(T, anomaly, xc, yc, a, b, r, xvi, sticky_air)

    @parallel_indices (i, j) function _elliptical_anomaly!(
        T, anomaly, xc, yc, a, b, r, x, y, sticky_air
    )
        depth = -y[j] - sticky_air
        @inbounds if (((x[i] - xc)^2 / a^2) + ((depth[j] + yc)^2 / b^2) ≤ r^2)
            T[i + 1, j ] = anomaly
        end
        return nothing
    end

    ni = length.(xvi)
    @parallel (@idx ni) _elliptical_anomaly!(T, anomaly, xc, yc, a, b, r, xvi..., sticky_air)
    return nothing
end


function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi, sticky_air)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, x, y, sticky_air)
        depth = -y[j] - sticky_air
        @inbounds if  ((x[i] - xc_anomaly)^2 + (depth[j] + yc_anomaly)^2 ≤ r_anomaly^2)
            T[i + 1, j ] = anomaly
        end
        return nothing
    end

    nx, ny = size(T)

    @parallel (1:nx-2, 1:ny) _circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi..., sticky_air)
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



# function main2D(igg; figdir=figdir, nx=nx, ny=ny, do_vtk= false, igg=igg)

    #-------rheology parameters--------------------------------------------------------------
    # plasticity setup
    do_DP       = true               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg       = 1.0e14           # regularisation "viscosity" for Drucker-Prager
    Coh         = 10MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ           = 30.0 * do_DP         # friction angle
    G0          = 25e9Pa        # elastic shear modulus
    G_magma     = 10e9Pa        # elastic shear modulus perturbation
    # εbg         = 2.5e-14            # background strain rate
    εbg         = 0.0            # background strain rate

    # soft_C      = LinearSoftening((ustrip(Coh)/2, ustrip(Coh)), (0e0, 1e-1)) # softening law
    pl          = DruckerPrager_regularised(; C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0.0)#, softening_C = soft_C)        # plasticity

    el          = SetConstantElasticity(; G=G0, ν=0.5)                            # elastic spring
    el_magma    = SetConstantElasticity(; G=G_magma, ν=0.5)                            # elastic spring
    creep_rock  = LinearViscous(; η=1e21 * Pa * s)
    creep_magma = LinearViscous(; η=1e16 * Pa * s)
    creep_air   = LinearViscous(; η=1e16 * Pa * s)
    cutoff_visc = (1e14, 1e24)
    β_rock      = inv(get_Kb(el))
    β_magma     = inv(get_Kb(el_magma))
    Kb          = get_Kb(el)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    sticky_air  = 0.0              # thickness oif the sticky air layer
    ly          = 12.5e3 + sticky_air # domain length in y-direction
    lx          = 10.5e3            # domain length in x-direction
    li          = lx, ly            # domain length in x- and y-direction
    ni          = nx, ny            # number of grid points in x- and y-direction
    di          = @. li / ni        # grid step in x- and y-direction
    origin      = 0.0, -ly          # origin coordinates of the domain
    grid        = Geometry(ni, li; origin = origin)
    (; xci, xvi)= grid             # nodes at the center and vertices of the cells
    #---------------------------------------------------------------------------------------

    # Set material parameters
    MatParam = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase   = 1,
            Density  = PT_Density(ρ0=2700kg/m^3,α=3e-5/K, T0=273.0, β=β_rock/Pa),
            HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=3.0Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl, )),
            Melting = MeltingParam_Caricchi(),
            Elasticity = el,
            ),

        #Name="Magma"
        SetMaterialParams(;
            Phase   = 2,
            Density  = PT_Density(ρ0=2600kg/m^3, T0=273.0, β=β_magma/Pa),
            HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=3.0Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Caricchi(),
            Elasticity = el_magma,
            ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase   = 4,
            Density   = PT_Density(ρ0=10kg/m^3, T0=273.0,β= 0.0),
            HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
            Conductivity = ConstantConductivity(k=15Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_air,)),
            ),
            )

    #----------------------------------------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    x_anomaly = lx * 0.5
    y_anomaly = -5e3  # origin of the small thermal anomaly

    r_anomaly = 1.5e3             # radius of perturbation

    init_phases!(pPhases, particles, x_anomaly, y_anomaly, r_anomaly,sticky_air)
    phase_ratios = PhaseRatio(ni, length(MatParam))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

    # Physical Parameters
    geotherm = GeoUnit(0.03K / m)
    geotherm = ustrip(Value(geotherm))
    ΔT = geotherm * (ly - sticky_air) # temperature difference between top and bottom of the domain
    η = MatParam[2].CompositeRheology[1][1].η.val       # viscosity for the Rayleigh number
    Cp0 = MatParam[2].HeatCapacity[1].Cp.val              # heat capacity
    ρ0 = MatParam[2].Density[1].ρ0.val                   # reference Density
    k0 = MatParam[2].Conductivity[1]              # Conductivity
    G = MatParam[1].Elasticity[1].G.val                 # Shear Modulus
    κ = ustrip(1.5Watt / K / m / (ρ0 * Cp0))                                   # thermal diffusivity
    g = MatParam[1].Gravity[1].g.val                    # Gravity

    α = MatParam[1].Density[1].α.val                    # thermal expansion coefficient for PT Density
    Ra =   ρ0 * g * α * (900+273 - ΔT) * 1.5e3^3 / (η * κ)                # Rayleigh number
    dt = dt_diff = (0.5 * min(di...)^2 / κ / 2.01)         # diffusive CFL timestep limiter

    # Initialisation
    thermal = ThermalArrays(ni)                                # initialise thermal arrays and boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )

    @parallel (@idx ni .+ 1) init_T!(
        thermal.T, xvi[2], sticky_air)

    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )

    stokes = StokesArrays(ni, ViscoElastic)                         # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=0.99 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1

    args = (; T=thermal.Tc, P=stokes.P, dt=Inf)

    pt_thermal = PTThermalCoeffs(
        MatParam, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=5e-2 / √2.1
    )
    # Boundary conditions of the flow
    stokes.V.Vx .= PTArray([ (x - lx/2) * εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([ -(ly - abs(y)) * εbg for _ in 1:nx+2, y in xvi[2]])

    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    η = @ones(ni...)                                     # initialise viscosity
    η_vep = deepcopy(η)                                       # initialise viscosity for the VEP
    G = @fill(MatParam[1].Elasticity[1].G.val, ni...)     # initialise shear modulus
    ϕ = similar(η)                                        # melt fraction center

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    # Arguments for functions
    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)
    @copy thermal.Told thermal.T

    for _ in 1:2
        @parallel (JustRelax.@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )
        @parallel init_P!(stokes.P, ρg[2], xci[2], sticky_air)
        # @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
    )
    η_vep = copy(η)

    anomaly = 750.0 .+ 273.0 # temperature anomaly

    circular_perturbation!(thermal.T, anomaly, x_anomaly, y_anomaly, r_anomaly, xvi, sticky_air)

    # make sure they are the same
    thermal.Told .= thermal.T
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )
    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
    )

    # Time loop
    t, it      = 0.0, 0
    local iters
    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end

    grid2particle!(pT, xvi, T_buffer, particles)
    @copy stokes.P0 stokes.P


    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------
    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size=(1200, 900))
        ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
        ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
        scatter!(
            ax1,
            Array(thermal.T[2:(end - 1), :][:].-273.0),
            Yv,
        )
        lines!(
            ax2,
            Array(stokes.P[:]),
            Y,
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    P_init = deepcopy(stokes.P);
    while it < 50 #nt

        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= 273.0
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)

        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
        )
        @parallel (@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )
        @copy stokes.P0 stokes.P
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)
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
            MatParam,
            args,
            dt,
            igg;
            iterMax = 250e3,
            nout = 5e3,
            viscosity_cutoff=cutoff_visc,
        )
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(
            @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
        )
        # dt = compute_dt(stokes, di, dt_diff, igg)
        # ------------------------------
        @show dt

        @parallel (@idx ni) compute_shear_heating!(
            thermal.shear_heating,
            @tensor_center(stokes.τ),
            @tensor_center(stokes.τ_o),
            @strain(stokes),
            phase_ratios.center,
            MatParam, # needs to be a tuple
            dt,
        )
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            MatParam,
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

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)
        # JustPIC.clean_particles!(particles, xvi, particle_args)
        grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # ------------------------------
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            checkpointing_jld2(figdir, stokes, thermal, η, particles, pPhases, t; igg=igg)

            if igg.me == 0
                if do_vtk
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

                pp = [argmax(p) for p in phase_ratios.center];
                @views η_vep[pp.==4.0] .= NaN
                @views stokes.τ.II[pp.==4.0] .= NaN
                @views stokes.ε.II[pp.==4.0] .= NaN

                # Make Makie figure
                fig = Figure(; size=(2000, 1800), createmissing=true)
                ar = li[1] / li[2]
                # ar = DataAspect()

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
                    # title=L"\log_{10}(\eta_{vep}) [\mathrm{Pas}]",
                    title=L"ΔP [MPa]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect=ar,
                    title=L"Phases",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\dot{\varepsilon}_{\textrm{II}}) [\mathrm{s}^{-1}]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax5 = Axis(
                    fig[4, 1][1, 1];
                    aspect=ar,
                    title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # Plot temperature
                p1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:].-273) , colormap=:batlow)
                # Plot effective viscosity
                p2  = heatmap!(ax2, xci[1].*1e-3, xci[2].*1e-3, Array((stokes.P .- P_init)./1e6) , colormap=:roma)#, colorrange= (log10(1e14), log10(1e21)))
                # Plot particles phase
                p3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:glasgow, colorrange= (log10(1e14), log10(1e21)))
                # p3  = scatter!(ax3, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
                # Plot 2nd invariant of strain rate
                p4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:roma)
                # Plot 2nd invariant of stress
                p5  = heatmap!(ax5, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.τ.II)) , colormap=:batlow)
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
                rowgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                fig
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)

                let
                    Yv = [y for x in xvi[1], y in xvi[2]][:]
                    Y = [y for x in xci[1], y in xci[2]][:]
                    fig = Figure(; size=(1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
                    ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
                    a3  = Axis(fig[2, 1]; aspect=2 / 3, title="τII")

                    scatter!(
                        ax1,
                        Array(thermal.T[2:(end - 1), :][:].-273.0),
                        Yv,
                    )
                    lines!(
                        ax2,
                        Array(stokes.P[:]),
                        Y,
                    )
                    scatter!(
                        a3,
                        Array(log10.(stokes.τ.II[:])),
                        Y,
                    )

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end

            end
        end
    end
    # finalize_global_grid()

end


figdir = "Thermal_stress_cooling_magma_body"
do_vtk = true # set to true to generate VTK files for ParaView
ar       = 1 # aspect ratio
n        = 64
nx       = n*ar - 2
ny       = n - 2
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

# run main script
main2D(igg; figdir=figdir, nx=nx, ny=ny, do_vtk=do_vtk, igg=igg);

function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clrT = pT.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma)
    Colorbar(f[1,2], h)
    f
end

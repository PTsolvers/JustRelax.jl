using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using Printf, Statistics, LinearAlgebra, GeoParams, GLMakie, CellArrays
using StaticArrays
using ImplicitGlobalGrid
using MPI: MPI
using WriteVTK

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

function init_phases!(phases, particles, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, sticky_air, top, bottom)
    ni = size(phases)

    @parallel_indices (I...) function init_phases!(
            phases, px, py, pz, index, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, sticky_air, top, bottom
        )
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            z = -(@index pz[ip, I...]) - sticky_air
            if top ≤ z ≤ bottom
                @index phases[ip, I...] = 1.0 # crust
            end

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y - yc_anomaly)^2 + (z + zc_anomaly)^2 ≤ r_anomaly^2)
                @index phases[ip, I...] = 2.0
            end

            if z < top
                @index phases[ip, I...] = 3.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(
        phases,
        particles.coords...,
        particles.index,
        xc_anomaly,
        yc_anomaly,
        zc_anomaly,
        r_anomaly,
        sticky_air,
        top,
        bottom,
    )
end

# Initial thermal profile
@parallel_indices (i, j, k) function init_T!(T, y, sticky_air, top, bottom, dTdz, offset)
    I = i, j, k
    depth = -y[k] - sticky_air

    if depth < top
        T[I...] = offset

    elseif top ≤ (depth) < bottom
        dTdZ = dTdz
        offset = offset
        T[I...] = (depth) * dTdZ + offset

    end

    return nothing
end


function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, xvi, sticky_air)

    @parallel_indices (i, j, k) function _circular_perturbation!(
            T, δT, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, x, y, z, sticky_air
        )
        depth = -z[k] - sticky_air
        @inbounds if ((x[i] - xc_anomaly)^2 + (y[j] - yc_anomaly)^2 + (depth + zc_anomaly)^2 ≤ r_anomaly^2)
            # T[i, j, k] *= δT / 100 + 1
            T[i, j, k] = δT
        end
        return nothing
    end

    ni = size(T)

    return @parallel (@idx ni) _circular_perturbation!(
        T, δT, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, xvi..., sticky_air
    )
end

function init_rheology(CharDim; is_compressible = false, steady_state = true)
    # plasticity setup
    do_DP = true          # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg = 1.0e16Pa * s    # regularisation "viscosity" for Drucker-Prager
    Coh = 10.0MPa       # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30.0 * do_DP  # friction angle
    G0 = 6.0e11Pa      # elastic shear modulus
    G_magma = 6.0e11Pa      # elastic shear modulus perturbation

    # soft_C = LinearSoftening((ustrip(Coh)/2, ustrip(Coh)), (0e0, 1e-1)) # softening law
    soft_C = NonLinearSoftening(; ξ₀ = ustrip(Coh), Δ = ustrip(Coh) / 2)   # softening law
    pl = DruckerPrager_regularised(; C = Coh, ϕ = ϕ, η_vp = η_reg, Ψ = 0.0, softening_C = soft_C)        # plasticity
    if is_compressible == true
        el = SetConstantElasticity(; G = G0, ν = 0.25)           # elastic spring
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.25) # elastic spring
        β_rock = 6.0e-11
        β_magma = 6.0e-11
    else
        el = SetConstantElasticity(; G = G0, ν = 0.5)            # elastic spring
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.5) # elastic spring
        β_rock = inv(get_Kb(el))
        β_magma = inv(get_Kb(el_magma))
    end
    if steady_state == true
        creep_rock = LinearViscous(; η = 1.0e23 * Pa * s)
        creep_magma = LinearViscous(; η = 1.0e18 * Pa * s)
        creep_air = LinearViscous(; η = 1.0e18 * Pa * s)
    else
        creep_rock = DislocationCreep(; A = 1.67e-24, n = 3.5, E = 1.87e5, V = 6.0e-6, r = 0.0, R = 8.3145)
        creep_magma = DislocationCreep(; A = 1.67e-24, n = 3.5, E = 1.87e5, V = 6.0e-6, r = 0.0, R = 8.3145)
        creep_air = LinearViscous(; η = 1.0e18 * Pa * s)
        β_rock = 6.0e-11
        β_magma = 6.0e-11
    end
    g = 9.81m / s^2
    return rheology = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2650kg / m^3, α = 3.0e-5 / K, T0 = 0.0C, β = β_rock / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
            LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3J / kg),
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
            CharDim = CharDim,
        ),

        #Name="Magma"
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2650kg / m^3, T0 = 0.0C, β = β_magma / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 1.5Watt / K / m),
            LatentHeat = ConstantLatentHeat(; Q_L = 350.0e3J / kg),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_magma,
            CharDim = CharDim,
        ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(ρ = 1kg / m^3),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1000J / kg / K),
            Conductivity = ConstantConductivity(; k = 15Watt / K / m),
            LatentHeat = ConstantLatentHeat(; Q_L = 0.0J / kg),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_air,)),
            Gravity = ConstantGravity(; g = g),
            CharDim = CharDim,
        ),
    )

end


function main3D(igg; figdir = "output", nx = 64, ny = 64, nz = 64, do_vtk = false)

    # Characteristic lengths for non dimensionalization
    CharDim = GEO_units(; length = 12.5km, viscosity = 1.0e21, temperature = 1.0e3C)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    sticky_air = nondimensionalize(1.5km, CharDim)          # thickness of the sticky air layer
    lz = nondimensionalize(12.5km, CharDim) + sticky_air     # domain length in y-direction
    lx = ly = nondimensionalize(15.5km, CharDim)            # domain length in x-direction
    li = lx, ly, lz                                         # domain length in x- and y-direction
    ni = nx, ny, nz                                         # number of grid points in x- and y-direction
    di = @. li / ni                                         # grid step in x- and y-direction
    origin = nondimensionalize(0.0km, CharDim), nondimensionalize(0.0km, CharDim), -lz    # origin coordinates of the domain
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                         # nodes at the center and vertices of the cells
    εbg = nondimensionalize(0.0 / s, CharDim)    # background strain rate
    #---------------------------------------------------------------------------------------

    # Physical Parameters
    rheology = init_rheology(CharDim; is_compressible = true, steady_state = true)
    cutoff_visc = nondimensionalize((1.0e16Pa * s, 1.0e24Pa * s), CharDim)
    κ = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = (0.5 * min(di...)^2 / κ / 2.01)         # diffusive CFL timestep limiter

    # Initialize particles -------------------------------
    nxcell = 20
    max_xcell = 40
    min_xcell = 15
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Circular temperature anomaly--------------------------
    x_anomaly = lx * 0.5
    y_anomaly = ly * 0.5
    z_anomaly = nondimensionalize(-5km, CharDim)  # origin of the small thermal anomaly
    r_anomaly = nondimensionalize(1.5km, CharDim)             # radius of perturbation
    anomaly = nondimensionalize((750 + 273)K, CharDim)               # thermal perturbation (in K)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, x_anomaly, y_anomaly, z_anomaly, r_anomaly, sticky_air, nondimensionalize(0.0km, CharDim), nondimensionalize(20km, CharDim))
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # Initialisation of thermal profile
    thermal = ThermalArrays(backend_JR, ni) # initialise thermal arrays and boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, front = true, back = true, top = false, bot = false),
    )
    @parallel (@idx ni .+ 1) init_T!(
        thermal.T,
        xvi[3],
        sticky_air,
        nondimensionalize(0.0e0km, CharDim),
        nondimensionalize(15km, CharDim),
        nondimensionalize((723 - 273)K, CharDim) / nondimensionalize(15km, CharDim),
        nondimensionalize(273K, CharDim)
    )
    circular_perturbation!(
        thermal.T, anomaly, x_anomaly, y_anomaly, z_anomaly, r_anomaly, xvi, sticky_air
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni) # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    args = (; T = thermal.Tc, P = stokes.P, dt = dt, ΔTc = thermal.ΔTc)
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 0.8 / √3.1
    )

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
        no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
        free_surface = true,
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)


    # Buoyancy force & viscosity
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...) # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
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

    # Plot initial T and η profiles
    let
        Zv = [z for _ in xvi[1], _ in xvi[2], z in xvi[3]][:]
        Z = [z for _ in xci[1], _ in xci[2], z in xci[3]][:]
        fig = Figure(; size = (1200, 900))
        ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
        scatter!(
            ax1,
            Array(ustrip.(dimensionalize(thermal.T[:], C, CharDim))),
            ustrip.(dimensionalize(Zv, km, CharDim)),
        )
        scatter!(
            ax2,
            Array(ustrip.(dimensionalize(stokes.P[:], MPa, CharDim))),
            ustrip.(dimensionalize(Z, km, CharDim)),
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

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
    Tsurf = thermal.T[1, 1, end]
    Tbot = thermal.T[1, 1, 1]

    while it < 25

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf, ΔTc = thermal.ΔTc)
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
                iterMax = 150.0e3,
                free_surface = false,
                nout = 5.0e3,
                viscosity_cutoff = cutoff_visc,
            )
        )
        tensor_invariant!(stokes.ε)

        dt = compute_dt(stokes, di, dt_diff, igg)
        # --------------------------------

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
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
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 150.0e3,
                nout = 1.0e3,
                verbose = true,
            )
        )
        # subgrid diffusion
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
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (thermal.T,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        particle2grid!(thermal.T, pT, xvi, particles)
        @views thermal.T[:, :, end] .= Tsurf
        @views thermal.T[:, :, 1] .= Tbot
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT)

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            (; η) = stokes.viscosity
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)
            t_dim = (dimensionalize(t, yr, CharDim).val / 1.0e3)
            t_Kyrs = t_dim / 1.0e3
            if igg.me == 0
                if do_vtk
                    velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                    data_v = (;
                        T = Array(ustrip.(dimensionalize(thermal.T, C, CharDim))),
                        τxy = Array(ustrip.(dimensionalize(stokes.τ.xy, s^-1, CharDim))),
                        εxy = Array(ustrip.(dimensionalize(stokes.ε.xy, s^-1, CharDim))),
                        Vx = Array(ustrip.(dimensionalize(Vx_v, cm / yr, CharDim))),
                        Vy = Array(ustrip.(dimensionalize(Vy_v, cm / yr, CharDim))),
                        Vz = Array(ustrip.(dimensionalize(Vz_v, cm / yr, CharDim))),
                    )
                    data_c = (;
                        P = Array(ustrip.(dimensionalize(stokes.P, MPa, CharDim))),
                        τxx = Array(ustrip.(dimensionalize(stokes.τ.xx, MPa, CharDim))),
                        τyy = Array(ustrip.(dimensionalize(stokes.τ.yy, MPa, CharDim))),
                        τzz = Array(ustrip.(dimensionalize(stokes.τ.zz, MPa, CharDim))),
                        τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CharDim))),
                        εxx = Array(ustrip.(dimensionalize(stokes.ε.xx, s^-1, CharDim))),
                        εyy = Array(ustrip.(dimensionalize(stokes.ε.yy, s^-1, CharDim))),
                        εzz = Array(ustrip.(dimensionalize(stokes.ε.zz, s^-1, CharDim))),
                        εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1, CharDim))),
                        η = Array(ustrip.(dimensionalize(stokes.viscosity.η_vep, Pa * s, CharDim))),
                    )
                    velocity_v = (
                        Array(ustrip.(dimensionalize(Vx_v, cm / yr, CharDim))),
                        Array(ustrip.(dimensionalize(Vy_v, cm / yr, CharDim))),
                        Array(ustrip.(dimensionalize(Vz_v, cm / yr, CharDim))),
                    )
                    save_vtk(
                        joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                        xvi,
                        xci,
                        data_v,
                        data_c,
                        velocity_v;
                        t = t_Kyrs
                    )
                end

                let
                    Zv = [z for _ in 1:(nx + 1), _ in 1:(ny + 1), z in ustrip.(dimensionalize(xvi[3], km, CharDim))][:]
                    Z = [z for _ in 1:nx  , _ in 1:ny  , z in ustrip.(dimensionalize(xci[3], km, CharDim))][:]
                    fig = Figure(; size = (1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
                    ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
                    a3 = Axis(fig[2, 1]; aspect = 2 / 3, title = "τII")

                    scatter!(ax1, ustrip.(dimensionalize((Array(thermal.T)), C, CharDim))[:], Zv, markersize = 4)
                    scatter!(ax2, ustrip.(dimensionalize((Array(stokes.P)), MPa, CharDim))[:], Z, markersize = 4)
                    scatter!(a3, ustrip.(dimensionalize(Array(stokes.τ.II), MPa, CharDim))[:], Z, markersize = 4)

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end
            end
        end
    end

    finalize_global_grid()
    return nothing
end

figdir = "Thermal_stresses_around_cooling_magma_3D"
do_vtk = true # set to true to generate VTK files for ParaView
n = 64
nx = n
ny = n
nz = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

# run main script
main3D(igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);

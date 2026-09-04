const isCUDA = false
# const isCUDA = true

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

using JustPIC
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPU # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, CairoMakie, CellArrays, Statistics, Dates, JLD2, Printf

# Load file with all the rheology configurations
include("SillRheology.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

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

# H2O gas density from the EOS, used to turn the exsolved gas *mass* fraction into a
# *volume* fraction. `gas_eos` is passed in rather than captured from global scope:
# this script runs at top level, so a closure over it would be a non-const global
# lookup that the GPU compiler cannot resolve.
@parallel_indices (i, j) function compute_gas_density!(ρ_gas, gas_eos, P, T)
    ρ_gas[i, j] = compute_density(gas_eos, (; P = P[i, j], T = T[i, j]))
    return nothing
end

# Bulk H2O carried by each phase. The sill is emplaced water-rich, the host rock dry;
# `phase_sill` is the (advected) sill marker, so the two blend across the contact.
@parallel_indices (i, j) function init_H2O!(mH2O, phase_sill, sill_H2O, host_H2O)
    s = phase_sill[i, j]
    mH2O[i, j] = host_H2O + (sill_H2O - host_H2O) * s
    return nothing
end

function init_sill!(
    phases,
    dimensions::NTuple{2, Float64},
    sill_size,
    grid;
    perturbation_amplitude::Float64 = 0.0,
    wavelength::Float64 = 100.0,
    bottom_pertubation = false
    )

    @parallel_indices (i, j) function _init_sill!(
        phases, dimensions, sill_size, x, z, perturbation_amplitude, wavelength, bottom_pertubation
        )
        x_coord = x[i]
        depth = -z[j]

        # Add sinusoidal perturbation to sill top and bottom
        perturbation = perturbation_amplitude * sin.(2π * x_coord / wavelength)
        perturbation_bot = perturbation * bottom_pertubation

        sill_bottom = (dimensions[2] - (dimensions[2] - sill_size) / 2) + perturbation_bot
        sill_top = (dimensions[2] - sill_size) / 2 + perturbation
        if depth <= sill_bottom && depth >= sill_top
            phases[i, j] = 2
        else
            phases[i, j] = 1
        end
        return nothing
    end

    @parallel (@idx size(phases)) _init_sill!(
        phases, dimensions, sill_size, grid..., perturbation_amplitude, wavelength, bottom_pertubation
    )
    return nothing
end

function init_T!(
    T,
    host_rock_temp::Float64,
    sill_temp::Float64,
    dimensions::NTuple{2, Float64},
    sill_size::Float64,
    grid;
    perturbation_amplitude::Float64 = 0.0,
    wavelength::Float64 = 100.0,
    bottom_pertubation = false
    )

    @parallel_indices (i, j) function _init_T!(T, host_rock_temp, sill_temp,  dimensions, sill_size, x, z, perturbation_amplitude, wavelength, bottom_pertubation)

        x_coord = x[i]
        depth = -z[j]

        # Add sinusoidal perturbation to sill top and bottom
        perturbation = perturbation_amplitude * sin.(2π * x_coord / wavelength)
        perturbation_bot = perturbation * bottom_pertubation

        sill_bottom = (dimensions[2] - (dimensions[2] - sill_size) / 2) + perturbation_bot
        sill_top = (dimensions[2] - sill_size) / 2 + perturbation
        if depth ≤ sill_bottom && depth ≥ sill_top
            T[i + 1, j + 1] = sill_temp + 10*rand()
        else
            T[i + 1, j + 1] = host_rock_temp
        end
        if (sill_top + (sill_size * 0.1) < depth ≤ sill_top + (sill_size * 0.2)) && ((dimensions[1] / 2 - (sill_size * 0.05)) < x[i] ≤  (dimensions[1] / 2 + (sill_size * 0.05)))
            T[i + 1, j + 1] = sill_temp + 75
        end
        return nothing
    end

    nx, ny = size(T) .- 2
    @parallel (1:nx, 1:ny) _init_T!(T, host_rock_temp, sill_temp, dimensions, sill_size, grid..., perturbation_amplitude, wavelength, bottom_pertubation)

    # Depth (top/bottom) ghost columns have no no_flux BC applied to them later,
    # so extrapolate them here; the left/right ghost rows are handled by the
    # subsequent thermal_bcs! call.
    @views T[:, 1]   .= T[:, 2]
    @views T[:, end] .= T[:, end - 1]

end

## END OF HELPER FUNCTION ------------------------------------------------------------


## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, igg; nx = 64, ny =64, figdir="SillConvection2D", do_vtk = false, cutoff_visc = (-Inf, Inf), plotting = true, sill_temp = 1000, host_rock_temp = 500, sill_size = 0.1, depth = 5e3, sill_H2O = 0.05, host_H2O = 0.01)

    # -----------------------------------------------------
    # Set up the JustRelax model
    # -----------------------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid             # nodes at the center and vertices of the cells

    # ---------------------------------------------------

    # Physical properties using GeoParams ----------------
    # Kilgore tuff -> the sill; pre-Kilgore (older) tuff -> the host rock. These must match
    # the bulk compositions the MAGEMin diagrams were computed at, i.e. the "Heise Sill" and
    # "Heise Host rock": the melt fraction comes from the diagram and the
    # melt viscosity/density from these tuples, so a mismatch describes two different rocks.
    #                   (SiO2   TiO2  Al2O3  FeO   MgO   CaO   Na2O  K2O   H2O)
    oxd_wt_sill       = (75.75, 0.28, 12.48, 2.14, 0.09, 0.48, 3.53, 5.19, 3.0)
    oxd_wt_host_rock  = (70.78, 0.55, 15.86, 3.93, 1.11, 1.20, 2.54, 3.84, 3.0)

    # Lookup tables are adapted onto the device so the interpolations run inside GPU kernels.
    PD_ArrayType = @static if isCUDA
        CuArray
    else
        Array
    end

    rheology = init_rheologies(oxd_wt_sill, oxd_wt_host_rock; scaling = 1e1Pas, magma = true, ArrayType = PD_ArrayType,  melting = :analytic)
    dt_time = 1.0 * 3600 * 24 * 365
    κ            = (4 / (1050 * rheology[1].Density[1].ρsolid.ρ0.val))
    dt_diff = 0.5 * min(di...)^2 / κ / 2.01
    dt = min(dt_time, dt_diff)
    # ----------------------------------------------------

    # Weno model -----------------------------------------
    weno = WENO5(backend, Val(2), ni) # T, phases all live at cell centers
    # ----------------------------------------------------

    # Assign material phases --------------------------
    phases_dev   = @zeros(ni...)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni);
    init_sill!(phases_dev, li, sill_size, xci; perturbation_amplitude = sill_size * 0.02, wavelength = sill_size, bottom_pertubation = true)

    # Sill and host partition the domain: mark the sill (phase 2), take the host
    # as its complement so the two markers sum to 1 in every cell.
    phase_sill = @zeros(ni...)
    @views phase_sill[phases_dev .== 2.0] .= 1.0
    phase_host = 1.0 .- phase_sill
    update_phase_ratios_2D!(phase_ratios, (phase_host, phase_sill), xci, xvi)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend, ni)
    pt_stokes       = PTStokesCoeffs(li, di; Re = π / 2, ϵ_rel=1e-5, ϵ_abs=1e-5, CFL=0.98 / √2.1)
    # ----------------------------------------------------

    thermal         = ThermalArrays(backend, ni) # T lives at cell centers with one ghost node on every boundary
    init_T!(thermal.T, host_rock_temp, sill_temp, li, sill_size, xci; perturbation_amplitude = sill_size * 0.02, wavelength = sill_size, bottom_pertubation = true)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)

    args = (; T=thermal.T, P=stokes.P, dt=dt)

    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )

    # Melt Fraction
    ϕ = @zeros(ni...)
    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    # --- Volatiles: bulk water, dissolved water, exsolved gas ---------------------
    # mH2O is the bulk (advected, conserved) water. The melt holds up to the
    # solubility mH2O_diss per unit melt mass, so the bulk capacity is mH2O_diss*ϕ and
    # crystallisation (falling ϕ) drives exsolution even at fixed P,T — second boiling.
    # The exsolved gas occupies a volume fraction ϕ_gas that lowers the mixture density.
    mH2O      = @zeros(ni...)                      # bulk (total) H2O mass fraction
    @parallel (@idx ni) init_H2O!(mH2O, phase_sill, sill_H2O, host_H2O)
    mH2O_diss = @zeros(ni...)                      # H2O solubility (per melt mass)
    mCO2_diss = @zeros(ni...)
    mH2O_exs  = @zeros(ni...)                      # exsolved gas mass fraction (of bulk)
    mH2O_melt = copy(mH2O)                         # dissolved water fed to melt/density
    X_co2     = @zeros(ni...)                      # pure-water system: no CO2 in the gas
    ϕ_gas     = @zeros(ni...)                      # exsolved-gas volume fraction
    ϕ_x       = @zeros(ni...)                      # crystal volume fraction
    ρ_gas     = @zeros(ni...)                      # H2O gas density (EOS), for mass->volume
    gas_eos   = RedlichKwong_Density()             # H2O gas EOS for the ϕ_gas conversion

    # Positive seed so the first density/solubility evaluation sees a sane pressure;
    # init_P! overwrites this below.
    stokes.P .= 1.0e6
    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (; T = thermal.T, P = stokes.P, mH2O = mH2O_melt, ϕ_gas, ϕ_x))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
        pressure_offset = ρg[end] .* depth  # in Pascals
        # Add pressure offset to simulate pressure at reference depth
        stokes.P .+= pressure_offset
    end

    # Melt fraction is the phase-diagram lookup ϕ(T, P); it does not see mH2O.
    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T=thermal.T, P=stokes.P)
    )
    @. ϕ_x = 1 - ϕ                                 # gas is still zero here
    # Costa needs ϕ and Giordano needs the dissolved water, so seed both here rather
    # than leaving the initial viscosity on the (melt-free, dry) defaults.
    # ΔT feeds the thermal-expansion source of the compressible continuity equation,
    # RP = -∇V - (P - P0)/ηb + α·(ΔT/dt) + Q/dt. It only does anything once the phases are
    # elastic (finite Kb); DYREL looks for the key `:ΔT` and expects the *padded* array
    # (it indexes ΔT[I .+ 1]), i.e. thermal.ΔT — not the centre-sized thermal.ΔTc that the
    # classic `solve!` pressure kernel takes. heatdiffusion_PT! refreshes it every step.
    args = (; ϕ = ϕ, T = thermal.T, P = stokes.P, dt = dt, mH2O = mH2O_melt, ΔT = thermal.ΔT)
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, cutoff_visc
    )
    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if plotting
        take(figdir)
        if do_vtk
            vtk_dir = joinpath(figdir, "vtk")
            take(vtk_dir)
        end
        checkpoint = joinpath(figdir, "checkpoint")
    end
    # ----------------------------------------------------
    # Plot initial T and η profiles
    let
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(
            ax1,
            Array(thermal.T[2:(end - 1), 2:(end - 1)][:].-273),
            Y,)
        scatter!(
            ax2,
            log10.(Array(stokes.viscosity.η[:])),
            Y,)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end
    let
        compo = [oxd_wt_sill[1] (oxd_wt_sill[7]+oxd_wt_sill[8]);
                 oxd_wt_host_rock[1] (oxd_wt_host_rock[7]+oxd_wt_host_rock[8])]
        fig=Plot_TAS_diagram(compo; sz=(1000, 1000))
        save(joinpath(figdir, "TAS_diagram.png"), fig)
    end
    # WENO arrays
    T_WENO  = @zeros(ni...)
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)

    # Time loop
    t, it = 0.0, 0

    # Sill-restricted diagnostics. The mask is the advected sill marker, NOT `ϕ > 0.2`:
    # with the phase-diagram melting law the host rock itself reaches ϕ ≈ 0.86 near the
    # contact, so a melt-fraction threshold no longer isolates the sill and the tracked
    # averages would silently include molten host rock.
    in_sill() = phase_sill .> 0.5
    # Viscosity spans many orders of magnitude and a few interface cells sit at the
    # cutoff, so those outliers dominate an arithmetic mean — use the median.
    sill_median(A) = (m = in_sill(); any(m) ? median(Array(A)[Array(m)]) : NaN)
    sill_mean(A)   = (m = in_sill(); any(m) ? mean(A[m]) : NaN)

    time_vec = Float64[0.0]
    melt_fraction_evo = Float64[sill_mean(ϕ)]
    viscosity_evo = Float64[sill_median(stokes.viscosity.η)]
    mH2O_bulk_evo = Float64[sill_mean(mH2O)]
    # mH2O_diss is the Liu2005 solubility *ceiling* per melt mass; mH2O_diss_bulk is the
    # water the melt actually carries, per bulk mass, so it and mH2O_exs share a denominator
    # and sum to the conserved bulk water. Both are still zero at t=0 (not yet computed).
    mH2O_diss_evo = Float64[0.0]
    mH2O_diss_bulk_evo = Float64[0.0]
    mH2O_exs_evo  = Float64[0.0]
    phi_gas_evo   = Float64[0.0]

    # Snapshot times for the temperature-field checkpoints, as fractions of the
    # run-time cap used in the `while` condition below.
    tmax = (650 * 3600 * 24 * 365)
    local snapshot_targets = sort(collect((0.1, 0.5, 0.9)) .* tmax)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5, γfact=50)

    while it < 200e3 && round(maximum(ϕ), digits=2) ≥ 0.3 && t < tmax

        args = (; ϕ = ϕ, T = thermal.T, P = stokes.P, dt = dt, mH2O = mH2O_melt)
        # Density sees the dissolved water and the exsolved-gas / crystal volume
        # fractions from the previous step (ϕ_gas, ϕ_x lagged one iteration).
        compute_ρg!(ρg[end], phase_ratios, rheology, (; T = thermal.T, P = stokes.P, mH2O = mH2O_melt, ϕ_gas, ϕ_x))
        # ------------------------------

        # # Stokes solver ----------------
        solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 150.0e3,
                nout = 100,
                rel_drop = 0.1,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1.0e-2,
                viscosity_cutoff = cutoff_visc,
            )
        )

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        tensor_invariant!(stokes.τ)

        dt   = compute_dt(stokes, di, dt_diff, igg)
        println("dt = $(dt/(3600*24)) days")
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs =(;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 1e3,
                verbose = true,
            )
        )
        # ------------------------------

        T_WENO .= @views thermal.T[2:end-1, 2:end-1]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        velocity2center!(Vx_c, Vy_c, @velocity(stokes)...)

        # Advect temperature, phases and isotopes on the same center grid
        WENO_advection!(T_WENO, (Vx_c, Vy_c), weno, di, dt)
        @views thermal.T[2:(end - 1), 2:(end - 1)] .= T_WENO

        WENO_advection!(phase_sill, (Vx_c, Vy_c), weno, di, dt)
        clamp!(phase_sill, 0.0, 1.0)
        phase_host .= 1.0 .- phase_sill

        WENO_advection!(mH2O, (Vx_c, Vy_c), weno, di, dt)
        clamp!(mH2O, 0.0, 1.0)

        thermal.ΔT .= thermal.T .- thermal.Told

        @show extrema(thermal.T .- 273.15)
        # NaN guard. Report which field went bad and stop: the `while` condition
        # below tests `maximum(ϕ) ≥ 0.3`, and a NaN there makes that false, so without
        # this the run would just end mid-simulation with no indication why.
        nan_field = ""
        for (name, A) in (("T", thermal.T), ("ϕ", ϕ), ("ρg", ρg[end]),
                          ("η", stokes.viscosity.η), ("P", stokes.P), ("mH2O", mH2O))
            any(isnan, A) && (nan_field = name; break)
        end
        if !isempty(nan_field)
            @warn "NaN detected in $nan_field at it = $it, t = $t — stopping"
            break
        end

        update_phase_ratios_2D!(phase_ratios, (phase_host, phase_sill), xci, xvi)

        compute_melt_fraction!(ϕ, phase_ratios, rheology, (T=thermal.T, P=stokes.P))

        # --- Volatile partitioning & three-phase density coupling ---------------
        # Solubility per melt mass at the current P,T:
        compute_dissolved_volatiles!(mH2O_diss, mCO2_diss, phase_ratios, rheology, (; thermal.T, stokes.P, X_co2))

        ϵϕ = 1.0e-6
        # Water actually dissolved in the melt (per melt mass), capped at solubility,
        # and the bulk excess that exsolves as gas (bulk capacity = solubility*ϕ).
        @. mH2O_melt = min(mH2O / max(ϕ, ϵϕ), mH2O_diss)
        # Only melt-bearing cells can exsolve: below the solidus the water is bound in
        # hydrous minerals, not free gas. Without this gate a cell with ϕ=0 reports all
        # of its bulk water as gas and the solid host rock becomes spuriously buoyant.
        @. mH2O_exs  = max(mH2O - mH2O_diss * ϕ, 0.0) * (ϕ > ϵϕ)

        # Convert exsolved gas mass fraction to a volume fraction. ρ_gas from the
        # H2O EOS at P,T; the condensed density is the current mixture density
        # (lagged, gas fraction small). ϕ_gas and ϕ_x feed next step's density.
        @parallel (@idx ni) compute_gas_density!(ρ_gas, gas_eos, stokes.P, T_WENO)
        ρ_cond = ρg[end] ./ 9.81
        @. ϕ_gas = (mH2O_exs / ρ_gas) / (mH2O_exs / ρ_gas + (1 - mH2O_exs) / ρ_cond)
        @. ϕ_x   = (1 - ϕ) * (1 - ϕ_gas)

        @show it += 1
        t        += dt
        push!(time_vec, t)
        push!(melt_fraction_evo, sill_mean(ϕ))
        push!(viscosity_evo, sill_median(stokes.viscosity.η))
        push!(mH2O_bulk_evo, sill_mean(mH2O))
        push!(mH2O_diss_evo, sill_mean(mH2O_diss))
        push!(mH2O_diss_bulk_evo, sill_mean(mH2O_melt .* ϕ))
        push!(mH2O_exs_evo, sill_mean(mH2O_exs))
        push!(phi_gas_evo, sill_mean(ϕ_gas))

        if !isempty(snapshot_targets) &&
                (t ≥ snapshot_targets[1] || round(maximum(ϕ), digits = 2) < 0.3)
            target = popfirst!(snapshot_targets)
            fname  = joinpath(figdir, "snapshot_$(round(Int, target))_$(nx)x$(ny).jld2")
            checkpointing_jld2(
                figdir, stokes, thermal, t, dt, fname;
                T = Array(thermal.T[2:(end - 1), 2:(end - 1)]) .- 273.15, xci = Array.(xci),
                # the fields the sill diagnostics are built from, so a snapshot can be
                # re-analysed without the VTK series
                phi = Array(ϕ), phi_gas = Array(ϕ_gas), phase_sill = Array(phase_sill),
                mH2O = Array(mH2O), mH2O_diss = Array(mH2O_diss), mH2O_diss_bulk =  Array(mH2O_melt .* ϕ), mH2O_exs = Array(mH2O_exs),
                eta = Array(stokes.viscosity.η),
            )
        end

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 50) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, joinpath(@__DIR__, "SillConvection.jl"), joinpath(@__DIR__, "SillRheology.jl"))
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)

            η_eff = @. stokes.τ.II / (2 * stokes.ε.II)
            (; η_vep, η) = stokes.viscosity


            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    stress_xy = Array(stokes.τ.xy),
                    strain_rate_xy = Array(stokes.ε.xy),
                    phase_vertices = [argmax(p) for p in Array(phase_ratios.vertex)],
                )
                data_c = (;
                    T = Array(thermal.T[2:(end - 1), 2:(end - 1)]),
                    P = Array(stokes.P),
                    viscosity_vep = Array(η_vep),
                    viscosity_eff = Array(η_eff),
                    viscosity = Array(η),
                    phases = [argmax(p) for p in Array(phase_ratios.center)],
                    Melt_fraction = Array(ϕ),
                    mH2O_bulk = Array(mH2O),
                    mH2O_dissolved = Array(mH2O_diss),
                    mH2O_exsolved = Array(mH2O_exs),
                    mH2O_diss_bulk=Array(mH2O_melt .* ϕ),
                    phi_gas = Array(ϕ_gas),
                    phi_x = Array(ϕ_x),
                    EII_pl = Array(stokes.EII_pl),
                    stress_II = Array(stokes.τ.II),
                    strain_rate_II = Array(stokes.ε.II),
                    density = Array(ρg[2] ./ 9.81),
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
                    t = t,
                    pvd = joinpath(vtk_dir, "SillConvection")
                )
            end

            # Make Makie figure
            fig = Figure(size = (2000, 1000), title = "t = $t", )
            ar = DataAspect()
            if t < 1e3
                TimeScale = 1
                TimeUnits = "s"
            elseif t >= 1e3 && t < 24*3600
                TimeScale = 3600
                TimeUnits = "hr"
            elseif t >= 24*3600 && t < 365*3600*24
                TimeScale = 3600*24
                TimeUnits = "days"
            elseif t >= 365*3600*24 && t < 1e3*3600*24*365
                TimeScale = 3600*24*365
                TimeUnits = "yr"
            else
                TimeScale = 1e3*3600*24*365
                TimeUnits = "kyr"
            end
            ax0 = Axis(
                fig[1, 1:2];
                aspect=ar,
                title = "T [C]  (time = $(round(t/TimeScale, digits=2)) $TimeUnits)",
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

            ax1 = Axis( fig[2, 1][1, 1], aspect = DataAspect(), title = L"T \;[\mathrm{C}]",  titlesize=40,
            yticklabelsize=25,
            xticklabelsize=25,
            xlabelsize=25,)
            ax2 = Axis(fig[2, 2][1, 1], aspect = DataAspect(), title = L"Density \;[\mathrm{kg/m}^{3}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            ax3 = Axis(fig[3, 1][1, 1], aspect = DataAspect(), title = L"Vy \;[\mathrm{m/s}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            ax4 = Axis(fig[3, 2][1, 1], aspect = DataAspect(), title = L"\phi", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            # Plot temperature
            h1  = heatmap!(
                ax1,
                xci...,
                Array(thermal.T[2:(end - 1), 2:(end - 1)].-273); colormap=:lipari, colorrange=(host_rock_temp .-273.15, sill_temp.-273.15))

            h2  = heatmap!(
                ax2,
                xci...,
                Array(ρg[end]./9.81);
                colormap=:batlowW)

            # Plot vy velocity
            h3  = heatmap!(
                ax3,
                xvi...,
                Array(stokes.V.Vy); colormap=:batlow)

            # Plot melt fraction
            # extendhigh/extendlow are required: the phase diagram returns ϕ exactly
            # 1.0 over a large fully-molten region, and contourf leaves values at or
            # beyond the outermost level unfilled — which reads as a hole in the field
            # rather than as "fully molten".
            h4  = contourf!(ax4,
                xci...,
                Array(ϕ);
                colormap=:lipari,
                levels=0.0:0.1:1.0,
                extendlow = :auto,
                extendhigh = :auto,
            )

            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hideydecorations!(ax2)
            hideydecorations!(ax4)
            Colorbar(fig[2, 1][1, 2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[2, 2][1, 2], h2, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 1][1, 2], h3, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 2][1, 2], h4, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            figsave = joinpath(figdir, @sprintf("%06d.png", it))
            save(figsave, fig)

            # Plot time evolution of mean melt fraction with flow regime backgrounds
            let
                fig = Figure(size = (2000, 1000), title = "t = $t")

                ax1 = Axis(
                    fig[1,1],
                    aspect = DataAspect(),
                    title = "T [C]  (time = $(round(t/TimeScale, digits=2)) $TimeUnits)",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                 # Plot temperature
                h1  = heatmap!(
                    ax1,
                    xci[1],
                    xci[2],
                    (Array(thermal.T[2:(end - 1), 2:(end - 1)].-273));
                    colormap=:lipari, colorrange=(host_rock_temp-273.15, sill_temp-273.15))
                Colorbar(fig[1,2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                save(joinpath(figdir, "Temperature_$(it).png"), fig)
                fig

                fig1 = Figure(size = (1600, 800), title = "Melt Fraction Evolution")

                ax = Axis(
                    fig1[1,1],
                    title = "Mean Melt Fraction Evolution",
                    titlesize=30,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                    ylabelsize=25,
                    xlabel = "Time ($TimeUnits)",
                    ylabel = "mean(ϕ)"
                )

                # Define regime boundaries
                porous_max = 0.08
                mush_max = 0.45
                suspension_max = 1.0

                # Plot colored backgrounds for regimes
                # Define x range for bands (span entire time axis)
                x_min = 0.0
                x_max = maximum(time_vec ./ TimeScale)
                x_band = [x_min, x_max]

                band!(
                    x_band,
                    [0.0, 0.0],
                    [porous_max, porous_max],
                    color = (:blue, 0.2),
                )
                text!(
                    ax,
                    x_min + 0.01 * (x_max - x_min),  # near left edge
                    porous_max / 2,
                    text = "Porous flow",
                    align = (:left, :center),
                    fontsize = 40,
                    color = :blue,
                    font = "bold"
                )
                band!(
                    x_band,
                    [porous_max, porous_max],
                    [mush_max, mush_max],
                    color = (:orange, 0.5),
                )
                text!(
                    ax,
                    x_min + 0.01 * (x_max - x_min),
                    (porous_max + mush_max) / 2,
                    text = "Mushy flow",
                    align = (:left, :center),
                    fontsize = 40,
                    color = :orange,
                    font = "bold"
                )
                band!(
                    x_band,
                    [mush_max, mush_max],
                    [suspension_max, suspension_max],
                    color = (:red, 0.2),
                )
                text!(
                    ax,
                    x_min + 0.01 * (x_max - x_min),
                    (mush_max + suspension_max) / 2,
                    text = "Suspension flow",
                    align = (:left, :center),
                    fontsize = 40,
                    color = :red,
                    font = "bold"
                )

                lines!(
                    ax,
                    time_vec ./ TimeScale,
                    melt_fraction_evo;
                    color = :black,
                    linewidth = 2,
                    label = "mean(ϕ)"
                )

                save(joinpath(figdir, "FlowRegime_diagram_$(it).png"), fig1)
                fig1
            end

            # --- Sill tracker: melt fraction, viscosity and water vs time ------------
            # All three averaged over the advected sill marker (see `in_sill` above),
            # so molten host rock near the contact does not contaminate the curves.
            let
                tv = time_vec ./ TimeScale
                fig3 = Figure(size = (1400, 1200))

                axϕ = Axis(
                    fig3[1, 1], ylabel = L"\phi \;[-]", titlesize = 30,
                    title = "Sill averages  (mask: sill marker > 0.5)",
                    ylabelsize = 24, yticklabelsize = 18, xticklabelsize = 18,
                )
                lines!(axϕ, tv, melt_fraction_evo, color = :black,      linewidth = 3, label = L"\phi_m")
                lines!(axϕ, tv, phi_gas_evo,       color = :firebrick,  linewidth = 3, label = L"\phi_{gas}")
                axislegend(axϕ, position = :rt, labelsize = 18)
                hidexdecorations!(axϕ, grid = false)

                axη = Axis(
                    fig3[2, 1], ylabel = L"\eta \;[\mathrm{Pa\,s}]", yscale = log10,
                    ylabelsize = 24, yticklabelsize = 18, xticklabelsize = 18,
                )
                # log10 axis cannot take non-positive or NaN values
                ok = findall(v -> isfinite(v) && v > 0, viscosity_evo)
                isempty(ok) || lines!(axη, tv[ok], viscosity_evo[ok], color = :black, linewidth = 3)
                hidexdecorations!(axη, grid = false)

                axw = Axis(
                    fig3[3, 1], xlabel = "Time ($TimeUnits)", ylabel = L"H_2O \;[\mathrm{wt\%}]",
                    xlabelsize = 24, ylabelsize = 24, yticklabelsize = 18, xticklabelsize = 18,
                )
                lines!(axw, tv, mH2O_bulk_evo .* 100, color = :black,     linewidth = 3, label = L"H_2O_{bulk}")
                lines!(axw, tv, mH2O_diss_evo .* 100, color = :turquoise3, linewidth = 3, label = L"H_2O^{sat}")
                lines!(axw, tv, mH2O_diss_bulk_evo .* 100, color = :dodgerblue3, linewidth = 3, label = L"H_2O_{diss}")
                lines!(axw, tv, mH2O_exs_evo  .* 100, color = :purple,     linewidth = 3, label = L"H_2O_{exs}")
                axislegend(axw, position = :rt, labelsize = 18)

                linkxaxes!(axϕ, axη, axw)
                save(joinpath(figdir, "SillTracking_$(it).png"), fig3)
                fig3
            end

        end
        # ------------------------------

    end

    checkpointing_jld2(
        figdir, stokes, thermal, t, dt, joinpath(figdir, "final_$(nx)x$(ny).jld2");
        time_vec = time_vec, viscosity_evo = viscosity_evo,
        melt_fraction_evo = melt_fraction_evo,
        mH2O_bulk_evo = mH2O_bulk_evo,
        mH2O_diss_evo = mH2O_diss_evo, mH2O_diss_bulk_evo = mH2O_diss_bulk_evo,
        mH2O_exs_evo = mH2O_exs_evo,
        phi_gas_evo = phi_gas_evo,
    )

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------
const plotting = true
do_vtk = true

# (Path)/folder where output data and figures are stored
figdir   = "Sill_convection"
n = 256
nx, ny = n, n

sill_temp = 900 + 273.15 # in K
host_rock_temp = 550.0 + 273.15 # in C
sill_H2O = 0.03
host_H2O = 0.03
sill_size = 10.0 # in m
depth = 5e3 # in m
li = dimensions = (20.0, 12.5) # in m
origin = (0.0, -li[2])
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
else
    igg
end

# run main script
main(li, origin, igg; nx = nx, ny = ny, figdir = figdir, do_vtk = do_vtk, cutoff_visc = (1e3, 1.0e16), plotting = plotting, sill_temp = sill_temp, host_rock_temp = host_rock_temp, sill_size = sill_size, depth = depth, sill_H2O = sill_H2O, host_H2O = host_H2O);

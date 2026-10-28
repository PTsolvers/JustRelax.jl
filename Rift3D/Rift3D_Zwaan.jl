# 3D version of the reference models of Zwaan et al. (2026, JGR Solid Earth,
# doi:10.1029/2025JB033255): the initial rifting phase of a 450 x 225 x 200 km continental
# lithosphere. The paper follows it with a tectonic pause and an inversion. This script
# gradually reverses the boundary velocities to compression over 0.5 Myr; it does not
# model the pause. The default rifting duration is that of the series A models (Table 1);
# series B rifts for 15 Myr.
#
# The initial structure varies along x only: it is the 2D section of Rift/Rift2D_Zwaan.jl
# extruded along y, with Ly = Lx / 2 and ny = nx / 2. The plates move along x, and the
# front and back boundaries are free slip.
#
# Differences from the paper:
#   - no surface processes (FastScape); the free surface is a marker surface with cut cells,
#   - no viscous strain weakening; plastic strain softens the friction angle only,
#   - Drucker-Prager with a tensile cap, and elasticity, in place of rigid-plastic
#     Drucker-Prager,
#   - the bottom boundary has a prescribed inflow balancing the lateral flux instead of a
#     lithostatic traction,
#   - the grid is uniform; the 3D variational DYREL solver does not support refined grids.

# Select the execution backend before initializing ParallelStencil.
# Example: RIFT_BACKEND=CPU julia --project=. --startup-file=no Rift3D/Rift3D_Zwaan.jl
const rift_backend = "CUDA"
rift_backend in ("CPU", "CUDA") || throw(ArgumentError("RIFT_BACKEND must be CPU or CUDA"))
const isCUDA = rift_backend == "CUDA"

@static if isCUDA
    using CUDA
else
    # KernelAbstractions runs CPU kernels through PoCL, which executes a whole work-group
    # on one thread. JustPIC's particle kernels overflow that thread's stack with PoCL's
    # default work-group size (0xC00000FD on Windows); smaller work-groups fit. PoCL reads
    # this when it initializes, so set it before loading JustPIC.
    ENV["POCL_MAX_WORK_GROUP_SIZE"] = get(ENV, "POCL_MAX_WORK_GROUP_SIZE", "256")
end

using CairoMakie
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO
using JustMelt
using WriteVTK

const backend = @static if isCUDA
    JustRelax.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences3D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3, nonconst_metadata = true)
else
    @init_parallel_stencil(Threads, Float64, 3, nonconst_metadata = true)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

# Load script dependencies
using GeoParams

# Load file with all the rheology configurations
include(joinpath(@__DIR__, "Rift3D_Zwaan_rheology.jl"))
include(joinpath(@__DIR__, "Rift3D_Zwaan_setup.jl"))
include(joinpath(@__DIR__, "rift_air_phase_patch.jl"))

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

# Pure-shear field with lateral boundaries moving at ±`v` along x and the rock column fed
# through the bottom boundary; the bottom inflow balances the lateral outflow of the rock.
# There is no flow along y.
function set_plate_velocity!(stokes, flow_bcs, v, lx, xvi, xci, backend)
    ε = 2 * v / lx
    @views stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .= PTArray(backend)(
        [ε * x for x in Array(xvi[1]), y in Array(xci[2]), z in Array(xci[3])]
    )
    stokes.V.Vy .= 0
    @views stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .= PTArray(backend)(
        [-ε * z * (z < 0) for x in Array(xci[1]), y in Array(xci[2]), z in Array(xvi[3])]
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    return nothing
end

# Zero only the velocity faces with no rock in their control volume. Selecting air by the
# particle phase ratios instead would also zero active free-surface faces of cut cells.
function zero_inactive_velocity!(stokes, ϕ)
    @views stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .*= ϕ.Vx .> 0
    @views stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .*= ϕ.Vy .> 0
    @views stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .*= ϕ.Vz .> 0
    return nothing
end

# A face with rock that borders no cell keeping its pressure row (e.g. between two cut
# slivers whose lids are dry) has no continuity equation on either side and acts as a mass
# source or sink. Faces of a retained cell are never dropped, so one pass is exact and
# the in-place update is race-free.
function prune_orphan_faces!(ϕ)
    @parallel (@idx size(ϕ.Vx)) _prune_orphan_vx!(ϕ.Vx, ϕ.Vy, ϕ.Vz, ϕ.center)
    @parallel (@idx size(ϕ.Vy)) _prune_orphan_vy!(ϕ.Vx, ϕ.Vy, ϕ.Vz, ϕ.center)
    @parallel (@idx size(ϕ.Vz)) _prune_orphan_vz!(ϕ.Vx, ϕ.Vy, ϕ.Vz, ϕ.center)
    return nothing
end

@inline _retained_cell(ϕc, ϕx, ϕy, ϕz, i, j, k) =
    ϕc[i, j, k] > 0 &&
    ϕx[i, j, k] > 0 && ϕx[i + 1, j, k] > 0 &&
    ϕy[i, j, k] > 0 && ϕy[i, j + 1, k] > 0 &&
    ϕz[i, j, k] > 0 && ϕz[i, j, k + 1] > 0

@parallel_indices (i, j, k) function _prune_orphan_vx!(ϕx, ϕy, ϕz, ϕc)
    if ϕx[i, j, k] > 0
        left = i > 1 && _retained_cell(ϕc, ϕx, ϕy, ϕz, i - 1, j, k)
        right = i ≤ size(ϕc, 1) && _retained_cell(ϕc, ϕx, ϕy, ϕz, i, j, k)
        left || right || (ϕx[i, j, k] = zero(eltype(ϕx)))
    end
    return nothing
end

@parallel_indices (i, j, k) function _prune_orphan_vy!(ϕx, ϕy, ϕz, ϕc)
    if ϕy[i, j, k] > 0
        front = j > 1 && _retained_cell(ϕc, ϕx, ϕy, ϕz, i, j - 1, k)
        back = j ≤ size(ϕc, 2) && _retained_cell(ϕc, ϕx, ϕy, ϕz, i, j, k)
        front || back || (ϕy[i, j, k] = zero(eltype(ϕy)))
    end
    return nothing
end

@parallel_indices (i, j, k) function _prune_orphan_vz!(ϕx, ϕy, ϕz, ϕc)
    if ϕz[i, j, k] > 0
        bottom = k > 1 && _retained_cell(ϕc, ϕx, ϕy, ϕz, i, j, k - 1)
        top = k ≤ size(ϕc, 3) && _retained_cell(ϕc, ϕx, ϕy, ϕz, i, j, k)
        bottom || top || (ϕz[i, j, k] = zero(eltype(ϕz)))
    end
    return nothing
end

# Height of the marker surface at `(x, y)` inside the horizontal cell `(i, j)`. The cell is
# split into eight triangles meeting at its center, as in `compute_rock_fraction!`, so the
# particles and the cut cells see the same surface. Along each ray from the center the
# surface is linear, and on the cell boundary it is the bilinear interpolant of the corners.
@inline function surface_height(topo, xv, yv, x, y, i, j)
    x0, x1 = xv[i], xv[i + 1]
    y0, y1 = yv[j], yv[j + 1]
    u = (2 * x - x0 - x1) / (x1 - x0)   # in [-1, 1] across the cell
    v = (2 * y - y0 - y1) / (y1 - y0)
    z00, z10 = topo[i, j], topo[i + 1, j]
    z01, z11 = topo[i, j + 1], topo[i + 1, j + 1]
    zc = (z00 + z10 + z01 + z11) / 4
    s = max(abs(u), abs(v))
    iszero(s) && return zc
    # where the ray from the center through (u, v) crosses the cell boundary
    ξ, η = (1 + u / s) / 2, (1 + v / s) / 2
    zb = (1 - ξ) * (1 - η) * z00 + ξ * (1 - η) * z10 + (1 - ξ) * η * z01 + ξ * η * z11
    return zc + s * (zb - zc)
end

"""
    update_phases_given_markersurface!(phase, surf, particles, air_phase)

Deactivate the particles that end up on the wrong side of the free surface tracked by
`surf`: air-phase particles below it and rock particles above it. Particle injection then
re-seeds those cells from their neighbors. 3D counterpart of
`update_phases_given_markerchain!`.
"""
function update_phases_given_markersurface!(phase, surf, particles, air_phase)
    (; coords, index) = particles
    ni = size(index) .- 2
    @parallel (@idx ni) _update_phases_given_markersurface!(
        phase, coords, index, surf.topo, surf.xv, surf.yv, air_phase
    )
    return nothing
end

@parallel_indices (i, j, k) function _update_phases_given_markersurface!(
        phase, coords, index, topo, xv, yv, air_phase
    )
    # particle cell arrays carry one ghost cell per side: grid cell I is particle cell I .+ 1
    Ip = (i + 1, j + 1, k + 1)
    for ip in cellaxes(index)
        (@index index[ip, Ip...]) || continue

        x = @index coords[1][ip, Ip...]
        y = @index coords[2][ip, Ip...]
        z = @index coords[3][ip, Ip...]
        phaseq = @index phase[ip, Ip...]

        above = z > surface_height(topo, xv, yv, x, y, i, j)
        wrong_side = (above && phaseq != air_phase) || (!above && phaseq == air_phase)
        wrong_side || continue

        @index phase[ip, Ip...] = NaN
        @index coords[1][ip, Ip...] = NaN
        @index coords[2][ip, Ip...] = NaN
        @index coords[3][ip, Ip...] = NaN
        @index index[ip, Ip...] = false
    end
    return nothing
end

# fill the ghost cells of a cell-centered array with their nearest interior values
function fill_ghost_cells!(A)
    for d in 1:ndims(A)
        n = size(A, d)
        selectdim(A, d, 1) .= selectdim(A, d, 2)
        selectdim(A, d, n) .= selectdim(A, d, n - 1)
    end
    return nothing
end

# Write the marker surface as a structured VTK surface; `conversion` is the
# nondimensional length per output unit.
function save_marker_surface(fname, surf, conversion; t = 0.0, pvd = nothing)
    xv = Array(surf.xv) ./ conversion
    yv = Array(surf.yv) ./ conversion
    topo = Array(surf.topo) ./ conversion
    X = [x for x in xv, y in yv, _ in 1:1]
    Y = [y for x in xv, y in yv, _ in 1:1]
    Z = reshape(topo, size(topo)..., 1)
    vtk_grid(fname, X, Y, Z) do vtk
        vtk["topography"] = Z
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end
    return nothing
end

function compute_meltfraction!(F, thermal, stokes, Xbulk, CharDim)
    Tscale = ustrip(dimensionalize(1.0, K, CharDim))
    Pscale = ustrip(dimensionalize(1.0, GPa, CharDim))

    @parallel_indices (I...) function compute_F!(F, T, P, Xbulk, Tscale, Pscale)
        Pᵢ = P[I...] * Pscale
        Tᵢ = T[I .+ 1...] * Tscale - 273.15

        F[I...] = melt_fraction(
            Pᵢ, Tᵢ, 0.17, Xbulk, WetMeltingLaw();
            check = Val(false),
        )
        return nothing
    end

    ni = size(F)
    @parallel (@idx ni) compute_F!(
        F, thermal.T, stokes.P, Xbulk, Tscale, Pscale,
    )
    return nothing
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(
        li_km, origin_km, ni, igg; v_plate = 5.0, t_rift = 5.0,
        t_reverse_Myr = Inf, figdir = "figs3D", do_vtk = false, nout = 10,
    )

    t_reverse_Myr ≥ 0 || throw(ArgumentError("t_reverse_Myr must be nonnegative"))
    thickness = li_km[1] * km
    η0 = 1.0e22 * Pa * s
    CharDim = GEO_units(;
        length = thickness, viscosity = η0, temperature = (ZWAAN_GEOMETRY.T_bottom + 273.0) * K
    )
    # Physical domain ------------------------------------
    li_nd = map(l -> nondimensionalize(l * km, CharDim), li_km)
    origin_nd = map(o -> nondimensionalize(o * km, CharDim), origin_km)
    grid = Geometry(ni, li_nd; origin = origin_nd)
    (; xci, xvi) = grid   # nodes at the center and vertices of the cells
    grid_vxi = grid.xi_vel
    dxi = grid.di.vertex                 # cell widths along each axis
    di_min = minimum.(grid.di.vertex)    # smallest cell width along each axis
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim)
    dt = nondimensionalize(10.0e3 * yr, CharDim) # initial timestep
    dtmax = nondimensionalize(15.0e3 * yr, CharDim) # maximum timestep
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 27, 40, 15
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    # material phase, temperature and accumulated plastic strain; `pΔEII` holds the
    # plastic strain increment of one time step and is not advected
    pPhases, pT, pEII, pΔEII = init_cell_arrays(particles, Val(4))

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, pEII, unwrap(pτ)...)
    particle_args_reduced = (pT, pEII, unwrap(pτ)...)

    # Assign particles phases and the initial plastic strain noise
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, CharDim)
    init_plastic_strain!(pEII, particles, CharDim)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # Free surface: marker surface and the cut-cell rock fractions built from it
    air_phase = 5
    initial_elevation = nondimensionalize(0.0e0 * m, CharDim)
    surf = init_marker_surface(backend_JP, xvi[1], xvi[2], initial_elevation)
    update_phases_given_markersurface!(pPhases, surf, particles, air_phase)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    patch_unsupported_air_ratios!(phase_ratios, air_phase)
    pEII.data[pPhases.data .== air_phase] .= 0.0
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, surf, xvi, dxi)
    prune_orphan_faces!(ϕ)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    particle2centroid!(stokes.EII_pl, pEII, particles; ghost_1 = false, ghost_2 = false, ghost_3 = false)
    EII_old = similar(stokes.EII_pl)          # plastic strain before the Stokes solve
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop = nondimensionalize((ZWAAN_GEOMETRY.T_surface + 273.0) * K, CharDim)
    Tbot = nondimensionalize((ZWAAN_GEOMETRY.T_bottom + 273.0) * K, CharDim)
    thermal = ThermalArrays(backend, ni)
    init_temperature!(thermal.T, xci, CharDim)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = true, right = true, front = true, back = true, top = false, bot = false),
        constant_value = (left = false, right = false, front = false, back = false, top = Ttop, bot = Tbot),
    )
    thermal_bcs!(thermal, thermal_bc)
    thermal.Told .= thermal.T
    # ----------------------------------------------------

    # Buoyancy forces and lithostatic pressure of the rock column. The density depends
    # on pressure, so iterate; the cut-cell weighting is undone after integration.
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    for _ in 1:10
        compute_ρg!(ρg[3], phase_ratios, rheology, (T = thermal.T, P = stokes.P); air_phase = air_phase)
        compute_lithostatic_pressure!(stokes.P, ρg[3] .* ϕ.center, dxi[3], igg)
        @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0)
    end
    @copy stokes.P0 stokes.P

    # Rheology
    args0 = (T = thermal.T, P = stokes.P, dt = Inf)
    # not stated in Zwaan et al.; the usual limits of their ASPECT setups
    viscosity_cutoff = nondimensionalize((1.0e18, 1.0e24) .* (Pa * s), CharDim)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di_min, li_nd; ϵ = 1.0e-5, CFL = 0.95 / √3.1
    )

    # Boundary conditions: lateral boundaries moving at ±v_plate, inflow at the base
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
        no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
        free_surface = false,
    )
    v = nondimensionalize(v_plate * mm / yr, CharDim)
    tmax = nondimensionalize(t_rift * 1.0e6 * yr, CharDim)
    t_reverse = isfinite(t_reverse_Myr) ?
        nondimensionalize(t_reverse_Myr * 1.0e6 * yr, CharDim) : Inf
    t_reverse_duration = nondimensionalize(500.0e3 * yr, CharDim)
    set_plate_velocity!(stokes, flow_bcs, v, li_nd[1], xvi, xci, backend)
    zero_inactive_velocity!(stokes, ϕ)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    vtk_dir = joinpath(figdir, "vtk")
    fig_dir = joinpath(figdir, "fig")
    do_vtk && take(vtk_dir)
    take(fig_dir)
    # ----------------------------------------------------

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    Vz_v = @zeros(ni .+ 1...)

    centroid2particle!(pT, thermal.T, particles)
    dt₀ = similar(thermal.T)

    P_lith = @zeros(ni...)

    # dimensional host copy of a nondimensional field
    dim(A, unit) = ustrip.(dimensionalize(Array(A), unit, CharDim))
    # dimensional coordinates for output
    xvi_dim = ntuple(i -> dim(xvi[i], km), Val(3))
    xci_dim = ntuple(i -> dim(xci[i], km), Val(3))
    # nondimensional length per km, to write the marker surface in km
    surf_conversion = nondimensionalize(1.0km, CharDim)
    # the figures show the x-z section through the middle of the domain
    j_slice = cld(ni[2], 2)
    slice(A) = Array(A)[:, j_slice, :]
    slice_T(T) = Array(T)[2:(end - 1), j_slice + 1, 2:(end - 1)]

    t, it = 0.0, 0
    zero_inactive_velocity!(stokes, ϕ)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; γfact = 20, ϵ = 1.0e-3, global_penalty = true)

    F = zero(stokes.P)
    Xbulk = 0.1

    # Time loop
    reversal_complete = false
    while t < tmax

        if !reversal_complete && t ≥ t_reverse
            ramp_elapsed = min(t - t_reverse, t_reverse_duration)
            v_current = v * (1 - 2 * ramp_elapsed / t_reverse_duration)
            set_plate_velocity!(stokes, flow_bcs, v_current, li_nd[1], xvi, xci, backend)
            zero_inactive_velocity!(stokes, ϕ)
            reversal_complete = ramp_elapsed ≥ t_reverse_duration
        end

        if it == 1
            P_lith .= stokes.P
        end

        # plastic strain before the solve, to extract this step's increment afterwards
        EII_old .= stokes.EII_pl

        args = (; T = thermal.T, P = stokes.P, dt = Inf)

        # Stokes solver -----------------
        t_stokes = @elapsed begin
            result = solve_VariationalDYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                ϕ,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    air_phase = air_phase,
                    verbose_PH = true,
                    verbose_DR = false,
                    iterMax = 150.0e3,
                    total_iterMax = 150.0e3,
                    nout = 20,
                    rel_drop = 1.0e-1,
                    λ_relaxation_PH = 1,
                    λ_relaxation_DR = 1,
                    viscosity_relaxation = 1.0e-2,
                    viscosity_cutoff = viscosity_cutoff,
                    free_surface = true,
                )
            )
        end
        zero_inactive_velocity!(stokes, ϕ)

        # add this step's plastic strain increment to the particles
        EII_old .= stokes.EII_pl .- EII_old
        centroid2particle!(pΔEII, EII_old, particles; ghosted = false)
        @. pEII.data += ifelse(isnan(pΔEII.data), 0.0, pΔEII.data)

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        dt = compute_dt(stokes, di_min, dtmax) * 0.98

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("           Δt:      $(ustrip(dimensionalize(dt, yr, CharDim)) / 1.0e3) kyrs")
        # ------------------------------

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
            grid;
            kwargs = (
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 2.0e2,
                verbose = true,
            )
        )
        @show extrema(thermal.T)

        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        # Populate the ghost cells before interpolating to particles.
        fill_ghost_cells!(dt₀)
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # advect the free surface and relabel the particles against it before injection
        advect_marker_surface!(surf, @velocity(stokes), grid_vxi, dt)
        update_phases_given_markersurface!(pPhases, surf, particles, air_phase)
        # check if we need to inject particles
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (thermal.T, stokes.EII_pl, stress_fields(stokes, pτ)...),
        )
        update_phases_given_markersurface!(pPhases, surf, particles, air_phase)

        # update phase ratios and rock fractions
        update_phase_ratios!(phase_ratios, particles, pPhases)
        patch_unsupported_air_ratios!(phase_ratios, air_phase)
        zero_inactive_velocity!(stokes, ϕ)
        compute_rock_fraction!(ϕ, surf, xvi, dxi)
        prune_orphan_faces!(ϕ)

        # interpolate fields from particles to centroids
        pT.data[pPhases.data .== air_phase] .= Ttop
        pEII.data[pPhases.data .== air_phase] .= 0.0
        particle2centroid!(thermal.T, pT, particles)
        thermal_bcs!(thermal, thermal_bc)
        # the friction softening reads the plastic strain carried by the particles
        particle2centroid!(stokes.EII_pl, pEII, particles; ghost_1 = false, ghost_2 = false, ghost_3 = false)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)

        compute_meltfraction!(F, thermal, stokes, Xbulk, CharDim)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, nout) == 0
            (; η_vep, η) = stokes.viscosity
            t_yr = ustrip(dimensionalize(t, yr, CharDim))
            if do_vtk
                # All rock phases share `pl`; shear flow contributes λ sin(Ψ).
                pl = only(filter(p -> p isa GeoParams.AbstractPlasticity, rheology[1].CompositeRheology[1].elements))
                sinΨ = pl.sinΨ.val
                λ_pl = Array(stokes.λ)
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    Vx = dim(Vx_v, m / s),
                    Vy = dim(Vy_v, m / s),
                    Vz = dim(Vz_v, m / s),
                )
                data_c = (;
                    dFdt = Array(F),
                    T = dim(thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)], C),
                    P = dim(stokes.P, Pa),
                    η = dim(η, Pa * s),
                    η_vep = dim(η_vep, Pa * s),
                    τII = dim(stokes.τ.II, Pa),
                    τxx = dim(stokes.τ.xx, Pa),
                    τzz = dim(stokes.τ.zz, Pa),
                    εII = dim(stokes.ε.II, s^-1),
                    εII_pl = dim(stokes.ε_pl.II, s^-1),
                    EII_pl = Array(stokes.EII_pl),
                    # Instantaneous failure: 0 = none, 1 = shear, 2 = tensile cap.
                    # Cap flow exceeds shear dilation, including when Ψ > 0.
                    failure_mode = ifelse.(
                        λ_pl .> 0,
                        ifelse.(Array(stokes.ε_vol_pl) .> λ_pl .* (sinΨ + 100 * eps(sinΨ)), 2, 1),
                        0,
                    ),
                    ρ = dim(ρg[3], m / s^2 * kg / m^3) ./ 9.81,
                    rock = Array(ϕ.center),
                )
                velocity_v = (data_v.Vx, data_v.Vy, data_v.Vz)
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi_dim,
                    xci_dim,
                    data_v,
                    data_c,
                    velocity_v;
                    t = t_yr,
                    pvd = joinpath(vtk_dir, "Rift3D_Zwaan"),
                )
                save_particles(
                    particles,
                    igg;
                    fname = joinpath(vtk_dir, "particles_" * lpad("$it", 6, "0")),
                    pPhases = pPhases,
                    pvd = joinpath(vtk_dir, "pseries"),
                    t = t,
                )
                save_marker_surface(
                    joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                    surf,
                    surf_conversion;
                    t = t_yr,
                    pvd = joinpath(vtk_dir, "Rift3D_Zwaan_topo"),
                )
            end

            # blank cells that are mostly air
            rock = slice(ϕ.center)
            mask_air(A) = ifelse.(rock .≥ 0.5, A, NaN)
            # the surface along the section, between the vertex rows around it
            topo_km = dim(surf.topo, km)
            surf_x = xvi_dim[1]
            surf_z = (topo_km[:, j_slice] .+ topo_km[:, j_slice + 1]) ./ 2

            ar = DataAspect()
            fig2 = Figure(size = (1200, 1200))
            ax1 = Axis(fig2[1, 1], aspect = ar, title = "τxx [MPa]")
            ax2 = Axis(fig2[2, 1], aspect = ar, title = "T [C]")
            ax3 = Axis(fig2[1, 3], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig2[2, 3], aspect = ar, title = "log10(η)")
            ax5 = Axis(fig2[3, 1], aspect = ar, title = "EII_pl")
            ax6 = Axis(fig2[3, 3], aspect = ar, title = "ΔP from P_lith [MPa]")
            ax7 = Axis(fig2[4, 1:3], aspect = ar, title = "topography [km]", xlabel = "x [km]", ylabel = "y [km]")
            Label(fig2[0, :], "t = $(round(t_yr / 1.0e6; digits = 3)) Myr, section y = $(round(xci_dim[2][j_slice]; digits = 1)) km", fontsize = 30)
            xz = (xci_dim[1], xci_dim[3])
            h1 = heatmap!(ax1, xz..., mask_air(slice(dim(stokes.τ.xx, MPa))), colormap = :batlow)
            h2 = heatmap!(ax2, xz..., dim(slice_T(thermal.T), C), colormap = :romaO, colorrange = (0, ustrip(dimensionalize(Tbot, C, CharDim))))
            h3 = heatmap!(ax3, xz..., mask_air(slice(dim(stokes.τ.II, MPa))), colormap = :batlow)
            h4 = heatmap!(ax4, xz..., mask_air(log10.(slice(dim(η_vep, Pa * s)))), colormap = :lipari, colorrange = (19, 24))
            h5 = heatmap!(ax5, xz..., mask_air(slice(stokes.EII_pl)), colormap = :batlow)
            h6 = heatmap!(ax6, xz..., mask_air(slice(dim(stokes.P .- P_lith, MPa))), colormap = :roma)
            h7 = heatmap!(ax7, xvi_dim[1], xvi_dim[2], topo_km, colormap = :oleron)
            for ax in (ax1, ax2, ax3, ax4, ax5, ax6)
                lines!(ax, surf_x, surf_z, color = :red, linewidth = 1.5)
            end

            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            hidexdecorations!(ax4)
            hideydecorations!(ax3)
            hideydecorations!(ax4)
            hideydecorations!(ax6)

            Colorbar(fig2[1, 2], h1)
            Colorbar(fig2[2, 2], h2)
            Colorbar(fig2[1, 4], h3)
            Colorbar(fig2[2, 4], h4)
            Colorbar(fig2[3, 2], h5)
            Colorbar(fig2[3, 4], h6)
            Colorbar(fig2[4, 4], h7)
            linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6)
            save(joinpath(fig_dir, "sol_" * lpad("$it", 6, "0") * ".png"), fig2)
        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView

# Low resolution for debugging. Ly = Lx / 2 and ny = nx / 2; nz gives cells about as
# tall as they are wide.
nx = 2 * 48
ny = 2 * nx ÷ 2
nz = 2 * 24

h_air = 15.0 # sticky air above the initial surface [km]

# domain extent and origin [km]: x centered on the rift, y centered on the section,
# z from the bottom boundary to the top of the sticky air
lx = ZWAAN_GEOMETRY.lx
li_km = (lx, lx / 2, ZWAAN_GEOMETRY.depth + h_air)
origin_km = (-lx / 2, -lx / 4, -ZWAAN_GEOMETRY.depth)

# output directory; set RIFT_FIGDIR to write somewhere else
figdir = get(ENV, "RIFT_FIGDIR", "output_Rift3D/Rift3D_Zwaan_$(nx)x$(ny)x$(nz)_reversal")

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

# series A: 5 Myr of rifting at 5 mm/yr on each side
main(li_km, origin_km, (nx, ny, nz), igg; v_plate = 5.0, t_rift = 15.0, t_reverse_Myr = 1.0, figdir = figdir, do_vtk = do_vtk, nout = 5)

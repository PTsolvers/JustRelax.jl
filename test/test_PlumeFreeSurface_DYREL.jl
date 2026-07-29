push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

# Regression test for the *variational* DYREL Stokes solver (`solve_DYREL!` with a `RockRatio` ϕ)
# on the sticky-air rising-plume configuration of `PlumeFreeSurface_VariationalStokes_DYREL.jl`.
# Phases are initialised deterministically on the grid (no random particles / marker chain) so the
# instantaneous Stokes solution is bit-reproducible, and a single solve is enough to exercise every
# masked DYREL kernel (∇V+strain+RP, stress, residual+damped update).

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

# deterministic, grid-based phase init: sticky-air layer on top, plume anomaly, mantle elsewhere
function init_phases!(phase_ratios, xci, xvi)
    ni = size(phase_ratios.center)
    circle = GGU.Circle((250.0e3, 250.0e3), 100.0e3)   # (x, depth)

    @parallel_indices (i, j) function _init!(phases, xc, yc, circle)
        x = xc[i]
        depth = -yc[j]
        ph = 2  # mantle
        if 0.0e0 ≤ depth ≤ 100.0e3
            ph = 1  # air
        elseif GGU.inside(GGU.Point(x, depth), circle)
            ph = 3  # plume
        end
        for k in 1:3
            @index phases[k, i, j] = (k == ph) ? 1.0 : 0.0
        end
        return nothing
    end

    @parallel (@idx ni) _init!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) _init!(phase_ratios.vertex, xvi..., circle)
    # velocity-face ratios too: `update_rock_ratio!` reads `phase_ratios.Vx`/`.Vy` for the
    # masks, and leaving them unset makes every face "rock" (ϕ.V = 1 even in air), which puts
    # valid velocity DOFs next to dead cells and makes the Gershgorin diagonal collapse to 0.
    @parallel (@idx size(phase_ratios.Vx)) _init!(phase_ratios.Vx, xvi[1], xci[2], circle)
    @parallel (@idx size(phase_ratios.Vy)) _init!(phase_ratios.Vy, xci[1], xvi[2], circle)
    return nothing
end

function PlumeFreeSurface_DYREL(igg, nx, ny; free_surface = false, nsolves = 1)
    # Physical domain ------------------------------------
    thick_air = 100.0e3
    ly = 400.0e3 + thick_air
    lx = 500.0e3
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    origin = 0.0, -ly
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid

    # Physical properties using GeoParams ----------------
    rheology = (
        SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = 1.0e1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e17),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        SetMaterialParams(;
            Phase = 2, Density = ConstantDensity(; ρ = 3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        SetMaterialParams(;
            Phase = 3, Density = ConstantDensity(; ρ = 3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )

    viscosity_cutoff = (1.0e16, 1.0e24)

    # Phase ratios + rock ratio (deterministic) ----------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi)
    air_phase = 1
    ϕ = RockRatio(backend_JR, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend_JR, ni)
    thermal = ThermalArrays(backend_JR, ni)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase = air_phase)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )

    dt = 1.0e3 * (3600 * 24 * 365.25)
    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    # nsolves > 1 re-solves the static configuration: the first solve seeds Vy, so the free-surface
    # term (Vy·∂ρg∂y·dt) is actually exercised on the subsequent solve(s).
    local iters
    for _ in 1:nsolves
        iters = solve_DYREL!(
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
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 100.0e3,
                nout = 50,
                rel_drop = 1.0e-2,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1.0e-2,
                linear_viscosity = true,
                free_surface = free_surface,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
    end

    maxVx = maximum(abs, Array(stokes.V.Vx))
    maxVy = maximum(abs, Array(stokes.V.Vy))
    allfinite = all(isfinite, Array(stokes.P)) &&
        all(isfinite, Array(stokes.V.Vx)) &&
        all(isfinite, Array(stokes.V.Vy))
    return iters, maxVx, maxVy, allfinite
end

@testset "PlumeFreeSurface DYREL (variational)" begin
    @suppress begin
        nx = ny = 40
        init_mpi = JustRelax.MPI.Initialized() ? false : true
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

        # --- free_surface = false (instantaneous Stokes solve from rest) ---
        iters, maxVx, maxVy, allfinite = PlumeFreeSurface_DYREL(igg, nx, ny)
        # all fields stay finite (catches NaN/Inf blow-ups in the masked kernels)
        @test allfinite
        # the instantaneous Stokes solve converges below the DYREL tolerance
        @test iters.err_evo_tot[end] < 1.0e-6
        # golden instantaneous rise velocities (bit-reproducible on CPU; loose rtol for portability)
        @test maxVy ≈ 6.199561037069179e-9 rtol = 1.0e-2
        @test maxVx ≈ 2.4543171180141625e-9 rtol = 1.0e-2
        # plume rises: vertical velocity dominates and is non-trivial
        @test maxVy > maxVx

        # --- free_surface = true (stabilization term active on the 2nd solve, Vy ≠ 0) ---
        iters_fs, maxVx_fs, maxVy_fs, allfinite_fs = PlumeFreeSurface_DYREL(igg, nx, ny; free_surface = true, nsolves = 2)
        @test allfinite_fs
        @test iters_fs.err_evo_tot[end] < 1.0e-6
        # golden with the free-surface term wired in. The stabilization changes maxVy by only
        # ~0.17% on this static one-step config (small Vy ⇒ small Vy·∂ρg∂y·dt correction), so the
        # guard that the term stays active must be tighter than the portability rtol above.
        @test maxVy_fs ≈ 6.210100979998384e-9 rtol = 1.0e-2
        @test !isapprox(maxVy_fs, 6.199561037069179e-9; rtol = 1.0e-3)

        finalize_global_grid(; finalize_MPI = true)
    end
end

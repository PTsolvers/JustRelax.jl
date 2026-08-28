# Verification harness for the variational Stokes free-surface treatment.
#
# A: rigid-body invariance.  A free-floating blob in vacuum under zero gravity
#    must retain a rigid translation / rotation exactly.
# B: hydrostatic exactness.  A fluid layer under a flat free surface placed at a
#    sub-grid height must reproduce P = rho*g*(y_s - y) exactly, for any height.

push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using GeoParams, Printf, Statistics, LinearAlgebra
using ImplicitGlobalGrid
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

using JustPIC

const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

const RHO, GRAV = 1.0, 1.0

@parallel_indices (i, j) function init_phases!(phases, px, py, index, phasefun)
    for ip in cellaxes(phases)
        @index(index[ip, i, j]) == 0 && continue
        @index phases[ip, i, j] = phasefun(@index(px[ip, i, j]), @index(py[ip, i, j]))
    end
    return nothing
end

# The staggered volume fractions are analytic, so they are assembled on the host
# and uploaded; on GPU backends they live in device memory.
function setmask!(dst, f, xs, ys)
    copyto!(dst, [f(x, y) for x in xs, y in ys])
    return dst
end

function materials(η_air, ρ_air, g)
    return (
        SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = ρ_air),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_air),)),
            Gravity = ConstantGravity(; g = g)
        ),
        SetMaterialParams(;
            Phase = 2, Density = ConstantDensity(; ρ = RHO),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            Gravity = ConstantGravity(; g = g)
        ),
    )
end

function build(n, origin, rheology, phasefun)
    ni = n, n
    li = 1.0, 1.0
    di = li ./ ni
    grid = Geometry(ni, li; origin = origin)
    particles = init_particles(backend_JP, 40, 60, 20, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    @parallel (@idx size(pPhases)) init_phases!(
        pPhases, particles.coords..., particles.index, phasefun
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(
        li, di; ϵ_abs = 1.0e-11, ϵ_rel = 1.0e-11,
        Re = 3π, r = 0.7, CFL = 0.9 / √2.1
    )
    thermal = ThermalArrays(backend_JR, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    ϕ = RockRatio(backend_JR, ni)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true), free_surface = false
    )
    return (; ni, li, di, grid, phase_ratios, stokes, pt_stokes, ρg, args, ϕ, flow_bcs, rheology)
end

function solve!(m, igg; iterMax = 100.0e3)
    @suppress begin
        compute_ρg!(m.ρg, m.phase_ratios, m.rheology, m.args)
        compute_viscosity!(m.stokes, m.phase_ratios, m.args, m.rheology, (1.0e-8, 1.0e8))
        solve_VariationalStokes!(
            m.stokes, m.pt_stokes, m.grid, m.flow_bcs, m.ρg, m.phase_ratios, m.ϕ,
            m.rheology, m.args, 1.0, igg;
            air_phase = 1,
            iterMax,
            iterMin = 10,
            nout = 2000,
            viscosity_cutoff = (1.0e-8, 1.0e8),
            free_surface = false,
            verbose = false,
        )
    end
    return nothing
end

# ---------------------------------------------------------------- A: rigid body
const RAD, XC0, YC0 = 0.25, 0.0137, -0.0091
incircle(x, y) = (x - XC0)^2 + (y - YC0)^2 ≤ RAD^2
function circfrac(x, y, hx, hy; k = 64)
    s = 0
    for a in 1:k, b in 1:k
        s += incircle(x - hx / 2 + hx * (a - 0.5) / k, y - hy / 2 + hy * (b - 0.5) / k)
    end
    return s / (k * k)
end

function rigid(igg, n, U, ω)
    m = build(n, (-0.5, -0.5), materials(1.0e-3, 0.0, 0.0), (x, y) -> incircle(x, y) ? 2.0 : 1.0)
    (; xci, xvi) = m.grid; di = m.di; ϕ = m.ϕ
    xc, yc = xci; xv, yv = xvi
    frac(x, y) = circfrac(x, y, di...)
    setmask!(ϕ.center, frac, xc, yc)
    setmask!(ϕ.vertex, frac, xv, yv)
    setmask!(ϕ.Vx, frac, xv, yc)
    setmask!(ϕ.Vy, frac, xc, yv)

    ϕVx, ϕVy = Array(ϕ.Vx), Array(ϕ.Vy)
    Vx, Vy = m.stokes.V.Vx, m.stokes.V.Vy
    vx0, vy0 = zeros(size(Vx)), zeros(size(Vy))
    mx, my = falses(size(Vx)), falses(size(Vy))
    for j in axes(ϕVx, 2), i in axes(ϕVx, 1)
        ϕVx[i, j] > 0 && (vx0[i, j + 1] = U - ω * (yc[j] - YC0); mx[i, j + 1] = true)
    end
    for j in axes(ϕVy, 2), i in axes(ϕVy, 1)
        ϕVy[i, j] > 0 && (vy0[i + 1, j] = ω * (xc[i] - XC0); my[i + 1, j] = true)
    end
    copyto!(Vx, vx0); copyto!(Vy, vy0)
    solve!(m, igg; iterMax = 5.0e3)
    vx, vy = Array(Vx), Array(Vy)
    den = dot(vx0[mx], vx0[mx]) + dot(vy0[my], vy0[my])
    retained = (dot(vx[mx], vx0[mx]) + dot(vy[my], vy0[my])) / den
    rel = sqrt(sum(abs2, vx[mx] .- vx0[mx]) + sum(abs2, vy[my] .- vy0[my])) / sqrt(den)
    return retained, rel
end

# -------------------------------------------------------------- B: hydrostatic
below(y, hy, y_s) = clamp((y_s - (y - hy / 2)) / hy, 0.0, 1.0)

function hydro(igg, n, y_s)
    m = build(n, (0.0, -1.0), materials(1.0e-3, 0.0, GRAV), (x, y) -> y < y_s ? 2.0 : 1.0)
    (; xci, xvi) = m.grid; dy = m.di[2]; ϕ = m.ϕ
    xc, yc = xci; xv, yv = xvi
    layer(_, y) = below(y, dy, y_s)
    setmask!(ϕ.center, layer, xc, yc)
    setmask!(ϕ.vertex, layer, xv, yv)
    setmask!(ϕ.Vx, layer, xv, yc)
    setmask!(ϕ.Vy, layer, xc, yv)
    solve!(m, igg)
    jp = argmin(abs.(yc .- (-0.5)))
    P = Array(m.stokes.P)
    err = (mean(P[:, jp]) - RHO * GRAV * (y_s - yc[jp])) / (RHO * GRAV * dy)
    Vmax = max(maximum(abs, m.stokes.V.Vx), maximum(abs, m.stokes.V.Vy))
    return err, Vmax
end


# Geometry takes its spacing from the global grid, so each resolution needs its own.
function at_resolution(f, n)
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    return f(igg)
end

@testset "Variational Stokes free-surface verification" begin

    println("\n===== A. rigid-body invariance (n = 64) =====")
    @printf("  %-14s %14s %14s\n", "motion", "retained", "rel. change")
    for (lbl, U, ω) in (("translation", 1.0, 0.0), ("rotation", 0.0, 1.0))
        r, c = at_resolution(igg -> rigid(igg, 64, U, ω), 64)
        @printf("  %-14s %14.8f %14.2e\n", lbl, r, c)
        @test r ≈ 1.0 atol = 1.0e-12
        @test c < 1.0e-12
    end

    println("\n===== B. hydrostatic exactness: surface misplacement, in cells =====")
    println("      (exact method => 0.0 at every delta and every n)\n")
    @printf("  %8s %12s %12s %12s %14s\n", "delta", "n=32", "n=64", "n=128", "max|V| n=128")
    for δ in (0.0, 0.25, 0.5, 0.75)
        errs = Float64[]; v = 0.0
        for n in (32, 64, 128)
            e, vm = at_resolution(igg -> hydro(igg, n, -0.40625 + δ / n), n)
            push!(errs, e); v = vm
        end
        @printf("  %8.2f %12.5f %12.5f %12.5f %14.2e\n", δ, errs..., v)
        @test all(isfinite, errs)
        @test v < 1.0e-10
        expected = δ in (0.25, 0.5) ? -δ / 2 : 0.0
        @test errs ≈ fill(expected, 3) atol = 1.0e-10
    end
end

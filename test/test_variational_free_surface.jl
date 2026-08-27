# Verification harness for the variational Stokes free-surface treatment.
#
# A: rigid-body invariance.  A free-floating blob in vacuum under zero gravity
#    must retain a rigid translation / rotation exactly.
# B: hydrostatic exactness.  A fluid layer under a flat free surface placed at a
#    sub-grid height must reproduce P = rho*g*(y_s - y) exactly, for any height.

using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)
using JustPIC
const backend_JP = JustPIC.CPU
const backend = JustRelax.CPUBackend
using GeoParams, Printf, Statistics, LinearAlgebra
using ImplicitGlobalGrid
using Test

const RHO, GRAV = 1.0, 1.0

quiet(f) = redirect_stdout(f, devnull)

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
    ni_ = size(pPhases)
    @parallel_indices (i, j) function _init!(phases, px, py, index)
        @inbounds for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue
            @index phases[ip, i, j] = phasefun(@index(px[ip, i, j]), @index(py[ip, i, j]))
        end
        return nothing
    end
    @parallel (@idx ni_) _init!(pPhases, particles.coords..., particles.index)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(
        li, di; ϵ_abs = 1.0e-11, ϵ_rel = 1.0e-11,
        Re = 3π, r = 0.7, CFL = 0.9 / √2.1
    )
    thermal = ThermalArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    ϕ = RockRatio(backend, ni)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true), free_surface = false
    )
    return (; ni, li, di, grid, phase_ratios, stokes, pt_stokes, ρg, args, ϕ, flow_bcs, rheology)
end

function solve!(m, igg; iterMax = 100.0e3)
    quiet() do
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
    for j in axes(ϕ.center, 2), i in axes(ϕ.center, 1)
        ϕ.center[i, j] = circfrac(xc[i], yc[j], di...)
    end
    for j in axes(ϕ.vertex, 2), i in axes(ϕ.vertex, 1)
        ϕ.vertex[i, j] = circfrac(xv[i], yv[j], di...)
    end
    for j in axes(ϕ.Vx, 2), i in axes(ϕ.Vx, 1)
        ϕ.Vx[i, j] = circfrac(xv[i], yc[j], di...)
    end
    for j in axes(ϕ.Vy, 2), i in axes(ϕ.Vy, 1)
        ϕ.Vy[i, j] = circfrac(xc[i], yv[j], di...)
    end

    Vx, Vy = m.stokes.V.Vx, m.stokes.V.Vy
    fill!(Vx, 0.0); fill!(Vy, 0.0)
    mx, my = falses(size(Vx)), falses(size(Vy))
    for j in axes(ϕ.Vx, 2), i in axes(ϕ.Vx, 1)
        ϕ.Vx[i, j] > 0 && (Vx[i, j + 1] = U - ω * (yc[j] - YC0); mx[i, j + 1] = true)
    end
    for j in axes(ϕ.Vy, 2), i in axes(ϕ.Vy, 1)
        ϕ.Vy[i, j] > 0 && (Vy[i + 1, j] = ω * (xc[i] - XC0); my[i + 1, j] = true)
    end
    Vx0, Vy0 = copy(Vx), copy(Vy)
    solve!(m, igg; iterMax = 5.0e3)
    den = dot(Vx0[mx], Vx0[mx]) + dot(Vy0[my], Vy0[my])
    retained = (dot(Vx[mx], Vx0[mx]) + dot(Vy[my], Vy0[my])) / den
    rel = sqrt(sum(abs2, Vx[mx] .- Vx0[mx]) + sum(abs2, Vy[my] .- Vy0[my])) / sqrt(den)
    return retained, rel
end

# -------------------------------------------------------------- B: hydrostatic
below(y, hy, y_s) = clamp((y_s - (y - hy / 2)) / hy, 0.0, 1.0)

function hydro(igg, n, y_s)
    m = build(n, (0.0, -1.0), materials(1.0e-3, 0.0, GRAV), (x, y) -> y < y_s ? 2.0 : 1.0)
    (; xci, xvi) = m.grid; dy = m.di[2]; ϕ = m.ϕ
    yc, yv = xci[2], xvi[2]
    for j in axes(ϕ.center, 2), i in axes(ϕ.center, 1)
        ϕ.center[i, j] = below(yc[j], dy, y_s)
    end
    for j in axes(ϕ.vertex, 2), i in axes(ϕ.vertex, 1)
        ϕ.vertex[i, j] = below(yv[j], dy, y_s)
    end
    for j in axes(ϕ.Vx, 2), i in axes(ϕ.Vx, 1)
        ϕ.Vx[i, j] = below(yc[j], dy, y_s)
    end
    for j in axes(ϕ.Vy, 2), i in axes(ϕ.Vy, 1)
        ϕ.Vy[i, j] = below(yv[j], dy, y_s)
    end
    solve!(m, igg)
    jp = argmin(abs.(yc .- (-0.5)))
    err = (mean(m.stokes.P[:, jp]) - RHO * GRAV * (y_s - yc[jp])) / (RHO * GRAV * dy)
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

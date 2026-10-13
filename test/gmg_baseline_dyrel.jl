using Printf, LinearAlgebra
using GeoParams, JustPIC
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
const JR = JustRelax

@parallel_indices (i, j) function fill_block!(phase, x, y, half)
    inside = abs(x[i] - 0.5) <= half && abs(y[j] - 0.5) <= half
    @index phase[1, i, j] = inside ? 0.0 : 1.0
    @index phase[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

function physical_state(n, contrast)
    ni = (n, n)
    grid = Geometry(ni, (1.0, 1.0))
    rheology = (
        SetMaterialParams(; Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),))),
        SetMaterialParams(; Phase = 2, Density = ConstantDensity(; ρ = 1.0), Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = contrast),))),
    )
    phase = PhaseRatios(JustPIC.CPU, 2, ni)
    @parallel (@idx ni) fill_block!(phase.center, grid.xci..., 0.1)
    @parallel (@idx ni .+ 1) fill_block!(phase.vertex, grid.xvi..., 0.1)
    s = StokesArrays(CPUBackend, ni)
    args = (; T = zeros(ni .+ 2), P = s.P, dt = Inf)
    ρg = zeros(ni), zeros(ni)
    compute_ρg!(ρg[2], phase, rheology, args)
    compute_viscosity!(s, phase, args, rheology, (-Inf, Inf))
    bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
    return (; ni, grid, rheology, phase, s, args, ρg, bcs)
end

# GMG fine level on the JR grid; edge viscosity copied from JR's vertex viscosity
# so both methods solve the same discrete Stokes problem.
function gmg_setup(st, γfact; pre = 2, post = 2)
    N = 2
    xvi = map(Array, st.grid.xvi)
    options = JR.GMGOptions(Val(N); coarsest_cells = 4)
    hierarchy = JR.GMGHierarchy(JR.CPUBackend, xvi, JR.GMGPartition(st.ni, ntuple(d -> 1:st.ni[d], N)); options)
    mg = JR.GMGMultigrid(hierarchy; pre, post)
    fine, op = first(mg.hierarchy.levels), first(mg.operators)
    η = st.s.viscosity.η
    γ = γfact .* η
    JustRelax2D.gmg_set_coefficients!(op, η; γ, pressure_scale = η .+ γ)
    op.coefficients.η_edge[1] .= st.s.viscosity.ηv
    JustRelax2D.gmg_refresh!(mg)
    audit = JR.GMGLevelOperator(fine)
    JustRelax2D.gmg_set_coefficients!(audit, η)
    audit.coefficients.η_edge[1] .= st.s.viscosity.ηv
    rhs = JR.gmg_fields(fine.rhs.p, st.ni)
    nx, ny = st.ni
    ρgy = st.ρg[2]
    for j in 2:ny, i in 2:(nx + 1)
        rhs.v[2][i, j] = -(ρgy[i - 1, j - 1] + ρgy[i - 1, j]) / 2
    end
    rhs.v[2] .*= op.velocity_mask[2]
    return (; mg, fine, op, audit, rhs)
end

# Physical residual of the plain Stokes operator (γ = 0): momentum RMS relative
# to the forcing RMS, continuity RMS relative to max|v| / L.
function physical_audit(g, Vx, Vy, P)
    e = JR.gmg_fields(g.fine.rhs.p, size(P))
    copyto!(e.v[1], Vx); copyto!(e.v[2], Vy); copyto!(e.p, P)
    r = JR.gmg_fields(g.fine.rhs.p, size(P))
    JustRelax2D.gmg_residual!(g.audit, r, g.rhs, e)
    m = g.audit.velocity_mask
    momentum = sqrt(sum(C -> sum(m[C] .* r.v[C] .^ 2), 1:2) / sum(C -> sum(m[C] .* g.rhs.v[C] .^ 2) + eps(), 1:2))
    nv = maximum(C -> maximum(abs, m[C] .* e.v[C]), 1:2)
    continuity = sqrt(sum(abs2, r.p) / length(r.p)) / nv
    return momentum, continuity
end

function run_dyrel(st, ϵ)
    s = st.s
    dyrel = DYREL(CPUBackend, s, st.rheology, st.phase, st.grid.di, Inf; ϵ)
    flow_bcs!(s, st.bcs)
    update_halo!(@velocity(s)...)
    t = @elapsed out = solve_DYREL!(
        s, st.ρg, dyrel, st.bcs, st.phase, st.rheology, st.args, st.grid, Inf, IGG_HANDLE[];
        kwargs = (; verbose_PH = false, verbose_DR = false, iterMax = 2.0e6, nout = 50, rel_drop = 1.0e-3,
            linear_viscosity = true, viscosity_cutoff = (-Inf, Inf)),
    )
    return out.iter, t
end

function run_gmg(g, target_m, target_c; γfact, vrtol, vmax, max_outer = 400)
    e = JR.gmg_fields(g.fine.rhs.p, size(g.fine.rhs.p))
    work, iters, cycles = 0.0, 0, 0
    m, c = Inf, Inf
    t = @elapsed for _ in 1:max_outer
        status, counters, _ = JustRelax2D.gmg_ph_solve!(g.mg, e, g.rhs; rtol = 1.0e-30, max_iterations = 1,
            velocity_cycles = vmax, velocity_rtol = vrtol)
        work += counters.work - 1  # the per-call initial residual is not part of a continued iteration
        iters += counters.iterations
        cycles += counters.cycles
        status == JR.gmg_nonfinite && break
        m, c = physical_audit(g, e.v[1], e.v[2], e.p)
        m <= target_m && c <= target_c && break
    end
    return (; iters, cycles, work, m, c, t, e)
end

const IGG_HANDLE = Ref{Any}()

function main()
    IGG_HANDLE[] = IGG(init_global_grid(8, 8, 1; init_MPI = !JustRelax.MPI.Initialized(), select_device = false)...)
    @printf("%-5s %-8s %-28s %8s %9s %9s %8s %10s\n", "n", "contrast", "method", "iters", "momentum", "contin.", "time[s]", "work[apps]")
    for n in (64, 128), contrast in (1.0e2, 1.0e4)
        finalize_global_grid(; finalize_MPI = false)
        IGG_HANDLE[] = IGG(init_global_grid(n, n, 1; init_MPI = false, select_device = false)...)
        st = physical_state(n, contrast)
        it, t = run_dyrel(st, 1.0e-8)
        @assert minimum(st.s.viscosity.ηv) > 0
        g0 = gmg_setup(st, 1.0)
        m, c = physical_audit(g0, st.s.V.Vx, st.s.V.Vy, st.s.P)
        @printf("%-5d %-8.0e %-28s %8d %9.2e %9.2e %8.2f %10d\n", n, contrast, "production DYREL ϵ=1e-8", it, m, c, t, it)
        for γfact in (1.0, 5.0, 20.0), (vrtol, vmax) in ((1.0e-1, 10), (1.0e-2, 20))
            g = gmg_setup(st, γfact)
            run_gmg(g, m, c; γfact, vrtol, vmax, max_outer = 2)  # compile
            g = gmg_setup(st, γfact)
            r = run_gmg(g, m, c; γfact, vrtol, vmax)
            dv = maximum(abs, r.e.v[2] .* g.op.velocity_mask[2] .- st.s.V.Vy .* g.op.velocity_mask[2]) / maximum(abs, st.s.V.Vy)
            @printf("%-5d %-8.0e %-28s %8d %9.2e %9.2e %8.2f %10.0f  (V-cycles %d, |ΔVy|/|Vy| %.1e)\n", n, contrast,
                "GMG-PH γ=$(γfact)η vtol=$(vrtol)", r.iters, r.m, r.c, r.t, r.work, r.cycles, dv)
        end
    end
    finalize_global_grid(; finalize_MPI = false)
end

main()

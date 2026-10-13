using Test, Random, Statistics

const gmg_backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if gmg_backend_name == "CUDA"
    import CUDA
    CUDA.allowscalar(false)
elseif gmg_backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D

const JR = JustRelax
const gmg_backend_type = gmg_backend_name == "CUDA" ? JR.CUDABackend :
    gmg_backend_name == "AMDGPU" ? JR.AMDGPUBackend : JR.CPUBackend

function gmg_kernel_module(N)
    extension = gmg_backend_name == "CUDA" ? :JustRelaxCUDAExt :
        gmg_backend_name == "AMDGPU" ? :JustRelaxAMDGPUExt : nothing
    parent = isnothing(extension) ? JR : Base.get_extension(JR, extension)
    return N == 2 ? parent.JustRelax2D : parent.JustRelax3D
end

function multigrid(ni; coarsest = 4, kwargs...)
    N = length(ni)
    xvi = ntuple(d -> collect(range(0.0, 1.0; length = ni[d] + 1)), N)
    options = JR.GMGOptions(Val(N); coarsest_cells = coarsest)
    hierarchy = JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, ntuple(d -> 1:ni[d], N)); options)
    return JR.GMGMultigrid(hierarchy; kwargs...)
end

function viscosity(level, contrast)
    xc = map(Array, level.grid.xci)
    N = length(xc)
    η = [exp(log(contrast) * prod(sin(2.3 * d * xc[d][I[d]] + 0.4d) for d in 1:N)^2) for I in CartesianIndices(level.grid.ni)]
    return copyto!(similar(level.correction.p), η)
end

# RHS = K e* for a random e* on owned retained DOFs: compatible with the
# pressure gauge and with consistently scaled velocity/pressure blocks.
function random_rhs!(level, op, rng)
    K = gmg_kernel_module(length(level.rhs.v))
    e = level.correction
    for C in eachindex(e.v)
        copyto!(e.v[C], randn(rng, size(e.v[C])))
        e.v[C] .*= op.velocity_mask[C]
    end
    copyto!(e.p, randn(rng, size(e.p)))
    JR.gmg_prepare!(op, e)
    K.gmg_apply_operator!(op, e)
    foreach((r, o, m) -> r .= m .* o, level.rhs.v, op.out.v, op.velocity_mask)
    level.rhs.p .= op.pressure_mask .* op.out.p
    foreach(x -> fill!(x, 0), (e.p, e.v...))
    return nothing
end

function solve(ni, contrast; rng = Xoshiro(11), γ = 1.0, kwargs...)
    N = length(ni)
    K = gmg_kernel_module(N)
    mg = multigrid(ni; kwargs...)
    fine, op = first(mg.hierarchy.levels), first(mg.operators)
    η = viscosity(fine, contrast)
    K.gmg_set_coefficients!(op, η; γ = γ .* η, pressure_scale = η)
    K.gmg_refresh!(mg)
    random_rhs!(fine, op, rng)
    status, counters, history = K.gmg_solve!(mg, fine.correction, fine.rhs; rtol = 1.0e-6, max_cycles = 60)
    factor = (history[end] / history[1])^(1 / (length(history) - 1))
    return status, counters, history, factor, length(mg.hierarchy.levels)
end

@testset "GMG linear V-cycle" begin
    @testset "2D constant viscosity" begin
        results = [solve((n, n), 1.0; pre = 2, post = 2) for n in (16, 32, 64)]
        for (status, counters, history, factor, nlevels) in results
            @test status == JR.gmg_converged
            @test counters.coarse_status == JR.gmg_converged
            @test factor < 0.1
            @test nlevels >= 3
        end
        # Depth independence (3, 4, 5 levels); intermediate corrections must start from zero.
        factors = [r[4] for r in results]
        @test maximum(factors) - minimum(factors) < 0.02
    end

    @testset "3D constant viscosity" begin
        status, counters, history, factor, nlevels = solve((16, 16, 16), 1.0)
        @test status == JR.gmg_converged
        @test factor < 0.4
        @test nlevels >= 3
    end

    @testset "Viscosity contrast (open gate)" begin
        # The coupled DR/PH sweep amplifies modes under contrast even with an exact coarse solve.
        status, _, _, _, _ = solve((16, 16), 1.0e2; pre = 2, post = 2)
        @test_broken status == JR.gmg_converged
    end

    @testset "Status and inference" begin
        K = gmg_kernel_module(2)
        mg = multigrid((16, 16))
        fine, op = first(mg.hierarchy.levels), first(mg.operators)
        K.gmg_set_coefficients!(op, 1.0; γ = 1.0)
        K.gmg_refresh!(mg)
        foreach(x -> fill!(x, 0), (fine.rhs.p, fine.rhs.v..., fine.correction.p, fine.correction.v...))
        status, counters, history = K.gmg_solve!(mg, fine.correction, fine.rhs)
        @test status == JR.gmg_converged && counters.cycles == 0
        random_rhs!(fine, op, Xoshiro(1))
        status, counters, _ = K.gmg_solve!(mg, fine.correction, fine.rhs; rtol = 1.0e-30, max_cycles = 2)
        @test status == JR.gmg_budget_exhausted && counters.cycles == 2
        @test counters.applications > 0 && counters.sweeps == 2 * 2 * (length(mg.hierarchy.levels) - 1)
        @test @inferred(K.gmg_vcycle!(mg, 1, fine.correction, fine.rhs, JR.GMGCounters())) === nothing
    end
end

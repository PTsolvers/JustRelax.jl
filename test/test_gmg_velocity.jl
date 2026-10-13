using Test, Random

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

function multigrid(ni; kwargs...)
    N = length(ni)
    xvi = ntuple(d -> collect(range(0.0, 1.0; length = ni[d] + 1)), N)
    options = JR.GMGOptions(Val(N); coarsest_cells = 4)
    hierarchy = JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, ntuple(d -> 1:ni[d], N)); options)
    return JR.GMGMultigrid(hierarchy; pre = 2, post = 2, kwargs...)
end

function inclusion(level, contrast)
    xc = map(Array, level.grid.xci)
    N = length(xc)
    η = [sum(d -> (xc[d][I[d]] - 0.5)^2, 1:N) < 0.04 ? contrast : 1.0 for I in CartesianIndices(level.grid.ni)]
    return copyto!(similar(level.correction.p), η)
end

fields(level) = JR.gmg_fields(level.correction.p, size(level.correction.p))

# rhs = K e* for random owned retained e*; returned in fresh fields.
function random_rhs(level, op, rng)
    K = gmg_kernel_module(length(level.rhs.v))
    e, rhs = fields(level), fields(level)
    foreach((v, m) -> v .= m .* copyto!(similar(v), randn(rng, size(v))), e.v, op.velocity_mask)
    copyto!(e.p, randn(rng, size(e.p)))
    JR.gmg_prepare!(op, e)
    K.gmg_apply_operator!(op, e)
    foreach((r, o, m) -> r .= m .* o, rhs.v, op.out.v, op.velocity_mask)
    rhs.p .= op.pressure_mask .* op.out.p
    return rhs
end

function setup(ni, contrast, γfact; kwargs...)
    K = gmg_kernel_module(length(ni))
    mg = multigrid(ni; kwargs...)
    fine, op = first(mg.hierarchy.levels), first(mg.operators)
    η = inclusion(fine, contrast)
    K.gmg_set_coefficients!(op, η; γ = γfact .* η, pressure_scale = η .+ γfact .* η)
    K.gmg_refresh!(mg)
    return K, mg, fine, op
end

velocity_only(level, rhs) = (b = fields(level); foreach(copyto!, b.v, rhs.v); b)
factor(history) = (history[end] / history[1])^(1 / (length(history) - 1))

@testset "GMG velocity-block multigrid" begin
    @testset "Face-measure symmetry" begin
        for ni in ((12, 10), (8, 8, 6))
            K, mg, fine, op = setup(ni, 1.0e2, 1.0)
            rng = Xoshiro(5)
            u, w = fields(fine), fields(fine)
            for (x, m) in zip((u.v..., w.v...), (op.velocity_mask..., op.velocity_mask...))
                x .= m .* copyto!(similar(x), randn(rng, size(x)))
            end
            dot(a, b) = K.gmg_velocity_dot(op, fine, a, b)
            JR.gmg_prepare!(op, u)
            K.gmg_apply_operator!(op, u)
            Ku = map(copy, op.out.v)
            JR.gmg_prepare!(op, w)
            K.gmg_apply_operator!(op, w)
            @test dot(Ku, w.v) ≈ dot(u.v, op.out.v) rtol = 1.0e-10
            @test dot(u.v, Ku) > 0
        end
    end

    @testset "V-cycle, constant viscosity, h-independent" begin
        for line_search in (true, false)
            factors = map((16, 32, 64)) do n
                K, mg, fine, op = setup((n, n), 1.0, 1.0; line_search)
                b = velocity_only(fine, random_rhs(fine, op, Xoshiro(3)))
                status, counters, history = K.gmg_velocity_solve!(mg, fields(fine), b; rtol = 1.0e-8, max_cycles = 40)
                @test status == JR.gmg_converged
                @test counters.coarse_status == JR.gmg_converged
                @test 0 < counters.work < counters.applications
                factor(history)
            end
            @test maximum(factors) < 0.15
        end
        K, mg, fine, op = setup((16, 16, 16), 1.0, 1.0)
        b = velocity_only(fine, random_rhs(fine, op, Xoshiro(3)))
        status, _, history = K.gmg_velocity_solve!(mg, fields(fine), b; rtol = 1.0e-8, max_cycles = 40)
        @test status == JR.gmg_converged
        @test factor(history) < 0.2
    end

    @testset "FCG under viscosity contrast" begin
        for (ni, contrast) in (((32, 32), 1.0e2), ((32, 32), 1.0e4), ((16, 16, 16), 1.0e2))
            K, mg, fine, op = setup(ni, contrast, 1.0)
            b = velocity_only(fine, random_rhs(fine, op, Xoshiro(3)))
            x, counters = fields(fine), JR.GMGCounters()
            status = K.gmg_velocity_fcg!(mg, x, b, counters; rtol = 1.0e-8, max_cycles = 100)
            @test status == JR.gmg_converged
            @test all(iszero, Array(x.p))
            # b now holds the final residual; check it against a fresh evaluation.
            r = fields(fine)
            K.gmg_velocity_residual!(op, r, velocity_only(fine, random_rhs(fine, op, Xoshiro(3))), x)
            @test K.gmg_velocity_norm(op, fine, r) ≈ K.gmg_velocity_norm(op, fine, b) rtol = 1.0e-6 atol = 1.0e-12
        end
    end

    @testset "Outer PH with velocity multigrid" begin
        # A unit pressure step is near the stability limit in 3D under contrast (1.5 diverges).
        cases = (((32, 32), 1.0, 1.0), ((32, 32), 1.0e2, 1.0), ((32, 32), 1.0e4, 1.0), ((16, 16, 16), 1.0e2, 0.5))
        for (ni, contrast, pressure_step) in cases
            K, mg, fine, op = setup(ni, contrast, 1.0)
            rhs = random_rhs(fine, op, Xoshiro(7))
            status, counters, history = K.gmg_ph_solve!(
                mg, fields(fine), rhs; rtol = 1.0e-6, max_iterations = 100, velocity_cycles = 20,
                velocity_rtol = 1.0e-2, pressure_step
            )
            @test status == JR.gmg_converged
            @test counters.iterations < 60
            @test counters.work > 0
        end
    end

    @testset "Inference" begin
        K, mg, fine, op = setup((16, 16), 1.0e2, 1.0)
        b = velocity_only(fine, random_rhs(fine, op, Xoshiro(1)))
        @test @inferred(K.gmg_velocity_vcycle!(mg, 1, fields(fine), b, JR.GMGCounters())) === nothing
        @test @inferred(K.gmg_velocity_fcg!(mg, fields(fine), b, JR.GMGCounters(); max_cycles = 2)) isa JR.GMGStatus
        rhs = random_rhs(fine, op, Xoshiro(1))
        @test first(@inferred(K.gmg_ph_solve!(mg, fields(fine), rhs; max_iterations = 2))) isa JR.GMGStatus
    end
end

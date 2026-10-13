using Test, GeoParams

const gmg_backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if gmg_backend_name == "CUDA"
    import CUDA
    CUDA.allowscalar(false)
elseif gmg_backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax

@testset "GMG hierarchy types and geometry" begin
    JR = JustRelax
    backend_type = gmg_backend_name == "CUDA" ? JR.CUDABackend : gmg_backend_name == "AMDGPU" ? JR.AMDGPUBackend : JR.CPUBackend
    for N in (2, 3), T in (Float32, Float64)
        dims = Val(N)
        vertices = ntuple(d -> collect(range(T(-d), T(d); length = 9)), dims)
        partition = @inferred JR.GMGPartition(ntuple(_ -> 8, dims), ntuple(_ -> 1:8, dims))
        options = @inferred JR.GMGOptions(dims; coarsest_cells = 2)
        hierarchy = @inferred JR.GMGHierarchy(backend_type, vertices, partition; options)
        @test isconcretetype(eltype(hierarchy.levels))
        @test isconcretetype(eltype(hierarchy.transitions))
        @test length(hierarchy.levels) == 3
        @test length(hierarchy.transitions) == 2
        for (i, level) in enumerate(hierarchy.levels)
            n = 8 ÷ 2^(i - 1)
            @test level.domain.ni == ntuple(_ -> n, dims)
            @test level.grid.ni == level.domain.ni
            @test level.domain.li == ntuple(d -> T(2d), dims)
            @test level.domain.origin == ntuple(d -> T(-d), dims)
            @test level.grid.origin isa NTuple{N, T}
            @test level.grid.max_li isa T
            @test eltype(level.correction.p) == T
            @test JR.backend(level) == JR.backend(level.correction.p)
            @test size(level.correction.p) == level.grid.ni
            @test level.correction.p !== level.rhs.p
            @test level.rhs.p !== level.residual.p
            @test all(iszero, Array(level.correction.p))
            for d in 1:N
                @test Array(level.grid.xvi[d]) == vertices[d][1:(2^(i - 1)):end]
                @test Array(level.grid.di.vertex[d]) ≈ fill(T(2d / n), n)
                @test size(level.correction.v[d]) == ntuple(k -> n + (k == d ? 1 : 2), dims)
                @test level.correction.v[d] !== level.rhs.v[d]
                @test Array(level.measures.cell[d]) ≈ diff(level.domain.xvi[d])
                @test sum(Array(level.measures.face[d])) ≈ level.domain.li[d]
            end
            @test @inferred(JR.level_sum(level, T(3))) === T(3)
            @test @inferred(JR.level_max(level, T(3))) === T(3)
        end
        @test @inferred(JR.gmg_parent(hierarchy.transitions[1], ntuple(_ -> 4, dims))) == ntuple(_ -> 2, dims)
        vertices[1][1] = T(-99)
        @test hierarchy.levels[1].domain.origin[1] == T(-1)
        @test first(Array(hierarchy.levels[1].grid.xvi[1])) == T(-1)
    end

    @testset "Directional coarsening and odd cells" begin
        for N in (2, 3)
            dims = Val(N)
            vertices = ntuple(_ -> collect(range(0.0, 1.0; length = 9)), dims)
            partition = JR.GMGPartition(ntuple(_ -> 8, dims), ntuple(_ -> 1:8, dims))
            for ratio in Iterators.product(ntuple(_ -> (1, 2), dims)...)
                options = JR.GMGOptions(dims; coarsening = ratio, coarsest_cells = 2)
                hierarchy = JR.GMGHierarchy(backend_type, vertices, partition; options)
                @test last(hierarchy.levels).domain.ni == ntuple(d -> ratio[d] == 1 ? 8 : 2, dims)
                @test all(t -> t.ratio == ratio, hierarchy.transitions)
            end
            options = JR.GMGOptions(dims; coarsest_cells = 2, max_levels = 1)
            @test length(JR.GMGHierarchy(backend_type, vertices, partition; options).levels) == 1
        end
        vertices = (collect(range(0.0, 1.0; length = 8)), collect(range(0.0, 1.0; length = 9)))
        hierarchy = JR.GMGHierarchy(backend_type, vertices, JR.GMGPartition((7, 8), (1:7, 1:8)); options = JR.GMGOptions(Val(2); coarsest_cells = 2))
        @test [l.domain.ni for l in hierarchy.levels] == [(7, 8), (7, 4), (7, 2)]
        @test all(t -> t.ratio == (1, 2), hierarchy.transitions)
    end

    @testset "Nonuniform nesting and physical scaling" begin
        x = [0.0, 0.03, 0.1, 0.2, 0.35, 0.5, 0.65, 0.83, 1.0]
        vertices = (x, 2 .* x .- 1)
        hierarchy = JR.GMGHierarchy(backend_type, vertices, JR.GMGPartition((8, 8), (1:8, 1:8)); options = JR.GMGOptions(Val(2); coarsest_cells = 2))
        for i in 2:3, d in 1:2
            fine, coarse = hierarchy.levels[i - 1], hierarchy.levels[i]
            fine_widths = Array(fine.grid.di.vertex[d])
            @test Array(coarse.grid.di.vertex[d]) ≈ fine_widths[1:2:end] + fine_widths[2:2:end]
            xv = Array(coarse.grid.xvi[d])
            xc = Array(coarse.grid.xci[d])
            @test diff(3 .* xv .+ 2) ./ Array(coarse.grid.di.vertex[d]) ≈ fill(3.0, length(xc))
            # Independent quadratic second derivative on nonuniform centers.
            dx = Array(coarse.grid.di.center[d])
            slopes = diff(xc .^ 2) ./ dx
            @test 2 .* diff(slopes) ./ (dx[1:(end - 1)] + dx[2:end]) ≈ fill(2.0, length(xc) - 2)
        end
    end

    @testset "Ownership and periodic seam" begin
        partition = @inferred JR.GMGPartition((8, 8), (1:4, 1:8))
        @test partition.storage == (1:5, 1:8)
        @test partition.owned_local == (1:4, 1:8)
        @test @inferred(JR.gmg_owned_velocity(partition, Val(1))) == (1:4, 2:9)
        @test @inferred(JR.gmg_owned_velocity(partition, Val(2))) == (2:5, 1:9)
        periodic = JR.GMGPartition((8, 8), (1:8, 1:8); periodic = (true, true))
        hierarchy = JR.GMGHierarchy(backend_type, (collect(0.0:8.0), collect(0.0:8.0)), periodic; options = JR.GMGOptions(Val(2); coarsest_cells = 2))
        @test periodic.storage == (0:9, 0:9)
        @test periodic.owned_local == (2:9, 2:9)
        @test JR.gmg_owned_velocity(periodic, Val(1)) == (2:9, 3:10)
        for level in hierarchy.levels, d in 1:2
            @test length(JR.gmg_owned_velocity(level.partition, Val(d))[d]) == level.domain.ni[d]
            @test Array(level.grid.xvi[d])[1] < level.domain.origin[d]
            @test last(Array(level.grid.xvi[d])) > level.domain.origin[d] + level.domain.li[d]
            @test all(>(0), Array(level.measures.face[d]))
        end
    end

    @testset "GeoParams input types" begin
        linear = SetMaterialParams(Name = "linear", CreepLaws = (LinearViscous(η = 1.0),))
        nonlinear = SetMaterialParams(Name = "nonlinear", CreepLaws = (PowerlawViscous(η0 = 2.0, n = 3.0),))
        TA = JR.PTArray(backend_type)
        args = (; T = TA(zeros(4, 4)), P = TA(ones(4, 4)))
        single = @inferred JR.GMGMaterialContext(linear, args)
        @test single.rheology === linear
        @test single.args === args
        @test single.phase_ratios === nothing
        backend_JP = gmg_backend_name == "CUDA" ? CUDA.CUDABackend : gmg_backend_name == "AMDGPU" ? AMDGPU.ROCBackend : JustPIC.CPU
        phase_ratios = JustPIC.PhaseRatios(backend_JP, 2, (4, 4))
        mixture = @inferred JR.GMGMaterialContext((linear, nonlinear), args, phase_ratios)
        @test mixture.rheology === (linear, nonlinear)
        @test mixture.phase_ratios === phase_ratios
        @test isconcretetype(typeof(mixture))
        @test_throws ArgumentError JR.GMGMaterialContext((), args, phase_ratios)
        @test_throws ArgumentError JR.GMGMaterialContext((linear, :invalid), args, phase_ratios)
        @test_throws ArgumentError JR.GMGMaterialContext((linear,), args, phase_ratios)
    end

    @testset "Rejected setup" begin
        @test_throws ArgumentError JR.GMGOptions(Val(1))
        @test_throws ArgumentError JR.GMGOptions(Val(2); coarsening = (2, 3))
        @test_throws ArgumentError JR.GMGOptions(Val(2); coarsening = [2, 2])
        @test_throws ArgumentError JR.GMGOptions(Val(2); coarsest_cells = 1)
        @test_throws ArgumentError JR.GMGOptions(Val(2); coarsest_cells = (2,))
        @test_throws ArgumentError JR.GMGOptions(Val(2); max_levels = 0)
        @test_throws ArgumentError JR.GMGPartition((8, 8), (1:9, 1:8))
        @test_throws ArgumentError JR.GMGPartition((8, 8), (1:1, 1:8))
        @test_throws ArgumentError JR.GMGPartition((8, 8), (1:8, 1:8); halo = (-1, 1))
        @test_throws ArgumentError JR.gmg_owned_velocity(JR.GMGPartition((8, 8), (1:8, 1:8)), Val(3))
        partition = JR.GMGPartition((2, 2), (1:2, 1:2))
        @test_throws ArgumentError JR.GMGHierarchy(backend_type, ([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]), partition)
        @test_throws ArgumentError JR.GMGHierarchy(backend_type, ([0.0, 1.0, Inf], [0.0, 1.0, 2.0]), partition)
        @test_throws ArgumentError JR.GMGHierarchy(backend_type, ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]), partition)
    end
end

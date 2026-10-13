using Test

const gmg_mpi_backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if gmg_mpi_backend_name == "CUDA"
    import CUDA
    CUDA.allowscalar(false)
elseif gmg_mpi_backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax

@testset "GMG level context with active IGG" begin
    JR = JustRelax
    backend_type = gmg_mpi_backend_name == "CUDA" ? JR.CUDABackend : gmg_mpi_backend_name == "AMDGPU" ? JR.AMDGPUBackend : JR.CPUBackend
    MPI = JR.MPI
    MPI.Initialized() || MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    igg = IGG(init_global_grid(10, 10, 1; init_MPI = false, dimx = nprocs, dimy = 1, select_device = false)...)
    try
        fine_grid = Geometry((10, 10), (1.0, 1.0))
        fine_sizes = (nx_g(), ny_g(), nz_g())
        fine_vertices = map(copy, fine_grid.xvi)
        field = fill(Float64(rank), 10, 10)
        update_halo!(field)
        expected_halo = copy(field)

        global_ni = (8nprocs, 8)
        vertices = (collect(range(0.0, 1.0; length = global_ni[1] + 1)), collect(range(-1.0, 1.0; length = 9)))
        owned = ((8rank + 1):(8rank + 8), 1:8)
        # Split communicator proves reductions do not accidentally use COMM_WORLD.
        local_comm = MPI.Comm_split(igg.comm_cart, rank, rank)
        try
            local_part = JR.GMGPartition(global_ni, owned; comm = local_comm)
            local_hierarchy = JR.GMGHierarchy(backend_type, vertices, local_part; options = JR.GMGOptions(Val(2); coarsest_cells = 2))
            @test @inferred(JR.level_sum(first(local_hierarchy.levels), Float64(rank + 1))) == rank + 1
        finally
            MPI.free(local_comm)
        end

        partition = @inferred JR.GMGPartition(global_ni, owned; comm = igg.comm_cart)
        hierarchy = @inferred JR.GMGHierarchy(backend_type, vertices, partition; options = JR.GMGOptions(Val(2); coarsest_cells = 2))
        @test length(hierarchy.levels) == 3
        @test isconcretetype(eltype(hierarchy.levels))
        for level in hierarchy.levels
            @test level.partition.comm === igg.comm_cart
            @test JR.backend(level) == JR.backend(JR.PTArray(backend_type))
            @test @inferred(JR.level_sum(level, 1)) == nprocs
            @test @inferred(JR.level_max(level, rank)) == nprocs - 1
            @test JR.level_sum(level, prod(length.(level.partition.owned))) == prod(level.domain.ni)
            widths = level.measures.cell
            owned_local = level.partition.owned_local
            local_volume = prod(sum(Array(widths[d])[owned_local[d]]) for d in 1:2)
            @test JR.level_sum(level, local_volume) ≈ prod(level.domain.li)
            for component in 1:2
                indices = JR.gmg_owned_velocity(level.partition, Val(component))
                count = prod(length.(indices))
                expected = prod(ntuple(d -> level.domain.ni[d] + Int(d == component), Val(2)))
                @test JR.level_sum(level, count) == expected
            end
        end

        # A single rank with invalid setup must cause every rank to reject.
        bad_owned = rank == 0 ? (1:1, 1:8) : owned
        @test_throws ArgumentError JR.GMGPartition(global_ni, bad_owned; comm = igg.comm_cart)
        bad_vertices = rank == 0 ? (fill(0.0, global_ni[1] + 1), vertices[2]) : vertices
        @test_throws ArgumentError JR.GMGHierarchy(backend_type, bad_vertices, partition)
        if nprocs > 1
            bad_options = JR.GMGOptions(Val(2); coarsest_cells = 2, max_levels = rank == 0 ? 1 : 3)
            @test_throws ArgumentError JR.GMGHierarchy(backend_type, vertices, partition; options = bad_options)
            inconsistent_vertices = rank == 0 ? (vertices[1] .+ 0.1, vertices[2]) : vertices
            @test_throws ArgumentError JR.GMGHierarchy(backend_type, inconsistent_vertices, partition)
        end
        @test JR.gmg_all(true, igg.comm_cart)

        @test (nx_g(), ny_g(), nz_g()) == fine_sizes
        @test fine_grid.xvi == fine_vertices
        @test Geometry((10, 10), (1.0, 1.0)).di == fine_grid.di
        fill!(field, rank)
        update_halo!(field)
        @test field == expected_halo
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

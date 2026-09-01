@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using JustRelax
import JustRelax.JustRelax3D as JR3

using ParallelStencil, ParallelStencil.FiniteDifferences3D
const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
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

@testset "PhaseRatios 3D" begin
    # src/phases/PhaseRatios.jl: `update_phase_ratios_3D!` and its kernels.
    # `compute_dx` needs `LinRange` (or similar AbstractRange) xci/xvi.
    nx, ny, nz = 4, 4, 4
    pr = JustPIC.PhaseRatios(backend_JP, 2, (nx, ny, nz))
    xvi = ntuple(_ -> range(0.0, 1.0; length = nx + 1), Val(3))
    xci = ntuple(_ -> range(0.125, 0.875; length = nx), Val(3))

    # Two-phase split: phase 1 on the left half, phase 2 on the right half.
    p1 = @zeros(nx, ny, nz)
    p2 = @zeros(nx, ny, nz)
    p1[1:2, :, :] .= 1.0
    p2[3:4, :, :] .= 1.0

    JR3.update_phase_ratios_3D!(pr, (p1, p2), xci, xvi)

    # CellArrays stores an SVector per cell; copy to host so we can index per-cell
    # on any backend.
    center_h = Base.Array(pr.center)
    vertex_h = Base.Array(pr.vertex)
    faces_h = map(Base.Array, (pr.Vx, pr.Vy, pr.Vz))
    midpoints_h = map(Base.Array, (pr.xy, pr.yz, pr.xz))

    # Staggered grids the ratios live on
    @test size(center_h) == (nx, ny, nz)
    @test size(vertex_h) == (nx + 1, ny + 1, nz + 1)
    @test size.(faces_h) == ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))
    @test size.(midpoints_h) ==
        ((nx + 1, ny + 1, nz), (nx, ny + 1, nz + 1), (nx + 1, ny, nz + 1))

    # Cell-center ratios reproduce the input cleanly (only one phase per cell).
    @test [center_h[i, 1, 1][1] for i in 1:nx] ≈ [1.0, 1.0, 0.0, 0.0]
    @test [center_h[i, 1, 1][2] for i in 1:nx] ≈ [0.0, 0.0, 1.0, 1.0]

    # Every ratio, on every grid, sums to one
    for A in (center_h, vertex_h, faces_h..., midpoints_h...)
        @test all(sum(A[I]) ≈ 1.0 for I in CartesianIndices(A))
    end

    # At a vertex on the phase boundary (i = 3), both phases contribute
    v = vertex_h[3, 2, 2]
    @test v[1] > 0.0 && v[2] > 0.0

    # Threshold path: a tiny third phase (< 1e-5) should be cleaned to zero.
    pr3 = JustPIC.PhaseRatios(backend_JP, 3, (nx, ny, nz))
    p1b = @fill(0.6, nx, ny, nz)
    p2b = @fill(0.4, nx, ny, nz)
    p3b = @fill(1.0e-6, nx, ny, nz)             # below the 1e-5 threshold
    JR3.update_phase_ratios_3D!(pr3, (p1b, p2b, p3b), xci, xvi)
    center3_h = Base.Array(pr3.center)
    @test center3_h[2, 2, 2][3] == 0.0          # tiny phase zeroed out
    @test center3_h[2, 2, 2][1] + center3_h[2, 2, 2][2] ≈ 1.0
end

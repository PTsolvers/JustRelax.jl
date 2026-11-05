push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
import ImplicitGlobalGrid
using MPI

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

nx, ny = 6, 5 # local cells, overlap included
igg = IGG(
    init_global_grid(
        nx, ny, 1; init_MPI = JustRelax.MPI.Initialized() ? false : true, select_device = false
    )...
)
me = igg.me
gg = ImplicitGlobalGrid.global_grid()
offset = ntuple(d -> gg.coords[d] * (gg.nxyz[d] - gg.overlaps[d]), Val(2))

# global vertex coordinates, refined towards the top and right
xv_g = cumsum([0.0; range(1.0, 2.0; length = nx_g())])
yv_g = cumsum([0.0; range(2.0, 1.0; length = ny_g())])

@testset "nonuniform Geometry on an MPI grid" begin
    grid_dev = Geometry(PTArray(backend), xv_g, yv_g)
    @test grid_dev.ni == (nx, ny)
    @test Array(grid_dev.xvi[2]) == Array(Geometry(PTArray(backend), xv_g, yv_g; distributed = false).xvi[2])[offset[2] .+ (1:(ny + 1))]

    grid = Geometry(Array, xv_g, yv_g)
    grid_g = Geometry(Array, xv_g, yv_g; distributed = false)

    @test grid.ni == (nx, ny)
    @test grid_g.ni == (nx_g(), ny_g())
    # the extent describes the global domain, as for the uniform constructor
    @test grid.li == grid_g.li
    @test grid.max_li == grid_g.max_li
    @test grid.origin == grid_g.origin

    ix = offset[1] .+ (1:(nx + 1))
    iy = offset[2] .+ (1:(ny + 1))
    @test grid.xvi == (grid_g.xvi[1][ix], grid_g.xvi[2][iy])
    @test grid.xci == (grid_g.xci[1][ix[1:(end - 1)]], grid_g.xci[2][iy[1:(end - 1)]])
    @test grid.di.vertex == (diff(grid.xvi[1]), diff(grid.xvi[2]))
    # the ghost nodes of the velocity grids sit on the cell centers of the neighboring
    # ranks, not on a local extrapolation
    @test grid.xi_vel[1][2] == grid_g.xi_vel[1][2][offset[2] .+ (1:(ny + 2))]
    @test grid.xi_vel[2][1] == grid_g.xi_vel[2][1][offset[1] .+ (1:(nx + 2))]
    @test grid.di.velocity[1][2] == diff(grid.xi_vel[1][2])
    @test grid._di.velocity[2][1] == inv.(diff(grid.xi_vel[2][1]))

    # uniform coordinates reproduce the subdomain of the uniform constructor
    li = (1.5, 0.75)
    grid_u = Geometry((nx, ny), li)
    grid_nu = Geometry(Array, range(0, li[1]; length = nx_g() + 1), range(0, li[2]; length = ny_g() + 1))
    @test all(d -> grid_nu.xvi[d] ≈ grid_u.xvi[d], 1:2)
    @test all(d -> grid_nu.xci[d] ≈ grid_u.xci[d], 1:2)
    @test all(d -> grid_nu.li[d] ≈ grid_u.li[d], 1:2)

    if any(>(1), igg.dims[1:2])
        @test_throws "pass the coordinates of the global grid" Geometry(
            Array, xv_g[1:(nx + 1)], yv_g[1:(ny + 1)]
        )
    end
end

@testset "compute_dt agrees across ranks" begin
    stokes = StokesArrays(backend, (nx, ny))
    stokes.V.Vx .= me + 1
    stokes.V.Vy .= 1
    # rank-dependent spacing, as on a refined grid
    di = (0.1 * (me + 1), 0.2)
    dt = compute_dt(stokes, di, Inf, igg)
    @test all(==(dt), MPI.Allgather(dt, igg.comm_cart))
    nranks = MPI.Comm_size(igg.comm_cart)
    expected = minimum(r -> min(0.1 * (r + 1) / nranks, 0.2 / 1) * 0.9, 0:(nranks - 1))
    @test dt ≈ expected
end

finalize_global_grid(; finalize_MPI = false)

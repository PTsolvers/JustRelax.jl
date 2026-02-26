@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using Statistics
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
import JustRelax.JustRelax2D:
    detect_args_size,
    _tuple,
    continuation_linear,
    continuation_log,
    assign!,
    mean_mpi,
    sum_mpi,
    norm_mpi,
    minimum_mpi,
    maximum_mpi

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
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

@testset "Utils" begin
    @testset "utils.jl" begin
        # Set up mock data
        # Physical domain ------------------------------------
        ly = 1.0       # domain length in y
        lx = 1.0       # domain length in x
        nx, ny, nz = 4, 4, 4   # number of cells
        ni = nx, ny     # number of cells
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
        li = lx, ly     # domain length in x- and y-
        di = @. li / ni # grid step in x- and -y
        origin = 0.0, -ly   # origin coordinates (15km f sticky air layer)
        grid = Geometry(ni, li; origin = origin)
        (; xci, xvi) = grid

        # 2D case
        dst = "test_Utils"
        stokes = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)
        take(dst)
        @test isdir(dst)
        rm(dst; recursive = true)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)
        # temperature
        pT, pPhases = init_cell_arrays(particles, Val(2))
        time = 1.0
        dt = 0.1

        stokes.viscosity.η .= @fill(1.0)
        stokes.V.Vy .= @fill(10)
        thermal.T .= @fill(100)
        stokes.P .= @fill(100.0)
        thermal.Tc .= @fill(100.0)

        # Utils
        args = (P = stokes.P, T = thermal.T)
        tuple_args = (args.P, args.T)
        @test detect_args_size(tuple_args) == (7, 5)

        @test _tuple(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test _tuple(stokes.V) === (stokes.V.Vx, stokes.V.Vy)

        A = @zeros(ni...)
        B = @zeros(ni...)
        @parallel (@idx ni) multi_copy!((A, B), (stokes.P, thermal.Tc))
        @test A == stokes.P
        @test B == thermal.Tc

        A .= 0.0e0
        @parallel (@idx ni) assign!(A, stokes.P)
        @test A == stokes.P

        @test JustRelax2D.tupleize(1) === (1,)
        @test JustRelax2D.tupleize((1, 2)) === (1, 2)

        # Stokes
        @test JustRelax2D.@unpack(stokes.V) === (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
        @test @velocity(stokes) === (stokes.V.Vx, stokes.V.Vy)
        @test @displacement(stokes) === (stokes.U.Ux, stokes.U.Uy)

        @test @strain(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy)
        @test @strain_center(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy_c)
        @test @strain_increment(stokes) === (stokes.Δε.xx, stokes.Δε.yy, stokes.Δε.xy)
        @test @plastic_strain(stokes) === (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)
        @test @stress(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy)
        @test @stress_center(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test @tensor(stokes.τ_o) === (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy)
        @test @tensor_center(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test @tensor_vertex(stokes.τ) === (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)
        @test @shear(stokes.τ) === (stokes.τ.xy)
        @test @normal(stokes.τ) === (stokes.τ.xx, stokes.τ.yy)
        @test @residuals(stokes.R) === (stokes.R.Rx, stokes.R.Ry)

        # Thermal
        @test @qT(thermal) === (thermal.qTx, thermal.qTy)
        @test @qT2(thermal) === (thermal.qTx2, thermal.qTy2)

        # other functions
        dt_diff = 0.1
        @test compute_dt(stokes, di, dt_diff, igg) === 0.011904761904761904
        @test compute_dt(stokes, di, dt_diff) === 0.011904761904761904
        @test compute_dt(stokes, di) ≈ 0.011904761904761904
        @test continuation_log(1.0, 0.8, 0.05) ≈ 0.8089757207980266
        @test continuation_linear(1.0, 0.8, 0.05) === 0.81

        #MPI
        @test sum_mpi(stokes.viscosity.η) === 16.0
        @test mean_mpi(stokes.viscosity.η) === 1.0
        @test norm_mpi(stokes.viscosity.η) === 4.0
        @test minimum_mpi(stokes.viscosity.η) === 1.0
        @test maximum_mpi(stokes.viscosity.η) === 1.0

        # 3D case
        ni = nx, ny, nz
        stokes = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)

        stokes.viscosity.η .= @fill(1.0)
        stokes.V.Vy .= @fill(10)
        thermal.T .= @fill(100)

        # Stokes
        @test _tuple(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test _tuple(stokes.V) === (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)

        @test @velocity(stokes) === (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
        @test @displacement(stokes) === (stokes.U.Ux, stokes.U.Uy, stokes.U.Uz)

        @test @strain(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz, stokes.ε.xz, stokes.ε.xy)
        @test @strain_center(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz_c, stokes.ε.xz_c, stokes.ε.xy_c)
        @test @strain_increment(stokes) === (stokes.Δε.xx, stokes.Δε.yy, stokes.Δε.zz, stokes.Δε.yz, stokes.Δε.xz, stokes.Δε.xy)
        @test @plastic_strain(stokes) === (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.zz, stokes.ε_pl.yz, stokes.ε_pl.xz, stokes.ε_pl.xy)
        @test @stress(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        @test @stress_center(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test @tensor(stokes.τ_o) === (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.zz, stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy)
        @test @tensor_center(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test @shear(stokes.τ) === (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        @test @normal(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz)
        @test @residuals(stokes.R) === (stokes.R.Rx, stokes.R.Ry, stokes.R.Rz)

        # Thermal
        @test @qT(thermal) === (thermal.qTx, thermal.qTy, thermal.qTz)
        @test @qT2(thermal) === (thermal.qTx2, thermal.qTy2, thermal.qTz2)

        # other functions
        dt_diff = 0.1
        @test compute_dt(stokes, di, dt_diff, igg) === 0.008064516129032258
        @test compute_dt(stokes, di, dt_diff) ≈ 0.008064516129032258
        @test compute_dt(stokes, di) ≈ 0.008064516129032258

        #MPI
        @test mean_mpi(stokes.viscosity.η) === 1.0
        @test norm_mpi(stokes.viscosity.η) === 8.0
        @test minimum_mpi(stokes.viscosity.η) === 1.0
        @test maximum_mpi(stokes.viscosity.η) === 1.0
    end
    @testset "MiniKernels" begin
        A2 = reshape(collect(1.0:16.0), 4, 4)
        A3 = reshape(collect(1.0:64.0), 4, 4, 4)
        i = 2
        j = 2
        k = 2

        @test JustRelax2D.center(A2, i, j) == 6.0
        @test JustRelax2D.next(A2, i, j) == 11.0
        @test JustRelax2D.left(A2, i, j) == 5.0
        @test JustRelax2D.right(A2, i, j) == 7.0
        @test JustRelax2D.back(A2, i, j) == 2.0
        @test JustRelax2D.front(A2, i, j) == 10.0

        @test JustRelax2D.left(A3, i, j, k) == 21.0
        @test JustRelax2D.right(A3, i, j, k) == 23.0
        @test JustRelax2D.back(A3, i, j, k) == 18.0
        @test JustRelax2D.front(A3, i, j, k) == 26.0
        @test JustRelax2D.bot(A3, i, j, k) == 6.0
        @test JustRelax2D.top(A3, i, j, k) == 38.0

        dx = 2.0
        dy = 3.0
        dz = 4.0

        @test JustRelax2D._d_xa(A2, dx, i, j) == 2.0
        @test JustRelax2D._d_ya(A2, dy, i, j) == 12.0
        @test JustRelax2D._d_za(A3, dz, i, j, k) == 64.0

        @test JustRelax2D._d_xi(A2, dx, i, j) == 2.0
        @test JustRelax2D._d_yi(A2, dy, i, j) == 12.0

        @test JustRelax2D._d_xi(A3, dx, i, j, k) == 2.0
        @test JustRelax2D._d_yi(A3, dy, i, j, k) == 12.0
        @test JustRelax2D._d_zi(A3, dz, i, j, k) == 64.0

        Ax = A2
        Ay = A2 .* 2.0
        @test JustRelax2D.div(Ax, Ay, dx, dy, i, j) == 26.0

        Ax3 = A3
        Ay3 = A3 .* 2.0
        Az3 = A3 .* 3.0
        @test JustRelax2D.div(Ax3, Ay3, Az3, dx, dy, dz, i, j, k) == 218.0

        @test JustRelax2D._av(A2, i, j) == 13.5
        @test JustRelax2D._av_a(A2, i, j) == 8.5
        @test JustRelax2D._av_xa(A2, i, j) == 6.5
        @test JustRelax2D._av_ya(A2, i, j) == 8.0
        @test JustRelax2D._av_xi(A2, i, j) == 10.5
        @test JustRelax2D._av_yi(A2, i, j) == 9.0

        @test JustRelax2D._harm(A2, i, j) == 1.2136363636363636
        @test JustRelax2D._harm_a(A2, i, j) == 2.001731601731602
        @test JustRelax2D._harm_xa(A2, i, j) == 0.6190476190476191
        @test JustRelax2D._harm_ya(A2, i, j) == 0.5333333333333333

        @test JustRelax2D._gather(A2, i, j) == (6.0, 7.0, 10.0, 11.0)

        @test JustRelax2D._av(A3, i, j, k) == 32.5
        @test JustRelax2D._av_x(A3, i, j, k) == 22.5
        @test JustRelax2D._av_y(A3, i, j, k) == 24.0
        @test JustRelax2D._av_z(A3, i, j, k) == 30.0
        @test JustRelax2D._av_xy(A3, i, j, k) == 24.5
        @test JustRelax2D._av_xz(A3, i, j, k) == 30.5
        @test JustRelax2D._av_yz(A3, i, j, k) == 32.0
        @test JustRelax2D._av_xyi(A3, i, j, k) == 19.5
        @test JustRelax2D._av_xzi(A3, i, j, k) == 13.5
        @test JustRelax2D._av_yzi(A3, i, j, k) == 12.0

        @test JustRelax2D._harm_x(A3, i, j, k) == 22.488888888888887
        @test JustRelax2D._harm_y(A3, i, j, k) == 23.833333333333332
        @test JustRelax2D._harm_z(A3, i, j, k) == 27.866666666666667
        @test JustRelax2D._harm_xy(A3, i, j, k) == 0.04081632653061224
        @test JustRelax2D._harm_xz(A3, i, j, k) == 0.03278688524590164
        @test JustRelax2D._harm_yz(A3, i, j, k) == 0.03125
        @test JustRelax2D._harm_xyi(A3, i, j, k) == 0.05128205128205128
        @test JustRelax2D._harm_xzi(A3, i, j, k) == 0.07407407407407407
        @test JustRelax2D._harm_yzi(A3, i, j, k) == 0.08333333333333333

        @test JustRelax2D._gather_yz(A3, i, j, k) == (22.0, 26.0, 38.0, 42.0)
        @test JustRelax2D._gather_xz(A3, i, j, k) == (22.0, 23.0, 38.0, 39.0)
        @test JustRelax2D._gather_xy(A3, i, j, k) == (22.0, 23.0, 26.0, 27.0)

        @test JustRelax2D._current(A3, i, j, k) == 22.0

        v = collect(1.0:5.0)
        @test JustRelax2D.mysum(v, 2:4) == 9.0
        @test JustRelax2D.mysum(inv, v, 2:4) == 1.0833333333333333
        @test JustRelax2D.mysum(A2, 2:3, 2:3) == 34.0
        @test JustRelax2D.mysum(inv, A2, 2:3, 2:3) == 0.5004329004329005
        @test JustRelax2D.mysum(A3, 2:3, 2:3, 2:3) == 260.0
        @test JustRelax2D.mysum(inv, A3, 2:3, 2:3, 2:3) == 0.2634535347004082
    end
    @testset "versioninfo" begin
        JustRelax.versioninfo(devnull)
        JustRelax.versioninfo(devnull; verbose = true)
    end
end

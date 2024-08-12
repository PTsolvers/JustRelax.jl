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
    @testset "Macros" begin
        # Set up mock data
        # Physical domain ------------------------------------
        ly           = 1.0       # domain length in y
        lx           = 1.0       # domain length in x
        nx, ny, nz   = 4, 4, 4   # number of cells
        ni           = nx, ny     # number of cells
        igg          = IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
        li           = lx, ly     # domain length in x- and y-
        di           = @. li / ni # grid step in x- and -y
        origin       = 0.0, -ly   # origin coordinates (15km f sticky air layer)
        grid         = Geometry(ni, li; origin = origin)
        (; xci, xvi) = grid

        # 2D case
        dst = "test_Utils"
        stokes  = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)
        take(dst)
        @test isdir(dst)
        rm(dst; recursive=true)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)
        # temperature
        pT, pPhases      = init_cell_arrays(particles, Val(2))
        time = 1.0
        dt = 0.1

        stokes.viscosity.η .= @fill(1.0)
        stokes.V.Vy        .= @fill(10)
        thermal.T          .= @fill(100)

        args = (P = stokes.P, T = thermal.T)
        tuple_args = (args.P, args.T)
        @test detect_args_size(tuple_args) == (7, 5)

        # Stokes

        @test _tuple(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test _tuple(stokes.V) === (stokes.V.Vx, stokes.V.Vy)

        @test @velocity(stokes) === (stokes.V.Vx, stokes.V.Vy)
        @test @displacement(stokes) === (stokes.U.Ux, stokes.U.Uy)

        @test @strain(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy)
        @test @strain_center(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy_c)
        @test @plastic_strain(stokes) === (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)
        @test @stress(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy)
        @test @stress_center(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test @tensor(stokes.τ_o) === (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy)
        @test @tensor_center(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
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
        @parallel assign!(stokes.P, stokes.viscosity.η)
        @test stokes.P == stokes.viscosity.η

        #MPI
        @test mean_mpi(stokes.P) === 1.0
        @test norm_mpi(stokes.P) === 4.0
        @test minimum_mpi(stokes.P) === 1.0
        @test maximum_mpi(stokes.P) === 1.0

        # 3D case
        ni = nx, ny, nz
        stokes = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)

        stokes.viscosity.η .= @fill(1.0)
        stokes.V.Vy .= @fill(10)
        thermal.T .= @fill(100)
        # Stokes
        @test _tuple(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test _tuple(stokes.V) === (stokes.V.Vx,stokes.V.Vy, stokes.V.Vz)

        @test @velocity(stokes) === (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
        @test @displacement(stokes) === (stokes.U.Ux, stokes.U.Uy, stokes.U.Uz)

        @test @strain(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz, stokes.ε.xz, stokes.ε.xy)
        @test @strain_center(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz_c, stokes.ε.xz_c, stokes.ε.xy_c)
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
end

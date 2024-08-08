@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using GeoParams

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
        rm(dst, recursive=true)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(
            backend, nxcell, max_xcell, min_xcell, xvi...)
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
        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @test @tensor(stokes.τ_o) == @tensor(stokes.τ)

        @test _tuple(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test _tuple(stokes.V) === (stokes.V.Vx,stokes.V.Vy)

        @test @velocity(stokes) === (stokes.V.Vx, stokes.V.Vy)
        @test @displacement(stokes) === (stokes.U.Ux, stokes.U.Uy)

        @test @strain(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy)
        @test @strain_center(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy_c)
        @test @strain_plastic(stokes) === (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)
        @test @stress(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy)
        @test @stress_center(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test @tensor(stokes.τ_o) === (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy)
        @test @tensor_center(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)
        @test @shear(stokes.τ) === (stokes.τ.xy)
        @test @unpack_shear_components_stag(stokes.τ) === (stokes.τ.xy)
        @test @normal(stokes.τ) === (stokes.τ.xx, stokes.τ.yy)
        @test @residuals(stokes.R) === (stokes.R.Rx, stokes.R.Ry)

        # Thermal
        @test @qT(thermal) === (thermal.qTx, thermal.qTy)
        @test @qT2(thermal) === (thermal.qTx2, thermal.qTy2)

        # # other FUNCTIONS
        # indices((ni...))

        # 3D case
        ni = nx, ny, nz
        stokes = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)

        # Stokes
        @test _tuple(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test _tuple(stokes.V) === (stokes.V.Vx,stokes.V.Vy, stokes.V.Vz)

        @test @velocity(stokes) === (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
        @test @displacement(stokes) === (stokes.U.Ux, stokes.U.Uy, stokes.U.Uz)

        @test @strain(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz, stokes.ε.xz, stokes.ε.xy)
        @test @strain_center(stokes) === (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz_c, stokes.ε.xz_c, stokes.ε.xy_c)
        @test @strain_plastic(stokes) === (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.zz, stokes.ε_pl.yz, stokes.ε_pl.xz, stokes.ε_pl.xy)
        @test @stress(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        @test @stress_center(stokes) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test @tensor(stokes.τ_o) === (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.zz, stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy)
        @test @tensor_center(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c)
        @test @shear(stokes.τ) === (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        @test @unpack_shear_components_stag === (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        @test @normal(stokes.τ) === (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz)
        @test @residuals(stokes.R) === (stokes.R.Rx, stokes.R.Ry, stokes.R.Rz)

        # Thermal
        @test @qT(thermal) === (thermal.qTx, thermal.qTy, thermal.qTz)
        @test @qT2(thermal) === (thermal.qTx2, thermal.qTy2, thermal.qTz2)
    end
end

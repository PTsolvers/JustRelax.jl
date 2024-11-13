push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax2D: interp_Vx∂ρ∂x_on_Vy!, interp_Vx_on_Vy!
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



@testset "Interpolations" begin
    if backend_JR == CPUBackend
        # Set up mock data
        # Physical domain ------------------------------------
        ly           = 1.0       # domain length in y
        lx           = 1.0       # domain length in x
        nx, ny, nz   = 4, 4, 4   # number of cells
        ni           = nx, ny     # number of cells
        li           = lx, ly     # domain length in x- and y-direction
        di           = @. li / ni # grid step in x- and y-direction
        origin       = 0.0, -ly   # origin coordinates (15km f sticky air layer)
        grid         = Geometry(ni, li; origin = origin)
        (; xci, xvi) = grid


        # 2D case
        stokes  = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)
        ρg      = @ones(ni)

        stokes.viscosity.η .= 1
        stokes.V.Vy        .= 10
        thermal.T          .= 100
        thermal.Told       .= 50
        stokes.τ.xy_c      .= 1
        temperature2center!(thermal)


        @test thermal.Tc[1,1] == 100

        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT)
        @test thermal.ΔTc[1,1] == 50

        center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
        @test stokes.τ.xy[2,2] == 1

        Vx_v = @ones(ni.+1...)
        Vy_v = @ones(ni.+1...)

        velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
        @test iszero(Vx_v[1,1])
        @test Vy_v[1,1] == 10

        Vx_v = @ones(ni.+1...)
        Vy_v = @ones(ni.+1...)
        # velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=false)
        velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
        @test iszero(Vx_v[1,1])
        @test Vy_v[1,1] == 10

        Vx = @ones(ni...)
        Vy = @ones(ni...)
        # velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=false)
        velocity2center!(Vx, Vy, stokes.V.Vx, stokes.V.Vy)
        @test iszero(Vx[1,1])
        @test Vy[1,1] == 10
    else
        @test true == true
    end
end

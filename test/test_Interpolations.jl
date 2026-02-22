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
        ly = 1.0       # domain length in y
        lx = 1.0       # domain length in x
        nx, ny, nz = 4, 4, 4   # number of cells
        ni = nx, ny     # number of cells
        li = lx, ly     # domain length in x- and y-direction
        di = @. li / ni # grid step in x- and y-direction
        origin = 0.0, -ly   # origin coordinates (15km f sticky air layer)
        grid = Geometry(ni, li; origin = origin)
        (; xci, xvi) = grid


        # 2D case
        stokes = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)
        ρg = @ones(ni)

        stokes.viscosity.η .= 1
        stokes.V.Vy .= 10
        thermal.T .= 100
        thermal.Told .= 50
        stokes.τ.xy_c .= 1
        temperature2center!(thermal)

        @test thermal.Tc[1, 1] == 100

        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT)
        @test thermal.ΔTc[1, 1] == 50

        center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
        @test stokes.τ.xy[2, 2] == 1

        Vx_v = @ones(ni .+ 1...)
        Vy_v = @ones(ni .+ 1...)

        velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
        @test iszero(Vx_v[1, 1])
        @test Vy_v[1, 1] == 10

        Vx_v = @ones(ni .+ 1...)
        Vy_v = @ones(ni .+ 1...)
        # velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=false)
        velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
        @test iszero(Vx_v[1, 1])
        @test Vy_v[1, 1] == 10

        Vx = @ones(ni...)
        Vy = @ones(ni...)
        # velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=false)
        velocity2center!(Vx, Vy, stokes.V.Vx, stokes.V.Vy)
        @test iszero(Vx[1, 1])
        @test Vy[1, 1] == 10

        # shear2center!
        stokes.ε.xy .= 2
        shear2center!(stokes.ε)
        @test stokes.ε.xy_c[1, 1] == 2

        # center2vertex_harm!
        @testset "center2vertex_harm!" begin
            ctr = rand(nx, ny) .+ 1.0   # avoid zeros for harmonic mean
            vtx = zeros(nx + 1, ny + 1)
            JustRelax2D.center2vertex_harm!(vtx, ctr)
            expected = 4 / (
                1 / ctr[1, 1] + 1 / ctr[1, 2] +
                    1 / ctr[2, 1] + 1 / ctr[2, 2]
            )
            @test vtx[2, 2] ≈ expected
        end

        # interp_Vx_on_Vy! kernel
        @testset "interp_Vx_on_Vy!" begin
            ni_x, ni_y = nx - 1, ny
            Vxi = rand(nx + 1, ny + 1)
            out = zeros(nx, ny)
            @parallel (1:ni_x, 1:ni_y) JustRelax2D.interp_Vx_on_Vy!(out, Vxi)
            i, j = 2, 2
            expected = 0.25 * (Vxi[i, j] + Vxi[i + 1, j] + Vxi[i, j + 1] + Vxi[i + 1, j + 1])
            @test out[i + 1, j] ≈ expected
        end

        # interp_Vx∂ρ∂x_on_Vy! without ϕ
        @testset "interp_Vx∂ρ∂x_on_Vy! (no ϕ)" begin
            ni_x, ni_y = nx - 1, ny
            Vxi = rand(nx + 1, ny + 1)
            ρgi = rand(nx, ny)
            out = zeros(nx, ny)
            _dx = 0.5
            @parallel (1:ni_x, 1:ni_y) JustRelax2D.interp_Vx∂ρ∂x_on_Vy!(out, Vxi, ρgi, _dx)
            i, j = 2, 2
            iW = clamp(i - 1, 1, nx); iE = clamp(i + 1, 1, nx)
            jS = clamp(j - 1, 1, ny); jN = clamp(j, 1, ny)
            ρg_L = 0.25 * (ρgi[iW, jS] + ρgi[i, jS] + ρgi[iW, jN] + ρgi[i, jN])
            ρg_R = 0.25 * (ρgi[iE, jS] + ρgi[i, jS] + ρgi[iE, jN] + ρgi[i, jN])
            expected = 0.25 * (Vxi[i, j] + Vxi[i + 1, j] + Vxi[i, j + 1] + Vxi[i + 1, j + 1]) * (ρg_R - ρg_L) * _dx
            @test out[i + 1, j] ≈ expected
        end

        # interp_Vx∂ρ∂x_on_Vy! with ϕ
        @testset "interp_Vx∂ρ∂x_on_Vy! (with ϕ)" begin
            ni_x, ni_y = nx - 1, ny
            Vxi = rand(nx + 1, ny + 1)
            ρgi = rand(nx, ny)
            out = zeros(nx, ny)
            _dx = 0.5
            struct FakeRockRatio
                center::Matrix{Float64}
            end
            ϕ = FakeRockRatio(rand(nx, ny))
            @parallel (1:ni_x, 1:ni_y) JustRelax2D.interp_Vx∂ρ∂x_on_Vy!(out, Vxi, ρgi, ϕ, _dx)
            i, j = 2, 2
            iW = clamp(i - 1, 1, nx); iE = clamp(i + 1, 1, nx)
            jS = clamp(j - 1, 1, ny); jN = clamp(j, 1, ny)
            ρg_L = 0.25 * (
                ρgi[iW, jS] * ϕ.center[iW, jS] + ρgi[i, jS] * ϕ.center[i, jS] +
                    ρgi[iW, jN] * ϕ.center[iW, jN] + ρgi[i, jN] * ϕ.center[i, jN]
            )
            ρg_R = 0.25 * (
                ρgi[iE, jS] * ϕ.center[iE, jS] + ρgi[i, jS] * ϕ.center[i, jS] +
                    ρgi[iE, jN] * ϕ.center[iE, jN] + ρgi[i, jN] * ϕ.center[i, jN]
            )
            expected = 0.25 * (Vxi[i, j] + Vxi[i + 1, j] + Vxi[i, j + 1] + Vxi[i + 1, j + 1]) * (ρg_R - ρg_L) * _dx
            @test out[i + 1, j] ≈ expected
        end

        # 3D interpolation functions
        @testset "velocity2vertex! 3D in-place" begin
            nx3, ny3, nz3 = 3, 3, 3
            Vx3 = rand(nx3 + 1, ny3 + 2, nz3 + 2)
            Vy3 = rand(nx3 + 2, ny3 + 1, nz3 + 2)
            Vz3 = rand(nx3 + 2, ny3 + 2, nz3 + 1)
            Vxv = zeros(nx3, ny3, nz3)
            Vyv = zeros(nx3, ny3, nz3)
            Vzv = zeros(nx3, ny3, nz3)
            velocity2vertex!(Vxv, Vyv, Vzv, Vx3, Vy3, Vz3)
            for k in 1:nz3, j in 1:ny3, i in 1:nx3
                @test Vxv[i, j, k] ≈ 0.25 * (Vx3[i, j, k] + Vx3[i, j + 1, k] + Vx3[i, j, k + 1] + Vx3[i, j + 1, k + 1])
                @test Vyv[i, j, k] ≈ 0.25 * (Vy3[i, j, k] + Vy3[i + 1, j, k] + Vy3[i, j, k + 1] + Vy3[i + 1, j, k + 1])
                @test Vzv[i, j, k] ≈ 0.25 * (Vz3[i, j, k] + Vz3[i, j + 1, k] + Vz3[i + 1, j, k] + Vz3[i + 1, j + 1, k])
            end
        end

        @testset "velocity2vertex 3D allocating" begin
            nx3, ny3, nz3 = 3, 3, 3
            Vx3 = rand(nx3 + 1, ny3 + 2, nz3 + 2)
            Vy3 = rand(nx3 + 2, ny3 + 1, nz3 + 2)
            Vz3 = rand(nx3 + 2, ny3 + 2, nz3 + 1)
            Vxv, Vyv, Vzv = JustRelax2D.velocity2vertex(Vx3, Vy3, Vz3)
            @test size(Vxv) == (nx3, ny3, nz3)
            @test size(Vyv) == (nx3, ny3, nz3)
            @test size(Vzv) == (nx3, ny3, nz3)
            for k in 1:nz3, j in 1:ny3, i in 1:nx3
                @test Vxv[i, j, k] ≈ 0.25 * (Vx3[i, j, k] + Vx3[i, j + 1, k] + Vx3[i, j, k + 1] + Vx3[i, j + 1, k + 1])
            end
        end

        @testset "velocity2center! 3D" begin
            nx3, ny3, nz3 = 3, 3, 3
            Vx3 = rand(nx3 + 1, ny3 + 2, nz3 + 2)
            Vy3 = rand(nx3 + 2, ny3 + 1, nz3 + 2)
            Vz3 = rand(nx3 + 2, ny3 + 2, nz3 + 1)
            Vxc = zeros(nx3, ny3, nz3)
            Vyc = zeros(nx3, ny3, nz3)
            Vzc = zeros(nx3, ny3, nz3)
            velocity2center!(Vxc, Vyc, Vzc, Vx3, Vy3, Vz3)
            for k in 1:nz3, j in 1:ny3, i in 1:nx3
                @test Vxc[i, j, k] ≈ (Vx3[i, j + 1, k + 1] + Vx3[i + 1, j + 1, k + 1]) / 2
                @test Vyc[i, j, k] ≈ (Vy3[i + 1, j, k + 1] + Vy3[i + 1, j + 1, k + 1]) / 2
                @test Vzc[i, j, k] ≈ (Vz3[i + 1, j + 1, k] + Vz3[i + 1, j + 1, k + 1]) / 2
            end
        end

        @testset "center2vertex! 3D" begin
            nx3, ny3, nz3 = 3, 3, 3
            cyz = rand(nx3, ny3, nz3)
            cxz = rand(nx3, ny3, nz3)
            cxy = rand(nx3, ny3, nz3)
            vyz = zeros(nx3, ny3 + 1, nz3 + 1)
            vxz = zeros(nx3 + 1, ny3, nz3 + 1)
            vxy = zeros(nx3 + 1, ny3 + 1, nz3)
            center2vertex!(vyz, vxz, vxy, cyz, cxz, cxy)
            i, j, k = 1, 1, 1
            @test vyz[i, j + 1, k + 1] ≈ 0.25 * (cyz[i, j, k] + cyz[i, j + 1, k] + cyz[i, j, k + 1] + cyz[i, j + 1, k + 1])
            @test vxz[i + 1, j, k + 1] ≈ 0.25 * (cxz[i, j, k] + cxz[i + 1, j, k] + cxz[i, j, k + 1] + cxz[i + 1, j, k + 1])
            @test vxy[i + 1, j + 1, k] ≈ 0.25 * (cxy[i, j, k] + cxy[i + 1, j, k] + cxy[i, j + 1, k] + cxy[i + 1, j + 1, k])
        end
        @test true == true
    end
end

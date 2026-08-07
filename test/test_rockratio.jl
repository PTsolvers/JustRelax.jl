@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax3D as JR3

using ParallelStencil, ParallelStencil.FiniteDifferences2D

const env_backend = ENV["JULIA_JUSTRELAX_BACKEND"]

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

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

@testset "RockRatio" begin
    @testset "RockRatio 2D constructor" begin
        nx, ny = 5, 4
        ϕ = JustRelax2D.RockRatio(backend_JR, nx, ny)
        @test ϕ isa JustRelax.RockRatio
        @test size(ϕ.center) == (nx, ny)
        @test size(ϕ.vertex) == (nx + 1, ny + 1)
        @test size(ϕ.Vx) == (nx + 1, ny)
        @test size(ϕ.Vy) == (nx, ny + 1)
        # 2D RockRatio has dummy fields for Vz/yz/xz/xy (sized (1,1))
        @test size(ϕ.Vz) == (1, 1)
        @test size(ϕ.yz) == (1, 1)
        @test size(ϕ.xz) == (1, 1)
        @test size(ϕ.xy) == (1, 1)
        # zero-initialized
        @test all(iszero.(ϕ.center))
        @test all(iszero.(ϕ.vertex))
    end

    @testset "RockRatio 3D constructor" begin
        nx, ny, nz = 4, 3, 2
        ϕ = JR3.RockRatio(backend_JR, nx, ny, nz)
        @test ϕ isa JustRelax.RockRatio
        @test size(ϕ.center) == (nx, ny, nz)
        @test size(ϕ.vertex) == (nx + 1, ny + 1, nz + 1)
        @test size(ϕ.Vx) == (nx + 1, ny, nz)
        @test size(ϕ.Vy) == (nx, ny + 1, nz)
        @test size(ϕ.Vz) == (nx, ny, nz + 1)
        @test size(ϕ.yz) == (nx, ny + 1, nz + 1)
        @test size(ϕ.xz) == (nx + 1, ny, nz + 1)
        @test size(ϕ.xy) == (nx + 1, ny + 1, nz)
    end

    @testset "size_* accessors" begin
        nx, ny = 5, 4
        ϕ = JustRelax2D.RockRatio(backend_JR, nx, ny)
        @test JustRelax2D.size_c(ϕ) == size(ϕ.center)
        @test JustRelax2D.size_v(ϕ) == size(ϕ.vertex)
        @test JustRelax2D.size_vx(ϕ) == size(ϕ.Vx)
        @test JustRelax2D.size_vy(ϕ) == size(ϕ.Vy)
        @test JustRelax2D.size_vz(ϕ) == size(ϕ.Vz)
        @test JustRelax2D.size_yz(ϕ) == size(ϕ.yz)
        @test JustRelax2D.size_xz(ϕ) == size(ϕ.xz)
        @test JustRelax2D.size_xy(ϕ) == size(ϕ.xy)
    end

    @testset "isvalid / isvalid_vx / isvalid_vy" begin
        nx, ny = 4, 4
        if backend_JR == CPUBackend
            ϕ = JustRelax2D.RockRatio(backend_JR, nx, ny)

            # initially everything is zero ⇒ isvalid(ϕ, i, j) == false
            @test JustRelax2D.isvalid(ϕ.center, 1, 1) == false
            # set a non-zero value: isvalid returns true
            ϕ.center[2, 2] = 0.5
            @test JustRelax2D.isvalid(ϕ.center, 2, 2) == true
            @test JustRelax2D.isvalid(ϕ.center, 1, 1) == false

            # isvalid_vx / _vy / _vz are thin wrappers over isvalid
            ϕ.Vx[3, 1] = 0.4
            @test JustRelax2D.isvalid_vx(ϕ, 3, 1) == true
            @test JustRelax2D.isvalid_vx(ϕ, 1, 1) == false
            ϕ.Vy[1, 3] = 0.6
            @test JustRelax2D.isvalid_vy(ϕ, 1, 3) == true
            @test JustRelax2D.isvalid_vy(ϕ, 1, 1) == false

            # isvalid_v at (i, j) requires both Vx and Vy adjacencies AND a positive vertex
            ϕ.vertex[2, 2] = 0.3
            ϕ.Vx[2, 1] = 0.1; ϕ.Vx[2, 2] = 0.1     # ϕ.Vx[i, j_bot] and ϕ.Vx[i, j0]
            ϕ.Vy[1, 2] = 0.1; ϕ.Vy[2, 2] = 0.1     # ϕ.Vy[i_left, j] and ϕ.Vy[i0, j]
            @test JustRelax2D.isvalid_v(ϕ, 2, 2) == true
        else
            println("Skipping isvalid tests for backend $env_backend")
            @test true
        end
    end

    @testset "compute_rock_ratio / compute_air_ratio" begin
        # Build a 2-phase JustPIC.PhaseRatios where phase 2 is the "air" phase.
        # We can populate it via update_phase_ratios_2D! with synthetic phase arrays
        # so that compute_rock_ratio = 1 - air_phase_fraction (with threshold).
        nx, ny = 4, 4
        pr = JustPIC.PhaseRatios(backend_JP, 2, (nx, ny))
        xvi = (range(0.0, 1.0; length = nx + 1), range(0.0, 1.0; length = ny + 1))
        xci = (range(0.125, 0.875; length = nx), range(0.125, 0.875; length = ny))

        # phase 1 = rock (left half), phase 2 = air (right half)
        p_rock = @zeros(nx, ny); p_rock[1:2, :] .= 1.0
        p_air = @zeros(nx, ny); p_air[3:4, :] .= 1.0
        JustRelax2D.update_phase_ratios_2D!(pr, (p_rock, p_air), xci, xvi)

        air_phase = 2

        # rock-side cell: no air ⇒ rock-ratio = 1, air-ratio = 0
        if backend_JR == CPUBackend
            @test JustRelax2D.compute_rock_ratio(pr.center, air_phase, 1, 1) ≈ 1.0
            @test JustRelax2D.compute_air_ratio(pr.center, air_phase, 1, 1) ≈ 0.0
            # air-side cell: pure air ⇒ rock-ratio = 0, air-ratio = 1
            @test JustRelax2D.compute_rock_ratio(pr.center, air_phase, 4, 1) ≈ 0.0
            @test JustRelax2D.compute_air_ratio(pr.center, air_phase, 4, 1) ≈ 1.0

            # out-of-range air_phase index → both helpers return 1.0 (fallback)
            @test JustRelax2D.compute_rock_ratio(pr.center, 0, 1, 1) == 1.0
            @test JustRelax2D.compute_rock_ratio(pr.center, 99, 1, 1) == 1.0
            @test JustRelax2D.compute_air_ratio(pr.center, 0, 1, 1) == 1.0
            @test JustRelax2D.compute_air_ratio(pr.center, 99, 1, 1) == 1.0
        else
            println("Skipping compute_rock_ratio / compute_air_ratio tests for backend $env_backend")
            @test true
        end
    end

    @testset "Variational MiniKernels (masked)" begin
        # src/variational_stokes/MiniKernels.jl — masked versions of the MiniKernels
        # accessors and finite-difference / averaging helpers. Each masked variant
        # returns the corresponding plain value multiplied by the mask `ϕ`.
        A = reshape(collect(1.0:16.0), 4, 4)
        ϕ = fill(0.5, 4, 4)

        # Masked accessors (generated by the @eval loop)
        @test JustRelax2D.center(A, ϕ, 2, 2) ≈ A[2, 2] * ϕ[2, 2]
        @test JustRelax2D.right(A, ϕ, 2, 2) ≈ A[3, 2] * ϕ[3, 2]
        @test JustRelax2D.left(A, ϕ, 2, 2) ≈ A[1, 2] * ϕ[1, 2]
        @test JustRelax2D.front(A, ϕ, 2, 2) ≈ A[2, 3] * ϕ[2, 3]
        @test JustRelax2D.back(A, ϕ, 2, 2) ≈ A[2, 1] * ϕ[2, 1]
        @test JustRelax2D.next(A, ϕ, 2, 2) ≈ A[3, 3] * ϕ[3, 3]

        # Finite differences (only the dimensions whose `top`/`bot` masked
        # accessors actually exist are testable — see notes below)
        @test JustRelax2D._d_xa(A, ϕ, 1.0, 2, 2) ≈
            -JustRelax2D.center(A, ϕ, 2, 2) + JustRelax2D.right(A, ϕ, 2, 2)
        @test JustRelax2D._d_ya(A, ϕ, 1.0, 2, 2) ≈
            -JustRelax2D.center(A, ϕ, 2, 2) + JustRelax2D.front(A, ϕ, 2, 2)
        @test JustRelax2D._d_xi(A, ϕ, 1.0, 2, 2) ≈
            -JustRelax2D.front(A, ϕ, 2, 2) + JustRelax2D.next(A, ϕ, 2, 2)
        @test JustRelax2D._d_yi(A, ϕ, 1.0, 2, 2) ≈
            -JustRelax2D.right(A, ϕ, 2, 2) + JustRelax2D.next(A, ϕ, 2, 2)
        # spacing factor is applied
        @test JustRelax2D._d_xa(A, ϕ, 2.0, 2, 2) ≈
            2.0 * (-JustRelax2D.center(A, ϕ, 2, 2) + JustRelax2D.right(A, ϕ, 2, 2))

        # Directional averages (xa, ya)
        @test JustRelax2D._av_xa(A, ϕ, 2, 2) ≈ 0.5 * (A[2, 2] + A[3, 2]) * 0.5
        @test JustRelax2D._av_ya(A, ϕ, 2, 2) ≈ 0.5 * (A[2, 2] + A[2, 3]) * 0.5

        # mymaskedsum: 1D, 2D, 3D
        @test JustRelax2D.mymaskedsum(A, ϕ, 2:3, 2:3) ≈
            sum(A[i, j] * ϕ[i, j] for i in 2:3, j in 2:3)
        v = collect(1.0:5.0)
        m = fill(0.25, 5)
        @test JustRelax2D.mymaskedsum(v, m, 2:4) ≈ sum(v[i] * m[i] for i in 2:4)
        A3 = reshape(collect(1.0:64.0), 4, 4, 4)
        ϕ3 = fill(0.5, 4, 4, 4)
        @test JustRelax2D.mymaskedsum(A3, ϕ3, 2:3, 2:3, 2:3) ≈
            sum(A3[i, j, k] * ϕ3[i, j, k] for i in 2:3, j in 2:3, k in 2:3)
        # with a transforming f
        @test JustRelax2D.mymaskedsum(inv, A, ϕ, 2:3, 2:3) ≈
            sum(inv(A[i, j]) * ϕ[i, j] for i in 2:3, j in 2:3)

        # 4-cell averages (these previously routed to `mysum` with a signature
        # mismatch — now corrected to `mymaskedsum`).
        @test JustRelax2D._av(A, ϕ, 1, 1) ≈
            0.25 * sum(A[i, j] * ϕ[i, j] for i in 2:3, j in 2:3)
        @test JustRelax2D._av_a(A, ϕ, 1, 1) ≈
            0.25 * sum(A[i, j] * ϕ[i, j] for i in 1:2, j in 1:2)

        # interface averages (previously built a 2-tuple with `,` instead of `+`)
        @test JustRelax2D._av_xi(A, ϕ, 1, 1) ≈
            0.5 * (JustRelax2D.front(A, ϕ, 1, 1) + JustRelax2D.next(A, ϕ, 1, 1))
        @test JustRelax2D._av_yi(A, ϕ, 1, 1) ≈
            0.5 * (JustRelax2D.right(A, ϕ, 1, 1) + JustRelax2D.next(A, ϕ, 1, 1))

        # 3D masked accessors for the z-direction (added via @eval loop)
        A3 = reshape(collect(1.0:64.0), 4, 4, 4)
        ϕ3 = fill(0.5, 4, 4, 4)
        @test JustRelax2D.top(A3, ϕ3, 2, 2, 2) ≈ A3[2, 2, 3] * ϕ3[2, 2, 3]
        @test JustRelax2D.bot(A3, ϕ3, 2, 2, 2) ≈ A3[2, 2, 1] * ϕ3[2, 2, 1]

        # 3D finite differences in z (fixed from `front` to `top`, and `_d_zi`
        # now resolvable since `top` is masked)
        @test JustRelax2D._d_za(A3, ϕ3, 1.0, 2, 2, 2) ≈
            -JustRelax2D.center(A3, ϕ3, 2, 2, 2) + JustRelax2D.top(A3, ϕ3, 2, 2, 2)
        @test JustRelax2D._d_zi(A3, ϕ3, 1.0, 2, 2, 2) ≈
            -JustRelax2D.top(A3, ϕ3, 2, 2, 2) + JustRelax2D.next(A3, ϕ3, 2, 2, 2)

        # 3D averages along z and the now-summable xi/yi/zi variants
        @test JustRelax2D._av_za(A3, ϕ3, 2, 2, 2) ≈
            0.5 * (JustRelax2D.center(A3, ϕ3, 2, 2, 2) + JustRelax2D.top(A3, ϕ3, 2, 2, 2))
        @test JustRelax2D._av_zi(A3, ϕ3, 2, 2, 2) ≈
            0.5 * (JustRelax2D.top(A3, ϕ3, 2, 2, 2) + JustRelax2D.next(A3, ϕ3, 2, 2, 2))
        @test JustRelax2D._av_xi(A3, ϕ3, 2, 2, 2) ≈
            0.5 * (JustRelax2D.front(A3, ϕ3, 2, 2, 2) + JustRelax2D.next(A3, ϕ3, 2, 2, 2))
        @test JustRelax2D._av_yi(A3, ϕ3, 2, 2, 2) ≈
            0.5 * (JustRelax2D.right(A3, ϕ3, 2, 2, 2) + JustRelax2D.next(A3, ϕ3, 2, 2, 2))
    end

    @testset "update_rock_ratio! 2D" begin
        # End-to-end: PhaseRatios → update_rock_ratio! → ϕ.center matches the
        # expected formula `clamp(1 - air_fraction, 0, 1)`.
        nx, ny = 4, 4
        pr = JustPIC.PhaseRatios(backend_JP, 2, (nx, ny))
        xvi = (range(0.0, 1.0; length = nx + 1), range(0.0, 1.0; length = ny + 1))
        xci = (range(0.125, 0.875; length = nx), range(0.125, 0.875; length = ny))
        p_rock = @zeros(nx, ny); p_rock[1:2, :] .= 1.0
        p_air = @zeros(nx, ny); p_air[3:4, :] .= 1.0
        JustRelax2D.update_phase_ratios_2D!(pr, (p_rock, p_air), xci, xvi)

        ϕ = JustRelax2D.RockRatio(backend_JR, nx, ny)
        JustRelax2D.update_rock_ratio!(ϕ, pr, 2)            # air_phase = 2

        center_h = Base.Array(ϕ.center)
        # rock side fully one, air side fully zero
        @test center_h[1, 1] ≈ 1.0
        @test center_h[2, 2] ≈ 1.0
        @test center_h[3, 1] ≈ 0.0
        @test center_h[4, 2] ≈ 0.0
        # vertex / Vx / Vy all populated (sum across rock/air half ≈ nx·ny/2)
        @test sum(Base.Array(ϕ.center)) ≈ (nx * ny) / 2
    end
end

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

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

@testset "DYREL 3D" begin
    @testset "allocator" begin
        nx, ny, nz = 6, 5, 4
        velocity_sizes = ((nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1))
        dyrel = JustRelax3D.DYREL(
            backend_JR, (nx, ny, nz);
            ϵ = 1.0e-7,
            ϵ_vel = 2.0e-7,
            CFL = 0.6,
            c_fact = 0.3,
        )

        @test dyrel isa JustRelax.DYREL
        @test size(dyrel.γ_eff) == (nx, ny, nz)
        @test size(dyrel.ηb) == (nx, ny, nz)
        for fields in (
                (dyrel.Dx, dyrel.Dy, dyrel.Dz),
                (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
                (dyrel.dVxdτ, dyrel.dVydτ, dyrel.dVzdτ),
                (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
                (dyrel.dVx, dyrel.dVy, dyrel.dVz),
                (dyrel.βVx, dyrel.βVy, dyrel.βVz),
                (dyrel.cVx, dyrel.cVy, dyrel.cVz),
                (dyrel.αVx, dyrel.αVy, dyrel.αVz),
                (dyrel.Rx0, dyrel.Ry0, dyrel.Rz0),
            )
            @test size.(fields) == velocity_sizes
        end
        @test size(dyrel.P_num) == (nx, ny, nz)
        @test dyrel.CFL === 0.6
        @test dyrel.γfact === 20.0
        @test dyrel.ϵ === 1.0e-7
        @test dyrel.ϵ_vel === 2.0e-7
        @test dyrel.c_fact === 0.3
        @test all(A -> all(iszero, Array(A)), (
            dyrel.γ_eff, dyrel.ηb, dyrel.P_num,
            dyrel.Dx, dyrel.Dy, dyrel.Dz,
            dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz,
        ))

        dyrel_forwarded = JustRelax3D.DYREL(backend_JR, nx, ny, nz; CFL = 0.7)
        @test size(dyrel_forwarded.Dz) == velocity_sizes[3]
        @test dyrel_forwarded.CFL === 0.7
    end

    @testset "update_α_β!" begin
        ni = 5, 4, 3
        βV = ntuple(_ -> @zeros(ni...), Val(3))
        αV = ntuple(_ -> @zeros(ni...), Val(3))
        dτV = ntuple(_ -> @ones(ni...) .* 0.5, Val(3))
        cV = ntuple(_ -> @ones(ni...) .* 0.2, Val(3))

        JustRelax3D.update_α_β!(βV..., αV..., dτV..., cV...)

        expected_β = 2 * 0.5 / (2 + 0.2 * 0.5)
        expected_α = (2 - 0.2 * 0.5) / (2 + 0.2 * 0.5)
        @test all(A -> all(Array(A) .≈ expected_β), βV)
        @test all(A -> all(Array(A) .≈ expected_α), αV)
    end

    @testset "update_dτV_α_β!" begin
        ni = 5, 4, 3
        dτV = ntuple(_ -> @zeros(ni...), Val(3))
        βV = ntuple(_ -> @zeros(ni...), Val(3))
        αV = ntuple(_ -> @zeros(ni...), Val(3))
        cV = ntuple(_ -> @ones(ni...) .* 0.1, Val(3))
        λmaxV = ntuple(_ -> @ones(ni...) .* 4.0, Val(3))
        CFL = 0.9

        JustRelax3D.update_dτV_α_β!(dτV..., βV..., αV..., cV..., λmaxV..., CFL)

        expected_dτ = 2 / sqrt(4.0) * CFL
        expected_β = 2 * expected_dτ / (2 + 0.1 * expected_dτ)
        expected_α = (2 - 0.1 * expected_dτ) / (2 + 0.1 * expected_dτ)
        @test all(A -> all(Array(A) .≈ expected_dτ), dτV)
        @test all(A -> all(Array(A) .≈ expected_β), βV)
        @test all(A -> all(Array(A) .≈ expected_α), αV)
    end

    @testset "struct wrappers" begin
        dyrel = JustRelax3D.DYREL(backend_JR, (5, 4, 3); CFL = 0.8)
        dτV = (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz)
        βV = (dyrel.βVx, dyrel.βVy, dyrel.βVz)
        αV = (dyrel.αVx, dyrel.αVy, dyrel.αVz)
        cV = (dyrel.cVx, dyrel.cVy, dyrel.cVz)
        λmaxV = (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz)
        foreach(A -> A .= 0.5, dτV)
        foreach(A -> A .= 0.1, cV)

        JustRelax3D.update_α_β!(dyrel)
        expected_β = 2 * 0.5 / (2 + 0.1 * 0.5)
        expected_α = (2 - 0.1 * 0.5) / (2 + 0.1 * 0.5)
        @test all(A -> all(Array(A) .≈ expected_β), βV)
        @test all(A -> all(Array(A) .≈ expected_α), αV)

        foreach(A -> A .= 4.0, λmaxV)
        JustRelax3D.update_dτV_α_β!(dyrel)
        expected_dτ = 2 / sqrt(4.0) * dyrel.CFL
        @test all(A -> all(Array(A) .≈ expected_dτ), dτV)
    end
end

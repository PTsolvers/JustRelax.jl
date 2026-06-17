@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax3D as JR3
using JustPIC, JustPIC._2D

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
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

@testset "DYREL" begin
    @testset "DYREL 2D allocator" begin
        # NTuple constructor
        nx, ny = 8, 6
        dyrel = JustRelax2D.DYREL(backend_JR, (nx, ny); ϵ = 1.0e-7, ϵ_vel = 2.0e-7, CFL = 0.5, c_fact = 0.25)
        @test dyrel isa JustRelax.DYREL
        @test size(dyrel.γ_eff) == (nx, ny)
        @test size(dyrel.ηb) == (nx, ny)
        @test size(dyrel.Dx) == (nx - 1, ny)
        @test size(dyrel.Dy) == (nx, ny - 1)
        @test size(dyrel.Dz) == (1, 1)            # dummy slot for 2D
        @test size(dyrel.λmaxVx) == (nx - 1, ny)
        @test size(dyrel.λmaxVy) == (nx, ny - 1)
        @test size(dyrel.λmaxVz) == (1, 1)
        @test size(dyrel.dVxdτ) == (nx - 1, ny)
        @test size(dyrel.dVydτ) == (nx, ny - 1)
        @test size(dyrel.βVx) == (nx - 1, ny)
        @test size(dyrel.αVy) == (nx, ny - 1)
        @test length(dyrel.∂τc_∂ε) == 9
        @test length(dyrel.∂τv_∂ε) == 9
        @test length(dyrel.∂ΔPψc_∂ε) == 3
        @test length(dyrel.∂ΔPψc_∂η) == 3
        @test length(dyrel.∂τc_∂η) == 3
        @test length(dyrel.∂τv_∂η) == 3
        @test length(dyrel.∂ηc_∂ε) == 3
        @test length(dyrel.∂ηv_∂ε) == 3
        @test size(dyrel.∂τc_∂ε[1]) == (nx, ny)
        @test size(dyrel.∂τv_∂ε[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂ΔPψc_∂ε[1]) == (nx, ny)
        @test size(dyrel.∂ΔPψc_∂η[1]) == (nx, ny)
        @test size(dyrel.∂τc_∂η[1]) == (nx, ny)
        @test size(dyrel.∂τv_∂η[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂ηc_∂ε[1]) == (nx, ny)
        @test size(dyrel.∂ηv_∂ε[1]) == (nx + 1, ny + 1)
        @test dyrel.CFL === 0.5
        @test dyrel.ϵ === 1.0e-7
        @test dyrel.ϵ_vel === 2.0e-7
        @test dyrel.c_fact === 0.25
        @test all(iszero.(dyrel.γ_eff))
        @test all(iszero.(dyrel.Dx)) && all(iszero.(dyrel.Dy))
        @test all(iszero.(dyrel.λmaxVx)) && all(iszero.(dyrel.λmaxVy))
    end

    @testset "DYREL 3D allocator" begin
        nx, ny, nz = 6, 5, 4
        dyrel = JR3.DYREL(backend_JR, (nx, ny, nz); ϵ = 1.0e-7, ϵ_vel = 2.0e-7, CFL = 0.6, c_fact = 0.3)
        @test dyrel isa JustRelax.DYREL
        @test size(dyrel.γ_eff) == (nx, ny, nz)
        @test size(dyrel.ηb) == (nx, ny, nz)
        @test size(dyrel.Dx) == (nx - 1, ny, nz)
        @test size(dyrel.Dy) == (nx, ny - 1, nz)
        @test size(dyrel.Dz) == (nx, ny, nz - 1)
        @test size(dyrel.λmaxVx) == (nx - 1, ny, nz)
        @test size(dyrel.λmaxVy) == (nx, ny - 1, nz)
        @test size(dyrel.λmaxVz) == (nx, ny, nz - 1)
        @test size(dyrel.dVxdτ) == (nx - 1, ny, nz)
        @test size(dyrel.dVydτ) == (nx, ny - 1, nz)
        @test size(dyrel.dVzdτ) == (nx, ny, nz - 1)
        @test size(dyrel.βVx) == (nx - 1, ny, nz)
        @test size(dyrel.αVy) == (nx, ny - 1, nz)
        @test size(dyrel.cVz) == (nx, ny, nz - 1)
        @test length(dyrel.∂τc_∂ε) == 1
        @test length(dyrel.∂τv_∂ε) == 1
        @test length(dyrel.∂ΔPψc_∂ε) == 1
        @test length(dyrel.∂ΔPψc_∂η) == 1
        @test length(dyrel.∂τc_∂η) == 1
        @test length(dyrel.∂τv_∂η) == 1
        @test length(dyrel.∂ηc_∂ε) == 1
        @test length(dyrel.∂ηv_∂ε) == 1
        @test dyrel.CFL === 0.6
        @test dyrel.ϵ === 1.0e-7
        @test dyrel.ϵ_vel === 2.0e-7
        @test dyrel.c_fact === 0.3
        @test all(iszero.(dyrel.γ_eff))
        @test all(iszero.(dyrel.Dz)) && all(iszero.(dyrel.λmaxVz))

        # 3-int forwarder
        dyrel2 = JR3.DYREL(backend_JR, nx, ny, nz; CFL = 0.7)
        @test size(dyrel2.Dz) == (nx, ny, nz - 1)
        @test dyrel2.CFL === 0.7
    end

    @testset "update_α_β! 2D" begin
        # Build a tiny pair of 2D arrays sized as DYREL expects: (nx-1, ny) for x-velocity
        # diagnostics, (nx, ny-1) for y-velocity diagnostics. The kernel loops over
        # `size(βV[1]) .+ (1, 0)` so we make βV[1] size (nx, ny) for simpler bookkeeping.
        nx, ny = 5, 4
        βVx = @zeros(nx, ny)
        βVy = @zeros(nx, ny)
        αVx = @zeros(nx, ny)
        αVy = @zeros(nx, ny)
        # filled inputs
        dτVx = @ones(nx, ny) .* 0.5         # dτ = 0.5
        dτVy = @ones(nx, ny) .* 0.5
        cVx = @ones(nx, ny) .* 0.2          # c = 0.2
        cVy = @ones(nx, ny) .* 0.2

        JustRelax2D.update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)

        # Expected per the kernel: β = 2dτ / (2 + c·dτ), α = (2 - c·dτ) / (2 + c·dτ)
        expected_β = 2 * 0.5 / (2 + 0.2 * 0.5)
        expected_α = (2 - 0.2 * 0.5) / (2 + 0.2 * 0.5)
        @test all(βVx .≈ expected_β)
        @test all(αVx .≈ expected_α)
        @test all(βVy .≈ expected_β)
        @test all(αVy .≈ expected_α)
    end

    @testset "update_dτV_α_β! 2D" begin
        nx, ny = 5, 4
        βVx = @zeros(nx, ny); βVy = @zeros(nx, ny)
        αVx = @zeros(nx, ny); αVy = @zeros(nx, ny)
        dτVx = @zeros(nx, ny); dτVy = @zeros(nx, ny)
        # λmax > 0 (otherwise sqrt produces NaN)
        λmaxVx = @ones(nx, ny) .* 4.0       # √λmax = 2 ⇒ dτ = 2/2 * CFL = CFL
        λmaxVy = @ones(nx, ny) .* 4.0
        cVx = @ones(nx, ny) .* 0.1
        cVy = @ones(nx, ny) .* 0.1
        CFL = 0.9

        JustRelax2D.update_dτV_α_β!(
            dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL,
        )

        expected_dτ = 2 / sqrt(4.0) * CFL    # = CFL
        expected_β = 2 * expected_dτ / (2 + 0.1 * expected_dτ)
        expected_α = (2 - 0.1 * expected_dτ) / (2 + 0.1 * expected_dτ)
        @test all(dτVx .≈ expected_dτ)
        @test all(dτVy .≈ expected_dτ)
        @test all(βVx .≈ expected_β)
        @test all(αVx .≈ expected_α)
        @test all(βVy .≈ expected_β)
        @test all(αVy .≈ expected_α)
    end

    @testset "DYREL struct wrappers" begin
        # update_α_β!(dyrel) and update_dτV_α_β!(dyrel) drive the kernels off the
        # DYREL fields directly.
        nx, ny = 5, 4
        dyrel = JustRelax2D.DYREL(backend_JR, (nx, ny); CFL = 0.8)
        dyrel.dτVx .= 0.5
        dyrel.dτVy .= 0.5
        dyrel.cVx .= 0.1
        dyrel.cVy .= 0.1

        JustRelax2D.update_α_β!(dyrel)
        expected_β = 2 * 0.5 / (2 + 0.1 * 0.5)
        @test all(dyrel.βVx .≈ expected_β)
        @test all(dyrel.βVy .≈ expected_β)

        # Reset and use the full dτV variant
        dyrel.dτVx .= 0.0; dyrel.dτVy .= 0.0
        dyrel.βVx .= 0.0; dyrel.βVy .= 0.0
        dyrel.αVx .= 0.0; dyrel.αVy .= 0.0
        dyrel.λmaxVx .= 4.0
        dyrel.λmaxVy .= 4.0
        JustRelax2D.update_dτV_α_β!(dyrel)

        expected_dτ = 2 / sqrt(4.0) * 0.8
        @test all(dyrel.dτVx .≈ expected_dτ)
        @test all(dyrel.dτVy .≈ expected_dτ)
    end

    @testset "GershgorinAD chain rule helpers" begin
        ∂τ_∂ε = ntuple(i -> fill(Float64(i), 1, 1), 9)
        ∂τ_∂η = (fill(10.0, 1, 1), fill(20.0, 1, 1), fill(30.0, 1, 1))
        ∂η_∂ε = (fill(0.5, 1, 1), fill(-0.25, 1, 1), fill(2.0, 1, 1))
        dε = (3.0, -4.0, 0.25)

        dτ_dε = 4.0 * dε[1] + 5.0 * dε[2] + 6.0 * dε[3]
        dη_dV = 0.5 * dε[1] - 0.25 * dε[2] + 2.0 * dε[3]
        @test JustRelax2D.dτ_dV(∂τ_∂ε, 2, 1, 1, dε...) ≈ dτ_dε
        @test JustRelax2D.dτ_dV(∂τ_∂ε, ∂τ_∂η, ∂η_∂ε, 2, 1, 1, dε...) ≈ dτ_dε + 20.0 * dη_dV

        ∂ΔPψ_∂ε = (fill(7.0, 1, 1), fill(8.0, 1, 1), fill(9.0, 1, 1))
        ∂ΔPψ_∂η = (fill(-3.0, 1, 1), fill(0.0, 1, 1), fill(0.0, 1, 1))
        dΔPψ_dε = 7.0 * dε[1] + 8.0 * dε[2] + 9.0 * dε[3]
        @test JustRelax2D.dΔPψ_dV(∂ΔPψ_∂ε, 1, 1, dε...) ≈ dΔPψ_dε
        @test JustRelax2D.dΔPψ_dV(∂ΔPψ_∂ε, ∂ΔPψ_∂η, ∂η_∂ε, 1, 1, dε...) ≈ dΔPψ_dε - 3.0 * dη_dV
    end

    @testset "GershgorinAD local Rx-Vx entry" begin
        dyrel = JustRelax2D.DYREL(CPUBackend, (3, 3))
        dyrel.γ_eff .= 2.0

        dyrel.∂τc_∂ε[1] .= 1.0
        dyrel.∂τc_∂ε[2] .= 2.0
        dyrel.∂τc_∂ε[3] .= 4.0
        dyrel.∂τc_∂η[1] .= 3.0
        dyrel.∂ηc_∂ε[1] .= 5.0
        dyrel.∂ηc_∂ε[2] .= 7.0
        dyrel.∂ηc_∂ε[3] .= 11.0

        dyrel.∂τv_∂ε[7] .= 13.0
        dyrel.∂τv_∂ε[8] .= 17.0
        dyrel.∂τv_∂ε[9] .= 19.0
        dyrel.∂τv_∂η[3] .= 23.0
        dyrel.∂ηv_∂ε[1] .= 29.0
        dyrel.∂ηv_∂ε[2] .= 31.0
        dyrel.∂ηv_∂ε[3] .= 37.0

        dyrel.∂ΔPψc_∂ε[1] .= 41.0
        dyrel.∂ΔPψc_∂ε[2] .= 43.0
        dyrel.∂ΔPψc_∂ε[3] .= 47.0
        dyrel.∂ΔPψc_∂η[1] .= 53.0

        jacobian_entry, gershgorin_entry = JustRelax2D.local_Rx_Vx_gershgorin_entry(
            dyrel,
            1,
            1,
            5,
            (2.0, 3.0),
            (5.0, 7.0),
            (11.0, 13.0),
            size(dyrel.γ_eff),
        )

        @test jacobian_entry ≈ -77870.0
        @test gershgorin_entry ≈ 80590.0
    end

    @testset "DYREL partial field storage" begin
        nx, ny = 4, 3
        ni = (nx, ny)
        xvi = (range(0.0, 1.0; length = nx + 1), range(0.0, 1.0; length = ny + 1))
        xci = (range(0.125, 0.875; length = nx), range(0.125, 0.875; length = ny))

        visc = GeoParams.LinearViscous(; η = 10.0)
        rheology = (
            GeoParams.SetMaterialParams(;
                Phase = 1,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((visc,)),
            ),
        )
        phase_ratios = JustPIC._2D.PhaseRatios(backend_JP, length(rheology), ni)
        JustRelax2D.update_phase_ratios_2D!(phase_ratios, (@ones(ni...),), xci, xvi)

        stokes = StokesArrays(backend_JR, ni)
        stokes.ε.xx .= 1.0
        stokes.ε.yy .= -0.5
        stokes.ε.xy .= 0.25
        stokes.ε.xy_c .= 0.25
        stokes.ε.xx_v .= 1.0
        stokes.ε.yy_v .= -0.5
        stokes.viscosity.η .= 10.0
        stokes.viscosity.ηv .= 10.0

        dyrel = JustRelax2D.DYREL(backend_JR, ni)
        foreach(A -> fill!(A, NaN), dyrel.∂τc_∂η)
        foreach(A -> fill!(A, NaN), dyrel.∂τv_∂η)
        foreach(A -> fill!(A, NaN), dyrel.∂ΔPψc_∂η)
        JustRelax2D.compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, 1.0, Inf, true)

        expected_∂τ_∂η = (2.0, -1.0, 0.5)
        @test all(dyrel.∂τc_∂η[1] .≈ expected_∂τ_∂η[1])
        @test all(dyrel.∂τc_∂η[2] .≈ expected_∂τ_∂η[2])
        @test all(dyrel.∂τc_∂η[3] .≈ expected_∂τ_∂η[3])
        @test all(dyrel.∂τv_∂η[1] .≈ expected_∂τ_∂η[1])
        @test all(dyrel.∂τv_∂η[2] .≈ expected_∂τ_∂η[2])
        @test all(dyrel.∂τv_∂η[3] .≈ expected_∂τ_∂η[3])
        @test all(iszero, dyrel.∂ΔPψc_∂η[1])

        pow = GeoParams.PowerlawViscous(; η0 = 10.0, n = 3, ε0 = 1.0)
        rheology_powerlaw = (
            GeoParams.SetMaterialParams(;
                Phase = 1,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((pow,)),
            ),
        )
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
        foreach(A -> fill!(A, NaN), dyrel.∂ηc_∂ε)
        foreach(A -> fill!(A, NaN), dyrel.∂ηv_∂ε)
        JustRelax2D.update_viscosity_εII!(
            stokes,
            phase_ratios,
            args,
            rheology_powerlaw,
            (-Inf, Inf);
            do_partials = true,
            ∂η_∂ε = (dyrel.∂ηc_∂ε, dyrel.∂ηv_∂ε),
        )

        expected_∂η_∂ε = (5 * (2 * 1.0 - 0.5), 5 * (2 * -0.5 + 1.0), 5 * (2 * 0.25))
        @test all(dyrel.∂ηc_∂ε[1] .≈ expected_∂η_∂ε[1])
        @test all(dyrel.∂ηc_∂ε[2] .≈ expected_∂η_∂ε[2])
        @test all(dyrel.∂ηc_∂ε[3] .≈ expected_∂η_∂ε[3])
        @test all(dyrel.∂ηv_∂ε[1] .≈ expected_∂η_∂ε[1])
        @test all(dyrel.∂ηv_∂ε[2] .≈ expected_∂η_∂ε[2])
        @test all(dyrel.∂ηv_∂ε[3] .≈ expected_∂η_∂ε[3])
    end

end

const JR_BACKEND = get(ENV, "JULIA_JUSTRELAX_BACKEND", "Threads")

@static if JR_BACKEND == "AMDGPU"
    using AMDGPU
elseif JR_BACKEND == "CUDA"
    using CUDA
end
using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax3D as JR3
using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

using ParallelStencil, ParallelStencil.FiniteDifferences2D
const backend_JR = @static if JR_BACKEND == "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif JR_BACKEND == "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end
const backend_JP = @static if JR_BACKEND == "AMDGPU"
    JustPIC.AMDGPUBackend
elseif JR_BACKEND == "CUDA"
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
        @test length(dyrel.∂εxx_∂Vx) == 2
        @test length(dyrel.∂εyy_∂Vx) == 2
        @test length(dyrel.∂∇V_∂Vx) == 2
        @test length(dyrel.∂εxx_∂Vy) == 2
        @test length(dyrel.∂εyy_∂Vy) == 2
        @test length(dyrel.∂∇V_∂Vy) == 2
        @test length(dyrel.∂εxy_∂Vx) == 2
        @test length(dyrel.∂εxy_∂Vy) == 2
        @test length(dyrel.∂Rx_∂τxx) == 2
        @test length(dyrel.∂Rx_∂P_num) == 2
        @test length(dyrel.∂Ry_∂τyy) == 2
        @test length(dyrel.∂Ry_∂P_num) == 2
        @test size(dyrel.∂τc_∂ε[1]) == (nx, ny)
        @test size(dyrel.∂τv_∂ε[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂ΔPψc_∂ε[1]) == (nx, ny)
        @test size(dyrel.∂ΔPψc_∂η[1]) == (nx, ny)
        @test size(dyrel.∂τc_∂η[1]) == (nx, ny)
        @test size(dyrel.∂τv_∂η[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂ηc_∂ε[1]) == (nx, ny)
        @test size(dyrel.∂ηv_∂ε[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂εxx_∂Vx[1]) == (nx, ny)
        @test size(dyrel.∂εyy_∂Vx[1]) == (nx, ny)
        @test size(dyrel.∂∇V_∂Vx[1]) == (nx, ny)
        @test size(dyrel.∂εxx_∂Vy[1]) == (nx, ny)
        @test size(dyrel.∂εyy_∂Vy[1]) == (nx, ny)
        @test size(dyrel.∂∇V_∂Vy[1]) == (nx, ny)
        @test size(dyrel.∂εxy_∂Vx[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂εxy_∂Vy[1]) == (nx + 1, ny + 1)
        @test size(dyrel.∂Rx_∂τxx[1]) == (nx - 1, ny)
        @test size(dyrel.∂Rx_∂P_num[1]) == (nx - 1, ny)
        @test size(dyrel.∂Ry_∂τyy[1]) == (nx, ny - 1)
        @test size(dyrel.∂Ry_∂P_num[1]) == (nx, ny - 1)
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
        @test length(dyrel.∂εxx_∂Vx) == 1
        @test length(dyrel.∂εyy_∂Vx) == 1
        @test length(dyrel.∂∇V_∂Vx) == 1
        @test length(dyrel.∂εxx_∂Vy) == 1
        @test length(dyrel.∂εyy_∂Vy) == 1
        @test length(dyrel.∂∇V_∂Vy) == 1
        @test length(dyrel.∂εxy_∂Vx) == 1
        @test length(dyrel.∂εxy_∂Vy) == 1
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

    @testset "local strain-rate partial storage" begin
        ni = 3, 3
        grid = Geometry(ni, (1.0, 1.0))
        stokes = StokesArrays(backend_JR, ni)
        dyrel = JustRelax2D.DYREL(backend_JR, ni)

        JustRelax2D.compute_local_strain_rates!(stokes, dyrel, grid, true)

        @test dyrel.∂εxx_∂Vx[1][2, 2] != 0
        @test dyrel.∂εxx_∂Vx[1][2, 2] ≈ -dyrel.∂εxx_∂Vx[2][2, 2]
        @test dyrel.∂εyy_∂Vx[1][2, 2] ≈ -dyrel.∂εyy_∂Vx[2][2, 2]
        @test dyrel.∂∇V_∂Vx[1][2, 2] ≈ -dyrel.∂∇V_∂Vx[2][2, 2]
        @test dyrel.∂εxx_∂Vy[1][2, 2] ≈ -dyrel.∂εxx_∂Vy[2][2, 2]
        @test dyrel.∂εyy_∂Vy[1][2, 2] != 0
        @test dyrel.∂εyy_∂Vy[1][2, 2] ≈ -dyrel.∂εyy_∂Vy[2][2, 2]
        @test dyrel.∂∇V_∂Vy[1][2, 2] ≈ -dyrel.∂∇V_∂Vy[2][2, 2]
        @test dyrel.∂εxy_∂Vx[1][2, 2] != 0
        @test dyrel.∂εxy_∂Vx[1][2, 2] ≈ -dyrel.∂εxy_∂Vx[2][2, 2]
        @test dyrel.∂εxy_∂Vy[1][2, 2] != 0
        @test dyrel.∂εxy_∂Vy[1][2, 2] ≈ -dyrel.∂εxy_∂Vy[2][2, 2]
    end

    @testset "GershgorinAD ∂Rx∂Vx center entry" begin
        dyrel = JustRelax2D.DYREL(backend_JR, (4, 4))
        i, j, m = 2, 2, 3

        dyrel.∂εxx_∂Vx[2][2, 2] = 2.0
        dyrel.∂∇V_∂Vx[2][2, 2] = 3.0
        dyrel.∂εxx_∂Vx[1][3, 2] = 7.0
        dyrel.∂∇V_∂Vx[1][3, 2] = 5.0
        dyrel.∂εxy_∂Vx[2][3, 2] = 11.0
        dyrel.∂εxy_∂Vx[1][3, 3] = 13.0

        dyrel.∂τc_∂ε[1] .= 1.0
        dyrel.∂τv_∂ε[9] .= 1.0
        dyrel.∂ΔPψc_∂ε[1] .= 1.0
        dyrel.γ_eff[2, 2] = 10.0
        dyrel.γ_eff[3, 2] = 20.0

        dyrel.∂Rx_∂τxx[1][i, j] = 101.0
        dyrel.∂Rx_∂τxx[2][i, j] = 103.0
        dyrel.∂Rx_∂τxy[1][i, j] = 107.0
        dyrel.∂Rx_∂τxy[2][i, j] = 109.0
        dyrel.∂Rx_∂P_num[1][i, j] = 113.0
        dyrel.∂Rx_∂P_num[2][i, j] = 127.0

        expected =
            101.0 * 2.0 +
            103.0 * 7.0 +
            107.0 * 11.0 +
            109.0 * 13.0 +
            113.0 * (-10.0 * 3.0) +
            127.0 * (-20.0 * 5.0)

        @test JustRelax2D.∂Rx∂Vx(dyrel, i, j, m) ≈ expected
    end

    @testset "GershgorinAD matches analytical DYREL parameters" begin
        ni = 4, 4
        grid = Geometry(ni, (1.0, 1.0))
        dt = 1.0

        rheology = (
            GeoParams.SetMaterialParams(;
                Phase = 1,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 10.0),)),
            ),
        )

        phase_ratios = JustPIC._2D.PhaseRatios(backend_JP, length(rheology), ni)
        @parallel_indices (i, j) function init_single_phase_dyrel_compare!(phases)
            @index phases[1, i, j] = 1.0
            return nothing
        end
        @parallel (@idx ni) init_single_phase_dyrel_compare!(phase_ratios.center)
        @parallel (@idx ni .+ 1) init_single_phase_dyrel_compare!(phase_ratios.vertex)

        stokes = StokesArrays(backend_JR, ni)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        stokes.ε.xx .= 1.0
        stokes.ε.xx_v .= 1.0
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

        stokes.V.Vx .= PTArray(backend_JR)([
            0.01 * i - 0.02 * j for i in 1:size(stokes.V.Vx, 1), j in 1:size(stokes.V.Vx, 2)
        ])
        stokes.V.Vy .= PTArray(backend_JR)([
            -0.03 * i + 0.04 * j for i in 1:size(stokes.V.Vy, 1), j in 1:size(stokes.V.Vy, 2)
        ])

        dyrel_analytic = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99)
        dyrel_ad = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99)

        ρg = @zeros(ni...), @zeros(ni...)
        P_num = similar(stokes.P)

        JustRelax2D.compute_local_strain_rates!(stokes, dyrel_ad, grid, true)
        JustRelax2D.compute_stress_DRYEL!(stokes, dyrel_ad, rheology, phase_ratios, 1.0, dt, true)
        JustRelax2D.compute_residual_P!(
            stokes.R.RP,
            stokes.P,
            stokes.P0,
            stokes.∇V,
            stokes.Q,
            dyrel_ad.ηb,
            rheology,
            phase_ratios,
            dt,
            args,
        )
        @. P_num = dyrel_ad.γ_eff * stokes.R.RP
        @parallel (@idx ni) JustRelax2D.compute_DR_residual_V!(
            @residuals(stokes.R)...,
            dyrel_ad,
            stokes.P,
            P_num,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            dyrel_ad.Dx,
            dyrel_ad.Dy,
            grid._di.center,
            grid._di.vertex,
            true,
        )
        JustRelax2D.Gershgorin_Stokes2D_SchurComplementAD(
            dyrel_ad,
            grid._di.center,
            grid._di.vertex,
            grid._di.velocity[1],
            grid._di.velocity[2],
        )
        JustRelax2D.update_dτV_α_β!(dyrel_ad)

        for field in (:Dx, :Dy, :λmaxVx, :λmaxVy, :dτVx, :dτVy, :βVx, :βVy, :αVx, :αVy)
            @test Array(getfield(dyrel_ad, field)) ≈ Array(getfield(dyrel_analytic, field)) rtol = 1.0e-10
        end
    end

    @testset "DYREL local residual helpers" begin
        τn = (2.0, 5.0)
        τs = (-1.0, 3.0)
        P = (7.0, 11.0)
        Pnum = (0.5, 2.5)
        ΔPψ = (1.0, 4.0)
        ρg = (6.0, 10.0)
        _dn = 0.25
        _ds = 0.5

        expected = (τn[2] - τn[1]) * _dn + (τs[2] - τs[1]) * _ds -
            (P[2] - P[1]) * _dn - (ΔPψ[2] - ΔPψ[1]) * _dn - 0.5 * (ρg[1] + ρg[2])
        @test JustRelax2D.local_Rx_residual(τn, τs, P, ΔPψ, ρg, _dn, _ds) ≈ expected
        @test JustRelax2D.local_Ry_residual(τn, τs, P, ΔPψ, ρg, _dn, _ds) ≈ expected
        @test JustRelax2D.local_DR_Rx_residual(τn, τs, P, Pnum, ΔPψ, ρg, _dn, _ds, 2.0) ≈
            (expected - (Pnum[2] - Pnum[1]) * _dn) / 2.0
        @test JustRelax2D.local_DR_Ry_residual(τn, τs, P, Pnum, ΔPψ, ρg, _dn, _ds, 2.0) ≈
            (expected - (Pnum[2] - Pnum[1]) * _dn) / 2.0

        ∂DRy = JustRelax2D.local_DR_Ry_residual_partials(collect(τn), collect(τs), collect(P), collect(Pnum), collect(ΔPψ), collect(ρg), _dn, _ds, 2.0)
        @test ∂DRy.τyy ≈ [-_dn, _dn]
        @test ∂DRy.P_num ≈ [_dn, -_dn]
    end

    # @testset "GershgorinAD local Rx-Vx entry" begin
    #     dyrel = JustRelax2D.DYREL(CPUBackend, (3, 3))
    #     dyrel.γ_eff .= 2.0

    #     dyrel.∂τc_∂ε[1] .= 1.0
    #     dyrel.∂τc_∂ε[2] .= 2.0
    #     dyrel.∂τc_∂ε[3] .= 4.0
    #     dyrel.∂τc_∂η[1] .= 3.0
    #     dyrel.∂ηc_∂ε[1] .= 5.0
    #     dyrel.∂ηc_∂ε[2] .= 7.0
    #     dyrel.∂ηc_∂ε[3] .= 11.0

    #     dyrel.∂τv_∂ε[7] .= 13.0
    #     dyrel.∂τv_∂ε[8] .= 17.0
    #     dyrel.∂τv_∂ε[9] .= 19.0
    #     dyrel.∂τv_∂η[3] .= 23.0
    #     dyrel.∂ηv_∂ε[1] .= 29.0
    #     dyrel.∂ηv_∂ε[2] .= 31.0
    #     dyrel.∂ηv_∂ε[3] .= 37.0

    #     dyrel.∂ΔPψc_∂ε[1] .= 41.0
    #     dyrel.∂ΔPψc_∂ε[2] .= 43.0
    #     dyrel.∂ΔPψc_∂ε[3] .= 47.0
    #     dyrel.∂ΔPψc_∂η[1] .= 53.0

    #     jacobian_entry = JustRelax2D.local_Rx_Vx_gershgorin_entry(
    #         dyrel,
    #         1,
    #         1,
    #         5,
    #         (2.0, 3.0),
    #         (5.0, 7.0),
    #         (11.0, 13.0),
    #         size(dyrel.γ_eff),
    #     )

    #     @test jacobian_entry ≈ -1509.0
    #     @test abs(jacobian_entry) ≈ 1509.0
    # end

    # @testset "DYREL partial field storage" begin
    #     nx, ny = 4, 3
    #     ni = (nx, ny)
    #     xvi = (range(0.0, 1.0; length = nx + 1), range(0.0, 1.0; length = ny + 1))
    #     xci = (range(0.125, 0.875; length = nx), range(0.125, 0.875; length = ny))

    #     visc = GeoParams.LinearViscous(; η = 10.0)
    #     rheology = (
    #         GeoParams.SetMaterialParams(;
    #             Phase = 1,
    #             Density = GeoParams.ConstantDensity(; ρ = 0.0),
    #             CompositeRheology = GeoParams.CompositeRheology((visc,)),
    #         ),
    #     )
    #     phase_ratios = JustPIC._2D.PhaseRatios(backend_JP, length(rheology), ni)
    #     JustRelax2D.update_phase_ratios_2D!(phase_ratios, (@ones(ni...),), xci, xvi)

    #     stokes = StokesArrays(backend_JR, ni)
    #     stokes.ε.xx .= 1.0
    #     stokes.ε.yy .= -0.5
    #     stokes.ε.xy .= 0.25
    #     stokes.ε.xy_c .= 0.25
    #     stokes.ε.xx_v .= 1.0
    #     stokes.ε.yy_v .= -0.5
    #     stokes.viscosity.η .= 10.0
    #     stokes.viscosity.ηv .= 10.0

    #     dyrel = JustRelax2D.DYREL(backend_JR, ni)
    #     foreach(A -> fill!(A, NaN), dyrel.∂τc_∂η)
    #     foreach(A -> fill!(A, NaN), dyrel.∂τv_∂η)
    #     foreach(A -> fill!(A, NaN), dyrel.∂ΔPψc_∂η)
    #     JustRelax2D.compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, 1.0, Inf, true)

    #     expected_∂τ_∂η = (2.0, -1.0, 0.5)
    #     @test all(dyrel.∂τc_∂η[1] .≈ expected_∂τ_∂η[1])
    #     @test all(dyrel.∂τc_∂η[2] .≈ expected_∂τ_∂η[2])
    #     @test all(dyrel.∂τc_∂η[3] .≈ expected_∂τ_∂η[3])
    #     @test all(dyrel.∂τv_∂η[1] .≈ expected_∂τ_∂η[1])
    #     @test all(dyrel.∂τv_∂η[2] .≈ expected_∂τ_∂η[2])
    #     @test all(dyrel.∂τv_∂η[3] .≈ expected_∂τ_∂η[3])
    #     @test all(iszero, dyrel.∂ΔPψc_∂η[1])

    #     pow = GeoParams.PowerlawViscous(; η0 = 10.0, n = 3, ε0 = 1.0)
    #     rheology_powerlaw = (
    #         GeoParams.SetMaterialParams(;
    #             Phase = 1,
    #             Density = GeoParams.ConstantDensity(; ρ = 0.0),
    #             CompositeRheology = GeoParams.CompositeRheology((pow,)),
    #         ),
    #     )
    #     args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    #     foreach(A -> fill!(A, NaN), dyrel.∂ηc_∂ε)
    #     foreach(A -> fill!(A, NaN), dyrel.∂ηv_∂ε)
    #     JustRelax2D.update_viscosity_εII!(
    #         stokes,
    #         phase_ratios,
    #         args,
    #         rheology_powerlaw,
    #         (-Inf, Inf);
    #         do_partials = true,
    #         ∂η_∂ε = (dyrel.∂ηc_∂ε, dyrel.∂ηv_∂ε),
    #     )

    #     expected_∂η_∂ε = (5 * (2 * 1.0 - 0.5), 5 * (2 * -0.5 + 1.0), 5 * (2 * 0.25))
    #     @test all(dyrel.∂ηc_∂ε[1] .≈ expected_∂η_∂ε[1])
    #     @test all(dyrel.∂ηc_∂ε[2] .≈ expected_∂η_∂ε[2])
    #     @test all(dyrel.∂ηc_∂ε[3] .≈ expected_∂η_∂ε[3])
    #     @test all(dyrel.∂ηv_∂ε[1] .≈ expected_∂η_∂ε[1])
    #     @test all(dyrel.∂ηv_∂ε[2] .≈ expected_∂η_∂ε[2])
    #     @test all(dyrel.∂ηv_∂ε[3] .≈ expected_∂η_∂ε[3])
    # end

    #=
    @testset "GershgorinAD linear forward finite difference" begin
        ly = 1.0e0
        lx = ly
        ni = 4, 4
        grid = Geometry(ni, (lx, ly); origin = (-lx / 2, -ly / 2))
        (; xci, xvi) = grid
        dt = 1.0
        εbg = 1.0e-2
        igg = IGG(init_global_grid(ni[1], ni[2], 1; init_MPI = !JustRelax.MPI.Initialized())...)


        # Physical properties using GeoParams ----------------
        visc_bg  = GeoParams.LinearViscous(; η = 1.0e2)
        visc_inc = GeoParams.LinearViscous(; η = 1.0e-1)

        rheology = (
            GeoParams.SetMaterialParams(;
                Phase = 1,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((visc_bg,)),
            ),
            GeoParams.SetMaterialParams(;
                Phase = 2,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology(( visc_inc,)),
            ),
        )

        phase_ratios = JustPIC._2D.PhaseRatios(backend_JP, length(rheology), ni)
        circle = GGU.Circle((0.0, 0.0), 0.1)

        @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
            p = GGU.Point(xc[i], yc[j])
            if GGU.inside(p, circle)
                @index phases[1, i, j] = 0.0
                @index phases[2, i, j] = 1.0
            else
                @index phases[1, i, j] = 1.0
                @index phases[2, i, j] = 0.0
            end
            return nothing
        end
        @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
        @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)

        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true),
            no_slip = (left = false, right = false, top = false, bot = false),
        )

        function run_once(; perturb = false, h = 1.0e-6, h0 = 1.0e-6)
            stokes = StokesArrays(backend_JR, ni)
            ρg = @zeros(ni...), @zeros(ni...)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

            stokes.ε.xx .= 1
            stokes.ε.xx_v .= 1
            compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

            stokes.V.Vx .= PTArray(backend_JR)([εbg * x for x in xvi[1], _ in 1:(ni[2] + 2)])
            stokes.V.Vy .= PTArray(backend_JR)([-εbg * y for _ in 1:(ni[1] + 2), y in xvi[2]])
            flow_bcs!(stokes, flow_bcs)

            δVx = h0 + (perturb ? h : 0.0)
            stokes.V.Vx[3, 3] += δVx
            vx12_start = Float64(εbg) * Float64(xvi[1][3]) + δVx

            dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99, ϵ = 1.0e-6)
            solve_DYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 0,
                    nout = 1,
                    rel_drop = 1.0e-5,
                    λ_relaxation_DR = 1,
                    λ_relaxation_PH = 1,
                    viscosity_relaxation = 1.0e-1,
                    viscosity_cutoff = (-Inf, Inf),
                    use_gershgorin_ad = true,
                    total_iterMax = 0,
                )
            )

            return Array(stokes.R.Rx), dyrel, vx12_start, Float64(dyrel.Dx[2, 2])
        end

        Rx, dyrel, Vx12_start, Dx11_start = run_once()
        RxP, _, Vx12P_start, Dx11P_start = run_once(; perturb = true)

        fd_jac_unscaled = (RxP[2, 2] * Dx11P_start - Rx[2, 2] * Dx11_start) / (Vx12P_start - Vx12_start)
        local_jac = JustRelax2D.local_Rx_Vx_gershgorin_entry(
            dyrel,
            2,
            2,
            5,
            grid._di.center,
            grid._di.vertex,
            grid._di.velocity[1],
            size(dyrel.γ_eff),
        )

        @test fd_jac_unscaled ≈ local_jac rtol = 1.0e-6
        finalize_global_grid(; finalize_MPI = false)
    end

     @testset "GershgorinAD nonlinear forward finite difference" begin
        ly = 1.0e0
        lx = ly
        ni = 4, 4
        grid = Geometry(ni, (lx, ly); origin = (-lx / 2, -ly / 2))
        (; xci, xvi) = grid
        dt = 1.0
        εbg = 1.0e-2
        igg = IGG(init_global_grid(ni[1], ni[2], 1; init_MPI = !JustRelax.MPI.Initialized())...)


        # Physical properties using GeoParams ----------------
        visc_bg  = PowerlawViscous(; η0 = 1.0e2, n = 3, ε0 = 1.0e0)
        visc_inc = PowerlawViscous(; η0 = 1.0e-1, n = 3, ε0 = 1.0e0)

        rheology = (
            GeoParams.SetMaterialParams(;
                Phase = 1,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((visc_bg,)),
            ),
            GeoParams.SetMaterialParams(;
                Phase = 2,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology(( visc_inc,)),
            ),
        )

        phase_ratios = JustPIC._2D.PhaseRatios(backend_JP, length(rheology), ni)
        circle = GGU.Circle((0.0, 0.0), 0.1)

        @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
            p = GGU.Point(xc[i], yc[j])
            if GGU.inside(p, circle)
                @index phases[1, i, j] = 0.0
                @index phases[2, i, j] = 1.0
            else
                @index phases[1, i, j] = 1.0
                @index phases[2, i, j] = 0.0
            end
            return nothing
        end
        @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
        @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)

        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true),
            no_slip = (left = false, right = false, top = false, bot = false),
        )

        function run_once(; perturb = false, h = 1.0e-6, h0 = 1.0e-6)
            stokes = StokesArrays(backend_JR, ni)
            ρg = @zeros(ni...), @zeros(ni...)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

            stokes.ε.xx .= 1
            stokes.ε.xx_v .= 1
            compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

            stokes.V.Vx .= PTArray(backend_JR)([εbg * x for x in xvi[1], _ in 1:(ni[2] + 2)])
            stokes.V.Vy .= PTArray(backend_JR)([-εbg * y for _ in 1:(ni[1] + 2), y in xvi[2]])
            flow_bcs!(stokes, flow_bcs)

            δVx = h0 + (perturb ? h : 0.0)
            stokes.V.Vx[3, 3] += δVx
            vx12_start = Float64(εbg) * Float64(xvi[1][3]) + δVx

            dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99, ϵ = 1.0e-6)
            solve_DYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 0,
                    nout = 1,
                    rel_drop = 1.0e-5,
                    λ_relaxation_DR = 1,
                    λ_relaxation_PH = 1,
                    viscosity_relaxation = 1.0,
                    viscosity_cutoff = (-Inf, Inf),
                    use_gershgorin_ad = true,
                    total_iterMax = 0,
                )
            )

            return Array(stokes.R.Rx), dyrel, vx12_start, Float64(dyrel.Dx[2, 2])
        end

        Rx, dyrel, Vx12_start, Dx11_start = run_once()
        RxP, _, Vx12P_start, Dx11P_start = run_once(; perturb = true)

        fd_jac_unscaled = (RxP[2, 2] * Dx11P_start - Rx[2, 2] * Dx11_start) / (Vx12P_start - Vx12_start)
        local_jac = JustRelax2D.local_Rx_Vx_gershgorin_entry(
            dyrel,
            2,
            2,
            5,
            grid._di.center,
            grid._di.vertex,
            grid._di.velocity[1],
            size(dyrel.γ_eff),
        )

        @test fd_jac_unscaled ≈ local_jac rtol = 1.0e-6
        finalize_global_grid(; finalize_MPI = false)
    end

    @testset "GershgorinAD viscoelastoplastic forward finite difference" begin
        ly = 1.0e0
        lx = ly
        ni = 4, 4
        grid = Geometry(ni, (lx, ly); origin = (0.0, 0.0))
        (; xci, xvi) = grid
        τ_y = 1.6
        ϕ = 30.0
        η0 = 1.0
        G0 = 1.0
        Gi = G0 / 2
        εbg = 1.0
        η_reg = 1.0e-2
        dt = η0 / G0 / 6.0
        igg = IGG(init_global_grid(ni[1], ni[2], 1; init_MPI = !JustRelax.MPI.Initialized())...)

        el_bg = GeoParams.ConstantElasticity(; G = G0, Kb = 5.0)
        el_inc = GeoParams.ConstantElasticity(; G = Gi, Kb = 5.0)
        visc = GeoParams.LinearViscous(; η = η0)
        pl = GeoParams.DruckerPrager_regularised(;
            C = τ_y / cosd(ϕ),
            ϕ = ϕ,
            η_vp = η_reg,
            Ψ = 10.0,
        )

        rheology = (
            GeoParams.SetMaterialParams(;
                Phase = 1,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((visc, el_bg, pl)),
                Elasticity = el_bg,
            ),
            GeoParams.SetMaterialParams(;
                Phase = 2,
                Density = GeoParams.ConstantDensity(; ρ = 0.0),
                Gravity = GeoParams.ConstantGravity(; g = 0.0),
                CompositeRheology = GeoParams.CompositeRheology((visc, el_inc, pl)),
                Elasticity = el_inc,
            ),
        )

        phase_ratios = JustPIC._2D.PhaseRatios(backend_JP, length(rheology), ni)
        circle = GGU.Circle((0.5, 0.5), 0.1)

        @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
            p = GGU.Point(xc[i], yc[j])
            if GGU.inside(p, circle)
                @index phases[1, i, j] = 0.0
                @index phases[2, i, j] = 1.0
            else
                @index phases[1, i, j] = 1.0
                @index phases[2, i, j] = 0.0
            end
            return nothing
        end
        @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
        @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)

        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true),
            no_slip = (left = false, right = false, top = false, bot = false),
        )

        function run_once(; perturb = false, h = 1.0e-6, h0 = 1.0e-6)
            stokes = StokesArrays(backend_JR, ni)
            ρg = @zeros(ni...), @zeros(ni...)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

            compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

            stokes.V.Vx .= PTArray(backend_JR)([εbg * x for x in xvi[1], _ in 1:(ni[2] + 2)])
            stokes.V.Vy .= PTArray(backend_JR)([-εbg * y for _ in 1:(ni[1] + 2), y in xvi[2]])
            @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0.0
            @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0.0
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99, ϵ = 1.0e-6)

            t = 0.0
            for _ in 1:10
                solve_DYREL!(
                    stokes,
                    ρg,
                    dyrel,
                    flow_bcs,
                    phase_ratios,
                    rheology,
                    args,
                    grid,
                    dt,
                    igg;
                    kwargs = (;
                        verbose_PH = false,
                        verbose_DR = false,
                        iterMax = 50.0e3,
                        nout = 10,
                        rel_drop = 1.0e-2,
                        λ_relaxation_DR = 1,
                        λ_relaxation_PH = 1,
                        viscosity_relaxation = 1.0,
                        linear_viscosity = true,
                        viscosity_cutoff = (-Inf, Inf),
                        use_gershgorin_ad = true,
                    )
                )
                tensor_invariant!(stokes.τ)
                tensor_invariant!(stokes.ε)
                tensor_invariant!(stokes.ε_pl)
                t += dt
            end

            δVx = h0 + (perturb ? h : 0.0)
            stokes.V.Vx[3, 3] += δVx
            vx12_start = Float64(stokes.V.Vx[3, 3])

            solve_DYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 0,
                    nout = 1,
                    rel_drop = 1.0e-5,
                    λ_relaxation_DR = 1,
                    λ_relaxation_PH = 1,
                    viscosity_relaxation = 1.0,
                    linear_viscosity = true,
                    viscosity_cutoff = (-Inf, Inf),
                    use_gershgorin_ad = true,
                    total_iterMax = 0,
                )
            )

            dx11_start = Float64(dyrel.Dx[2, 2])
            tensor_invariant!(stokes.ε_pl)
            return Array(stokes.R.Rx), dyrel, vx12_start, dx11_start, stokes
        end

        Rx, dyrel, Vx12_start, Dx11_start, pl = run_once()
        RxP, _, Vx12P_start, Dx11P_start, plP = run_once(; perturb = true)

        fd_jac_unscaled = (RxP[2, 2] * Dx11P_start - Rx[2, 2] * Dx11_start) / (Vx12P_start - Vx12_start)
        local_jac = JustRelax2D.local_Rx_Vx_gershgorin_entry(
            dyrel,
            2,
            2,
            5,
            grid._di.center,
            grid._di.vertex,
            grid._di.velocity[1],
            size(dyrel.γ_eff),
        )

        @test pl.ε_pl.II[2,2] > 0.0
        @test pl.ε_pl.II[2,2] > 0.0
        @test fd_jac_unscaled ≈ local_jac rtol = 1.0e-6
        finalize_global_grid(; finalize_MPI = false)
    end
=#

end

# using CairoMakie
# heatmap(pl.ε_pl.II)

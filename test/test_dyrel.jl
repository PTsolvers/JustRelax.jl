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
        @test size(dyrel.∂τxxc_∂εxx) == (nx, ny)
        @test size(dyrel.∂τyyc_∂εyy) == (nx, ny)
        @test size(dyrel.∂τxyv_∂εxy) == (nx + 1, ny + 1)
        @test size(dyrel.P_num) == (nx, ny)
        @test size(dyrel.Rx0) == (nx - 1, ny)
        @test size(dyrel.Ry0) == (nx, ny - 1)
        @test size(dyrel.Rz0) == (1, 1)
        @test dyrel.CFL === 0.5
        @test dyrel.ϵ === 1.0e-7
        @test dyrel.ϵ_vel === 2.0e-7
        @test dyrel.c_fact === 0.25
        @test all(iszero.(dyrel.γ_eff))
        @test all(iszero.(dyrel.P_num))
        @test all(iszero.(dyrel.Dx)) && all(iszero.(dyrel.Dy))
        @test all(iszero.(dyrel.λmaxVx)) && all(iszero.(dyrel.λmaxVy))

        for field in (
                :∂εxx_∂Vx, :∂εyy_∂Vx, :∂∇V_∂Vx, :∂εxx_∂Vy, :∂εyy_∂Vy,
                :∂∇V_∂Vy, :∂εxy_∂Vx, :∂εxy_∂Vy, :∂Rx_∂τxx, :∂Rx_∂P_num,
                :∂Ry_∂τyy, :∂Ry_∂P_num, :∂τxxv_∂εxx,
            )
            @test !hasproperty(dyrel, field)
        end
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
        @test size(dyrel.∂τxxc_∂εxx) == (1, 1, 1)
        @test size(dyrel.∂τyyc_∂εyy) == (1, 1, 1)
        @test size(dyrel.∂τxyv_∂εxy) == (1, 1, 1)
        @test size(dyrel.P_num) == (nx, ny, nz)
        @test size(dyrel.Rx0) == (nx - 1, ny, nz)
        @test size(dyrel.Ry0) == (nx, ny - 1, nz)
        @test size(dyrel.Rz0) == (nx, ny, nz - 1)
        @test dyrel.CFL === 0.6
        @test dyrel.ϵ === 1.0e-7
        @test dyrel.ϵ_vel === 2.0e-7
        @test dyrel.c_fact === 0.3
        @test all(iszero.(dyrel.γ_eff))
        @test all(iszero.(dyrel.P_num))
        @test all(iszero.(dyrel.Dz)) && all(iszero.(dyrel.λmaxVz))

        for field in (
                :∂εxx_∂Vx, :∂εyy_∂Vx, :∂∇V_∂Vx, :∂εxx_∂Vy, :∂εyy_∂Vy,
                :∂∇V_∂Vy, :∂εxy_∂Vx, :∂εxy_∂Vy,
            )
            @test !hasproperty(dyrel, field)
        end

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

    @testset "GershgorinAD ∂Rx∂Vx center entry" begin
        dyrel = JustRelax2D.DYREL(backend_JR, (4, 4))
        grid = Geometry((4, 4), (1.0, 1.0))
        i, j, m = 2, 2, 3

        dyrel.∂τxxc_∂εxx[2, 2] = 1.0
        dyrel.∂τxxc_∂εxx[3, 2] = 2.0
        dyrel.∂τxyv_∂εxy[3, 2] = 3.0
        dyrel.∂τxyv_∂εxy[3, 3] = 4.0
        dyrel.γ_eff[2, 2] = 5.0
        dyrel.γ_eff[3, 2] = 6.0

        dτxx = JustRelax2D.∂Rx_∂τxx(grid._di.center, i)
        dτxy = JustRelax2D.∂Rx_∂τxy(grid._di.vertex, j)
        dPnum = JustRelax2D.∂Rx_∂Pnum(grid._di.center, i)
        dεxx_E, _, d∇V_E = JustRelax2D.∂normal_∂Vx(grid._di.vertex, i + 1, j)
        dεxx_W, _, d∇V_W = JustRelax2D.∂normal_∂Vx(grid._di.vertex, i, j)

        expected =
            -dτxx * dyrel.∂τxxc_∂εxx[i + 1, j] * dεxx_E -
            dτxx * dyrel.∂τxxc_∂εxx[i, j] * dεxx_W -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * JustRelax2D.∂shear_∂Vx(grid._di.velocity[1], j + 1) -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j] * JustRelax2D.∂shear_∂Vx(grid._di.velocity[1], j) -
            dPnum * (dyrel.γ_eff[i + 1, j] * d∇V_E) -
            dPnum * (dyrel.γ_eff[i, j] * d∇V_W)

        @test JustRelax2D.∂Rx∂Vx(dyrel, grid._di.center, grid._di.vertex, grid._di.velocity[1], i, j, m) ≈ expected
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

        stokes.V.Vx .= PTArray(backend_JR)(
            [
                0.01 * i - 0.02 * j for i in 1:size(stokes.V.Vx, 1), j in 1:size(stokes.V.Vx, 2)
            ]
        )
        stokes.V.Vy .= PTArray(backend_JR)(
            [
                -0.03 * i + 0.04 * j for i in 1:size(stokes.V.Vy, 1), j in 1:size(stokes.V.Vy, 2)
            ]
        )

        dyrel_analytic = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99)
        dyrel_ad = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; CFL = 0.99)

        ρg = @zeros(ni...), @zeros(ni...)
        P_num = similar(stokes.P)

        JustRelax2D.compute_∇V_strain_rate_RP!(stokes, dyrel_ad, rheology, phase_ratios, grid._di, ni, dt, args)
        JustRelax2D.compute_stress_DRYEL!(stokes, dyrel_ad, rheology, phase_ratios, 1.0, dt, true)
        @. P_num = dyrel_ad.γ_eff * stokes.R.RP
        @parallel (@idx ni) JustRelax2D.compute_DR_residual_V!(
            @residuals(stokes.R)...,
            stokes.P,
            P_num,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            dyrel_ad.Dx,
            dyrel_ad.Dy,
            grid._di.center,
            grid._di.vertex,
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

end

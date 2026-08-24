push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil
using StaticArrays

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

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

const JR3K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D
else
    JustRelax3D
end

@parallel_indices (i, j, k) function _init_single_phase_3D!(phases)
    @index phases[1, i, j, k] = 1.0
    return nothing
end

@testset "DYREL 3D kernels" begin
    ni = 4, 3, 5
    dt = 1.0

    @testset "pure helpers" begin
        P, P0, ∇V, Q, ηb, local_dt = 3.0, 1.0, 0.5, 0.2, 4.0, 0.25
        @test JR3K._compute_RP!(P, P0, ∇V, Q, ηb, local_dt) ≈ -∇V - (P - P0) / ηb + Q / local_dt

        ΔT, α = 10.0, 3.0e-5
        @test JR3K._compute_RP!(P, P0, ∇V, Q, ΔT, α, ηb, local_dt) ≈
            -∇V - (P - P0) / ηb + α * ΔT / local_dt + Q / local_dt

        dVdτ, R, a, b, dτ = 2.0, 0.5, 0.9, 0.8, 0.3
        dVdτ_new, ΔV = JR3K.damped_update_V(dVdτ, R, a, b, dτ)
        @test dVdτ_new ≈ a * dVdτ + R
        @test ΔV ≈ dVdτ_new * b * dτ
    end

    @testset "compute_∇V_strain_rate! 3D" begin
        local_ni = 6, 5, 4
        grid = Geometry(local_ni, (1.0, 1.0, 1.0))
        (; xvi) = grid
        stokes = StokesArrays(backend_JR, local_ni)
        a, b, c = 2.0, -0.7, 0.4
        nx, ny, nz = local_ni
        stokes.V.Vx .= PTArray(backend_JR)([a * x for x in xvi[1], _ in 1:(ny + 2), _ in 1:(nz + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([b * y for _ in 1:(nx + 2), y in xvi[2], _ in 1:(nz + 2)])
        stokes.V.Vz .= PTArray(backend_JR)([c * z for _ in 1:(nx + 2), _ in 1:(ny + 2), z in xvi[3]])

        JR3K.compute_∇V_strain_rate!(stokes, grid._di, local_ni, Val(3))

        div = a + b + c
        @test all(Array(stokes.∇V) .≈ div)
        @test all(Array(stokes.ε.xx) .≈ a - div / 3)
        @test all(Array(stokes.ε.yy) .≈ b - div / 3)
        @test all(Array(stokes.ε.zz) .≈ c - div / 3)
        @test all(A -> all(abs.(Array(A)) .< 1.0e-12), (stokes.ε.yz, stokes.ε.xz, stokes.ε.xy))
    end

    @testset "fused kernels 3D" begin
        local_ni = 6, 5, 4
        nx, ny, nz = local_ni
        grid = Geometry(local_ni, (1.0, 1.0, 1.0))
        (; xvi) = grid
        local_dt = 1.0
        elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
        local_rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = 1.0),
                Gravity = ConstantGravity(; g = 1.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
                Elasticity = elasticity,
            ),
        )
        local_phases = PhaseRatios(backend_JP, 1, local_ni)
        for ratios in (
                local_phases.center, local_phases.vertex,
                local_phases.yz, local_phases.xz, local_phases.xy,
            )
            @parallel (@idx size(ratios)) _init_single_phase_3D!(ratios)
        end

        stokes = StokesArrays(backend_JR, local_ni)
        args = (; T = @zeros(local_ni .+ 2...), P = stokes.P, dt = local_dt)
        compute_viscosity!(stokes, local_phases, args, local_rheology, (-Inf, Inf))

        a, b, c = 1.3, -0.4, 0.2
        stokes.V.Vx .= PTArray(backend_JR)([a * x for x in xvi[1], _ in 1:(ny + 2), _ in 1:(nz + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([b * y for _ in 1:(nx + 2), y in xvi[2], _ in 1:(nz + 2)])
        stokes.V.Vz .= PTArray(backend_JR)([c * z for _ in 1:(nx + 2), _ in 1:(ny + 2), z in xvi[3]])

        dyrel = JustRelax3D.DYREL(backend_JR, stokes, local_rheology, local_phases, grid.di, local_dt; γfact = 37.0)
        γ_eff_before = copy(dyrel.γ_eff)
        JR3K.DYREL!(dyrel, stokes, local_rheology, local_phases, grid.di, local_dt)
        @test dyrel.γfact === 37.0
        @test Array(dyrel.γ_eff) == Array(γ_eff_before)
        stokes.P0 .= stokes.P
        stokes.Q .= 0.0
        JR3K.compute_∇V_strain_rate_RP!(stokes, dyrel, local_rheology, local_phases, grid._di, local_ni, local_dt, args)
        @test all(Array(stokes.R.RP) .≈ -(a + b + c))
        @test all(Array(stokes.ε.xx) .≈ a - (a + b + c) / 3)

        θc = copy(dyrel.P_num)
        JR3K.compute_stress_viscosity_DRYEL!(
            stokes, θc, dyrel.γ_eff, local_rheology, local_phases,
            1.0, local_dt, 1.0, args, (-Inf, Inf), false,
        )
        @test all(isfinite, Array(stokes.viscosity.η))
        @test all(>(0), Array(stokes.viscosity.η))
        @test Array(θc) ≈ Array(dyrel.γ_eff) .* Array(stokes.R.RP) .+ Array(stokes.ΔPψ)

        ρg = ntuple(_ -> @zeros(local_ni...), Val(3))
        @parallel (@idx local_ni) JR3K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.R.Rz,
            stokes.P, stokes.ΔPψ, @stress(stokes)..., ρg...,
            grid._di.center, grid._di.vertex,
        )
        @test all(A -> all(isfinite, Array(A)), (stokes.R.Rx, stokes.R.Ry, stokes.R.Rz))

        dyrel.Dx .= 1.0
        dyrel.Dy .= 1.0
        dyrel.Dz .= 1.0
        dyrel.βVx .= 0.0
        dyrel.βVy .= 0.0
        dyrel.βVz .= 0.0
        dyrel.αVx .= 0.0
        dyrel.αVy .= 0.0
        dyrel.αVz .= 0.0
        dyrel.dτVx .= 1.0
        dyrel.dτVy .= 1.0
        dyrel.dτVz .= 1.0
        V_before = map(copy, (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz))
        @parallel (@idx local_ni) JR3K.compute_DR_residual_update_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.R.Rz,
            stokes.V.Vx, stokes.V.Vy, stokes.V.Vz,
            dyrel.dVxdτ, dyrel.dVydτ, dyrel.dVzdτ,
            stokes.P, θc, @stress(stokes)..., ρg...,
            dyrel.Dx, dyrel.Dy, dyrel.Dz,
            dyrel.αVx, dyrel.αVy, dyrel.αVz,
            dyrel.βVx, dyrel.βVy, dyrel.βVz,
            dyrel.dτVx, dyrel.dτVy, dyrel.dτVz,
            grid._di.center, grid._di.vertex,
        )
        @test all(A -> all(isfinite, Array(A)), (stokes.R.Rx, stokes.R.Ry, stokes.R.Rz))
        @test all(i -> Array((stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)[i]) == Array(V_before[i]), 1:3)
    end

    elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
            Elasticity = elasticity,
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    for ratios in (
            phase_ratios.center, phase_ratios.vertex,
            phase_ratios.yz, phase_ratios.xz, phase_ratios.xy,
        )
        @parallel (@idx size(ratios)) _init_single_phase_3D!(ratios)
    end

    stokes = StokesArrays(backend_JR, ni)
    stokes.viscosity.η .= 1.0
    stokes.ε.yz .= 1.0
    stokes.ε.xz .= 2.0
    stokes.ε.xy .= 3.0
    stokes.ε.yz_c .= NaN
    stokes.ε.xz_c .= NaN
    stokes.ε.xy_c .= NaN

    θc = @zeros(ni...)
    γ_eff = @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    JR3K.compute_stress_viscosity_DRYEL!(
        stokes, θc, γ_eff, rheology, phase_ratios,
        1.0, dt, 1.0, args, (-Inf, Inf), true,
    )

    # η_ve = ηGdt/(η + Gdt) = 1/2, hence τij = 2η_ve εij = εij.
    # In particular, the physical boundary shear stresses must not be zeroed.
    @test Array(stokes.τ.yz) ≈ ones(size(stokes.τ.yz))
    @test Array(stokes.τ.xz) ≈ fill(2.0, size(stokes.τ.xz))
    @test Array(stokes.τ.xy) ≈ fill(3.0, size(stokes.τ.xy))
    @test Array(stokes.τ.yz_c) ≈ ones(ni...)
    @test Array(stokes.τ.xz_c) ≈ fill(2.0, ni...)
    @test Array(stokes.τ.xy_c) ≈ fill(3.0, ni...)
    @test Array(stokes.τ.II) ≈ fill(sqrt(14.0), ni...)
    @test all(A -> all(isnan, Array(A)), (stokes.ε.yz_c, stokes.ε.xz_c, stokes.ε.xy_c))

    stokes.τ.yz .= 0.0
    stokes.τ.xz .= 0.0
    stokes.τ.xy .= 0.0
    JR3K.compute_stress_DRYEL!(stokes, rheology, phase_ratios, 1.0, dt)

    @test Array(stokes.τ.yz) ≈ ones(size(stokes.τ.yz))
    @test Array(stokes.τ.xz) ≈ fill(2.0, size(stokes.τ.xz))
    @test Array(stokes.τ.xy) ≈ fill(3.0, size(stokes.τ.xy))
    @test Array(stokes.τ.yz_c) ≈ ones(ni...)
    @test Array(stokes.τ.xz_c) ≈ fill(2.0, ni...)
    @test Array(stokes.τ.xy_c) ≈ fill(3.0, ni...)
    @test Array(stokes.τ.II) ≈ fill(sqrt(14.0), ni...)
    @test all(A -> all(isnan, Array(A)), (stokes.ε.yz_c, stokes.ε.xz_c, stokes.ε.xy_c))

    shear2center!(stokes.ε)
    @test Array(stokes.ε.yz_c) ≈ ones(ni...)
    @test Array(stokes.ε.xz_c) ≈ fill(2.0, ni...)
    @test Array(stokes.ε.xy_c) ≈ fill(3.0, ni...)

    @testset "viscoelastoplastic stress" begin
        plasticity = DruckerPrager_regularised(;
            C = 0.1,
            ϕ = 30.0,
            η_vp = 1.0e-2,
            Ψ = 10.0,
        )
        plastic_rheology = (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity, plasticity)),
                Elasticity = elasticity,
            ),
        )
        out = JR3K.compute_local_stress(
            (1.0, -0.5, -0.5, 0.2, 0.3, 0.4),
            ntuple(_ -> 0.0, Val(6)),
            1.0,
            0.0,
            0.0,
            1.0,
            plastic_rheology,
            SVector(1.0),
            dt,
            0.0,
        )

        @test all(isfinite, out)
        @test out[14] > 0
        @test any(!iszero, out[7:12])
        @test !iszero(out[15])
        @test out[17] > 0

        incompressible_elasticity = ConstantElasticity(; G = 1.0, ν = 0.5)
        nondilatant_rheology = (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((
                    LinearViscous(; η = 1.0),
                    incompressible_elasticity,
                    DruckerPrager_regularised(; C = 0.1, ϕ = 30.0, η_vp = 1.0e-2, Ψ = 0.0),
                )),
                Elasticity = incompressible_elasticity,
            ),
        )
        out = JR3K.compute_local_stress(
            (1.0, -0.5, -0.5, 0.2, 0.3, 0.4),
            ntuple(_ -> 0.0, Val(6)),
            1.0,
            0.0,
            0.0,
            1.0,
            nondilatant_rheology,
            SVector(1.0),
            dt,
            0.0,
        )

        @test all(isfinite, out)
        @test out[14] > 0
        @test any(!iszero, out[7:12])
        @test iszero(out[15])
        @test iszero(out[17])
    end

    @testset "Gershgorin shear staggering" begin
        η_host = [10.0^mod(i + 2j + 3k, 4) for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3]]
        γ_host = [isodd(i + j + k) ? 0.01 : 1000.0 for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3]]
        copyto!(stokes.viscosity.η, η_host)
        grid = Geometry(ni, Float64.(ni))
        dyrel = JustRelax3D.DYREL(backend_JR, ni; CFL = 0.99)
        copyto!(dyrel.γ_eff, γ_host)
        gersh_rheology = (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            ),
        )
        JR3K.Gershgorin_Stokes3D_SchurComplement!(
            dyrel.Dx, dyrel.Dy, dyrel.Dz,
            dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz,
            stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff,
            phase_ratios, gersh_rheology, grid.di, dt,
        )

        phases = map(x -> Array(x), (phase_ratios.center, phase_ratios.yz, phase_ratios.xz, phase_ratios.xy))
        center, yz, xz, xy = phases
        ηc(i, j, k) = JustRelax.JustRelax3D._ηve_center(η_host, center, gersh_rheology, dt, i, j, k)
        ηyz(i, j, k) = JustRelax.JustRelax3D._ηve_yz(η_host, yz, gersh_rheology, dt, ni, i, j, k)
        ηxz(i, j, k) = JustRelax.JustRelax3D._ηve_xz(η_host, xz, gersh_rheology, dt, ni, i, j, k)
        ηxy(i, j, k) = JustRelax.JustRelax3D._ηve_xy(η_host, xy, gersh_rheology, dt, ni, i, j, k)
        i, j, k = 2, 2, 3

        ηW, ηE = ηc(i, j, k), ηc(i + 1, j, k)
        ηS, ηN = ηxy(i + 1, j, k), ηxy(i + 1, j + 1, k)
        ηB, ηF = ηxz(i + 1, j, k), ηxz(i + 1, j, k + 1)
        γW, γE = γ_host[i, j, k], γ_host[i + 1, j, k]
        expected_Dx = ηN + ηS + ηB + ηF + γE + γW + 4 / 3 * (ηE + ηW)
        expected_Cx = abs(γE + 4 / 3 * ηE) + abs(γW + 4 / 3 * ηW) +
            abs(ηN) + abs(ηS) + abs(ηB) + abs(ηF) +
            abs(γE - 2 / 3 * ηE + ηN) + abs(γE - 2 / 3 * ηE + ηS) +
            abs(γW + ηN - 2 / 3 * ηW) + abs(γW + ηS - 2 / 3 * ηW) +
            abs(γE + ηB - 2 / 3 * ηE) + abs(γW + ηB - 2 / 3 * ηW) +
            abs(γE - 2 / 3 * ηE + ηF) + abs(γW + ηF - 2 / 3 * ηW) + abs(expected_Dx)

        ηW, ηE = ηxy(i, j + 1, k), ηxy(i + 1, j + 1, k)
        ηS, ηN = ηc(i, j, k), ηc(i, j + 1, k)
        ηB, ηF = ηyz(i, j + 1, k), ηyz(i, j + 1, k + 1)
        γS, γN = γ_host[i, j, k], γ_host[i, j + 1, k]
        expected_Dy = ηE + ηW + ηB + ηF + γN + γS + 4 / 3 * (ηN + ηS)
        expected_Cy = abs(ηE) + abs(ηW) + abs(γN + 4 / 3 * ηN) + abs(γS + 4 / 3 * ηS) +
            abs(ηB) + abs(ηF) +
            abs(γN + ηE - 2 / 3 * ηN) + abs(γS + ηE - 2 / 3 * ηS) +
            abs(γN - 2 / 3 * ηN + ηW) + abs(γS - 2 / 3 * ηS + ηW) +
            abs(γN + ηB - 2 / 3 * ηN) + abs(γS + ηB - 2 / 3 * ηS) +
            abs(γN + ηF - 2 / 3 * ηN) + abs(γS + ηF - 2 / 3 * ηS) + abs(expected_Dy)

        ηW, ηE = ηxz(i, j, k + 1), ηxz(i + 1, j, k + 1)
        ηS, ηN = ηyz(i, j, k + 1), ηyz(i, j + 1, k + 1)
        ηB, ηF = ηc(i, j, k), ηc(i, j, k + 1)
        γB, γF = γ_host[i, j, k], γ_host[i, j, k + 1]
        expected_Dz = ηE + ηW + ηN + ηS + γB + γF + 4 / 3 * (ηB + ηF)
        expected_Cz = abs(ηE) + abs(ηW) + abs(ηN) + abs(ηS) +
            abs(γB + 4 / 3 * ηB) + abs(γF + 4 / 3 * ηF) +
            abs(γB - 2 / 3 * ηB + ηE) + abs(γB - 2 / 3 * ηB + ηW) +
            abs(γF + ηE - 2 / 3 * ηF) + abs(γF - 2 / 3 * ηF + ηW) +
            abs(γB - 2 / 3 * ηB + ηN) + abs(γB - 2 / 3 * ηB + ηS) +
            abs(γF - 2 / 3 * ηF + ηN) + abs(γF - 2 / 3 * ηF + ηS) + abs(expected_Dz)

        D = map(Array, (dyrel.Dx, dyrel.Dy, dyrel.Dz))
        λmax = map(Array, (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz))
        @test D[1][i, j, k] ≈ expected_Dx
        @test D[2][i, j, k] ≈ expected_Dy
        @test D[3][i, j, k] ≈ expected_Dz
        @test λmax[1][i, j, k] ≈ expected_Cx / expected_Dx
        @test λmax[2][i, j, k] ≈ expected_Cy / expected_Dy
        @test λmax[3][i, j, k] ≈ expected_Cz / expected_Dz

        if backend_JR == CPUBackend
            @testset "actual operator row sums" begin
                dyrel.ηb .= Inf
                stokes.P .= 0.0
                stokes.P0 .= 0.0
                stokes.Q .= 0.0
                θc = dyrel.P_num
                ρg = ntuple(_ -> zeros(ni), 3)
                linear_args = (; T = zeros(ni .+ 2), P = stokes.P, dt = dt)
                flow_bcs = VelocityBoundaryConditions(;
                    free_slip = (left = true, right = true, top = true, bot = false, front = true, back = true),
                    no_slip = (left = false, right = false, top = false, bot = true, front = false, back = false),
                )
                active = (
                    @view(stokes.V.Vx[2:ni[1], 2:(ni[2] + 1), 2:(ni[3] + 1)]),
                    @view(stokes.V.Vy[2:(ni[1] + 1), 2:ni[2], 2:(ni[3] + 1)]),
                    @view(stokes.V.Vz[2:(ni[1] + 1), 2:(ni[2] + 1), 2:ni[3]]),
                )
                lengths = length.(active)
                offsets = (0, lengths[1], lengths[1] + lengths[2])
                operator = zeros(sum(lengths), sum(lengths))

                function apply_column!(out, column)
                    foreach(A -> fill!(A, 0.0), (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz))
                    for d in 1:3
                        if offsets[d] < column ≤ offsets[d] + lengths[d]
                            active[d][column - offsets[d]] = 1.0
                        end
                    end
                    flow_bcs!(stokes, flow_bcs)
                    JR3K.compute_∇V_strain_rate_RP!(stokes, dyrel, gersh_rheology, phase_ratios, grid._di, ni, dt, linear_args)
                    JR3K.compute_stress_viscosity_DRYEL!(
                        stokes, θc, dyrel.γ_eff, gersh_rheology, phase_ratios,
                        1.0, dt, 1.0, linear_args, (-Inf, Inf), true,
                    )
                    @parallel (@idx ni) JR3K.compute_DR_residual_V!(
                        stokes.R.Rx, stokes.R.Ry, stokes.R.Rz,
                        stokes.P, θc, stokes.ΔPψ, @stress(stokes)..., ρg...,
                        dyrel.Dx, dyrel.Dy, dyrel.Dz, grid._di.center, grid._di.vertex,
                    )
                    out .= vcat(vec(stokes.R.Rx), vec(stokes.R.Ry), vec(stokes.R.Rz))
                    return nothing
                end

                for column in axes(operator, 2)
                    apply_column!(@view(operator[:, column]), column)
                end
                measured = vec(sum(abs, operator; dims = 2))
                estimated = vcat(vec(dyrel.λmaxVx), vec(dyrel.λmaxVy), vec(dyrel.λmaxVz))
                @test all(measured .≤ estimated .* (1 + 1.0e-12))
                @test maximum(measured ./ estimated) ≈ 1.0 atol = 1.0e-12
            end
        end
    end

end

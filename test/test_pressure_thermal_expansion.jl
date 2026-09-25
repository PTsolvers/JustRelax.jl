# Thermal pressurization in `compute_P!` and `compute_variational_P!` reads the ghosted
# `ΔT` (size `ni .+ 2`, as `thermal.ΔT`) at the same cell it updates.

push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

using JustPIC

const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

@testset "Thermal pressurization reads ghosted ΔT" begin
    ni = 5, 4
    hot = (2, 3) # cell whose temperature changes
    el = ConstantElasticity(; G = 1.0, ν = 0.25)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 1.0, α = 1.0, β = 1.0, T0 = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    fill!(phase_ratios.center.data, 1.0)

    ΔT_host = zeros(ni .+ 2)
    ΔT_host[(hot .+ 1)...] = 1.0
    ΔT = PTArray(backend)(ΔT_host)

    fields() = ntuple(_ -> @zeros(ni...), 5) # P, P0, RP, ∇V, Q
    η = @ones(ni...)
    dt, r, θ_dτ = 1.0, 1.0, 1.0

    responds_only_at_hot_cell(P) = findall(!iszero, Array(P)) == [CartesianIndex(hot)]

    @testset "compute_P!" begin
        P, P0, RP, ∇V, Q = fields()
        JustRelax2D.compute_P!(P, P0, RP, ∇V, Q, η, rheology, phase_ratios, dt, r, θ_dτ; ΔT)
        @test responds_only_at_hot_cell(P)
    end

    @testset "compute_variational_P!" begin
        P, P0, RP, ∇V, Q = fields()
        ϕ = RockRatio(backend, ni)
        foreach(f -> fill!(getfield(ϕ, f), 1.0), fieldnames(typeof(ϕ)))
        JustRelax2D.compute_variational_P!(
            P, P0, RP, ∇V, Q, η, rheology, phase_ratios, ϕ, dt, r, θ_dτ, (; ΔT)
        )
        @test responds_only_at_hot_cell(P)
    end

    @testset "cell-sized ΔT is rejected" begin
        P, P0, RP, ∇V, Q = fields()
        @test_throws "ΔT must include ghost nodes" JustRelax2D.compute_P!(
            P, P0, RP, ∇V, Q, η, rheology, phase_ratios, dt, r, θ_dτ; ΔT = @zeros(ni...)
        )
    end
end

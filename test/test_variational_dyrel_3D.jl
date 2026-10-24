push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end


using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
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

const JR3K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D
else
    JustRelax3D
end

@parallel_indices (i, j, k) function _init_single_phase_solver_3D!(phases)
    @index phases[1, i, j, k] = 1.0
    return nothing
end

@parallel_indices (i, j, k) function _init_two_phases3D!(phases)
    fraction = isodd(i + 2j + 3k) ? 0.2 : 0.8
    @index phases[1, i, j, k] = fraction
    @index phases[2, i, j, k] = 1 - fraction
    return nothing
end


function variational_setup3D(ni; G = 0.2, Kb = Inf, heterogeneous = false)
    grid = Geometry(ni, (1.0, 1.3, 0.9))
    elasticity = ConstantElasticity(; G, Kb)
    rheology = (
        SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)), Elasticity = elasticity,
        ),
    )
    if heterogeneous
        elasticity2 = ConstantElasticity(; G = 5.0, Kb)
        rheology = (
            rheology..., SetMaterialParams(;
                Phase = 2, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity2)), Elasticity = elasticity2,
            ),
        )
    end
    phases = PhaseRatios(backend_JP, length(rheology), ni)
    for ratio in (phases.center, phases.vertex, phases.yz, phases.xz, phases.xy)
        if heterogeneous
            @parallel (@idx size(ratio)) _init_two_phases3D!(ratio)
        else
            @parallel (@idx size(ratio)) _init_single_phase_solver_3D!(ratio)
        end
    end
    stokes = StokesArrays(backend, ni)
    ϕ = RockRatio(backend, ni)
    for f in fieldnames(typeof(ϕ))
        getfield(ϕ, f) .= 1.0
    end
    free_slip = (left = true, right = true, front = true, back = true, top = true, bot = true)
    bcs = VelocityBoundaryConditions(; free_slip, no_slip = map(_ -> false, free_slip), periodic = map(_ -> false, free_slip))
    return grid, rheology, phases, stokes, ϕ, bcs
end

@testset "Variational DYREL 3D Gershgorin operator" begin
    ni = (4, 4, 4)
    # Check the actual assembled velocity operator, rather than repeating the
    # bound's formula. Finite G makes using creep viscosity alone observable.
    for G in (Inf, 0.2), configuration in (:full, :fractional, :cut), free_surface in (false, true)
        grid, rheology, phases, stokes, ϕ, bcs = variational_setup3D(ni; G, heterogeneous = isfinite(G))
        fraction = configuration == :fractional ? 0.25 : 1.0
        for f in fieldnames(typeof(ϕ))
            a = getfield(ϕ, f)
            a .= fraction
            if configuration == :cut
                a[:, :, 4:end] .= 0.0
                a[:, :, 3] .= 0.25
            end
        end
        η = [10.0^mod(i + 2j + 3k, 3) for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3]]
        copyto!(stokes.viscosity.η, η)
        dt = 0.7
        dyrel = DYREL(backend, stokes, rheology, phases, ϕ, grid.di, dt)
        g = ntuple(_ -> @zeros(ni...), Val(3))
        if free_surface
            copyto!(g[3], [1.0 - k / 10 for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3]])
        end
        JR3K.Gershgorin_variational!(Val(3), dyrel, stokes, phases, ϕ, rheology, grid.di, dt, free_surface ? g[3] : nothing)
        dt_fs = free_surface ? dt : 0.0
        θ = dyrel.P_num
        @parallel (@idx ni) JR3K.compute_PH_residual_V!(
            @residuals(stokes.R)..., @velocity(stokes)..., stokes.P, θ, @stress(stokes)..., g..., ϕ, grid._di.center, grid._di.vertex, dt_fs,
        )
        forcing = vcat(vec.(Array.(@residuals(stokes.R)))...)
        V = @velocity(stokes)
        active = ntuple(d -> view(V[d], ntuple(k -> k == d ? (2:ni[k]) : (2:(ni[k] + 1)), 3)...), 3)
        lengths = length.(active)
        offsets = (0, lengths[1], lengths[1] + lengths[2])
        A = zeros(sum(lengths), sum(lengths))
        for column in axes(A, 2)
            foreach(v -> fill!(v, 0.0), V)
            for d in 1:3
                if offsets[d] < column ≤ offsets[d] + lengths[d]
                    # This test constructs basis vectors on the host, then copies
                    # the complete field so accelerator scalar indexing is avoided.
                    v = zeros(size(active[d]))
                    v[column - offsets[d]] = 1.0
                    copyto!(active[d], v)
                end
            end
            flow_bcs!(stokes, bcs)
            JR3K.compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phases, ϕ, grid._di, ni, dt)
            JR3K.compute_stress_DYREL!(stokes, rheology, phases, ϕ, 1.0, dt)
            @. θ = dyrel.γ_eff * stokes.R.RP
            @parallel (@idx ni) JR3K.compute_PH_residual_V!(
                @residuals(stokes.R)..., V..., stokes.P, θ, @stress(stokes)..., g..., ϕ, grid._di.center, grid._di.vertex, dt_fs,
            )
            A[:, column] .= forcing .- vcat(vec.(Array.(@residuals(stokes.R)))...)
        end
        D = vcat(vec.(Array.((dyrel.Dx, dyrel.Dy, dyrel.Dz)))...)
        λ = vcat(vec.(Array.((dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz)))...)
        measured = vec(sum(abs, A; dims = 2)) ./ D
        @test all(isfinite, λ)
        @test all(measured .≤ λ .* (1 + 1.0e-12))
        if configuration != :cut
            if isinf(G)
                @test maximum(measured ./ λ) ≈ 1.0 atol = 1.0e-12
            else
                # Mixed shear moduli can make the triangle bound strict.
                @test maximum(measured ./ λ) > 0.99
            end
        end
    end
end

@testset "Variational DYREL 3D loaded free surface" begin
    ni = (8, 8, 8)
    igg = IGG(init_global_grid(ni...; init_MPI = !JustRelax.MPI.Initialized())...)
    try
        grid, rheology, phases, stokes, ϕ, bcs = variational_setup3D(ni; Kb = 5.0)
        # An embedded surface, including a partially filled top rock cell.
        for f in fieldnames(typeof(ϕ))
            a = getfield(ϕ, f)
            a[:, :, 7:end] .= 0.0
            a[:, :, 6] .= 0.25
        end
        g = ntuple(_ -> @zeros(ni...), Val(3))
        dt = 0.7
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
        dyrel = DYREL(backend, stokes, rheology, phases, ϕ, grid.di, dt; ϵ = 1.0e-6)
        out = solve_VariationalDYREL!(
            stokes, g, dyrel, bcs, phases, ϕ, rheology, args, grid, dt, igg;
            surface_pressure = 0.1, nout = 20, iterMax_DR = 500, total_iterMax = 20_000,
            linear_viscosity = true, verbose_PH = false, verbose_DR = false
        )
        @test out.converged
        @test all(isfinite, Array(stokes.P))
        @test all(iszero, Array(stokes.P)[:, :, 7:end])
        @test maximum(abs, Array(stokes.V.Vz)) > 0
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

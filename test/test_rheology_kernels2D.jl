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
using StaticArrays

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

const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

function to_device(A::AbstractArray{T}) where {T}
    B = @zeros(size(A)..., eltype = T)
    copyto!(B, A)
    return B
end

@parallel_indices (I...) function _set_ratios!(ratios, vals)
    for p in 1:size(vals, 1)
        @index ratios[p, I...] = vals[p, I...]
    end
    return nothing
end

function set_ratios!(ratios, host_vals)
    vals = to_device(host_vals)
    @parallel (@idx size(ratios)) _set_ratios!(ratios, vals)
    return nothing
end

# second invariant of a plane-strain deviatoric tensor, with εzz = -(εxx + εyy)
invariant(xx, yy, xy) = sqrt(0.5 * (xx^2 + yy^2 + (xx + yy)^2) + xy^2)
vertex_average(A) = [0.25 * (A[i, j] + A[i + 1, j] + A[i, j + 1] + A[i + 1, j + 1]) for i in 1:(size(A, 1) - 1), j in 1:(size(A, 2) - 1)]
# second invariant using the mean of squares of the 4 vertices around cell (i, j),
# not the square of their mean
gather4(A, i, j) = (A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1])
invariant_meansq(xx, yy, xy4) = sqrt(0.5 * (xx^2 + yy^2 + (xx + yy)^2) + sum(a -> a^2, xy4) / 4)

# Arrhenius-type law: η = η0 exp(E / (T + T_O) - E / (T_η + T_O)), independent of the strain rate
const arrhenius = (η0 = 2.0, E = 3.0, T_O = 0.5, T_η = 1.0)
η_arrhenius(T) = arrhenius.η0 * exp(arrhenius.E / (T + arrhenius.T_O) - arrhenius.E / (arrhenius.T_η + arrhenius.T_O))
# power law: τII = ε0 η0 (εII / ε0)^n, hence η = τII / (2 εII)
const powerlaw = (η0 = 2.0, n = 3.0, ε0 = 1.0)
η_powerlaw(εII) = powerlaw.ε0 * powerlaw.η0 * (εII / powerlaw.ε0)^powerlaw.n / (2 * εII)

@testset "Rheology kernels 2D" begin
    nx, ny = 6, 5
    ni = nx, ny
    c = CartesianIndices(ni)
    v = CartesianIndices(ni .+ 1)

    Th = [1.0 + 0.4 * sin(0.9i) + 0.3 * cos(0.7j) for i in 1:(nx + 2), j in 1:(ny + 2)]
    Ph = [2.0 + 0.3 * i - 0.2 * j for i in 1:nx, j in 1:ny]
    args = (; T = to_device(Th), P = to_device(Ph))
    T_center = Th[2:(end - 1), 2:(end - 1)]

    εxx = [0.3 + 0.2 * sin(i + 0.5j) for i in 1:nx, j in 1:ny]
    εyy = [-0.1 + 0.15 * cos(0.6i - j) for i in 1:nx, j in 1:ny]
    εxy = [0.25 * sin(0.8i + 0.3j) + 0.1 for i in 1:(nx + 1), j in 1:(ny + 1)]
    εxx_v = [0.2 + 0.1 * cos(i - 0.4j) for i in 1:(nx + 1), j in 1:(ny + 1)]
    εyy_v = [-0.3 + 0.2 * sin(0.3i * j) for i in 1:(nx + 1), j in 1:(ny + 1)]
    τ_scale = 1.7
    function stokes_with_fields(η_old)
        stokes = StokesArrays(backend, ni)
        foreach(copyto!, (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, stokes.ε.xy_c), (εxx, εyy, εxy, vertex_average(εxy)))
        foreach(copyto!, (stokes.ε.xx_v, stokes.ε.yy_v), (εxx_v, εyy_v))
        foreach(
            (A, B) -> copyto!(A, τ_scale .* B),
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c, stokes.τ.xx_v, stokes.τ.yy_v),
            (εxx, εyy, εxy, vertex_average(εxy), εxx_v, εyy_v),
        )
        copyto!(stokes.viscosity.η, η_old)
        copyto!(stokes.viscosity.ηv, fill(1.0, ni .+ 1))
        return stokes
    end
    η_old = [1.0 + 0.5 * sin(i * j) for i in 1:nx, j in 1:ny]
    # squares the pre-averaged vertex shear: matches the phase-ratio path, which reads
    # the plain (already-averaged) center field ε.xy_c
    εII_c = [invariant(εxx[I], εyy[I], vertex_average(εxy)[I]) for I in c]
    # true mean-of-squares over the 4 vertices around each cell: matches the non-phase
    # path, which gathers the raw vertex-located ε.xy directly
    εII_c_meansq = [invariant_meansq(εxx[I], εyy[I], gather4(εxy, I[1], I[2])) for I in c]

    rheo_arrhenius = SetMaterialParams(;
        Phase = 1,
        CompositeRheology = CompositeRheology((ArrheniusType(; η0 = arrhenius.η0, E_η = arrhenius.E, T_O = arrhenius.T_O, T_η = arrhenius.T_η),)),
    )
    rheo_powerlaw = SetMaterialParams(;
        Phase = 2,
        CompositeRheology = CompositeRheology((PowerlawViscous(; η0 = powerlaw.η0, n = powerlaw.n, ε0 = powerlaw.ε0),)),
    )

    @testset "single rheology: relaxation and cutoff" begin
        ν = 0.4
        η_new = η_arrhenius.(T_center)
        relaxed = (1 - ν) .* η_old .+ ν .* η_new
        cutoff = (sort(vec(relaxed))[5], sort(vec(relaxed))[end - 4])
        expected = clamp.(relaxed, cutoff...)
        @test count(==(cutoff[1]), expected) ≥ 4 && count(==(cutoff[2]), expected) ≥ 4

        for fn! in (compute_viscosity_εII!, compute_viscosity_τII!, compute_viscosity!)
            stokes = stokes_with_fields(η_old)
            fn!(stokes, args, rheo_arrhenius, cutoff; relaxation = ν)
            @test Array(stokes.viscosity.η) ≈ expected
        end
        for fn! in (JR2K.update_viscosity_εII!, JR2K.update_viscosity_τII!)
            stokes = stokes_with_fields(η_old)
            fn!(stokes, args, rheo_arrhenius, cutoff; relaxation = ν)
            @test Array(stokes.viscosity.η) ≈ expected
        end
    end

    @testset "single rheology: strain-rate invariant at the cell centers" begin
        # the vertex shear strain rate enters as the mean of squares over the 4
        # vertices around each cell, not the square of their mean
        stokes = stokes_with_fields(η_old)
        compute_viscosity_εII!(stokes, args, rheo_powerlaw, (0.0, Inf))
        @test Array(stokes.viscosity.η) ≈ η_powerlaw.(εII_c_meansq)
    end

    @testset "single rheology: stress invariant at the cell centers" begin
        # compute_viscosity_τII! must read the deviatoric stress tensor, not the
        # strain rate tensor, along the non-phase path
        stokes = stokes_with_fields(η_old)
        τII_c_meansq = [invariant_meansq((τ_scale .* εxx)[I], (τ_scale .* εyy)[I], gather4(τ_scale .* εxy, I[1], I[2])) for I in c]
        ηp(τII) = τII / (2 * compute_εII(rheo_powerlaw.CompositeRheology[1].elements[1], τII))
        compute_viscosity_τII!(stokes, args, rheo_powerlaw, (0.0, Inf))
        @test Array(stokes.viscosity.η) ≈ ηp.(τII_c_meansq)
    end

    @testset "precomputed invariant field" begin
        ν = 0.7
        cutoff = (0.05, 0.6)
        AII = [0.2 + 0.5 * (1 + sin(i + 2j)) for i in 1:nx, j in 1:ny]
        η = to_device(η_old)
        JR2K.compute_viscosity_εII!(η, ν, to_device(AII), args, rheo_powerlaw, cutoff)
        @test Array(η) ≈ clamp.((1 - ν) .* η_old .+ ν .* η_powerlaw.(AII), cutoff...)
        η = to_device(η_old)
        # scalar entries of args apply to every cell
        JR2K.compute_viscosity_τII!(η, ν, to_device(AII), (; T = args.T, P = 2.0), rheo_arrhenius, cutoff)
        @test Array(η) ≈ clamp.((1 - ν) .* η_old .+ ν .* η_arrhenius.(T_center), cutoff...)
    end

    @testset "phase ratios: harmonic average without the air phase" begin
        η_air = 1.0e-3
        rheo_air = SetMaterialParams(; Phase = 3, CompositeRheology = CompositeRheology((LinearViscous(; η = η_air),)))
        rheology = (rheo_arrhenius, rheo_powerlaw, rheo_air)
        air_phase = 3

        ratios_c = [SA[0.5 + 0.3 * sin(i), 0.3 - 0.3 * sin(i), 0.2] for i in 1:nx, j in 1:ny]
        ratios_c[1, 1] = SA[1.0, 0.0, 0.0]
        ratios_c[2, 1] = SA[0.0, 1.0, 0.0]
        ratios_c[3, 1] = SA[0.0, 0.0, 1.0]
        ratios_v = [SA[0.4 + 0.2 * cos(j), 0.4 - 0.2 * cos(j), 0.2] for i in 1:(nx + 1), j in 1:(ny + 1)]
        ratios_v[end, end] = SA[0.0, 0.0, 1.0]
        phase_ratios = PhaseRatios(backend_JP, 3, ni)
        set_ratios!(phase_ratios.center, collect(reshape(reinterpret(Float64, ratios_c), 3, nx, ny)))
        set_ratios!(phase_ratios.vertex, collect(reshape(reinterpret(Float64, ratios_v), 3, nx + 1, ny + 1)))

        # a cell that holds only air keeps its own ratio; elsewhere air is dropped and the rest renormalized
        function drop_air(r)
            r[air_phase] ≈ 1 && return r
            rr = SA[r[1], r[2], 0.0]
            return rr ./ sum(rr)
        end
        harmonic(r, ηs) = inv(sum(r[p] / ηs[p] for p in 1:3 if !iszero(r[p])))

        # vertex temperature: mean of the four ghost-padded cell centers around the vertex
        T_v = vertex_average(Th)

        for (fn!, scale) in ((compute_viscosity_εII!, 1.0), (compute_viscosity_τII!, τ_scale), (compute_viscosity!, 1.0), (JR2K.update_viscosity_τII!, τ_scale), (JR2K.update_viscosity_εII!, 1.0))
            ν = 0.6
            AII_c = scale .* εII_c
            AII_v = [scale * invariant(εxx_v[I], εyy_v[I], εxy[I]) for I in v]
            # the stress-based evaluation of the power law uses its strain rate at that stress
            ηp(A) = scale == 1.0 ? η_powerlaw(A) : A / (2 * compute_εII(rheo_powerlaw.CompositeRheology[1].elements[1], A))
            η_c = [harmonic(drop_air(ratios_c[I]), (η_arrhenius(T_center[I]), ηp(AII_c[I]), η_air)) for I in c]
            η_v = [harmonic(drop_air(ratios_v[I]), (η_arrhenius(T_v[I]), ηp(AII_v[I]), η_air)) for I in v]
            cutoff = (1.0e-2, 5.0)
            stokes = stokes_with_fields(η_old)
            fn!(stokes, phase_ratios, args, rheology, cutoff; air_phase, relaxation = ν)
            @test Array(stokes.viscosity.η) ≈ clamp.((1 - ν) .* η_old .+ ν .* η_c, cutoff...)
            @test Array(stokes.viscosity.ηv) ≈ clamp.((1 - ν) .+ ν .* η_v, cutoff...)
        end
        # the air-only cell carries the air viscosity
        stokes = stokes_with_fields(η_old)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (0.0, Inf); air_phase)
        @test Array(stokes.viscosity.η)[3, 1] ≈ η_air
        @test Array(stokes.viscosity.η)[1, 1] ≈ η_arrhenius(T_center[1, 1])
        @test Array(stokes.viscosity.η)[2, 1] ≈ η_powerlaw(εII_c[2, 1])
    end

    @testset "buoyancy ρ(T, P) g" begin
        ρ0, α, β = (2.0, 2.6), (0.01, 0.03), (0.001, 0.002)
        g = (1.5, 0.4, 9.0)
        rheology = ntuple(
            p -> SetMaterialParams(;
                Phase = p,
                Density = PT_Density(; ρ0 = ρ0[p], α = α[p], β = β[p], T0 = 0.0, P0 = 0.0),
                Gravity = ConstantGravity(; g = g[3]),
            ), Val(2)
        )
        rheology_dip = ntuple(
            p -> SetMaterialParams(;
                Phase = p,
                Density = PT_Density(; ρ0 = ρ0[p], α = α[p], β = β[p], T0 = 0.0, P0 = 0.0),
                Gravity = DippingGravity(; g = g[3], gx = g[1], gy = g[2], gz = g[3]),
            ), Val(2)
        )
        ρ(p, I) = ρ0[p] * (1 - α[p] * T_center[I] + β[p] * Ph[I])

        ρg = @zeros(ni...)
        compute_ρg!(ρg, rheology[1], args)
        @test Array(ρg) ≈ [ρ(1, I) * g[3] for I in c]
        fill!(ρg, 0.0)
        JR2K.update_ρg!(ρg, rheology[1], args)
        @test Array(ρg) ≈ [ρ(1, I) * g[3] for I in c]

        φ1 = [0.5 + 0.5 * sin(0.8i + 1.1j) for i in 1:nx, j in 1:ny]
        φ1[1, 1] = 0.0
        phase_ratios = PhaseRatios(backend_JP, 2, ni)
        set_ratios!(phase_ratios.center, permutedims(cat(φ1, 1 .- φ1; dims = 3), (3, 1, 2)))
        ρ_mix = [φ1[I] * ρ(1, I) + (1 - φ1[I]) * ρ(2, I) for I in c]

        # a vector gravity fills one buoyancy component per direction; 2D takes the x and z components
        ρgx, ρgy = @zeros(ni...), @zeros(ni...)
        compute_ρg!((ρgx, ρgy), phase_ratios, rheology_dip, args)
        @test Array(ρgx) ≈ ρ_mix .* g[1]
        @test Array(ρgy) ≈ ρ_mix .* g[3]
        # air_phase = 2 leaves the density of phase 1 wherever phase 1 is present
        compute_ρg!((ρgx, ρgy), phase_ratios, rheology_dip, args; air_phase = 2)
        ρ_rock = [iszero(φ1[I]) ? 0.0 : ρ(1, I) for I in c]
        @test Array(ρgx) ≈ ρ_rock .* g[1]
        @test Array(ρgy) ≈ ρ_rock .* g[3]
        # a scalar gravity fills the vertical component only
        fill!(ρgx, -1.0)
        compute_ρg!((ρgx, ρgy), phase_ratios, rheology, args)
        @test all(==(-1.0), Array(ρgx))
        @test Array(ρgy) ≈ ρ_mix .* g[3]
        fill!(ρgy, 0.0)
        JR2K.update_ρg!((ρgx, ρgy), phase_ratios, rheology, args)
        @test Array(ρgy) ≈ ρ_mix .* g[3]

        # the single-rheology form of a vector-gravity buoyancy
        compute_ρg!((ρgx, ρgy), rheology_dip[1], args)
        @test Array(ρgx) ≈ [ρ(1, I) * g[1] for I in c]
        @test Array(ρgy) ≈ [ρ(1, I) * g[3] for I in c]
    end
end

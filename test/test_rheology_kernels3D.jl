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
using StaticArrays

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
    JustRelax.JustRelax3D
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

const arrhenius = (η0 = 2.0, E = 3.0, T_O = 0.5, T_η = 1.0)
η_arrhenius(T) = arrhenius.η0 * exp(arrhenius.E / (T + arrhenius.T_O) - arrhenius.E / (arrhenius.T_η + arrhenius.T_O))
const powerlaw = (η0 = 2.0, n = 3.0, ε0 = 1.0)
η_powerlaw(εII) = powerlaw.ε0 * powerlaw.η0 * (εII / powerlaw.ε0)^powerlaw.n / (2 * εII)

@testset "Rheology kernels 3D" begin
    nx, ny, nz = 5, 4, 3
    ni = nx, ny, nz
    c = CartesianIndices(ni)

    Th = [1.0 + 0.4 * sin(0.9i) + 0.3 * cos(0.7j) - 0.2 * k for i in 1:(nx + 2), j in 1:(ny + 2), k in 1:(nz + 2)]
    Ph = [2.0 + 0.3 * i - 0.2 * j + 0.1 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    args = (; T = to_device(Th), P = to_device(Ph))
    T_center = Th[2:(end - 1), 2:(end - 1), 2:(end - 1)]

    εn = (
        [0.3 + 0.2 * sin(i + 0.5j - k) for (i, j, k) in Tuple.(c)],
        [-0.1 + 0.15 * cos(0.6i - j + k) for (i, j, k) in Tuple.(c)],
        [0.05 * sin(i * j * k) for (i, j, k) in Tuple.(c)],
    )
    ε_yz = [0.25 * sin(0.8i + 0.3j - k) for i in 1:nx, j in 1:(ny + 1), k in 1:(nz + 1)]
    ε_xz = [0.2 * cos(0.5i - 0.6j + 0.9k) for i in 1:(nx + 1), j in 1:ny, k in 1:(nz + 1)]
    ε_xy = [0.15 + 0.1 * sin(0.2i + j + 0.4k) for i in 1:(nx + 1), j in 1:(ny + 1), k in 1:nz]
    # the edge components enter through the mean of their squares over the four edges of the cell
    msq_yz(i, j, k) = 0.25 * (ε_yz[i, j, k]^2 + ε_yz[i, j + 1, k]^2 + ε_yz[i, j, k + 1]^2 + ε_yz[i, j + 1, k + 1]^2)
    msq_xz(i, j, k) = 0.25 * (ε_xz[i, j, k]^2 + ε_xz[i + 1, j, k]^2 + ε_xz[i, j, k + 1]^2 + ε_xz[i + 1, j, k + 1]^2)
    msq_xy(i, j, k) = 0.25 * (ε_xy[i, j, k]^2 + ε_xy[i + 1, j, k]^2 + ε_xy[i, j + 1, k]^2 + ε_xy[i + 1, j + 1, k]^2)
    εII = [sqrt(0.5 * (εn[1][i, j, k]^2 + εn[2][i, j, k]^2 + εn[3][i, j, k]^2) + msq_yz(i, j, k) + msq_xz(i, j, k) + msq_xy(i, j, k)) for (i, j, k) in Tuple.(c)]
    τ_scale = 1.7
    η_old = [1.0 + 0.5 * sin(i * j + k) for (i, j, k) in Tuple.(c)]
    function stokes_with_fields()
        stokes = StokesArrays(backend, ni)
        fields = (εn..., ε_yz, ε_xz, ε_xy)
        foreach(copyto!, (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz, stokes.ε.xz, stokes.ε.xy), fields)
        foreach((A, B) -> copyto!(A, τ_scale .* B), (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz, stokes.τ.xz, stokes.τ.xy), fields)
        copyto!(stokes.viscosity.η, η_old)
        return stokes
    end

    pl = PowerlawViscous(; η0 = powerlaw.η0, n = powerlaw.n, ε0 = powerlaw.ε0)
    rheo_arrhenius = SetMaterialParams(;
        Phase = 1,
        CompositeRheology = CompositeRheology((ArrheniusType(; η0 = arrhenius.η0, E_η = arrhenius.E, T_O = arrhenius.T_O, T_η = arrhenius.T_η),)),
    )
    rheo_powerlaw = SetMaterialParams(; Phase = 2, CompositeRheology = CompositeRheology((pl,)))
    η_air = 1.0e-3
    rheo_air = SetMaterialParams(; Phase = 3, CompositeRheology = CompositeRheology((LinearViscous(; η = η_air),)))
    rheology = (rheo_arrhenius, rheo_powerlaw, rheo_air)
    air_phase = 3

    ratios = [SA[0.5 + 0.3 * sin(i + k), 0.3 - 0.3 * sin(i + k), 0.2] for (i, j, k) in Tuple.(c)]
    ratios[1, 1, 1] = SA[1.0, 0.0, 0.0]
    ratios[2, 1, 1] = SA[0.0, 1.0, 0.0]
    ratios[3, 1, 1] = SA[0.0, 0.0, 1.0]
    phase_ratios = PhaseRatios(backend_JP, 3, ni)
    set_ratios!(phase_ratios.center, collect(reshape(reinterpret(Float64, ratios), 3, ni...)))
    function drop_air(r)
        r[air_phase] ≈ 1 && return r
        rr = SA[r[1], r[2], 0.0]
        return rr ./ sum(rr)
    end
    harmonic(r, ηs) = inv(sum(r[p] / ηs[p] for p in 1:3 if !iszero(r[p])))

    @testset "phase ratios: harmonic average without the air phase" begin
        cutoff = (1.0e-2, 5.0)
        for (fn!, scale) in ((compute_viscosity_εII!, 1.0), (compute_viscosity_τII!, τ_scale), (compute_viscosity!, 1.0))
            ν = 0.6
            AII = scale .* εII
            ηp(A) = scale == 1.0 ? η_powerlaw(A) : A / (2 * compute_εII(pl, A))
            η_c = [harmonic(drop_air(ratios[I]), (η_arrhenius(T_center[I]), ηp(AII[I]), η_air)) for I in c]
            stokes = stokes_with_fields()
            fn!(stokes, phase_ratios, args, rheology, cutoff; air_phase, relaxation = ν)
            @test Array(stokes.viscosity.η) ≈ clamp.((1 - ν) .* η_old .+ ν .* η_c, cutoff...)
        end
        stokes = stokes_with_fields()
        compute_viscosity!(stokes, phase_ratios, args, rheology, (0.0, Inf); air_phase)
        η = Array(stokes.viscosity.η)
        @test η[1, 1, 1] ≈ η_arrhenius(T_center[1, 1, 1])
        @test η[2, 1, 1] ≈ η_powerlaw(εII[2, 1, 1])
        @test η[3, 1, 1] ≈ η_air
    end

    @testset "single rheology" begin
        # without phase ratios the viscosity follows the Arrhenius law at the cell temperature
        stokes = stokes_with_fields()
        compute_viscosity!(stokes, args, rheo_arrhenius, (0.0, Inf))
        @test Array(stokes.viscosity.η) ≈ η_arrhenius.(T_center)

        # the stress-invariant path must read the deviatoric stress tensor, not the
        # strain rate tensor
        stokes = stokes_with_fields()
        ηp(τII) = τII / (2 * compute_εII(pl, τII))
        compute_viscosity_τII!(stokes, args, rheo_powerlaw, (0.0, Inf))
        @test Array(stokes.viscosity.η) ≈ ηp.(τ_scale .* εII)

        # update_viscosity_τII!/εII! (the entry points the single-MaterialParams 3D
        # solver calls) must relax towards the same values
        ν = 0.6
        for (fn!, rheo, ηtarget) in (
                (JR3K.update_viscosity_τII!, rheo_powerlaw, ηp.(τ_scale .* εII)),
                (JR3K.update_viscosity_εII!, rheo_arrhenius, η_arrhenius.(T_center)),
            )
            stokes = stokes_with_fields()
            fn!(stokes, args, rheo, (0.0, Inf); relaxation = ν)
            @test Array(stokes.viscosity.η) ≈ (1 - ν) .* η_old .+ ν .* ηtarget
        end
    end

    @testset "vertex arguments average the eight surrounding cells" begin
        i, j, k = 3, 2, 4
        # device-side helper: call it on host copies so it runs on every backend
        la = JR3K.local_viscosity_args_vertex((; T = Th, P = Ph), i, j, k)
        # cell indices around vertex (i, j, k), clamped to the domain
        cells = Iterators.product((max(i - 1, 1), min(i, nx)), (max(j - 1, 1), min(j, ny)), (max(k - 1, 1), min(k, nz)))
        @test la.P ≈ sum(Ph[I...] for I in cells) / 8
        # T carries a ghost ring, so the eight cell centers around the vertex are T[i:i+1, j:j+1, k:k+1]
        @test la.T ≈ sum(Th[i:(i + 1), j:(j + 1), k:(k + 1)]) / 8
        @test JR3K.average_or_scalar(2.5, i, j, k) == 2.5
    end

    @testset "buoyancy with a vector gravity" begin
        ρ0, α, β = (2.0, 2.6), (0.01, 0.03), (0.001, 0.002)
        g = (1.5, 0.4, 9.0)
        rheology_dip = ntuple(
            p -> SetMaterialParams(;
                Phase = p,
                Density = PT_Density(; ρ0 = ρ0[p], α = α[p], β = β[p], T0 = 0.0, P0 = 0.0),
                Gravity = DippingGravity(; g = g[3], gx = g[1], gy = g[2], gz = g[3]),
            ), Val(2)
        )
        ρ(p, I) = ρ0[p] * (1 - α[p] * T_center[I] + β[p] * Ph[I])
        φ1 = [0.5 + 0.5 * sin(0.8i + 1.1j - k) for (i, j, k) in Tuple.(c)]
        φ1[1, 1, 1] = 0.0
        pr = PhaseRatios(backend_JP, 2, ni)
        set_ratios!(pr.center, permutedims(cat(φ1, 1 .- φ1; dims = 4), (4, 1, 2, 3)))
        ρ_mix = [φ1[I] * ρ(1, I) + (1 - φ1[I]) * ρ(2, I) for I in c]

        ρg = (@zeros(ni...), @zeros(ni...), @zeros(ni...))
        compute_ρg!(ρg, pr, rheology_dip, args)
        for d in 1:3
            @test Array(ρg[d]) ≈ ρ_mix .* g[d]
        end
        compute_ρg!(ρg, pr, rheology_dip, args; air_phase = 2)
        for d in 1:3
            @test Array(ρg[d]) ≈ [iszero(φ1[I]) ? 0.0 : ρ(1, I) * g[d] for I in c]
        end
        compute_ρg!(ρg, rheology_dip[1], args)
        for d in 1:3
            @test Array(ρg[d]) ≈ [ρ(1, I) * g[d] for I in c]
        end
    end
end

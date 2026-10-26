@static if get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "CUDA"
    import CUDA
elseif get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax3D
using GeoParams, ParallelStencil, Test, LinearAlgebra, Adapt

const backend = @static if get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    JustRelax.CUDABackend
elseif get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustRelax.AMDGPUBackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    CPUBackend
end
const JR = backend == CPUBackend ? JustRelax.JustRelax3D :
    backend == JustRelax.CUDABackend ? Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D :
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D

@parallel_indices (i, j, k) function load_phase!(phase)
    @index phase[1, i, j, k] = 1.0
    return nothing
end

@testset "Surface pressure 3D momentum" begin
    ni = (4, 5, 6)
    grid = Geometry(ni, (1.2, 2.0, 3.3))
    ϕ = RockRatio(backend, ni)
    faces = (ϕ.Vx, ϕ.Vy, ϕ.Vz)
    # An inclined interface exercises all three pressure-gradient directions.
    fraction(i, j, k) = clamp(3.3 + 0.2i + 0.15j - k, 0.0, 1.0)
    for f in fieldnames(typeof(ϕ))
        A = getfield(ϕ, f)
        f == :center || fill!(A, 0.7)
    end
    copyto!(ϕ.center, [fraction(i, j, k) for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3]])
    wc = Array(ϕ.center)
    wfaces = map(Array, faces)
    valid = wc .> 0
    for I in CartesianIndices(wc), d in 1:3
        F = CartesianIndex(ntuple(k -> I[k] + (k == d), 3))
        valid[I] &= wfaces[d][I] > 0 && wfaces[d][F] > 0
    end
    wp = wc .* valid
    load = [2.0 + 0.1i^2 + 0.2j + 0.3k for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3]]

    for input in (2.5, copyto!(similar(ϕ.center), load))
        p = input isa Number ? fill(input, ni) : load
        stokes = StokesArrays(backend, ni)
        ητ = @ones(ni...)
        ρg = ntuple(_ -> @zeros(ni...), 3)
        pressure = JR.SurfacePressure(stokes.P, input)
        @parallel (@idx ni .+ 1) JR.compute_V!(
            @velocity(stokes)..., @residuals(stokes.R)..., pressure, ρg...,
            @stress(stokes)..., ητ, 0.02, ϕ, grid._di.center, 0.0,
        )
        for d in 1:3
            R = Array(@residuals(stokes.R)[d])
            G = zeros(length(R), length(wc))
            ci, ri = LinearIndices(wc), LinearIndices(R)
            wf = zeros(size(R))
            for I in CartesianIndices(R)
                J = CartesianIndex(ntuple(k -> I[k] + (k == d), 3))
                G[ri[I], ci[I]] = -grid._di.center[d]
                G[ri[I], ci[J]] = grid._di.center[d]
                wf[I] = wfaces[d][J]
            end
            reference = reshape((G * Diagonal(vec(wp)) - Diagonal(vec(wf)) * G) * vec(p), size(R))
            @test R ≈ reference atol = 1.0e-12
        end
    end

    # Uniform compression exactly cancels its pressure gradient on retained rows.
    for frac in (0.25, 0.75)
        for f in fieldnames(typeof(ϕ))
            A = getfield(ϕ, f)
            vertex_z = f in (:vertex, :Vz, :yz, :xz)
            shift = vertex_z ? 1.5 : 1.0
            copyto!(A, [clamp(3 + frac - (k - shift), 0.0, 1.0) for i in axes(A, 1), j in axes(A, 2), k in axes(A, 3)])
        end
        stokes = StokesArrays(backend, ni)
        hostϕ = Adapt.adapt(Array, ϕ)
        p = [JR.isvalid_c(hostϕ, I.I...) ? 2.5 : 0.0 for I in CartesianIndices(ni)]
        copyto!(stokes.P, p)
        ητ = @ones(ni...)
        ρg = ntuple(_ -> @zeros(ni...), 3)
        @parallel (@idx ni .+ 1) JR.compute_V!(
            @velocity(stokes)..., @residuals(stokes.R)..., JR.SurfacePressure(stokes.P, 2.5),
            ρg..., @stress(stokes)..., ητ, 0.02, ϕ, grid._di.center, 0.0,
        )
        # GPU fused arithmetic can leave roundoff when the two gradients cancel.
        @test all(A -> maximum(abs, Array(A)) < 1.0e-12, @residuals(stokes.R))
        @test all(A -> maximum(abs, Array(A)) < 1.0e-14, @velocity(stokes))
    end
end

@testset "Surface pressure 3D solver" begin
    ni = (6, 6, 8)
    igg = IGG(init_global_grid(ni...; init_MPI = !JustRelax.MPI.Initialized())...)
    try
        grid = Geometry(ni, Float64.(ni))
        ϕ = RockRatio(backend, ni)
        for f in fieldnames(typeof(ϕ))
            A = getfield(ϕ, f)
            shift = f in (:vertex, :Vz, :yz, :xz) ? 1.5 : 1.0
            copyto!(A, [clamp(5.75 - (k - shift), 0.0, 1.0) for i in axes(A, 1), j in axes(A, 2), k in axes(A, 3)])
        end
        jp = backend == CPUBackend ? JustPIC.CPU : backend == JustRelax.CUDABackend ? CUDA.CUDABackend : AMDGPU.ROCBackend
        phase = PhaseRatios(jp, 1, ni)
        @parallel (@idx ni) load_phase!(phase.center)
        @parallel (@idx ni .+ 1) load_phase!(phase.vertex)
        for field in (phase.xy, phase.yz, phase.xz)
            @parallel (@idx size(field)) load_phase!(field)
        end
        rheology = (
            SetMaterialParams(;
                Phase = 1, Density = ConstantDensity(; ρ = 1.0), Gravity = ConstantGravity(; g = 0.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            ),
        )
        slip = (left = true, right = true, front = true, back = true, top = true, bot = true)
        bcs = VelocityBoundaryConditions(; free_slip = slip, no_slip = map(_ -> false, slip), periodic = map(_ -> false, slip))
        pt = PTStokesCoeffs(grid.li, minimum.(grid.di.vertex); ϵ_abs = 1.0e-9, ϵ_rel = 1.0e-9, CFL = 0.9 / √3.1)
        hostϕ = Adapt.adapt(Array, ϕ)
        valid = [JR.isvalid_c(hostϕ, I.I...) for I in CartesianIndices(ni)]
        for field_load in (false, true)
            stokes = StokesArrays(backend, ni)
            load = field_load ? fill!(similar(stokes.P), 2.5) : 2.5
            ρg = ntuple(_ -> @zeros(ni...), 3)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P)
            result = redirect_stdout(devnull) do
                solve_VariationalStokes!(
                    stokes, pt, grid, bcs, ρg, phase, ϕ, rheology, args, 1.0, igg;
                    surface_pressure = load, free_surface = field_load,
                    iterMax = 10_000, nout = 100, verbose = false,
                )
            end
            @test result.iter < 10_000
            @test all(isapprox.(Array(stokes.P)[valid], 2.5; atol = 1.0e-6))
            @test all(iszero, Array(stokes.P)[.!valid])
            @test maximum(maximum(abs, Array(A)) for A in @velocity(stokes)) < 1.0e-7
        end
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

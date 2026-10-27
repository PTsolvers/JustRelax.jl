@static if get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "CUDA"
    import CUDA
elseif get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D
using GeoParams, ParallelStencil, Test, LinearAlgebra

const backend = @static if get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
elseif get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU") == "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end
const backend_JP = backend == CPUBackend ? JustPIC.CPU :
    backend == JustRelax.CUDABackend ? CUDA.CUDABackend : AMDGPU.ROCBackend
const JR = backend == CPUBackend ? JustRelax.JustRelax2D :
    backend == JustRelax.CUDABackend ? Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D :
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D

@parallel_indices (i, j) function load_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
end

function load_mask(ni; frac = 0.75, jcut = ni[2] - 3)
    ϕ = RockRatio(backend, ni)
    h = jcut + frac
    for f in (:center, :Vx, :Vy, :vertex)
        A = getfield(ϕ, f)
        shift = f in (:Vy, :vertex) ? 1.5 : 1.0
        copyto!(A, [clamp(h - (j - shift), 0.0, 1.0) for i in axes(A, 1), j in axes(A, 2)])
    end
    return ϕ
end

# Independently assemble G*Wp - Wv*G on the retained pressure space.
function load_reference(ϕ, load, grid, d; periodic = false)
    wc = Array(ϕ.center)
    wx, wy = Array(ϕ.Vx), Array(ϕ.Vy)
    wp = [
        wc[i, j] > 0 && min(wx[i, j], wx[i + 1, j], wy[i, j], wy[i, j + 1]) > 0 ? wc[i, j] : 0.0
            for i in axes(wc, 1), j in axes(wc, 2)
    ]
    rows = ntuple(k -> size(wc, k) - (k == d && !periodic), 2)
    G = zeros(prod(rows), length(wc))
    wf = zeros(rows)
    ci, ri = LinearIndices(wc), LinearIndices(rows)
    for I in CartesianIndices(rows)
        J = CartesianIndex(ntuple(k -> k == d ? mod1(I[k] + 1, size(wc, k)) : I[k], 2))
        F = CartesianIndex(ntuple(k -> I[k] + (k == d), 2))
        spacing = grid._di.center[d]
        inv_h = spacing isa Number ? spacing : Array(spacing)[I[d]]
        G[ri[I], ci[I]] = -inv_h
        G[ri[I], ci[J]] = inv_h
        wf[I] = (d == 1 ? wx : wy)[F]
    end
    p = load isa Number ? fill(load, size(wc)) : Array(load)
    force = reshape((G * Diagonal(vec(wp)) - Diagonal(vec(wf)) * G) * vec(p), rows)
    force[wf .== 0] .= 0.0
    return force
end

function ph_load!(stokes, ϕ, pressure, grid)
    ni = size(stokes.P)
    ρg = ntuple(_ -> @zeros(ni...), 2)
    @parallel (@idx ni) JR.compute_PH_residual_V!(
        @residuals(stokes.R)..., @velocity(stokes)..., pressure, stokes.ΔPψ,
        @stress(stokes)..., ρg..., ϕ, grid._di.center, grid._di.vertex, 0.0,
    )
    return map(Array, @residuals(stokes.R))
end

@testset "Surface pressure 2D operators" begin
    ni = (6, 7)
    for grid in (Geometry(ni, (1.8, 3.5)), Geometry(PTArray(backend), [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1], collect(range(0.0, 1.0; length = 8)) .^ 1.4))
        ϕ = load_mask(ni)
        p = @zeros(ni...)
        copyto!(p, [2.0 + sin(i) + 0.2j for i in 1:ni[1], j in 1:ni[2]])
        for load in (2.5, p)
            stokes = StokesArrays(backend, ni)
            pressure = JR.SurfacePressure(stokes.P, load)
            reference = ntuple(d -> load_reference(ϕ, load, grid, d), 2)
            @test all(isapprox.(ph_load!(stokes, ϕ, pressure, grid), reference; atol = 1.0e-12))

            # The PT update must use the same physical residual, before face-mass scaling.
            ητ = @ones(ni...)
            ρg = ntuple(_ -> @zeros(ni...), 2)
            @parallel (@idx ni .+ 1) JR.compute_V!(
                @velocity(stokes)..., @residuals(stokes.R)..., pressure,
                @stress(stokes)..., 0.02, ρg..., ητ, ϕ,
                grid._di.center, grid._di.vertex, 0.0,
            )
            @test all(isapprox.(map(Array, @residuals(stokes.R)), reference; atol = 1.0e-12))
            @test Array(stokes.V.Vy)[2:(end - 1), 2:(end - 1)] ≈
                0.02 .* reference[2] ./ max.(Array(ϕ.Vy)[:, 2:(end - 1)], 0.1)

            # DYREL stores a diagonally scaled residual in its inner update.
            foreach(A -> fill!(A, 0.0), @velocity(stokes))
            z = ntuple(d -> zero(getfield(stokes.R, d == 1 ? :Rx : :Ry)), 2)
            a = map(A -> fill!(similar(A), 1.0), z)
            D = map(A -> fill!(similar(A), 3.0), z)
            @parallel (@idx ni) JR.compute_DR_residual_update_V!(
                @residuals(stokes.R)..., @velocity(stokes)..., z..., pressure, stokes.ΔPψ,
                @stress(stokes)..., ρg..., D..., a..., a..., a..., ϕ,
                grid._di.center, grid._di.vertex, 0.0,
            )
            @test all(isapprox.(map(A -> 3 .* Array(A), @residuals(stokes.R)), reference; atol = 1.0e-12))
        end
    end

    # Full rock: a varying prescribed field must not become a volume force.
    ϕ = load_mask(ni; jcut = ni[2] + 1)
    grid = Geometry(ni, Float64.(ni))
    stokes = StokesArrays(backend, ni)
    load = copy(stokes.P)
    copyto!(load, [1.0e8 * i * j for i in 1:ni[1], j in 1:ni[2]])
    @test all(A -> all(iszero, A), ph_load!(stokes, ϕ, JR.SurfacePressure(stokes.P, load), grid))

    # Zero load preserves the existing residual exactly.
    copyto!(stokes.P, Array(load))
    bare = ph_load!(stokes, ϕ, stokes.P, grid)
    @test ph_load!(stokes, ϕ, JR.SurfacePressure(stokes.P, 0.0), grid) == bare

    # A periodic DYREL seam differentiates the prescribed field and masks together.
    periodic_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = true, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, periodic_bcs)
    ϕ = load_mask(ni)
    pressure = JR.SurfacePressure(stokes.P, load)
    force = ph_load!(stokes, ϕ, pressure, grid)
    @test force[1] ≈ load_reference(ϕ, load, grid, 1; periodic = true)
    @test force[2] ≈ load_reference(ϕ, load, grid, 2)
end

function solve_load_column(igg, solver; frac = 0.75, load = 2.5, initial = 0.0, gravity = 0.0)
    ni = (8, 8)
    grid = Geometry(ni, Float64.(ni))
    ϕ = load_mask(ni; frac)
    stokes = StokesArrays(backend, ni)
    fill!(stokes.P, initial)
    phase = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx ni) load_phase!(phase.center)
    @parallel (@idx ni .+ 1) load_phase!(phase.vertex)
    rheology = (
        SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = 1.0), Gravity = ConstantGravity(; g = gravity),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
    ρg = ntuple(_ -> @zeros(ni...), 2)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P)
    field = load isa Number || isnothing(load) ? load : copyto!(similar(stokes.P), load)
    result = redirect_stdout(devnull) do
        if solver == :pt
            pt = PTStokesCoeffs(grid.li, minimum.(grid.di.vertex); ϵ_abs = 1.0e-9, ϵ_rel = 1.0e-9)
            solve_VariationalStokes!(
                stokes, pt, grid, bcs, ρg, phase, ϕ, rheology, args, 1.0, igg;
                surface_pressure = field, iterMax = 20_000, iterMin = 20, nout = 50, verbose = false,
            )
        else
            dyrel = DYREL(backend, stokes, rheology, phase, ϕ, grid.di, 1.0; ϵ = 1.0e-8)
            solve_VariationalDYREL!(
                stokes, ρg, dyrel, bcs, phase, ϕ, rheology, args, grid, 1.0, igg;
                kwargs = (;
                    surface_pressure = field, total_iterMax = 30_000, nout = 25,
                    verbose_PH = false, verbose_DR = false, linear_viscosity = true,
                ),
            )
        end
    end
    return stokes, ϕ, result
end

@testset "Surface pressure 2D solves and validation" begin
    igg = IGG(init_global_grid(8, 8, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    try
        for frac in (0.25, 0.75), solver in (:pt, :dyrel)
            stokes, ϕ, result = solve_load_column(igg, solver; frac)
            wc, wx, wy = Array(ϕ.center), Array(ϕ.Vx), Array(ϕ.Vy)
            valid = [
                wc[i, j] > 0 && min(wx[i, j], wx[i + 1, j], wy[i, j], wy[i, j + 1]) > 0
                    for i in axes(wc, 1), j in axes(wc, 2)
            ]
            @test all(isapprox.(Array(stokes.P)[valid], 2.5; atol = 1.0e-6))
            @test all(iszero, Array(stokes.P)[.!valid])
            @test maximum(maximum(abs, Array(A)) for A in @velocity(stokes)) < 1.0e-7
            @test solver == :pt ? result.iter < 20_000 : result.converged
        end
        for solver in (:pt, :dyrel)
            scalar, _, _ = solve_load_column(igg, solver)
            field, _, _ = solve_load_column(igg, solver; load = fill(2.5, 8, 8))
            @test Array(scalar.P) ≈ Array(field.P)
            unloaded, _, _ = solve_load_column(igg, solver; load = nothing, gravity = 1.0)
            loaded, ϕ, _ = solve_load_column(igg, solver; load = 2.5, gravity = 1.0)
            valid = Array(unloaded.P) .> 0
            @test Array(loaded.P)[valid] - Array(unloaded.P)[valid] ≈ fill(2.5, count(valid)) atol = 1.0e-6
        end

        P = @zeros(8, 8)
        ϕ = load_mask((8, 8))
        bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
        @test JR.prepare_surface_pressure(P, nothing, ϕ, bcs) === P
        @test_throws "cell-centered" JR.prepare_surface_pressure(P, @zeros(8, 9), ϕ, bcs)
        @test_throws "real scalar" JR.prepare_surface_pressure(P, :bad, ϕ, bcs)
        @test_throws "finite" JR.prepare_surface_pressure(P, Inf, ϕ, bcs)
        top_bcs = VelocityBoundaryConditions(; free_slip = bcs.free_slip, free_surface = true)
        @test_throws "box-top" JR.prepare_surface_pressure(P, 1.0, ϕ, top_bcs)
        prescribed = fill!(similar(P), 3.0)
        prepared = JR.prepare_surface_pressure(P, prescribed, ϕ, bcs)
        fill!(prescribed, 4.0)
        @test all(==(3.0), Array(prepared.load))
        @test all(==(4.0), Array(JR.prepare_surface_pressure(P, view(prescribed, :, :), ϕ, bcs).load))
        if backend != CPUBackend
            @test_throws "same backend" JR.prepare_surface_pressure(P, Array(prescribed), ϕ, bcs)
        end
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

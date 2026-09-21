push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test, Suppressor
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

const pT = -0.5     # tensile strength of the cap

# Homogeneous volumetric extension of a single-phase box: the only way to
# accommodate the imposed ∇V is elastic decompression, so the pressure walks
# into tension until the tensile cap opens the material (mode-I).
function tensile_box(; with_cap::Bool, multiphase = true, nx = 32, ny = 32, nsteps = 8, finalize_mpi = true)
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    ly = lx = 1.0
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    (; xci, xvi) = grid

    ϕ_fric = 30
    C = 1.0
    η0 = 1.0
    G0 = 1.0
    Kb = 4.0
    η_reg = 1.0e-3
    εbg = 1.0                      # volumetric background extension rate
    dt = η0 / G0 / 8.0

    el = ConstantElasticity(; G = G0, Kb = Kb)
    visc = LinearViscous(; η = η0)
    pl = if with_cap
        DruckerPragerCap(; C = C, ϕ = ϕ_fric, η_vp = η_reg, Ψ = 0.0, pT = pT)
    else
        # same cone, no tensile cap: nothing stops the pressure from going tensile
        DruckerPrager_regularised(; C = C, ϕ = ϕ_fric, η_vp = η_reg, Ψ = 0.0)
    end

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el, pl)),
            Elasticity = el,
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    fill!(phase_ratios.center.data, 1.0)
    fill!(phase_ratios.vertex.data, 1.0)

    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.95 / √2.1)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    # outward velocities on every boundary: isotropic extension
    stokes.V.Vx .= PTArray(backend)([(x - 0.5lx) * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([(y - 0.5ly) * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    local iters
    kwargs = (verbose = false, iterMax = 50.0e3, nout = 1.0e3, viscosity_cutoff = (-Inf, Inf))
    for _ in 1:nsteps
        # the single-phase path goes through `compute_τ_nonlinear!`, the multiphase one
        # through `update_stresses_center_vertex_ps!`; both must record the opening
        iters = if multiphase
            solve!(stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, igg; kwargs)
        else
            solve!(stokes, pt_stokes, grid, flow_bcs, ρg, rheology[1], args, dt, igg; kwargs)
        end
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε_pl)
    end

    finalize_global_grid(; finalize_MPI = finalize_mpi)

    return (;
        iters,
        Pmin = minimum(Array(stokes.P)),
        Pmax = maximum(Array(stokes.P)),
        τII_max = maximum(Array(stokes.τ.II)),
        ε_pl_max = maximum(Array(stokes.ε_pl.II)),
        ε_vol_extrema = extrema(Array(stokes.ε_vol_pl)),
        EVol_max = maximum(abs, Array(stokes.EVol_pl)),
        λ_max = maximum(Array(stokes.λ)),
    )
end

# same material, but incompressible: the cap has no volumetric compliance to correct
# the pressure with, and the solver must say so instead of failing on a NaN later
function incompressible_cap_rheology()
    el = SetConstantElasticity(; G = 1.0, ν = 0.5)
    pl = DruckerPragerCap(; C = 1.0, ϕ = 30.0, η_vp = 1.0e-3, Ψ = 0.0, pT = pT)
    return (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el, pl)),
            Elasticity = el,
        ),
    )
end

@testset "Cap tensile opening on a homogeneous grid" begin
    @suppress begin
        cap = tensile_box(; with_cap = true, finalize_mpi = false)

        @test cap.iters.norm_Rx[end] < 1.0e-5
        @test cap.iters.norm_Ry[end] < 1.0e-5
        @test cap.iters.norm_∇V[end] < 1.0e-5
        @test isfinite(cap.Pmin) && isfinite(cap.Pmax)
        # mode-I: the box opens volumetrically, well below the shear envelope
        @test cap.ε_pl_max > 0
        @test cap.τII_max < 1.0
        @test cap.ε_vol_extrema[2] > 0          # volumetric opening
        @test cap.ε_vol_extrema[1] ≥ -1.0e-12
        @test cap.EVol_max > 0                  # accumulated once per timestep
        @test cap.λ_max > 0                     # the solver publishes the multiplier
        # the tensile cap bounds the pressure; only the Perzyna overstress may exceed it
        @test cap.Pmin > pT - 0.05

        # control: the same setup without a tensile cap cannot stop the decompression
        nocap = tensile_box(; with_cap = false, finalize_mpi = false)
        @test nocap.Pmin < pT
        @test cap.Pmin > nocap.Pmin

        # the single-phase stress kernel must record the opening too
        single = tensile_box(; with_cap = true, multiphase = false)
        @test single.ε_vol_extrema[2] > 0
        @test single.EVol_max > 0
        @test single.Pmin > pT - 0.05
    end
end

@testset "Incompressible tensile cap is rejected" begin
    rheology = incompressible_cap_rheology()
    for solver in (JustRelax.JustRelax2D.solve!, JustRelax.JustRelax2D.solve_VariationalStokes!)
        # the check runs before anything is solved, so the remaining arguments are irrelevant
        err = try
            solver(StokesArrays(backend, (2, 2)), nothing, rheology; kwargs = (;))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("finite `Kb`", err.msg)
    end
    @test JustRelax.JustRelax2D.reject_incompressible_cap(nothing, "solver") === nothing
end

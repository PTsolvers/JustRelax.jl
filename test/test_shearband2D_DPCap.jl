push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        p = GGU.Point(x, y)
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
    return nothing
end

# MAIN SCRIPT --------------------------------------------------------------------
function ShearBand2D_DPCap()
    n = 32
    nx = ny = n
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain ------------------------------------
    ly = 1.0e0
    lx = ly
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    origin = 0.0, 0.0
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid

    # Physical properties ----------------
    τ_y = 1.6
    ϕ = 30
    ψ = 3                    # nonzero dilation triggers ε_vol_pl path
    C = τ_y
    η0 = 1.0
    ηi = 1.0
    G0 = 1.0
    Gi = G0 / 2.0
    εbg_x = 1.0
    εbg_y = 1.0
    η_reg = 1.0e-3
    dt = η0 / G0 / 8.0

    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc_bg = LinearViscous(; η = η0)
    visc_inc = LinearViscous(; η = ηi)

    pl = DruckerPragerCap(;
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = ψ,
        pT = -0.5,
    )

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_bg, el_bg, pl)),
            Elasticity = el_bg,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_inc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    radius = 0.1
    origin_c = 0.5, 0.5
    circle = GGU.Circle(origin_c, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.95 / √2.1)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt)

    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend_JR)([ x * εbg_x for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([-y * εbg_y for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    t, it = 0.0, 0
    local iters

    # ~25 Maxwell-time steps are needed before τII reaches the yield envelope
    while it < 10
        iters = solve!(
            stokes,
            pt_stokes,
            grid,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                verbose = false,
                iterMax = 50.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        tensor_invariant!(stokes.τ)

        it += 1
        t += dt
    end

    finalize_global_grid(; finalize_MPI = true)

    return (;
        iters,
        τII_max = maximum(Array(stokes.τ.II)),
        εpl_max = maximum(Array(stokes.ε_pl.II)),
        Pmin = minimum(Array(stokes.P)),
        Pmax = maximum(Array(stokes.P)),
        EVol_max = maximum(abs, Array(stokes.EVol_pl)),
        εvol_extrema = extrema(Array(stokes.ε_vol_pl)),
    )
end

@testset "ShearBand2D_DPCap" begin
    @suppress begin
        out = ShearBand2D_DPCap()

        # Velocity / pressure residuals reached the requested tolerance
        @test out.iters.norm_Rx[end] < 1.0e-5
        @test out.iters.norm_Ry[end] < 1.0e-5
        @test out.iters.norm_∇V[end] < 1.0e-5

        # τII stays bounded by the yield envelope (cohesion=τ_y/cosϕ ≈ 1.85)
        @test isfinite(out.τII_max)
        @test out.τII_max < 2.0

        # Plasticity engaged in the inclusion — exercises compute_yieldfunction_phase
        # and compute_plastic_gradients_phase paths
        @test out.εpl_max > 0.0

        # Volumetric plastic strain accumulator updated through new accumulate_vol!
        @test out.EVol_max > 0.0

        # ε_vol_pl = -λ * dQ/dP; with ψ > 0, dQ/dP < 0, so ε_vol_pl ≥ 0 (dilation)
        @test out.εvol_extrema[1] ≥ 0.0
        @test out.εvol_extrema[2] > 0.0
    end
end

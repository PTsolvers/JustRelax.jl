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
import JustRelax: ImplicitGlobalGrid, MPI

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

# Largest difference between the entries this rank shares with its lower neighbor along each
# split dimension and that neighbor's copies of them; zero when every overlap entry agrees.
function overlap_mismatch(A_device)
    A = Array(A_device)
    gg = ImplicitGlobalGrid.global_grid()
    mismatch = 0.0
    for d in 1:ndims(A)
        gg.dims[d] > 1 || continue
        nshared = gg.overlaps[d] + size(A, d) - gg.nxyz[d]
        lower, upper = gg.neighbors[1, d], gg.neighbors[2, d]
        n = size(A, d)
        sendbuf = collect(selectdim(A, d, (n - nshared + 1):n))
        recvbuf = similar(sendbuf)
        MPI.Sendrecv!(sendbuf, recvbuf, gg.comm; dest = upper, source = lower)
        if lower != MPI.PROC_NULL
            mine = selectdim(A, d, 1:nshared)
            mismatch = max(mismatch, maximum(abs.(mine .- recvbuf)))
        end
    end
    return MPI.Allreduce(mismatch, max, gg.comm)
end

function thermal_overlap_mismatch(nx, ny)
    ni = nx, ny
    li = 1.0, 1.0
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    (; xvi) = grid

    thermal = ThermalArrays(backend, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    # a Gaussian anomaly centered on the seam between ranks along x
    T = [
        exp(-((xvi[1][clamp(i - 1, 1, nx + 1)] - 0.5)^2 + (xvi[2][clamp(j - 1, 1, ny + 1)] - 0.5)^2) / 0.02)
            for i in 1:(nx + 2), j in 1:(ny + 2)
    ]
    thermal.T .= PTArray(backend)(T)
    thermal_bcs!(thermal, thermal_bc)
    update_halo!(thermal.T)

    # conductivity varying across the seam, so a one-sided average at a rank edge differs
    K = PTArray(backend)([1 + 0.5 * sinpi(2 * x) for x in grid.xci[1], _ in 1:ny])
    ρCp = @ones(ni...)
    dt = 1.0e-3
    pt_thermal = PTThermalCoeffs(backend, K, ρCp, dt, grid.di.center, li; ϵ = 1.0e-10, CFL = 0.9 / √2)
    heatdiffusion_PT!(
        thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid;
        kwargs = (; igg, iterMax = 2.0e3, nout = 10, verbose = false),
    )
    return overlap_mismatch(thermal.T)
end

function init_phases!(phase_ratios, xci, xvi, radius)
    ni = size(phase_ratios.center)
    @parallel_indices (i, j) function _init_phases!(phases, xc, yc, radius)
        inside = ((xc[i] - 0.5)^2 + (yc[j] - 0.5)^2) ≤ radius^2
        @index phases[1, i, j] = inside ? 0.0 : 1.0
        @index phases[2, i, j] = inside ? 1.0 : 0.0
        return nothing
    end
    @parallel (@idx ni) _init_phases!(phase_ratios.center, xci..., radius)
    @parallel (@idx ni .+ 1) _init_phases!(phase_ratios.vertex, xvi..., radius)
    return nothing
end

# Visco-elasto-plastic shear band, yielding from the first step, whose weak inclusion straddles
# the seam between ranks.
function stokes_overlap_mismatch(nx, ny)
    ni = nx, ny
    li = 1.0, 1.0
    di = @. li / (nx_g(), ny_g())
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    (; xci, xvi) = grid

    η0, G0, εbg = 1.0, 1.0, 1.0
    dt = η0 / G0 / 4.0
    pl = DruckerPrager_regularised(; C = 0.3 / cosd(30), ϕ = 30, η_vp = 8.0e-3, Ψ = 0)
    rheology = ntuple(2) do phase
        el = ConstantElasticity(; G = phase == 1 ? G0 : G0 / 2, Kb = 4)
        SetMaterialParams(;
            Phase = phase,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η0), el, pl)),
            Elasticity = el,
        )
    end
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, 0.1)

    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_rel = 1.0e-6, CFL = 0.75 / √2.1)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    for _ in 1:3
        solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, igg;
            kwargs = (; verbose = false, iterMax = 50.0e3, nout = 1.0e3, viscosity_cutoff = (-Inf, Inf)),
        )
    end
    return (
        εxy = overlap_mismatch(stokes.ε.xy),
        ε_plxy = overlap_mismatch(stokes.ε_pl.xy),
        ε_plxy_max = maximum(abs, Array(stokes.ε_pl.xy)),
    )
end

const n = 16
const igg = IGG(
    init_global_grid(
        n, n, 1; init_MPI = JustRelax.MPI.Initialized() ? false : true, select_device = false
    )...,
)

@testset "Overlap copies agree across ranks" begin
    @test (@suppress thermal_overlap_mismatch(n, n)) < 1.0e-12
    mismatch = @suppress stokes_overlap_mismatch(n, n)
    @test mismatch.ε_plxy_max > 0 # the setup yields
    @test mismatch.εxy < 1.0e-12
    @test mismatch.ε_plxy < 1.0e-12
end

finalize_global_grid(; finalize_MPI = false)

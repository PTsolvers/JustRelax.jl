# Adjoint DYREL solve of a strongly plastic, non-associated Drucker–Prager state.
#
# A dense block sinks through a viscoelastic-plastic matrix in background shear. With friction
# angle ϕ = 20°, dilation angle Ψ = 0° and weak regularization η_vp = 1e-2, most of the matrix
# yields, and the consistent plastic tangent of non-associated flow is non-symmetric. The
# adjoint once diverged here; it now converges, in about 3–4× the iterations of associated flow
# (Ψ = ϕ) or of stronger regularization (η_vp = 0.1), which the script runs for comparison.
#
# Run with
#     JULIA_JUSTRELAX_BACKEND=CPU julia --project=miniapps --startup-file=no -t 1 \
#         miniapps/Adjoint2D/NonAssociatedPlasticity2D_adjoint.jl [grid_size]

const backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if backend_name == "CUDA"
    using CUDA
elseif backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D
using JustPIC
using ParallelStencil
using GeoParams
using Printf

const backend = @static if backend_name == "CUDA"
    JustRelax.CUDABackend
elseif backend_name == "AMDGPU"
    JustRelax.AMDGPUBackend
else
    JustRelax.CPUBackend
end

const backend_JP = @static if backend_name == "CUDA"
    CUDA.CUDABackend
elseif backend_name == "AMDGPU"
    AMDGPU.ROCBackend
else
    JustPIC.CPU
end

@static if backend_name == "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
elseif backend_name == "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@parallel_indices (i, j) function init_dense_block!(phases, xc, yc)
    block = abs(xc[i] - 0.4) ≤ 0.15 && abs(yc[j] - 0.55) ≤ 0.12
    @index phases[1, i, j] = block ? 0.0 : 1.0
    @index phases[2, i, j] = block ? 1.0 : 0.0
    return nothing
end

"""
    non_associated_plasticity_adjoint(igg; nx = 16, ny = 16, Ψ = 0.0, η_vp = 1.0e-2, verbose = true)

Solve one forward step and its adjoint for the sinking block with matrix dilation angle `Ψ` and
regularization viscosity `η_vp`. The friction angle is 20°. Returns
`(; converged, message, yielded_fraction, forward_iterations)`, where `converged` reports
whether the adjoint solve finished and `message` its error otherwise.
"""
function non_associated_plasticity_adjoint(
        igg; nx = 16, ny = 16, Ψ = 0.0, η_vp = 1.0e-2, verbose = true,
    )
    ni = (nx, ny)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    xv, yv = Array.(grid.xvi)
    dt = 0.5

    # Rheology ---------------------------------------
    elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
    function material(phase; ρ0, C)
        return SetMaterialParams(;
            Phase = phase,
            Density = PT_Density(; ρ0, β = 0.05, α = 0.0, T0 = 0.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology(
                (
                    LinearViscous(; η = 1.0),
                    elasticity,
                    DruckerPrager_regularised(; C, ϕ = 20.0, Ψ, η_vp),
                )
            ),
            Elasticity = elasticity,
        )
    end
    rheology = (material(1; ρ0 = 1.0, C = 0.02), material(2; ρ0 = 1.5, C = 1.0))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_dense_block!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_dense_block!(phase_ratios.vertex, grid.xvi...)

    # Stokes and adjoint arrays ----------------------
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    stokes_ad = AdjointStokesArrays(backend, ni, flow_bcs)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    compute_ρg!(ρg, phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    # A stress history, as after earlier time steps
    foreach(
        (A, value) -> fill!(A, value),
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c, stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),
        (0.05, -0.05, 0.02, 0.05, -0.05, 0.02),
    )

    # Background shear on the boundaries
    Vx = [x - 0.5 for x in xv, _ in 1:(ny + 2)]
    Vy = [-y for _ in 1:(nx + 2), y in yv]
    Vx[2:(end - 1), 2:(end - 1)] .= 0.0
    Vy[2:(end - 1), 2:(end - 1)] .= 0.0
    stokes.V.Vx .= PTArray(backend)(Vx)
    stokes.V.Vy .= PTArray(backend)(Vy)
    flow_bcs!(stokes, flow_bcs)

    # Objective: Vy summed over a box next to the block
    observation = (; field = :Vy, center = (0.6, 0.55), half_width = (0.1, 0.1))
    gradients = material_controls(backend, ni, (:C, :ϕ); nphases = length(rheology))

    # Forward and adjoint solve ----------------------
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)
    converged, message, forward_iterations = true, "", 0
    try
        out = solve_DYREL!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            free_surface = true, verbose_PH = verbose, verbose_DR = false,
            iterMax = 1.0e5, total_iterMax = 2.0e5, nout = 50, rel_drop = 0.5,
            linear_viscosity = true, adjoint = true, observation, gradients,
        )
        forward_iterations = out.iter
    catch err
        err isa ErrorException || rethrow()
        converged, message = false, err.msg
    end
    yielded_fraction = count(>(0), Array(stokes.λ)) / prod(ni)
    return (; converged, message, yielded_fraction, forward_iterations)
end

function main(igg; nx = 16, ny = 16)
    cases = (
        ("non-associated, Ψ = 0°, η_vp = 1e-2", (; Ψ = 0.0, η_vp = 1.0e-2)),
        ("associated,     Ψ = 20°, η_vp = 1e-2", (; Ψ = 20.0, η_vp = 1.0e-2)),
        ("non-associated, Ψ = 0°, η_vp = 1e-1", (; Ψ = 0.0, η_vp = 1.0e-1)),
    )
    results = map(cases) do (label, parameters)
        igg.me == 0 && println("\n######## ", label, " ########")
        label => non_associated_plasticity_adjoint(igg; nx, ny, parameters...)
    end
    if igg.me == 0
        println("\nAdjoint solve of a sinking block in a yielding Drucker–Prager matrix (ϕ = 20°):")
        for (label, r) in results
            status = r.converged ? "converged" : "diverged ($(r.message))"
            @printf("  %-38s yielded %4.0f %%   adjoint %s\n", label, 100 * r.yielded_fraction, status)
        end
    end
    return results
end

# Define `NO_AUTORUN` before including this file to load the functions without running them.
if !@isdefined(NO_AUTORUN)
    n = isempty(ARGS) ? 16 : parse(Int, ARGS[1])
    igg = IGG(init_global_grid(n, n, 1; init_MPI = true)...)
    main(igg; nx = n, ny = n)
end

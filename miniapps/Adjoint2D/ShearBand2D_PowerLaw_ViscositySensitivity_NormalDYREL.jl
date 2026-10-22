const isCUDA = false

@static if isCUDA
    using CUDA
    CUDA.allowscalar(false)
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using JustPIC
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

function adjoint_circle_observation(grid, Vx, radius)
    weights = zeros(eltype(Vx), size(Vx))
    for (i, x) in enumerate(Array(grid.xvi[1])), (j, y) in enumerate(Array(grid.xci[2]))
        if x^2 + y^2 < radius^2
            weights[i, j + 1] = sign(x)
        end
    end
    selected = count(!iszero, weights)
    iszero(selected) && throw(ArgumentError("the grid has no Vx nodes inside the circle"))
    weights ./= selected
    return (; field = :Vx, weights = PTArray(backend)(weights))
end
using JustRelax.GeoParams
using CairoMakie

@parallel_indices (i, j) function set_normal_dyrel_shear_band_phases!(
        ratios, xc, yc, radius
    )
    inside = xc[i]^2 + yc[j]^2 < radius^2
    @index ratios[1, i, j] = ifelse(inside, 0.0, 1.0)
    @index ratios[2, i, j] = ifelse(inside, 1.0, 0.0)
    return nothing
end

"""
    shear_band_normal_dyrel_viscosity_sensitivity(
        igg; n=64, figdir="ShearBand2D_PowerLaw_ViscositySensitivity_NormalDYREL",
    )

Solve the power-law shear-band problem with the standard, non-variational `solve_DYREL!`
solver. The objective observes `Vx` inside the circular inclusion, with opposite weights on
its left and right halves. The plotted field `η ∂J/∂η` is the sensitivity to a local change in
log viscosity; it is not a derivative with respect to a power-law parameter such as `η0` or
`n`.

The forward solver retains its relaxed, stress-invariant viscosity update. The adjoint
differentiates the converged constitutive state and therefore does not differentiate the
viscosity-relaxation history.
"""
function shear_band_normal_dyrel_viscosity_sensitivity(
        igg;
        n = 64,
        figdir = "ShearBand2D_PowerLaw_ViscositySensitivity_NormalDYREL",
    )
    ni = (n, n)
    grid = Geometry(ni, (1.0, 1.0); origin = (-0.5, -0.5))
    (; xci, xvi) = grid
    radius = 0.1

    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    @parallel (@idx ni) set_normal_dyrel_shear_band_phases!(
        phase_ratios.center, PTArray(backend).(xci)..., radius
    )
    @parallel (@idx ni .+ 1) set_normal_dyrel_shear_band_phases!(
        phase_ratios.vertex, PTArray(backend).(xvi)..., radius
    )

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology(
                (
                    PowerlawViscous(; η0 = 1.0e2, n = 3, ε0 = 1.0),
                )
            ),
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology(
                (
                    PowerlawViscous(; η0 = 1.0e-1, n = 3, ε0 = 1.0),
                )
            ),
        ),
    )

    stokes = StokesArrays(backend, ni)
    stokes_ad = AdjointStokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    stokes.ε.xx .= 1
    stokes.ε.xx_v .= 1
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([x for x in xvi[1], _ in 1:(n + 2)])
    stokes.V.Vy .= PTArray(backend)([-y for _ in 1:(n + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # Observe only velocity nodes inside the inclusion. Opposite signs on the two halves give an
    # objective that is invariant under reflection about x = 0.
    observation = adjoint_circle_observation(grid, stokes.V.Vx, radius)

    dyrel = DYREL(
        backend, stokes, rheology, phase_ratios, grid.di, Inf;
        ϵ = 1.0e-6, CFL = 0.99, γfact = 10.0,
    )
    solve_DYREL!(
        stokes,
        stokes_ad,
        ρg,
        dyrel,
        flow_bcs,
        phase_ratios,
        rheology,
        args,
        grid,
        Inf,
        igg;
        adjoint = true,
        observation,
        viscosity_relaxation = 1.0e-2,
        viscosity_cutoff = (-Inf, Inf),
        iterMax = 50.0e3,
        nout = 100,
        rel_drop = 1.0e-2,
        verbose_PH = true,
        verbose_DR = false,
    )

    η = Array(stokes.viscosity.η)
    dJdη = Array(stokes_ad.viscosity.η)
    sensitivity = η .* dJdη
    @assert all(isfinite, sensitivity) && any(!iszero, sensitivity)

    if igg.me == 0
        x_error = maximum(abs, sensitivity .- reverse(sensitivity; dims = 1)) /
            maximum(abs, sensitivity)
        y_error = maximum(abs, sensitivity .- reverse(sensitivity; dims = 2)) /
            maximum(abs, sensitivity)
        println("maximum |∂J/∂η| = ", maximum(abs, dJdη))
        println("maximum |η ∂J/∂η| = ", maximum(abs, sensitivity))
        println("x-reflection error = ", x_error)
        println("y-reflection error = ", y_error)

        limit = maximum(abs, sensitivity)
        figure = Figure(; size = (1000, 430))
        Label(
            figure[0, 1:4],
            "Magenta circle: Vx observation inside inclusion (left −, right +)",
        )
        viscosity_axis = Axis(figure[1, 1]; aspect = 1, title = "log₁₀ viscosity")
        viscosity_plot = heatmap!(
            viscosity_axis, xci..., log10.(η); colormap = :viridis
        )
        Colorbar(figure[1, 2], viscosity_plot)
        sensitivity_axis = Axis(figure[1, 3]; aspect = 1, title = "η ∂J/∂η")
        sensitivity_plot = heatmap!(
            sensitivity_axis,
            xci...,
            sensitivity;
            colormap = :vik,
            colorrange = (-limit, limit),
        )
        Colorbar(figure[1, 4], sensitivity_plot)

        θ = range(0, 2π; length = 101)
        for axis in (viscosity_axis, sensitivity_axis)
            lines!(
                axis,
                radius .* cos.(θ),
                radius .* sin.(θ);
                color = :magenta,
                linewidth = 3,
            )
        end
        mkpath(figdir)
        save(joinpath(figdir, "viscosity_sensitivity.png"), figure)
    end

    return (;
        viscosity = η,
        viscosity_sensitivity = dJdη,
        log_viscosity_sensitivity = sensitivity,
    )
end

# Run with
# `julia --project=. --startup-file=no \
#     miniapps/Adjoint2D/ShearBand2D_PowerLaw_ViscositySensitivity_NormalDYREL.jl [grid_size]`.
if !@isdefined(NO_AUTORUN)
    n = isempty(ARGS) ? 64 : parse(Int, ARGS[1])
    igg = IGG(init_global_grid(n, n, 1; init_MPI = true)...)
    shear_band_normal_dyrel_viscosity_sensitivity(igg; n)
end

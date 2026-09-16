get!(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax3D as JR3
using JustPIC
using ParallelStencil

@init_parallel_stencil(Threads, Float64, 2)

const Enzyme = JustRelax2D.Enzyme

@parallel_indices (i, j) function _init_enzyme_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
end

@parallel_indices (i, j) function _selected_control_kernel!(yc, yv, xc, xv, controls)
    @inbounds begin
        if i <= size(yc, 1) && j <= size(yc, 2)
            yc[i, j] = xc[i, j] * controls.G.center[1, i, j]
        end
        yv[i, j] = xv[i, j] * controls.G.vertex[1, i, j]
    end
    return nothing
end

function _selected_control_sensitivity!(yc, dyc, yv, dyv, xc, xv, controls, dcontrols, n)
    @parallel (@idx n) configcall = _selected_control_kernel!(yc, yv, xc, xv, controls) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _selected_control_kernel!,
        Enzyme.DuplicatedNoNeed(yc, dyc),
        Enzyme.DuplicatedNoNeed(yv, dyv),
        Enzyme.Const(xc),
        Enzyme.Const(xv),
        Enzyme.DuplicatedNoNeed(controls, dcontrols),
    )
    return nothing
end

function _selected_control_const!(yc, dyc, yv, dyv, xc, dxc, xv, dxv, controls, n)
    @parallel (@idx n) configcall = _selected_control_kernel!(yc, yv, xc, xv, controls) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _selected_control_kernel!,
        Enzyme.DuplicatedNoNeed(yc, dyc),
        Enzyme.DuplicatedNoNeed(yv, dyv),
        Enzyme.DuplicatedNoNeed(xc, dxc),
        Enzyme.DuplicatedNoNeed(xv, dxv),
        Enzyme.Const(controls),
    )
    return nothing
end

@testset "Selectively active material controls MWE" begin
    ni = (3, 2)
    yc = @zeros(ni...)
    yv = @zeros(ni .+ 1...)
    xc = @ones(ni...) .* 2.0
    xv = @ones(ni .+ 1...) .* 3.0
    controls, dcontrols = material_controls(CPUBackend, ni, (:G,); nphases = 2)
    empty_controls, empty_gradients = material_controls(CPUBackend, ni, ())
    controls3D, gradients3D = JR3.material_controls(
        CPUBackend, (3, 2, 4), (:G, :C); nphases = 2
    )

    @test keys(controls) == (:G,)
    @test !haskey(controls, :C)
    @test isempty(empty_controls)
    @test isempty(empty_gradients)
    @test size(controls.G.center) == (2, ni...)
    @test size(controls.G.vertex) == (2, (ni .+ 1)...)
    @test all(isone, controls.G.center)
    @test all(iszero, dcontrols.G.center)
    @test controls.G.center !== dcontrols.G.center
    @test keys(controls3D) == (:G, :C)
    @test size(controls3D.G.center) == (2, 3, 2, 4)
    @test size(controls3D.C.vertex) == (2, 4, 3, 5)
    @test controls3D.G.center !== gradients3D.G.center

    @parallel (@idx ni .+ 1) _selected_control_kernel!(yc, yv, xc, xv, controls)
    dyc = @ones(ni...)
    dyv = @ones(ni .+ 1...)
    _selected_control_sensitivity!(yc, dyc, yv, dyv, xc, xv, controls, dcontrols, ni .+ 1)
    @test dcontrols.G.center[1, :, :] ≈ xc
    @test dcontrols.G.vertex[1, :, :] ≈ xv
    @test all(iszero, @view(dcontrols.G.center[2, :, :]))
    @test all(iszero, @view(dcontrols.G.vertex[2, :, :]))

    dxc = @zeros(ni...)
    dxv = @zeros(ni .+ 1...)
    dyc .= 1.0
    dyv .= 1.0
    _selected_control_const!(yc, dyc, yv, dyv, xc, dxc, xv, dxv, controls, ni .+ 1)
    @test dxc ≈ one.(dxc)
    @test dxv ≈ one.(dxv)
    @test all(isone, controls.G.center)
    @test all(isone, controls.G.vertex)
end

# Minimal working example for reverse-mode differentiation of
# `compute_∇V_strain_rate_RP!` with Enzyme and ParallelStencil.
#
# Run from the repository root with:
#
#     JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no \
#         test/test_enzyme_parallelstencil.jl
@testset "Enzyme + ParallelStencil strain-rate/RP MWE" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    (; xvi) = grid
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    dyrel = DYREL(CPUBackend, ni)

    a, b = 1.3, -0.4
    stokes.V.Vx .= [a * x for x in xvi[1], _ in 1:(ni[2] + 2)]
    stokes.V.Vy .= [b * y for _ in 1:(ni[1] + 2), y in xvi[2]]
    dyrel.ηb .= 1.0

    phase_ratios = (; center = nothing)
    # Seed L = sum(εxx) + sum(εyy) + sum(εxy) + sum(RP).
    adjoint.ε.xx .= 1.0
    adjoint.ε.yy .= 1.0
    adjoint.ε.xy .= 1.0
    adjoint.R.RP .= 1.0

    JustRelax2D.enzyme_compute_∇V_strain_rate_RP!(
        stokes,
        adjoint,
        dyrel,
        nothing,
        phase_ratios,
        grid._di,
        ni,
        1.0,
        (;),
    )

    Vx̄ = zeros(size(stokes.V.Vx))
    Vȳ = zeros(size(stokes.V.Vy))
    third = 1.0 / 3.0
    for i in 1:(ni[1] + 1), j in 1:(ni[2] + 1)
        dy_vx = JustRelax.get_dy(grid._di.velocity[1], j)
        dx_vy = JustRelax.get_dx(grid._di.velocity[2], i)
        Vx̄[i, j] -= 0.5 * dy_vx
        Vx̄[i, j + 1] += 0.5 * dy_vx
        Vȳ[i, j] -= 0.5 * dx_vy
        Vȳ[i + 1, j] += 0.5 * dx_vy

        if i ≤ ni[1] && j ≤ ni[2]
            dx, dy = JustRelax.get_dxi(grid._di.vertex, i, j)
            dVx_dx_bar = 2third - third - 1.0
            dVy_dy_bar = 2third - third - 1.0
            Vx̄[i, j + 1] -= dx * dVx_dx_bar
            Vx̄[i + 1, j + 1] += dx * dVx_dx_bar
            Vȳ[i + 1, j] -= dy * dVy_dy_bar
            Vȳ[i + 1, j + 1] += dy * dVy_dy_bar
        end
    end

    @test adjoint.V.Vx ≈ Vx̄
    @test adjoint.V.Vy ≈ Vȳ
end

@testset "Enzyme DYREL kernel wrappers" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))

    adjoint.R.Rx .= 1.0
    adjoint.R.Ry .= 1.0
    JustRelax2D.enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, grid._di, ni)
    @test any(!iszero, adjoint.P)
    @test any(!iszero, adjoint.τ.xy)

    adjoint.R.Rx .= 1.0
    adjoint.R.Ry .= 1.0
    adjoint.P .= 0.0
    adjoint.τ.xy .= 0.0
    dρg = (@zeros(ni...), @zeros(ni...))
    JustRelax2D.enzyme_compute_PH_residual_V_sensitivity!(
        stokes, adjoint, ρg, dρg, grid._di, ni
    )
    @test any(!iszero, adjoint.P)
    @test any(!iszero, adjoint.τ.xy)
    @test any(!iszero, dρg[1])
    @test any(!iszero, dρg[2])

    elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
            Elasticity = elasticity,
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _init_enzyme_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _init_enzyme_phase!(phase_ratios.vertex)

    stokes.viscosity.η .= 1.0
    stokes.viscosity.ηv .= 1.0
    stokes.ε.xx .= 0.2
    stokes.ε.yy .= -0.1
    stokes.ε.xy .= 0.3
    JustRelax2D.compute_stress_DRYEL!(stokes, rheology, phase_ratios, 1.0, 1.0)
    τxx_default = copy(stokes.τ.xx)
    τxy_default = copy(stokes.τ.xy)

    empty_controls, = material_controls(CPUBackend, ni, ())
    JustRelax2D.compute_stress_DRYEL!(
        stokes, rheology, phase_ratios, 1.0, 1.0, empty_controls
    )
    @test stokes.τ.xx ≈ τxx_default
    @test stokes.τ.xy ≈ τxy_default

    controls, = material_controls(CPUBackend, ni, (:G,))
    JustRelax2D.compute_stress_DRYEL!(
        stokes, rheology, phase_ratios, 1.0, 1.0, controls
    )
    @test stokes.τ.xx ≈ τxx_default
    @test stokes.τ.xy ≈ τxy_default

    controls.G.center .= 2.0
    controls.G.vertex .= 2.0
    JustRelax2D.compute_stress_DRYEL!(
        stokes, rheology, phase_ratios, 1.0, 1.0, controls
    )
    @test stokes.τ.xx ≈ (4 / 3) .* τxx_default
    @test stokes.τ.xy ≈ (4 / 3) .* τxy_default

    θc = @zeros(ni...)
    γ_eff = @ones(ni...)
    JustRelax2D.compute_stress_viscosity_DRYEL!(
        stokes,
        θc,
        γ_eff,
        rheology,
        phase_ratios,
        1.0,
        1.0,
        1.0,
        (;),
        (-Inf, Inf),
        true,
        controls,
    )
    @test stokes.τ.xx ≈ (4 / 3) .* τxx_default
    @test stokes.τ.xy ≈ (4 / 3) .* τxy_default

    adjoint.τ.xx .= 1.0
    adjoint.τ.yy .= 1.0
    adjoint.τ.xy .= 1.0
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, 1.0, 1.0, controls
    )
    @test any(!iszero, adjoint.ε.xx)
    @test any(!iszero, adjoint.ε.xy)

    adjoint.τ.xx .= 1.0
    adjoint.τ.yy .= 1.0
    adjoint.τ.xy .= 1.0
    JustRelax2D.enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, adjoint, rheology, phase_ratios, 1.0, 1.0, controls
    )
    @test any(!iszero, adjoint.viscosity.η)
    @test any(!iszero, adjoint.viscosity.ηv)

    bcs = VelocityBoundaryConditions()
    adjoint.V.Vx .= 1.0
    adjoint.V.Vy .= 1.0
    JustRelax2D.enzyme_flow_bcs!(stokes, adjoint, bcs)
    @test all(iszero, adjoint.V.Vx[:, 1])
    @test all(iszero, adjoint.V.Vx[:, end])
    @test all(iszero, adjoint.V.Vy[1, :])
    @test all(iszero, adjoint.V.Vy[end, :])

    no_slip = (left = true, right = true, top = true, bot = true)
    bcs = VelocityBoundaryConditions(; no_slip, free_slip = map(!, no_slip))
    adjoint.V.Vx .= 1.0
    adjoint.V.Vy .= 1.0
    JustRelax2D.enzyme_no_slip!(
        stokes.V.Vx, adjoint.V.Vx, stokes.V.Vy, adjoint.V.Vy, bcs.no_slip
    )
    @test all(isfinite, adjoint.V.Vx)
    @test all(isfinite, adjoint.V.Vy)
end

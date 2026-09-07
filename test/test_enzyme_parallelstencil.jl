get!(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using JustPIC
using ParallelStencil

@init_parallel_stencil(Threads, Float64, 2)

@parallel_indices (i, j) function _init_enzyme_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
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
    adjoint.PA .= 1.0

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

    @test adjoint.VA.Vx ≈ Vx̄
    @test adjoint.VA.Vy ≈ Vȳ
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
    @test any(!iszero, adjoint.dτ.xy)

    adjoint.R.Rx .= 1.0
    adjoint.R.Ry .= 1.0
    dρg = (@zeros(ni...), @zeros(ni...))
    JustRelax2D.enzyme_compute_PH_residual_V_sensitivity!(
        stokes, adjoint, ρg, dρg, grid._di, ni
    )
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

    adjoint.dτ.xx .= 1.0
    adjoint.dτ.yy .= 1.0
    adjoint.dτ.xy .= 1.0
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, 1.0, 1.0
    )
    @test any(!iszero, adjoint.ε.xx)
    @test any(!iszero, adjoint.ε.xy)

    adjoint.dτ.xx .= 1.0
    adjoint.dτ.yy .= 1.0
    adjoint.dτ.xy .= 1.0
    JustRelax2D.enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, adjoint, rheology, phase_ratios, 1.0, 1.0
    )
    @test any(!iszero, adjoint.viscosity.η)
    @test any(!iszero, adjoint.viscosity.ηv)

    bcs = VelocityBoundaryConditions()
    adjoint.VA.Vx .= 1.0
    adjoint.VA.Vy .= 1.0
    JustRelax2D.enzyme_flow_bcs!(stokes, adjoint, bcs)
    @test all(iszero, adjoint.VA.Vx[:, 1])
    @test all(iszero, adjoint.VA.Vx[:, end])
    @test all(iszero, adjoint.VA.Vy[1, :])
    @test all(iszero, adjoint.VA.Vy[end, :])

    no_slip = (left = true, right = true, top = true, bot = true)
    bcs = VelocityBoundaryConditions(; no_slip, free_slip = map(!, no_slip))
    adjoint.VA.Vx .= 1.0
    adjoint.VA.Vy .= 1.0
    JustRelax2D.enzyme_no_slip!(
        stokes.V.Vx, adjoint.VA.Vx, stokes.V.Vy, adjoint.VA.Vy, bcs.no_slip
    )
    @test all(isfinite, adjoint.VA.Vx)
    @test all(isfinite, adjoint.VA.Vy)
end

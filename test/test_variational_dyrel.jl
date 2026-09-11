push!(LOAD_PATH, "..")

using GeoParams
using JustPIC
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
using Test

@init_parallel_stencil(Threads, Float64, 2)

@parallel_indices (i, j) function _fill_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
end

@parallel_indices (i, j) function _fill_air_rock!(phase)
    @index phase[1, i, j] = 0.5
    @index phase[2, i, j] = 0.5
    return nothing
end

function _full_volume_dyrel(igg; variational, hydrostatic = false, partial = false, plastic = false, legacy_grid = false)
    ni = (8, 8)
    grid = Geometry(ni, (1.0, 1.0))
    creep = LinearViscous(; η = 1.0)
    composite = plastic ?
        CompositeRheology((creep, DruckerPrager_regularised(; C = 0.5, ϕ = 0.0, η_vp = 0.1, Ψ = 0.0))) :
        CompositeRheology((creep,))
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = hydrostatic ? 1.0 : 0.0),
            Gravity = ConstantGravity(; g = hydrostatic ? 1.0 : 0.0),
            CompositeRheology = composite,
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _fill_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _fill_phase!(phase_ratios.vertex)

    stokes = StokesArrays(CPUBackend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    dt = 1.0
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    if !hydrostatic
        stokes.V.Vx .= [x - 0.5 for x in grid.xvi[1], _ in 1:(ni[2] + 2)]
        stokes.V.Vy .= [0.5 - y for _ in 1:(ni[1] + 2), y in grid.xvi[2]]
        @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0.0
        @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0.0
    end
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    kwargs = (;
        verbose_PH = false,
        verbose_DR = false,
        iterMax = hydrostatic ? 10_000 : 500,
        total_iterMax = hydrostatic ? 20_000 : 1_000,
        nout = 20,
        rel_drop = 0.5,
        linear_viscosity = !plastic,
        free_surface = partial,
    )
    ϕ = nothing
    if variational
        ϕ = RockRatio(CPUBackend, ni)
        update_rock_ratio!(ϕ, phase_ratios, 0)
        if partial
            ϕ.center[:, end] .= 0.25
            ϕ.vertex[:, (end - 1):end] .= 0.25
            ϕ.Vx[:, end] .= 0.25
            ϕ.Vy[:, end] .= 0.25
            ϕ.Vy[:, end - 1] .= 0.25
            ϕ.Vx[4, 4] = 0.0
        end
        dyrel = DYREL(CPUBackend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-6)
        # `legacy_grid` doubles as the switch between the two accepted keyword forms: the bundled
        # `kwargs = (; ...)` used by the miniapps and the plain keywords used by the docs.
        result = if legacy_grid
            solve_VariationalDYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args,
                grid.di, dt, igg; kwargs
            )
        else
            solve_VariationalDYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args,
                grid, dt, igg; kwargs...
            )
        end
    else
        dyrel = DYREL(CPUBackend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)
        result = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg; kwargs
        )
    end
    return stokes, result, ϕ
end

const TEST_IGG = IGG(init_global_grid(8, 8, 1; init_MPI = !JustRelax.MPI.Initialized())...)

@testset "DYREL pressure relaxation" begin
    P = zeros(2)
    JustRelax2D.relax_volumetric_mode!(P, fill(2.0, 2), fill(4.0, 2), trues(2), 0.25)
    @test P == fill(2.0, 2)
end

@testset "Variational DYREL volumetric compliance" begin
    # A uniform shift of the retained pressures moves each residual by ϕ.center / ηb,
    # because the continuity residual carries the rock fraction.
    ni = (2, 1)
    ϕ = RockRatio(CPUBackend, ni)
    ϕ.center[1, 1] = 1.0
    ϕ.center[2, 1] = 0.25
    ηb = reshape([4.0, 2.0], ni)
    mask = trues(ni)

    @test JustRelax2D.volumetric_compliance_total(ηb, ϕ, mask) ≈ 1 / 4 + 0.25 / 2
    # An incompressible rheology carries no uniform volumetric correction.
    @test JustRelax2D.volumetric_compliance_total(fill(Inf, ni), ϕ, mask) == 0.0
    mask[2, 1] = false
    @test JustRelax2D.volumetric_compliance_total(ηb, ϕ, mask) ≈ 1 / 4
end

@testset "Variational DYREL hydrostatic convergence" begin
    stokes, _, _ = _full_volume_dyrel(TEST_IGG; variational = false, hydrostatic = true)
    stokes_variational, result, _ = _full_volume_dyrel(
        TEST_IGG; variational = true, hydrostatic = true, legacy_grid = true
    )

    @test result.converged
    @test maximum(abs, stokes_variational.V.Vx) < 1.0e-5
    @test maximum(abs, stokes_variational.V.Vy) < 1.0e-5
    @test all(isfinite, stokes_variational.P)
    @test diff(stokes_variational.P; dims = 2) ≈ diff(stokes.P; dims = 2) rtol = 1.0e-4 atol = 1.0e-6

end

@testset "Variational DYREL partial-volume rows" begin
    stokes, result, ϕ = _full_volume_dyrel(
        TEST_IGG; variational = true, hydrostatic = true, partial = true
    )

    @test all(isfinite, stokes.V.Vx)
    @test all(isfinite, stokes.V.Vy)
    @test all(isfinite, stokes.P)
    @test stokes.V.Vx[4, 5] == 0.0
    # A single dry face eliminates the continuity row of both cells it separates,
    # even though they are full: the pressure of each acts back on that face through
    # the weighted gradient, so the divergence it contributes cannot be relieved.
    @test ϕ.center[3, 4] == 1.0
    @test ϕ.center[4, 4] == 1.0
    @test !JustRelax2D.isvalid_vx(ϕ, 4, 4)
    @test stokes.P[3, 4] == 0.0
    @test stokes.P[4, 4] == 0.0
    @test ϕ.center[1, end] == 0.25
    @test isfinite(result.err)

    # The fused DYREL kernel must apply the same cut-cell weights as the
    # reference variational strain-rate kernel.
    ni = size(stokes.P)
    grid = Geometry(ni, (1.0, 1.0))
    reference = StokesArrays(CPUBackend, ni)
    copyto!(reference.V.Vx, stokes.V.Vx)
    copyto!(reference.V.Vy, stokes.V.Vy)
    @parallel (@idx ni) JustRelax2D.compute_∇V!(reference.∇V, @velocity(reference), ϕ, grid._di.vertex)
    @parallel (@idx ni .+ 1) JustRelax2D.compute_strain_rate!(
        @strain(reference)...,
        reference.∇V,
        @velocity(reference)...,
        ϕ,
        grid._di.vertex,
        grid._di.velocity...,
    )
    @test stokes.ε.xx ≈ reference.ε.xx
    @test stokes.ε.yy ≈ reference.ε.yy
    @test stokes.ε.xy ≈ reference.ε.xy
end

# Rock fractions of a flat free surface lying `frac` of a cell above vertex row
# `jcut`, on a grid of unit cells with vertices at integer heights. Each staggered
# control volume gets the fraction of its own box that sits below the surface, the
# way a geometric cut-cell routine builds them: the `Vy` and vertex volumes straddle
# two cell halves, so they run dry as soon as the surface drops below a cell centre.
function _flat_surface_ratios(ni, jcut, frac)
    ϕ = RockRatio(CPUBackend, ni)
    nx, ny = ni
    h = jcut + frac
    below(ybot, ytop) = clamp((h - ybot) / (ytop - ybot), 0, 1)
    for j in 1:ny, i in 1:nx
        ϕ.center[i, j] = below(j - 1, j)
    end
    for j in 1:ny, i in 1:(nx + 1)
        ϕ.Vx[i, j] = below(j - 1, j)
    end
    for j in 1:(ny + 1), i in 1:nx
        ϕ.Vy[i, j] = below(j - 1.5, j - 0.5)
    end
    for j in 1:(ny + 1), i in 1:(nx + 1)
        ϕ.vertex[i, j] = below(j - 1.5, j - 0.5)
    end
    return ϕ
end

# A column at rest under a flat free surface. `ρ = g = 1` and the cells are unit
# squares, so the exact pressure of a full cell is its depth below the surface, and
# the exact `ϕ.center * P` of the cut cell is half its rock thickness — half a cell
# in the stored, ϕ-divided convention the momentum operator works in.
function _hydrostatic_column(igg, frac; ni = (4, 8), jcut = 5, start_at_rest = true)
    grid = Geometry(ni, Float64.(ni))
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _fill_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _fill_phase!(phase_ratios.vertex)

    ϕ = _flat_surface_ratios(ni, jcut, frac)
    h = jcut + frac
    stokes = StokesArrays(CPUBackend, ni)
    for j in axes(stokes.P, 2), i in axes(stokes.P, 1)
        f = ϕ.center[i, j]
        ytop = min(h, float(j))
        stokes.P[i, j] = iszero(f) ? 0.0 : ((h - (j - 1)) + (h - ytop)) / 2 / f
    end
    exact = copy(stokes.P)
    start_at_rest || fill!(stokes.P, 0.0)

    ρg = @zeros(ni...), @zeros(ni...)
    dt = 1.0
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    dyrel = DYREL(CPUBackend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-10)
    result = solve_VariationalDYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
        kwargs = (;
            air_phase = 0, verbose_PH = false, verbose_DR = false,
            linear_viscosity = true, iterMax = 50_000, total_iterMax = 100_000, nout = 50,
        ),
    )
    return stokes, ϕ, exact, result
end

@testset "Variational DYREL hydrostatic cut cells" begin
    for frac in (0.05, 0.25, 0.5, 0.75, 0.95)
        stokes, ϕ, exact, result = _hydrostatic_column(TEST_IGG, frac)

        @test result.converged
        # A column already at rest must stay there, whether the surface cuts the top
        # cell above or below its centre.
        @test maximum(abs, stokes.V.Vy) < 1.0e-8

        # `isvalid_c` keeps the cut cell only while all four of its faces carry
        # liquid. Its top `Vy` control volume straddles the two cell halves either
        # side of the cell centre, so it runs dry as soon as the surface does.
        if frac > 0.5
            @test stokes.P ≈ exact atol = 1.0e-8
            # ϕ.center * P is the pressure the momentum operator sees; in the cut
            # cell it is the mean pressure over the rock it holds.
            @test ϕ.center[2, 6] * stokes.P[2, 6] ≈ frac / 2 atol = 1.0e-8
        else
            # Eliminating the cut cell takes its weight out of the column, and every
            # pressure below is short by the half-thickness of rock it held. This is
            # the accuracy the null-space rule costs at a free surface.
            @test stokes.P[2, 6] == 0.0
            @test all(
                isapprox(exact[i, j] - stokes.P[i, j], frac / 2; atol = 1.0e-8)
                    for j in 1:5, i in axes(stokes.P, 1)
            )
        end
    end
end

@testset "Variational DYREL cut-cell penalty" begin
    # `γ_eff` is the step length of the pressure update `P += γ_eff * RP`, and `RP`
    # carries `ϕ.center` once. Their product is what sets how fast a pressure relaxes,
    # so it must not depend on how much rock the cell holds; otherwise a cut cell
    # converges `ϕ.center` times slower than the interior and is left visibly
    # under-converged along a free surface.
    ni = (3, 1)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _fill_phase!(phase_ratios.center)

    ϕ = RockRatio(CPUBackend, ni)
    foreach(f -> fill!(getfield(ϕ, f), 1.0), (:center, :vertex, :Vx, :Vy))
    ϕ.center[2, 1] = 0.5
    ϕ.center[3, 1] = 0.05

    ηb, γ_eff = @zeros(ni...), @zeros(ni...)
    η = @ones(ni...)
    @parallel (@idx ni) JustRelax2D.compute_bulk_viscosity_and_penalty!(
        ηb, γ_eff, rheology, phase_ratios.center, η, ϕ, 1.0, 4.0, 1.0
    )

    @test all(γ_eff .* ϕ.center .≈ γ_eff[1, 1])
    # The bulk viscosity is a coefficient of the continuity equation, not a step
    # length, and stays the plain material value.
    @test all(ηb .≈ ηb[1, 1])
end

@testset "Variational DYREL vertex viscosity" begin
    # `compute_stress_DRYEL!` builds the vertex stress from the harmonic mean of the
    # four surrounding cell viscosities, so the Gershgorin diagonal has to sample the
    # same combination. Sampling an independently computed vertex viscosity instead
    # preconditions a different operator wherever the viscosity is heterogeneous.
    ni = (4, 4)
    grid = Geometry(ni, Float64.(ni))
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _fill_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _fill_phase!(phase_ratios.vertex)

    ϕ = RockRatio(CPUBackend, ni)
    foreach(f -> fill!(getfield(ϕ, f), 1.0), (:center, :vertex, :Vx, :Vy))

    # heterogeneous by four orders of magnitude, so a harmonic mean is nowhere close
    # to any single sample it is built from
    η = [10.0^(i + j) for i in 1:ni[1], j in 1:ni[2]]
    γ_eff = @zeros(ni...)
    Dx, λmaxVx = @zeros(ni[1] - 1, ni[2]), @zeros(ni[1] - 1, ni[2])
    Dy, λmaxVy = @zeros(ni[1], ni[2] - 1), @zeros(ni[1], ni[2] - 1)

    JustRelax2D.Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, γ_eff, phase_ratios, ϕ, rheology, grid.di, 1.0
    )

    # unit rock fraction and no penalty, so the diagonal is the bare viscous stencil
    harm(i, j) = JustRelax2D.harm_clamped(η, JustRelax2D.clamped_indices(ni, i, j)...)
    @test grid.di.center == grid.di.vertex
    _dx2, _dy2 = inv.(grid.di.center) .^ 2
    for j in axes(Dx, 2), i in axes(Dx, 1)
        @test Dx[i, j] ≈ (harm(i + 1, j + 1) + harm(i + 1, j)) * _dy2 +
            4 / 3 * (η[i + 1, j] + η[i, j]) * _dx2
    end
    for j in axes(Dy, 2), i in axes(Dy, 1)
        @test Dy[i, j] ≈ (harm(i, j + 1) + harm(i + 1, j + 1)) * _dx2 +
            4 / 3 * (η[i, j + 1] + η[i, j]) * _dy2
    end
    @test all(isfinite, λmaxVx)
    @test all(isfinite, λmaxVy)
end

@testset "Variational DYREL cut-cell convergence rate" begin
    # From a cold start the cut cell must not be the slow mode: a thin sliver of rock
    # has to reach the same pressure in the same order of iterations as a nearly full
    # one.
    iters = map((0.95, 0.5, 0.05)) do frac
        stokes, ϕ, exact, result = _hydrostatic_column(TEST_IGG, frac; start_at_rest = false)
        @test result.converged
        # Where the cut cell survives `isvalid_c` the column is exact; where it does
        # not, the whole column below sits `ρg·dz·ϕ.center/2` low, uniformly.
        if frac > 0.5
            @test stokes.P ≈ exact atol = 1.0e-8
        else
            @test all(
                isapprox(exact[i, j] - stokes.P[i, j], frac / 2; atol = 1.0e-8)
                    for j in 1:5, i in axes(stokes.P, 1)
            )
        end
        result.iter
    end
    @test maximum(iters) < 2 * minimum(iters)
end

@testset "Variational DYREL plasticity" begin
    stokes, result, _ = _full_volume_dyrel(TEST_IGG; variational = true, plastic = true)

    @test all(isfinite, stokes.V.Vx)
    @test all(isfinite, stokes.V.Vy)
    @test all(isfinite, stokes.P)
    @test all(isfinite, stokes.λ)
    @test minimum(stokes.λ) ≥ 0.0
    @test isfinite(result.err)
end

@testset "Variational DYREL thermal and melt dispatch" begin
    ni = (2, 2)
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _fill_phase!(phase_ratios.center)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 1.0, α = 3.0e-5, β = 0.0, T0 = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    ΔT = fill(2.0, ni .+ 2)
    melt_fraction = fill(0.1, ni)
    RP_thermal = JustRelax2D._RP_cell(
        0.0, 0.0, 0.5, 0.0, Inf, 1.0, rheology, phase_ratios.center, ΔT, nothing, 1, 1
    )
    RP_melt = JustRelax2D._RP_cell(
        0.0, 0.0, 0.5, 0.0, Inf, 1.0, rheology, phase_ratios.center, ΔT, melt_fraction, 1, 1
    )

    @test RP_thermal ≈ -0.5 + 6.0e-5
    @test RP_melt ≈ RP_thermal
    @test isfinite(RP_melt)
end

@testset "Variational DYREL air-phase buoyancy" begin
    ni = (2, 2)
    phase_ratios = PhaseRatios(JustPIC.CPU, 2, ni)
    @parallel (@idx ni) _fill_air_rock!(phase_ratios.center)
    rheology = (
        SetMaterialParams(; Phase = 1, Density = ConstantDensity(; ρ = 1.0), Gravity = ConstantGravity(; g = 1.0)),
        SetMaterialParams(; Phase = 2, Density = ConstantDensity(; ρ = 3.0), Gravity = ConstantGravity(; g = 1.0)),
    )
    ρg = zeros(ni)
    args = (; T = zeros(ni .+ 2), P = zeros(ni))

    JustRelax2D.compute_ρg!(ρg, phase_ratios, rheology, args; air_phase = 1)
    @test all(==(3.0), ρg)
end

@testset "Variational DYREL full-volume equivalence" begin
    stokes, _, _ = _full_volume_dyrel(TEST_IGG; variational = false)
    stokes_variational, _, _ = _full_volume_dyrel(TEST_IGG; variational = true)

    @test stokes_variational.V.Vx ≈ stokes.V.Vx rtol = 1.0e-5
    @test stokes_variational.V.Vy ≈ stokes.V.Vy rtol = 1.0e-5
    @test maximum(abs, stokes.P) < 1.0e-5
    @test maximum(abs, stokes_variational.P) < 1.0e-5
    @test all(isfinite, stokes_variational.P)
    @test all(isfinite, stokes_variational.V.Vx)
    @test all(isfinite, stokes_variational.V.Vy)

    finalize_global_grid(; finalize_MPI = true)
end

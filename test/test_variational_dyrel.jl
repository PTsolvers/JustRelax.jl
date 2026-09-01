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

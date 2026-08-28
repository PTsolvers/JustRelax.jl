using GeoParams
using JustPIC
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

@init_parallel_stencil(Threads, Float64, 2)

@parallel_indices (i, j) function fill_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
end

function run_case(igg, grid, rheology, phase_ratios; variational)
    ni = size(phase_ratios.center)
    stokes = StokesArrays(CPUBackend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = 1.0)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    kwargs = (;
        verbose_PH = false,
        verbose_DR = false,
        iterMax = 10_000,
        total_iterMax = 20_000,
        nout = 20,
        rel_drop = 0.5,
        linear_viscosity = true,
        free_surface = variational,
    )

    if variational
        ϕ = RockRatio(CPUBackend, ni)
        update_rock_ratio!(ϕ, phase_ratios, 0)
        ϕ.center[:, end] .= 0.25
        ϕ.vertex[:, (end - 1):end] .= 0.25
        ϕ.Vx[:, end] .= 0.25
        ϕ.Vy[:, (end - 1):end] .= 0.25
        dyrel = DYREL(CPUBackend, stokes, rheology, phase_ratios, ϕ, grid.di, 1.0; ϵ = 1.0e-6)
        result = solve_VariationalDYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ,
            rheology, args, grid, 1.0, igg; kwargs,
        )
    else
        dyrel = DYREL(CPUBackend, stokes, rheology, phase_ratios, grid.di, 1.0; ϵ = 1.0e-6)
        result = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios,
            rheology, args, grid, 1.0, igg; kwargs,
        )
    end
    return stokes, result
end

function main()
    ni = (32, 32)
    grid = Geometry(ni, (1.0, 1.0))
    igg = IGG(init_global_grid(ni..., 1)...)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) fill_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) fill_phase!(phase_ratios.vertex)

    standard, _ = run_case(igg, grid, rheology, phase_ratios; variational = false)
    variational, result = run_case(igg, grid, rheology, phase_ratios; variational = true)
    println("variational converged: ", result.converged)
    println("standard max |Vy|: ", maximum(abs, standard.V.Vy))
    println("variational max |Vy|: ", maximum(abs, variational.V.Vy))

    finalize_global_grid()
    return nothing
end

main()

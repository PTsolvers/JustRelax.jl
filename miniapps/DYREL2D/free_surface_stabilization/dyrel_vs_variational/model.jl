# Model setup shared by both runs: the Rayleigh-Taylor instability of the two DYREL miniapps.
# Must be included after `@init_parallel_stencil`.

function init_phases!(phases, particles, A)
    ni = size(phases)

    @parallel_indices (i, j) function _init_phases!(phases, px, py, index, A)

        f(x, A, λ) = A * sin(π * x / λ)

        for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 2.0

            if 0.0e0 ≤ depth ≤ 100.0e3
                @index phases[ip, i, j] = 1.0

            elseif depth > (-f(x, A, 500.0e3) + (200.0e3 - A))
                @index phases[ip, i, j] = 3.0

            end
        end
        return nothing
    end

    return @parallel (@idx ni) _init_phases!(phases, particles.coords..., particles.index, A)
end

rayleigh_taylor_rheology() = (
    SetMaterialParams(;
        Phase = 1,
        Density = ConstantDensity(; ρ = 1.0e0),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e16),)),
        Gravity = ConstantGravity(; g = 9.81),
    ),
    SetMaterialParams(;
        Phase = 2,
        Density = ConstantDensity(; ρ = 3.3e3),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
        Gravity = ConstantGravity(; g = 9.81),
    ),
    SetMaterialParams(;
        Phase = 3,
        Density = ConstantDensity(; ρ = 3.2e3),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),)),
        Gravity = ConstantGravity(; g = 9.81),
    ),
)

"""
    model_setup(nx, ny)

Grid, particles, phase ratios and rheology for the Rayleigh-Taylor problem. `SEED` fixes the
particle seeding so both runs start from an identical particle distribution.
"""
function model_setup(nx, ny)
    ni = nx, ny
    li = LX, LY
    di = @. li / ni
    origin = 0.0, -LY
    grid = Geometry(ni, li; origin = origin)

    rheology = rayleigh_taylor_rheology()

    Random.seed!(SEED)
    nxcell, max_xcell, min_xcell = NXCELL
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, AMPLITUDE)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    return (; ni, li, di, origin, grid, rheology, particles, pPhases, particle_args, phase_ratios)
end

"""
    stokes_setup(m, free_surface_bc)

Stokes arrays, buoyancy and boundary conditions for the model `m`. `free_surface_bc` selects
how the sticky-air top is stabilized, which is the one boundary-condition difference between
the two miniapps.
"""
function stokes_setup(m, free_surface_bc)
    (; ni, di, rheology, phase_ratios) = m
    stokes = StokesArrays(backend_JR, ni)
    thermal = ThermalArrays(backend_JR, ni)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_lithostatic_pressure!(stokes.P, ρg[2], di[2])

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        free_surface = free_surface_bc,
    )

    return (; stokes, thermal, ρg, args, flow_bcs)
end

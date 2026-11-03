# The two time loops under comparison. Both are stripped to the parts that affect the solver:
# no figures are written, and nothing is printed per iteration. Each returns one record per
# time step.
#
# `solve_DYREL!` returns `(; err_evo_it, err_evo_V, err_evo_P, err_evo_tot, err, iter, converged)`;
# `iter` is the total number of dynamic-relaxation iterations the step consumed across all
# Powell-Hestenes sweeps, which is the quantity being compared.

step_record(it, dt, t, out, elapsed) = (;
    step = it,
    dt_kyr = dt / (1.0e3 * SECYR),
    t_kyr = t / (1.0e3 * SECYR),
    iter = out.iter,
    err = out.err,
    converged = out.converged,
    seconds = elapsed,
)

function log_step(label, r)
    return @printf(
        "%-12s step %3d  t = %8.2f kyr  dt = %7.2f kyr  iter = %7d  err = %.3e  %-13s %6.1f s\n",
        label, r.step, r.t_kyr, r.dt_kyr, r.iter, r.err,
        r.converged ? "converged" : "NOT CONVERGED", r.seconds
    )
end

"""
    run_standard(igg; nsteps)

`RayleighTaylor2D_DYREL.jl`: non-variational DYREL, sticky air handled through the free-surface
boundary condition, plain RK2 particle advection.
"""
function run_standard(igg; nsteps = NSTEPS, γfact = GAMMA_FACT_STANDARD)
    m = model_setup(N, N)
    (; grid, di, rheology, particles, pPhases, particle_args, phase_ratios) = m
    (; stokes, ρg, args, flow_bcs) = stokes_setup(m, BC_FREE_SURFACE_STANDARD)

    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    dyrel = DYREL(
        backend_JR, stokes, rheology, phase_ratios, grid.di, DT;
        ϵ = DYREL_TOL, γfact = γfact, CFL = DR_CFL, c_fact = C_FACT
    )

    records = typeof(step_record(0, DT, 0.0, (; iter = 0, err = 0.0, converged = true), 0.0))[]
    t, dt = 0.0, DT

    for it in 1:nsteps
        elapsed = @elapsed out = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = SOLVER_KWARGS
        )

        r = step_record(it, dt, t + dt, out, elapsed)
        push!(records, r)
        log_step("standard", r)

        ADAPTIVE_DT && (dt = compute_dt(stokes, di, DT_MAX))

        advection!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)
        inject_particles_phase!(particles, pPhases, (), ())
        update_phase_ratios!(phase_ratios, particles, pPhases)

        t += dt
    end

    return records
end

"""
    run_variational(igg; nsteps)

`RayleighTaylor2D_VariationalStokes_DYREL.jl`: variational DYREL on a `RockRatio` mask, sticky
air handled through a marker chain, MQS particle advection.
"""
function run_variational(igg; nsteps = NSTEPS, γfact = GAMMA_FACT_VARIATIONAL)
    m = model_setup(N, N)
    (; grid, di, origin, rheology, particles, pPhases, particle_args, phase_ratios) = m
    (; xci, xvi) = grid
    (; stokes, ρg, args, flow_bcs) = stokes_setup(m, BC_FREE_SURFACE_VARIATIONAL)

    nxcell, min_xcell, max_xcell = NXCELL_CHAIN
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], INIT_ELEVATION)

    ϕ = RockRatio(backend_JR, m.ni)
    compute_rock_fraction!(ϕ, chain, xvi, di)
    grid_vxi = velocity_grids(xci, xvi, di)

    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf); air_phase = AIR_PHASE)

    dyrel = DYREL(
        backend_JR, stokes, rheology, phase_ratios, ϕ, grid.di, DT;
        ϵ = DYREL_TOL, γfact = γfact, CFL = DR_CFL, c_fact = C_FACT
    )

    records = typeof(step_record(0, DT, 0.0, (; iter = 0, err = 0.0, converged = true), 0.0))[]
    t, dt = 0.0, DT

    for it in 1:nsteps
        elapsed = @elapsed out = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
            kwargs = (; VARIATIONAL_SOLVER_KWARGS..., air_phase = AIR_PHASE)
        )

        r = step_record(it, dt, t + dt, out, elapsed)
        push!(records, r)
        log_step("variational", r)

        ADAPTIVE_DT && (dt = compute_dt(stokes, di, DT_MAX))

        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)

        # JustPIC v0.6.7 applies its mean-height correction with the opposite sign. Preserve
        # the pre-advection mean and translate every chain representation consistently.
        chain_mean0 = sum(chain.h_vertices) / length(chain.h_vertices)
        semilagrangian_advection_markerchain!(
            chain, RungeKutta2(), @velocity(stokes), grid_vxi, xvi, dt
        )
        chain_shift = chain_mean0 - sum(chain.h_vertices) / length(chain.h_vertices)
        chain.h_vertices .+= chain_shift
        chain.h_vertices0 .+= chain_shift
        chain.coords[2].data .+= chain_shift
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, AIR_PHASE)
        inject_particles_phase!(particles, pPhases, (), ())

        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        t += dt
    end

    return records
end

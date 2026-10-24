# Initial deformation of a viscous layer under sinusoidal surface pressure.
# Run in the miniapps environment. No plotting or output files are required.
using JustRelax, JustRelax.JustRelax2D
using GeoParams, ParallelStencil, LinearAlgebra

@init_parallel_stencil(Threads, Float64, 2)

@parallel_indices (i, j) function surface_load_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
end

function surface_pressure_benchmark(n = 32; solver = :pt, amplitude = 0.5)
    solver in (:pt, :dyrel) || throw(ArgumentError("solver must be :pt or :dyrel"))
    ni = (n, n)
    igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    try
        grid = Geometry(ni, (1.0, 1.0))
        stokes = StokesArrays(CPUBackend, ni)
        ϕ = RockRatio(CPUBackend, ni)
        jcut = 3n ÷ 4
        h = jcut + 0.75
        for f in (:center, :Vx, :Vy, :vertex)
            A = getfield(ϕ, f)
            shift = f in (:Vy, :vertex) ? 1.5 : 1.0
            A .= [clamp(h - (j - shift), 0.0, 1.0) for i in axes(A, 1), j in axes(A, 2)]
        end
        phase = PhaseRatios(JustPIC.CPU, 1, ni)
        @parallel (@idx ni) surface_load_phase!(phase.center)
        @parallel (@idx ni .+ 1) surface_load_phase!(phase.vertex)
        rheology = (
            SetMaterialParams(;
                Phase = 1, Density = ConstantDensity(; ρ = 1.0), Gravity = ConstantGravity(; g = 1.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            ),
        )
        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true),
        )
        # Extend the surface profile vertically through every stencil sample.
        load = [1.0 + amplitude * cospi(2x) for x in grid.xci[1], _ in grid.xci[2]]
        ρg = @zeros(ni...), @zeros(ni...)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P)
        result = if solver == :pt
            pt = PTStokesCoeffs(grid.li, minimum.(grid.di.vertex); ϵ_abs = 1.0e-8, ϵ_rel = 1.0e-8)
            solve_VariationalStokes!(
                stokes, pt, grid, flow_bcs, ρg, phase, ϕ, rheology, args, 0.01, igg;
                surface_pressure = load, free_surface = true,
                iterMax = 50_000, nout = 250, verbose = false,
            )
        else
            dyrel = DYREL(CPUBackend, stokes, rheology, phase, ϕ, grid.di, 0.01; ϵ = 1.0e-7)
            solve_VariationalDYREL!(
                stokes, ρg, dyrel, flow_bcs, phase, ϕ, rheology, args, grid, 0.01, igg;
                surface_pressure = load, free_surface = true, linear_viscosity = true,
                total_iterMax = 100_000, nout = 100, verbose_PH = false, verbose_DR = false,
            )
        end
        vy = copy(stokes.V.Vy[2:(end - 1), jcut + 2])
        profile = cospi.(2 .* grid.xci[1])
        response = dot(vy, profile) / dot(profile, profile)
        converged = solver == :pt ? result.iter < 50_000 : result.converged
        @assert converged "The surface-load solve did not converge"
        @assert response < 0 "High-pressure regions must initially move downwards"
        return (; n, solver, response, surface_velocity = vy, pressure = copy(stokes.P), result)
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for n in (16, 32)
        pt = surface_pressure_benchmark(n; solver = :pt)
        dyrel = surface_pressure_benchmark(n; solver = :dyrel)
        difference = norm(pt.surface_velocity - dyrel.surface_velocity) / norm(pt.surface_velocity)
        println((; n, pt_response = pt.response, dyrel_response = dyrel.response, relative_difference = difference))
        @assert difference < 1.0e-3 "PT and DYREL surface velocities disagree"
    end
end

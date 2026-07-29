# SolKz convergence comparison: plain DYREL vs. variational DYREL with ϕ ≡ 1 (RockRatio
# built via `update_rock_ratio!(ϕ, phase_ratios, 0)` — air_phase = 0 is the "no void phase"
# convention, see `compute_rock_ratio` in variational_stokes/mask.jl). Both solvers scale the
# penalty γ_num by the local viscosity (γfact ≈ 20 optimal at Δη = 1e6), so at ϕ ≡ 1 they solve
# the same discrete problem; this script overlays their L1-error curves across a resolution
# range as a check on the variational path.

include(joinpath(@__DIR__, "SolKz_DYREL.jl"))

function solKz_DYREL_VS(;
        Δη = 1.0e6, km = 2, nx = 64, ny = 64, lx = 1.0e0, ly = 1.0e0,
        ϵ = 1.0e-6, γfact = 50,
        init_MPI = true, finalize_MPI = false, figdir = nothing
    )

    ni = nx, ny
    li = lx, ly
    origin = 0.0, 0.0
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_MPI)...)
    di = @. li / (nx_g(), ny_g())
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid

    ttot = 1
    Δt = 1
    dt = 0.1

    η_target = solKz_viscosity(xci, ni; B = log(Δη))
    ρ = solKz_density(xci, ni; km = km)

    creep = CustomRheology(custom_εII, custom_τII, NamedTuple())
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = Vector_Density(; rho = vec(ρ)),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((creep,)),
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios)
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, 0)

    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    index_field = reshape(1:prod(ni), ni)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt, η_target = η_target, index = index_field)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    !isnothing(figdir) && take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ, γfact)

    t = 0.0
    local iters
    while t < ttot
        iters = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            ϕ,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 5.0e3 * nx,
                total_iterMax = 5.0e3 * nx,
                nout = 100,
                rel_drop = 0.1,
                linear_viscosity = true,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        t += Δt
    end

    if !isnothing(figdir)
        fig = Figure(size = (1200, 500))
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\log_{10}(\eta)")
        ax2 = Axis(fig[1, 3], aspect = 1, title = L"P")
        h1 = heatmap!(ax1, xci..., Array(log10.(stokes.viscosity.η)), colormap = :batlow)
        h2 = heatmap!(ax2, xci..., Array(stokes.P), colormap = :vik)
        Colorbar(fig[1, 2], h1); Colorbar(fig[1, 4], h2)
        save(joinpath(figdir, "SolKz_DYREL_VS.png"), fig)
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

function multiple_solKz_DYREL_vs_VS(;
        Δη = 1.0e6, km = 2, nrange::UnitRange = 4:8, ϵ = 1.0e-7,
        γfact_plain = 20, γfact_vs = 20,
    )
    L2_vx_plain, L2_vy_plain, L2_p_plain = Float64[], Float64[], Float64[]
    L2_vx_vs, L2_vy_vs, L2_p_vs = Float64[], Float64[], Float64[]

    for i in nrange
        nx = ny = 2^i - 1

        geometry, stokes, iters = solKz_DYREL(;
            Δη = Δη, km = km, nx = nx, ny = ny, ϵ = ϵ, γfact = γfact_plain,
            init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false,
        )
        get(iters, :converged, true) || error("SolKz DYREL (plain) did not converge at nx = $nx")
        vx, vy, p = Li_error(geometry, stokes; order = 1, Δη = Δη, km = km)
        push!(L2_vx_plain, vx); push!(L2_vy_plain, vy); push!(L2_p_plain, p)

        geometry_vs, stokes_vs, iters_vs = solKz_DYREL_VS(;
            Δη = Δη, km = km, nx = nx, ny = ny, ϵ = ϵ, γfact = γfact_vs,
            init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false,
        )
        get(iters_vs, :converged, true) || error("SolKz DYREL-VS (ϕ≡1) did not converge at nx = $nx")
        vx_vs, vy_vs, p_vs = Li_error(geometry_vs, stokes_vs; order = 1, Δη = Δη, km = km)
        push!(L2_vx_vs, vx_vs); push!(L2_vy_vs, vy_vs); push!(L2_p_vs, p_vs)
    end

    nx = @. 2^nrange - 1
    h = @. (1 / nx)

    f = Figure(; fontsize = 14)
    ax = Axis(
        f[1, 1];
        yscale = log10,
        xscale = log10,
        yminorticksvisible = true,
        yminorticks = IntervalsBetween(8),
    )
    lines!(ax, h, L2_vx_plain; linewidth = 3, color = :steelblue, label = "Vx (plain)")
    lines!(ax, h, L2_vy_plain; linewidth = 3, color = :orangered, label = "Vy (plain)")
    lines!(ax, h, L2_p_plain; linewidth = 3, color = :seagreen, label = "P (plain)")
    lines!(ax, h, L2_vx_vs; linewidth = 3, linestyle = :dash, color = :steelblue, label = "Vx (VS, ϕ≡1)")
    lines!(ax, h, L2_vy_vs; linewidth = 3, linestyle = :dash, color = :orangered, label = "Vy (VS, ϕ≡1)")
    lines!(ax, h, L2_p_vs; linewidth = 3, linestyle = :dash, color = :seagreen, label = "P (VS, ϕ≡1)")
    axislegend(ax; position = :lt)
    ax.xlabel = "h"
    ax.ylabel = "L1 norm"

    save(joinpath(@__DIR__, "SolKz_DYREL_vs_VS_convergence.png"), f)
    jldsave(
        joinpath(@__DIR__, "solkz_dyrel_vs_vs_convergence.jld2");
        h, L2_vx_plain, L2_vy_plain, L2_p_plain, L2_vx_vs, L2_vy_vs, L2_p_vs,
        Δη, km, nrange, γfact_plain, γfact_vs,
    )

    return f
end

if abspath(PROGRAM_FILE) == @__FILE__
    multiple_solKz_DYREL_vs_VS()
end

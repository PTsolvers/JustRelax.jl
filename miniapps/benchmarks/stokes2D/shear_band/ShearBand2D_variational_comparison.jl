using GeoParams, CairoMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        p = GGU.Point(x, y)
        if GGU.inside(p, circle)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0

        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)
    return nothing
end

n = 256
nx = n
ny = n
figdir = "VariationalComparison"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt = Inf
    dt = Inf

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30            # friction angle
    C = τ_y           # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 8.0e-3          # regularisation "viscosity"
    dt = η0 / G0 / 4.0     # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((visc, el_bg, )),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((visc, el_inc, )),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @rand(ni...)

    # Initialize phase ratios -------------------------------

    phase_ratios_vel = PhaseRatios(backend_JP, length(rheology), ni)
    phase_ratios_displ = PhaseRatios(backend_JP, length(rheology), ni)

    radius = 0.1
    origin = 0.5, 0.5
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios_vel, xci, xvi, circle)
    init_phases!(phase_ratios_displ, xci, xvi, circle)

    air_phase = 0

    ϕ_vel = RockRatio(backend, ni)
    ϕ_displ = RockRatio(backend, ni)

    update_rock_ratio!(ϕ_vel, phase_ratios_vel, air_phase)
    update_rock_ratio!(ϕ_displ, phase_ratios_displ, air_phase)


    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem

    stokes_vel = StokesArrays(backend, ni)
    stokes_displ = StokesArrays(backend, ni)

    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.95 / √2.1)
    # pt_stokes = PTStokesCoeffs(li, di; ϵ_rel=1e-6, Re=3e0, r=0.7, CFL = 0.95 / √2.1)

    # Buoyancy forces
    ρg_vel = @zeros(ni...), @zeros(ni...)
    ρg_displ = @zeros(ni...), @zeros(ni...)

    args_vel = (; T = @zeros(ni...), P = stokes_vel.P, dt = dt, perturbation_C = perturbation_C)
    args_displ = (; T = @zeros(ni...), P = stokes_displ.P, dt = dt, perturbation_C = perturbation_C)

    # Rheology
    compute_viscosity!(
        stokes_vel, phase_ratios_vel, args_vel, rheology, (-Inf, Inf); air_phase = air_phase
    )
    compute_viscosity!(
        stokes_displ, phase_ratios_displ, args_displ, rheology, (-Inf, Inf); air_phase = air_phase
    )

    # Boundary conditions
    flow_bcs_vel = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes_vel.V.Vx .= PTArray(backend)([ x * εbg  for x in xvi[1], _ in 1:(ny + 2)])
    stokes_vel.V.Vy .= PTArray(backend)([-y * εbg  for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes_vel, flow_bcs_vel) # apply boundary conditions
    velocity2displacement!(stokes_vel, dt)
    update_halo!(@velocity(stokes_vel)...)

    flow_bcs_displ = DisplacementBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes_displ.U.Ux .= PTArray(backend)([ x * εbg * dt for x in xvi[1], _ in 1:(ny + 2)])
    stokes_displ.U.Uy .= PTArray(backend)([-y * εbg * dt for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes_displ, flow_bcs_displ) # apply boundary conditions
    displacement2velocity!(stokes_displ, dt)
    update_halo!(@velocity(stokes_displ)...)

    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    tmax = 5
    τII_vel = Float64[]
    τII_displ = Float64[]
    sol = Float64[]
    iter_vel = Float64[]
    iter_displ = Float64[]
    ttot = Float64[]

    # while t < tmax
    for _ in 1:15

        # Stokes solver ----------------
        iters_vel = solve_VariationalStokes!(
            stokes_vel,
            pt_stokes,
            di,
            flow_bcs_vel,
            ρg_vel,
            phase_ratios_vel,
            ϕ_vel,
            rheology,
            args_vel,
            dt,
            igg;
            kwargs = (;
                iterMax = 50.0e5,
                nout = 1.0,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes_vel.ε)
        push!(τII_vel, maximum(stokes_vel.τ.xx))
        push!(iter_vel, iters_vel.iter)

        # Stokes solver ----------------
        iters_displ = solve_VariationalStokes_displacement!(
            stokes_displ,
            pt_stokes,
            di,
            flow_bcs_displ,
            ρg_displ,
            phase_ratios_displ,
            ϕ_displ,
            rheology,
            args_displ,
            dt,
            igg;
            kwargs = (;
                iterMax = 50.0e5,
                nout = 1.0,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes_displ.ε)
        push!(τII_displ, maximum(stokes_displ.τ.xx))
        push!(iter_displ, iters_displ.iter)
        it += 1
        t += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # visualisation
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5

        fig = Figure(size = (3200, 1600), title = "t = $t")

        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[1, 2], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 3], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[1, 4], aspect = 1)
        ax5 = Axis(fig[2, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax6 = Axis(fig[2, 2], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax7 = Axis(fig[2, 3], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax8 = Axis(fig[2, 4], aspect = 1, title = L"PT-iterations")

        heatmap!(ax1, xci..., Array(stokes_vel.τ.II), colormap = :batlow)
        heatmap!(ax2, xci..., Array((stokes_vel.EII_pl)), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes_vel.ε.II)), colormap = :batlow)

        heatmap!(ax5, xci..., Array(stokes_displ.τ.II), colormap = :batlow)
        heatmap!(ax6, xci..., Array((stokes_displ.EII_pl)), colormap = :batlow)
        heatmap!(ax7, xci..., Array(log10.(stokes_displ.ε.II)), colormap = :batlow)

        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax6, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII_vel, linestyle = :dash, color = :blue, label = "velocity")
        lines!(ax4, ttot, τII_displ, linestyle = :dot, color = :red, label = "displacement")
        lines!(ax4, ttot, sol, color = :black, label = "solution")

        lines!(ax8, ttot, iter_vel ./ 1.0e3, color = :blue, label = "velocity")
        lines!(ax8, ttot, iter_displ ./ 1.0e3, color = :red, label = "displacement")
        lines!(ax8, ttot, (iter_vel - iter_displ) ./ 1.0e3, color = :black, label = "difference")

        hidexdecorations!(ax1)
        hidexdecorations!(ax2)
        hidexdecorations!(ax3)
        axislegend(ax4)
        hidexdecorations!(ax5)
        hidexdecorations!(ax6)
        axislegend(ax8)
        save(joinpath(figdir, "$(it).png"), fig)

        fig2 = Figure(size = (3200, 1600), title = "Error at t = $t")

        error_tau = (abs.((stokes_vel.τ.II - stokes_displ.τ.II) ./ stokes_vel.τ.II))

        mask = stokes_vel.EII_pl .!= 0
        error_ε_pl = similar(stokes_vel.EII_pl)
        error_ε_pl[mask] = abs.((stokes_vel.EII_pl[mask] - stokes_displ.EII_pl[mask]) ./ stokes_vel.EII_pl[mask])
        error_ε_pl[.!mask] .= NaN  # or 0, depending on context

        log_error = similar(error_ε_pl)
        mask = error_ε_pl .> 0
        log_error[mask] = log10.(error_ε_pl[mask])
        log_error[.!mask] .= NaN

        error_ε = abs.((stokes_vel.ε.II - stokes_displ.ε.II) ./ stokes_vel.ε.II)
        error_velocity_x = abs.((stokes_vel.V.Vx - stokes_displ.V.Vx) ./ stokes_vel.V.Vx)
        error_velocity_y = abs.((stokes_vel.V.Vy - stokes_displ.V.Vy) ./ stokes_vel.V.Vy)
        error_displacement_x = abs.((stokes_vel.U.Ux - stokes_displ.U.Ux) ./ stokes_vel.U.Ux)
        error_displacement_y = abs.((stokes_vel.U.Uy - stokes_displ.U.Uy) ./ stokes_vel.U.Uy)

        ax8 = Axis(fig2[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax9 = Axis(fig2[1, 3], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax10 = Axis(fig2[1, 5], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax15 = Axis(fig2[1, 7], aspect = 1)
        ax11 = Axis(fig2[2, 1], aspect = 1, title = L"V_x", titlesize = 35)
        ax12 = Axis(fig2[2, 3], aspect = 1, title = L"V_y", titlesize = 35)
        ax13 = Axis(fig2[2, 5], aspect = 1, title = L"U_x", titlesize = 35)
        ax14 = Axis(fig2[2, 7], aspect = 1, title = L"U_y", titlesize = 35)

        h1 = heatmap!(ax8, xci..., log10.(Array(error_tau)), colormap = :batlow)
        h2 = heatmap!(ax9, xci..., Array(log_error), colormap = :batlow)
        h3 = heatmap!(ax10, xci..., log10.(Array(error_ε)), colormap = :batlow)
        h4 = heatmap!(ax11, xvi..., log10.(Array(error_velocity_x)), colormap = :batlow)
        h5 = heatmap!(ax12, xci..., log10.(Array(error_velocity_y)), colormap = :batlow)
        h6 = heatmap!(ax13, xvi..., log10.(Array(error_displacement_x)), colormap = :batlow)
        h7 = heatmap!(ax14, xvi..., log10.(Array(error_displacement_y)), colormap = :batlow)

        Colorbar(fig2[1, 2], h1)
        if !all(isnan, h2.args.value.x[3])
            Colorbar(fig2[1, 4], h2)
        end
        Colorbar(fig2[1, 6], h3)
        Colorbar(fig2[2, 2], h4)
        Colorbar(fig2[2, 4], h5)
        Colorbar(fig2[2, 6], h6)
        Colorbar(fig2[2, 8], h7)
        error_tau = log10.(abs.((τII_vel - τII_displ)) ./ τII_vel)
        lines!(ax15, ttot, error_tau, color = :black)

        save(joinpath(figdir, "error_$(it).png"), fig2)
    end

    return nothing
end


main(igg; figdir = figdir, nx = nx, ny = ny);

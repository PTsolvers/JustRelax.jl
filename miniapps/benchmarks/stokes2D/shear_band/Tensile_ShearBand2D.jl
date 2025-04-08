using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
const backend_JP = JustPIC.CPUBackend

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, radius)
    ni = size(phase_ratios.center)
    origin = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius)
        x, y = xc[i], yc[j]
        if ((x - o_x)^2 + (y - o_y)^2) > radius^2
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0

        else
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., origin..., radius)
    return nothing
end

function line(p, K, Δt, η_ve, sinψ, p1, t1)
    p2 = p1 + K*Δt*sinψ
    t2 = t1 - η_ve
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end

# MAIN SCRIPT --------------------------------------------------------------------
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

    # Physical properties using GeoParams ----------------
    τ_y = 1           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    τ_T = 0.5         # tensile strength
    δτ_T= 0.15          # pressure limiter (if τII_tr<δσ_T) only pressure correction is applied, Pa
    ϕ = 30            # friction angle
    C = τ_y / cosd(ϕ)           # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 8.0e-3          # regularisation "viscosity"
    dt = η0 / G0 / 4.0 *0.25    # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C,
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 5
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    radius = 0.1
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-6, CFL = 0.95 / √2.1)
    # pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6, Re=3π, r=0.7, CFL = 0.95 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    # args = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C, τ_tensile = τ_T, δτ_tensile = δτ_T)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([ -y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    tmax = 5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]

    # while t < tmax
    for _ in 1:35

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                verbose = false,
                iterMax = 50.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # visualisation
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5

        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        # ax2   = Axis(fig[2,1], aspect = 1, title = "η_vep")
        ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 3], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 3], aspect = 1)
        ax5 = Axis(fig[3, 1], aspect = 1, title = "Plastic domain", titlesize = 35)
        ax6 = Axis(fig[3, 3], aspect = 1, title = "Pressure", titlesize = 35)
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
        heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
        heatmap!(ax2, xci..., Array((stokes.EII_pl)), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        h = heatmap!(ax5, xci..., Array(stokes.pl_domain), colormap = :lipari10, colorrange = (0, 5))
        heatmap!(ax6, xci..., Array(stokes.P), colormap = :batlow)
        Colorbar(fig[3, 2], h)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black)
        lines!(ax4, ttot, sol, color = :red)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
        fig = let
            stress_data = []
            for I in (1,1)# CartesianIndices(1,1)
                phase = @inbounds phase_ratios.center[Tuple(I)...]
                _Gdt = inv(fn_ratio(JustRelax.JustRelax2D.get_shear_modulus, rheology, phase) * dt)
                is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JustRelax.JustRelax2D.plastic_params_phase(rheology, stokes.EII_pl[Tuple(I)...], phase)
                K = fn_ratio(JustRelax.JustRelax2D.get_bulk_modulus, rheology, phase)
                volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
                ηij = stokes.viscosity.η[Tuple(I)...]
                dτ_r = 1.0 / (pt_stokes.θ_dτ + ηij * _Gdt + 1.0)
                η_ve =  1 / (1/ηij + 1/_Gdt)
                ratio = η_ve * inv(K) *inv(dt)
                τ_tensile, δτ_tensile = τ_T, δτ_T
                Pc1 = - (τ_tensile - δτ_tensile)                   # Pressure corner 1
                τc1 =   Pc1 + τ_tensile #δτ_tensile
                Pc2 =  (τ_tensile + C * cosϕ) / (1.0 + sinϕ)      # Pressure corner 2
                τc2 = Pc2 + τ_tensile                              # dev stress corner2

                push!(stress_data, (Pc1, τc1, Pc2, τc2, C, sinϕ, cosϕ, sinψ, η_reg, η_ve, K))
            end

            # Plotting outside the loop
            fig = Figure()
            ax = Axis(fig[1, 1], aspect = 1, limits = (nothing, nothing, 0, 4))

            Pc1, τc1, Pc2, τc2, C, sinϕ, cosϕ, sinψ, η_reg, η_ve, K = stress_data[1]
                pr_tr1 = LinRange(-1.5, Pc1, 100)
                pr_tr2 = LinRange(0, Pc2, 100)
                pr_tr3 = LinRange(0, Pc2, 100)
                # Plot the Drucker-Prager yield surface
                lines!(ax, [Pc1, Pc1, Pc2, 4], [0.0, τc1, τc2, (4 * sinϕ + C * cosϕ)], color = :black, linewidth = 2)
                # l1 = line.(pr_tr1, K, dt, η_ve, sind(90.0), Pc1, τc1)
                # l2 = line.(pr_tr2, K, dt, η_ve, sind(90.0), Pc2, τc2)
                # l3 = line.(pr_tr3, K, dt, η_ve, sinψ, Pc2, τc2)
                l1 = @. (η_ve + η_reg)/(K*dt + η_reg)*(-pr_tr1 + Pc1) + τc1
                l2 = @. (η_ve + η_reg)/(K*dt + η_reg)*(-pr_tr2 + Pc2) + τc2
                l3 = @. (η_ve + η_reg)/((K*dt + η_reg)*sinψ)*(-pr_tr3 + Pc2) + τc2
                # Plot the lines for the conditions
                if !isempty(Pc1)
                    lines!(ax, [-1.5, Pc1], [τc1, τc1], color = :purple, linewidth = 2, label = "Condition 0")
                end
                line1 = lines!(ax, pr_tr1, l1, color = :red)
                line2 = lines!(ax, pr_tr2, l2, color = :blue)
                line3 = lines!(ax, pr_tr3, l3, color = :green)

            # end
            scatter!(ax, Array(stokes.P)[:], Array(stokes.τ.II)[:]; color = :red, markersize = 5)
            save(joinpath(figdir, "stress$(it).png"), fig)
            # fig
        end
    end

    return nothing
end

n = 96
nx = n
ny = n
figdir = "Tensile_plasticity"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);

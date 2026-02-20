# using CUDA
using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
# @init_parallel_stencil(CUDA, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend
# const backend = CUDABackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend
# const backend_JP = CUDABackend

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


# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = 2.0e0           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = -lx / 2, -ly / 2     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 35            # friction angle
    C = 5             # Cohesion
    η0 = 4.0e3           # viscosity
    εbg = -1.0          # background strain-rate
    η_reg = 5.0e-3          # regularisation "viscosity"
    dt = 1.5e9         # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = 1.0e-10, Kb = 1.0e-10)
    el_inc = ConstantElasticity(; G = 1.0e-10, Kb = 1.0e-10)
    visc = LinearViscous(; η = η0)
    visc_inc = LinearViscous(; η = 4.0e-2)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C,
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 5.0e0,
    )

    rheology = (
        # Host
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # Inclusion
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_inc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    origin = 0.0e0, 0.0e0
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    # pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1e-1, CFL = 0.95 / √2.1)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-1, Re = 1.0e0, r = 0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)

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
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    stokes.V.Vx[2:(end - 1), :] .= 0 # ensure non zero initial pressure residual
    stokes.V.Vy[:, 2:(end - 1)] .= 0 # ensure non zero initial pressure residual
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
    for _ in 1:150

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
                iterMax = 150.0e3,
                nout = 2.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t += dt

        # push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # visualisation
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
        # heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        # lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black)
        # lines!(ax4, ttot, sol, color = :red)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

n = 128
nx = n
ny = n * 2
figdir = "ShearBands2D"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
# main(igg; figdir = figdir, nx = nx, ny = ny);


function Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ, dx, dy, ncx, ncy)

    ηN = ones(ncx - 1, ncy)
    ηS = ones(ncx - 1, ncy)
    ηN[:, 1:(end - 1)] .= ηv[2:(end - 1), 2:(end - 1)]
    ηS[:, 2:(end - 0)] .= ηv[2:(end - 1), 2:(end - 1)]
    ηW = ηc[1:(end - 1), :]
    ηE = ηc[2:(end - 0), :]
    γW = γ[1:(end - 1), :]
    γE = γ[2:(end - 0), :]
    Cxx = ones(ncx - 1, ncy)
    Cxy = ones(ncx - 1, ncy)
    @. Cxx = abs.(ηN ./ dy .^ 2) + abs.(ηS ./ dy .^ 2) + abs.(γE ./ dx .^ 2 + (4 / 3) * ηE ./ dx .^ 2) + abs.(γW ./ dx .^ 2 + (4 / 3) * ηW ./ dx .^ 2) + abs.(-(-ηN ./ dy - ηS ./ dy) ./ dy + (γE ./ dx + γW ./ dx) ./ dx + ((4 / 3) * ηE ./ dx + (4 / 3) * ηW ./ dx) ./ dx)
    @. Cxy = abs.(γE ./ (dx .* dy) - 2 / 3 * ηE ./ (dx .* dy) + ηN ./ (dx .* dy)) + abs.(γE ./ (dx .* dy) - 2 / 3 * ηE ./ (dx .* dy) + ηS ./ (dx .* dy)) + abs.(γW ./ (dx .* dy) + ηN ./ (dx .* dy) - 2 / 3 * ηW ./ (dx .* dy)) + abs.(γW ./ (dx .* dy) + ηS ./ (dx .* dy) - 2 / 3 * ηW ./ (dx .* dy))

    Dx = ones(ncx - 1, ncy)
    @. Dx .= -(-ηN ./ dy - ηS ./ dy) ./ dy + (γE ./ dx + γW ./ dx) ./ dx + ((4 / 3) * ηE ./ dx + (4 / 3) * ηW ./ dx) ./ dx

    ηE = ones(ncx, ncy - 1)
    ηW = ones(ncx, ncy - 1)
    ηE[1:(end - 1), :] .= ηv[2:(end - 1), 2:(end - 1)]
    ηW[2:(end - 0), :] .= ηv[2:(end - 1), 2:(end - 1)]
    ηS = ηc[:, 1:(end - 1)]
    ηN = ηc[:, 2:(end - 0)]
    γS = γ[:, 1:(end - 1)]
    γN = γ[:, 2:(end - 0)]
    Cyy = ones(ncx, ncy - 1)
    Cyx = ones(ncx, ncy - 1)
    @. Cyy = abs.(ηE ./ dx .^ 2) + abs.(ηW ./ dx .^ 2) + abs.(γN ./ dy .^ 2 + (4 / 3) * ηN ./ dy .^ 2) + abs.(γS ./ dy .^ 2 + (4 / 3) * ηS ./ dy .^ 2) + abs.((γN ./ dy + γS ./ dy) ./ dy + ((4 / 3) * ηN ./ dy + (4 / 3) * ηS ./ dy) ./ dy - (-ηE ./ dx - ηW ./ dx) ./ dx)
    @. Cyx = abs.(γN ./ (dx .* dy) + ηE ./ (dx .* dy) - 2 / 3 * ηN ./ (dx .* dy)) + abs.(γN ./ (dx .* dy) - 2 / 3 * ηN ./ (dx .* dy) + ηW ./ (dx .* dy)) + abs.(γS ./ (dx .* dy) + ηE ./ (dx .* dy) - 2 / 3 * ηS ./ (dx .* dy)) + abs.(γS ./ (dx .* dy) - 2 / 3 * ηS ./ (dx .* dy) + ηW ./ (dx .* dy))

    Dy = ones(ncx, ncy - 1)
    @. Dy .= (γN ./ dy + γS ./ dy) ./ dy + ((4 / 3) * ηN ./ dy + (4 / 3) * ηS ./ dy) ./ dy - (-ηE ./ dx - ηW ./ dx) ./ dx

    λmaxVx = 1.0 ./ Dx .* (Cxx .+ Cxy)
    λmaxVy = 1.0 ./ Dy .* (Cyx .+ Cyy)

    return Dx, Dy, λmaxVx, λmaxVy
end

# Bulk viscosity
Kb = 1.0e-10
ηb = @fill(Kb * dt, ni...)
# Select γ
γfact = 20           # penalty: multiplier to the arithmetic mean of η
γi = γfact * mean(stokes.viscosity.η)
# (Pseudo-)compressibility
γ_eff = @zeros(ni...)
# if comp
γ_num = @fill(γi, ni...)
γ_phy = ηb
γ_eff = ((γ_phy .* γ_num) ./ (γ_phy .+ γ_num))
# else
#     γ_eff .= γi
# end

Dx = @zeros(ni[1] - 1, ni[2])
λmaxVx = @zeros(ni[1] - 1, ni[2])

Dy = @zeros(ni[1], ni[2] - 1)
λmaxVy = @zeros(ni[1], ni[2] - 1)

center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)

(; ηv, η) = stokes.viscosity
i = 1
j = 1

function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, dx, dy)
    ni = size(η)
    return @parallel (@idx ni) _Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, dx, dy)
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, dx, dy)
    @inbounds begin
        # viscosity coefficients at surrounding points
        ηN = ηv[i + 1, j + 1]
        ηS = ηv[i + 1, j]
        ηW = η[i, j]
        # # bulk viscosity coefficients at surrounding points
        γE = γ_eff[i, j]

        if all((i, j) .≤ size(Dx))
            ηE = η[i + 1, j]
            γW = γ_eff[i + 1, j]
            # compute Gershgorin entries
            Cxx = abs(ηN / dy^2) + abs(ηS / dy^2) + abs(γE / dx^2 + (4 / 3) * ηE / dx^2) + abs(γW / dx^2 + (4 / 3) * ηW / dx^2) + abs(-(-ηN / dy - ηS / dy) / dy + (γE / dx + γW / dx) / dx + ((4 / 3) * ηE / dx + (4 / 3) * ηW / dx) / dx)
            Cxy = abs(γE / (dx * dy) - 2 / 3 * ηE / (dx * dy) + ηN / (dx * dy)) + abs(γE / (dx * dy) - 2 / 3 * ηE / (dx * dy) + ηS / (dx * dy)) + abs(γW / (dx * dy) + ηN / (dx * dy) - 2 / 3 * ηW / (dx * dy)) + abs(γW / (dx * dy) + ηS / (dx * dy) - 2 / 3 * ηW / (dx * dy))
            # this is the preconditioner diagonal entry
            Dx_ij = Dx[i, j] = -(-ηN / dy - ηS / dy) / dy + (γE / dx + γW / dx) / dx + ((4 / 3) * ηE / dx + (4 / 3) * ηW / dx) / dx
            # maximum eigenvalue estimate
            λmaxVx[i, j] = inv(Dx_ij) * (Cxx + Cxy)
        end

        # viscosity coefficients at surrounding points
        ηS = ηW # reuse cached value
        ηW = ηv[i, j + 1]
        ηE = ηN # reuse cached value
        # # bulk viscosity coefficients at surrounding points
        γN = γE # reuse cached value

        if all((i, j) .≤ size(Dy))
            ηN = η[i, j + 1]
            γS = γ_eff[i, j + 1]
            # compute Gershgorin entries
            Cyy = abs(ηE / dx^2) + abs(ηW / dx^2) + abs(γN / dy^2 + (4 / 3) * ηN / dy^2) + abs(γS / dy^2 + (4 / 3) * ηS / dy^2) + abs((γN / dy + γS / dy) / dy + ((4 / 3) * ηN / dy + (4 / 3) * ηS / dy) / dy - (-ηE / dx - ηW / dx) / dx)
            Cyx = abs(γN / (dx * dy) + ηE / (dx * dy) - 2 / 3 * ηN / (dx * dy)) + abs(γN / (dx * dy) - 2 / 3 * ηN / (dx * dy) + ηW / (dx * dy)) + abs(γS / (dx * dy) + ηE / (dx * dy) - 2 / 3 * ηS / (dx * dy)) + abs(γS / (dx * dy) - 2 / 3 * ηS / (dx * dy) + ηW / (dx * dy))
            # this is the preconditioner diagonal entry
            Dy_ij = Dy[i, j] = (γN / dy + γS / dy) / dy + ((4 / 3) * ηN / dy + (4 / 3) * ηS / dy) / dy - (-ηE / dx - ηW / dx) / dx
            # maximum eigenvalue estimate
            λmaxVy[i, j] = inv(Dy_ij) * (Cyx + Cyy)
        end
    end
    return nothing
end

# @b Gershgorin_Stokes2D_SchurComplement!($(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, dx, dy)...)

# note: η, ηv have to be the effective viscoelastic ones
Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, dx, dy)

CFL_v = 0.98
# if dτ_local
dτVx = @. 2 / √(λmaxVx) * CFL_v
dτVy = @. 2 / √(λmaxVy) * CFL_v
# else
#     dτVx =  2.0./sqrt.(maximum(λmaxVx))*CFL_v
#     dτVy =  2.0./sqrt.(maximum(λmaxVy))*CFL_v
# end

Dx = @zeros(ni[1] - 1, ni[2])
λmaxVx = @zeros(ni[1] - 1, ni[2])


βVx = @zeros(ni[1] - 1, ni[2])  # this disappears is dτ is not local
βVy = @zeros(ni[1], ni[2] - 1)  # this disappears is dτ is not local
cVx = @zeros(ni[1] - 1, ni[2])  # this disappears is dτ is not local
cVy = @zeros(ni[1], ni[2] - 1)  # this disappears is dτ is not local
αVx = @zeros(ni[1] - 1, ni[2])  # this disappears is dτ is not local
αVy = @zeros(ni[1], ni[2] - 1)  # this disappears is dτ is not local


function update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    ni = size(βVx) .+ (1, 0)
    @parallel (@idx ni) _update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    return nothing
end

@parallel_indices (I...) function _update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)

    @inbounds if all(I .≤ size(βVx))
        dτVx_ij = dτVx[I...]
        cVx_ij = cVx[I...]

        βVx[I...] = @muladd  2 * dτVx_ij / (2 + cVx_ij * dτVx_ij)
        αVx[I...] = @muladd (2 - cVx_ij * dτVx_ij) / (2 + cVx_ij * dτVx_ij)
    end
    @inbounds if all(I .≤ size(βVy))
        dτVy_ij = dτVy[I...]
        cVy_ij = cVy[I...]
        βVy[I...] = @muladd  2 * dτVy_ij / (2 + cVy_ij * dτVy_ij)
        αVy[I...] = @muladd (2 - cVy_ij * dτVy_ij) / (2 + cVy_ij * dτVy_ij)
    end

    return nothing
end


update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)


# Residuals
Rx .= (.-(Pt[2:end, :] .- Pt[1:(end - 1), :]) ./ Δx .- (ΔPψ[2:end, :] .- ΔPψ[1:(end - 1), :]) ./ Δx .+ (Txx[2:end, :] .- Txx[1:(end - 1), :]) ./ Δx .+ (Txy[2:(end - 1), 2:end] .- Txy[2:(end - 1), 1:(end - 1)]) ./ Δy)
Ry .= (.-(Pt[:, 2:end] .- Pt[:, 1:(end - 1)]) ./ Δy .- (ΔPψ[:, 2:end] .- ΔPψ[:, 1:(end - 1)]) ./ Δy .+ (Tyy[:, 2:end] .- Tyy[:, 1:(end - 1)]) ./ Δy .+ (Txy[2:end, 2:(end - 1)] .- Txy[1:(end - 1), 2:(end - 1)]) ./ Δx)
Rp .= .-∇V .- comp * (Pt .- Pt0) ./ ηb

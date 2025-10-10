const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D

const backend_JR = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end
using GeoParams, CairoMakie

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            depth = abs(y[j])
            dTdZ = (2047 - 2017) / 50.0e3
            offset = 2017
            T[i + 1, j] = (depth - 585.0e3) * dTdZ + offset
        end
        return nothing
    end
    ni = length.(xvi)
    @parallel (@idx ni) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, xc, yc, r)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            # plume - rectangular
            @index phases[ip, i, j] = if ((x - xc)^2 ≤ r^2) && ((depth - yc)^2 ≤ r^2)
                2.0
            else
                1.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, xc, yc, r)
end


function init_phases!(phases::AbstractArray, xc, yc, r, x, y)
    ni = size(phases)

    @parallel_indices (i, j) function _init_phases!(phases, x, y, xc, yc, r)
        @inbounds begin
            xi = x[i]
            depth = -y[j]
            phases[i, j] = ((xi - xc)^2 ≤ r^2) && ((depth - yc)^2 ≤ r^2) ? 2.0 : 1.0
        end
        return nothing
    end

    @parallel (@idx ni) _init_phases!(phases, x, y, xc, yc, r)
    return nothing
end


import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_j(z))
    return nothing
end

function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clr = pϕ.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)
    Colorbar(f[1,2], h)
    f
end
# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function sinking_block2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", thermal_perturbation = :circular)

    # Physical domain ------------------------------------
    ly = 500.0e3
    lx = ly * ar
    origin = 0.0, -ly                         # origin coordinates
    ni = nx, ny                           # number of cells
    li = lx, ly                           # domain length in x- and y-
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    δρ = 100
    rheology = (
        SetMaterialParams(;
            Name = "Mantle",
            Phase = 1,
            Density = ConstantDensity(; ρ = 3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        SetMaterialParams(;
            Name = "Block",
            Phase = 2,
            Density = ConstantDensity(; ρ = 3.2e3 + δρ),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )
    # heat diffusivity
    dt = 1
    # ----------------------------------------------------

    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)

    # Rectangular density anomaly
    xc_anomaly = 250.0e3   # origin of thermal anomaly
    yc_anomaly = -(ly - 400.0e3) # origin of thermal anomaly
    r_anomaly = 50.0e3   # radius of perturbation
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    # init_phases!(pPhases, particles, xc_anomaly, abs(yc_anomaly), r_anomaly)
    # update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    phases = @zeros(ni...)
    # phases = Float64.([argmax(p) for p in Array(phase_ratios.center)])
    weno = WENO5(backend_JR, Val(2), ni.+1) # ni.+1 for Temp
    weno_c = WENO5(backend_JR, Val(2), ni) # ni.+1 for Temp
    init_phases!(phases, xc_anomaly, abs(yc_anomaly), r_anomaly, xci[1], xci[2])

    phases_blob = @zeros(ni...) # for plotting purposes
    @views phases_blob[phases .== 1.0] .= 0.0 # set the blob to 1.0 where phases == 2.0
    @views phases_blob[phases .== 2.0] .= 1.0 # set the blob to 1.0 where phases == 2.0
    phases_bg = @zeros(ni...) # for plotting purposes
    @views phases_bg[phases .== 1.0] .= 1.0 # set the blob to 1.0 where phases == 2.0
    @views phases_bg[phases .== 2.0] .= 0.0 # set the blob to 1.0 where phases == 2.0
    clamp!(phases, 1.0, 2.0) # clamp phases to 1.0 and 2.0
    clamp!(phases_blob, 0.0, 1.0) # clamp
    clamp!(phases_bg, 0.0, 1.0) # clamp phases_bg to 0.0 and 1.0
    update_phase_ratios_2D!(phase_ratios, (phases_bg, phases_blob), xci, xvi)
    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-5, ϵ_rel = 1.0e-5, CFL = 0.95 / √2.1)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = @ones(ni...), P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # ----------------------------------------------------

    # Viscosity
    args = (; dt = dt, ΔTc = @zeros(ni...))
    η_cutoff = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    # ----------------------------------------------------

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    it = 0 # iteration counter
    while it < 50
    # Stokes solver ----------------
    args = (; T = @ones(ni...), P = stokes.P, dt = dt, ΔTc = @zeros(ni...))
    solve!(
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
            iterMax = 150.0e3,
            nout = 1.0e3,
            viscosity_cutoff = η_cutoff,
            verbose = false,
        )
    )
    dt = compute_dt(stokes, di, igg)
    # ------------------------------

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    velocity2center!(Vx_c, Vy_c, @velocity(stokes)...)
    velocity = @. √(Vx_v^2 + Vy_v^2)

    # Advection ---------------------
    WENO_advection!(phases, (Vx_c, Vy_c), weno_c, di, dt)
    WENO_advection!(phases_blob, (Vx_c, Vy_c), weno_c, di, dt)
    WENO_advection!(phases_bg, (Vx_c, Vy_c), weno_c, di, dt)
    clamp!(phases, 1.0, 2.0) # clamp phases to 1.0 and 2.0
    clamp!(phases_blob, 0.0, 1.0) # clamp phases_blob to 0.0 and 1.0
    clamp!(phases_bg, 0.0, 1.0) # clamp phases_bg to 0.0 and 1

    # update phase ratios
    update_phase_ratios_2D!(phase_ratios, (phases_bg, phases_blob), xci, xvi)

    compute_ρg!(ρg[2], phase_ratios, rheology, (T = @ones(ni...), P = stokes.P))
    # Plotting ---------------------
    fig = Figure(size = (1200, 1200))
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(), title = "Velocity")
    ax2 = Axis(fig[1, 3]; aspect = DataAspect(), title = "Phase ratio center")
    ax3 = Axis(fig[2, 1]; aspect = DataAspect(), title = "Phase ratio vertex")
    ax4 = Axis(fig[2, 3]; aspect = DataAspect(), title = "Phase ratio Vx")
    ax5 = Axis(fig[3, 1]; aspect = DataAspect(), title = "Phase ratio Vy")
    ax6 = Axis(fig[3, 3]; aspect = DataAspect(), title = "Density")

    pp_c = [argmax(p) for p in Array(phase_ratios.center)]
    pp_v = [argmax(p) for p in Array(phase_ratios.vertex)]
    pp_Vx = [argmax(p) for p in Array(phase_ratios.Vx)]
    pp_Vy = [argmax(p) for p in Array(phase_ratios.Vy)]

    h1 = heatmap!(ax1, (xvi./1e3)..., Array(velocity), colormap = :vikO)
    Colorbar(fig[1, 2], h1)

    h2 = heatmap!(ax2, (xci./1e3)..., Array(pp_c); colormap = :roma)
    Colorbar(fig[1, 4], h2)

    h3 = heatmap!(ax3, (xvi./1e3)..., Array(pp_v); colormap = :roma)
    Colorbar(fig[2, 4], h3)

    h4 = heatmap!(ax4, (xvi./1e3)..., Array(pp_Vx); colormap = :roma)
    Colorbar(fig[2, 2], h4)

    h5 = heatmap!(ax5, (xvi./1e3)..., Array(pp_Vy); colormap = :roma)
    Colorbar(fig[3, 2], h5)

    h6 = heatmap!(ax6, (xci./1e3)..., Array(ρg[2]./9.81); colormap = :lapaz)
    Colorbar(fig[3, 4], h6)

    linkaxes!(ax1, ax2, ax3, ax4, ax5)
    # f3, ax3, h3 = heatmap(xvi..., Array(pp_c); colormap = :roma)
    # Colorbar(f3[1, 2], h3)

    display(fig)
    it += 1
    @show it
    # ------------------------------
    end
    return nothing
end

ar = 1 # aspect ratio
n = 96
nx = n * ar - 2
ny = n - 2
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

sinking_block2D(igg; ar = ar, nx = nx, ny = ny);

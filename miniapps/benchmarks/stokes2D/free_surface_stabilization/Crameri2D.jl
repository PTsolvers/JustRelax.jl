const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, GLMakie

function init_phases!(phases, particles, A)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, A)

        f(x, A, λ) = A * sin(π * x / λ)

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 3.0

            # if depth ≤ cos(x * 2π/2800e3) * 7e3 #- 100e3
            #     @index phases[ip, i, j] = 1.0
            # end

            if depth < 100.0e3 + 100.0e3
                @index phases[ip, i, j] = 2.0
            end

            # if depth < (-cos(x * 2π/700e3) * 7e3 + 100e3)
            if depth < (-cos(x * 2π / 2800.0e3) * 7.0e3 + 100.0e3)
                @index phases[ip, i, j] = 1.0
            end

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n = 64
nx = n
ny = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny)

    # Physical domain ------------------------------------
    thick_air = 100.0e3             # thickness of sticky air layer
    ly = 700.0e3 + thick_air # domain length in y
    lx = 2800.0e3            # domain length in x
    ni = nx, ny            # number of cells
    li = lx, ly            # domain length in x- and y-
    di = @. li / ni        # grid step in x- and -y
    origin = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = rheology = (
        # Name              = "Air",
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0e0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e22),)),
            Gravity = ConstantGravity(; g = 10),
        ),
        # Name              = "Crust",
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),)),
            Gravity = ConstantGravity(; g = 10),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
            Gravity = ConstantGravity(; g = 10),
        ),
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 60, 80, 40
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    A = 5.0e3    # Amplitude of the anomaly
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, A)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # rock ratios for variational stokes
    # RockRatios
    air_phase = 1
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, (phase_ratios.Vx, phase_ratios.Vy), air_phase)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-6, Re = 15π, r = 1.0e0, CFL = 0.98 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))
    compute_viscosity!(stokes, phase_ratios, args, rheology, air_phase, (1.0e18, 1.0e24))

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        free_surface = true,
    )

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    figdir = "Crameri2012"
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    dt = 10.0e3 * (3600 * 24 * 365.25)

    while it < 20

        # Variational Stokes solver ----------------
        solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            Inf,
            igg;
            kwargs = (
                iterMax = 50.0e3,
                iterMin = 1.0e3,
                viscosity_relaxation = 1.0e-2,
                nout = 2.0e3,
                viscosity_cutoff = (1.0e18, 1.0e24),
            )
        )
        dt = compute_dt(stokes, di, dt_max)
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ, phase_ratios, (phase_ratios.Vx, phase_ratios.Vy), air_phase)

        @show it += 1
        t += dt


        (; η_vep, η) = stokes.viscosity
        # if do_vtk
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        velocity_v = @. √(Vx_v^2 .+ Vy_v^2)
        data_v = (;
            τII = Array(stokes.τ.II),
            εII = Array(stokes.ε.II),
            Vx = Array(Vx_v),
            Vy = Array(Vy_v),
            Vel = Array(velocity_v),
        )
        data_c = (;
            P = Array(stokes.P),
            η = Array(η_vep),
        )
        velocity_v = (
            Array(Vx_v),
            Array(Vy_v),
        )
        JustRelax.DataIO.save_vtk(
            joinpath(figdir, "vtk_" * lpad("$it", 6, "0")),
            xvi,
            xci,
            data_v,
            data_c,
            velocity_v;
            t = t
        )
        # end

        # if it == 1 || rem(it, 1) == 0
        #     px, py = particles.coords

        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        #     nt = 5
        #     fig = Figure(size = (900, 900), title = "t = $t")
        #     ax  = Axis(fig[1,1], aspect = 1, title = " t=$(round.(t/(1e3 * 3600 * 24 *365.25); digits=3)) Kyrs")
        #     heatmap!(ax, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.V.Vy), colormap = :vikO)
        #     # heatmap!(ax, xci[1].*1e-3, xci[2].*1e-3, Array([argmax(p) for p in phase_ratios.vertex]), colormap = :grayC)
        #     # scatter!(ax, Array(px.data[:]).*1e-3, Array(py.data[:]).*1e-3, color =Array(pPhases.data[:]), colormap = :grayC)
        #     # arrows!(
        #     #     ax,
        #     #     xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))...,
        #     #     lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
        #     #     color = :red,
        #     # )
        #     fig
        #     save(joinpath(figdir, "$(it).png"), fig)

        fig = Figure(size = (900, 900), title = "t = $t")
        ax = Axis(fig[1, 1], aspect = 1, title = " t=$(round.(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)) Kyrs")

        # Make particles plottable
        nt = 5
        p = particles.coords
        ppx, ppy = p
        pxv = ppx.data[:] ./ 1.0e3
        pyv = ppy.data[:] ./ 1.0e3
        clr = pPhases.data[:]
        idxv = particles.index.data[:]
        heatmap!(ax, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.V.Vy))
        scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 2, colormap = :grayC)
        arrows!(
            ax,
            xvi[1][1:nt:(end - 1)] ./ 1.0e3, xvi[2][1:nt:(end - 1)] ./ 1.0e3, Array.((Vx_v[1:nt:(end - 1), 1:nt:(end - 1)], Vy_v[1:nt:(end - 1), 1:nt:(end - 1)]))...,
            lengthscale = 25 / max(maximum(Vx_v), maximum(Vy_v)),
            color = :red,
        )
        save(joinpath(figdir, "$(it).png"), fig)
        fig
        # end
    end
    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
main(igg, nx, ny)


# @parallel_indices (I...) function compute_P!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ,
#     ::Nothing,
# ) where {N}
#     # if isvalid_c(ϕ, I...)
#     #     K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
#     #     RP[I...], P[I...] = _compute_P!(
#     #         P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ
#     #     )
#     # else
#     #     RP[I...] = P[I...] = zero(eltype(P))
#     # end
#     return nothing
# end

# @parallel (@idx ni) compute_P_kernel!(
#         stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, stokes.viscosityη,
#         rheology, phase_ratio.center, ϕ, dt, pt_stokes.r, pt_stokes.θ_dτ, nothing
#     )

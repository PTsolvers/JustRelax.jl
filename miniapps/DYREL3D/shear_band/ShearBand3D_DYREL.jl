const isCUDA = false

@static if isCUDA
    using CUDA
end

using Printf, CellArrays
using GeoParams
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO
using ParallelStencil

const backend_JR = @static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    CPUBackend
end

using JustPIC, JustPIC._3D

const backend_JP = @static if isCUDA
    JustPIC.CUDABackend
else
    JustPIC.CPUBackend
end

solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

function init_phases!(phases, particles, radius)
    ni = size(phases)
    origin = 0.5, 0.5, 0.5

    @parallel_indices (I...) function init_phases!(phases, index, xc, yc, zc, o_x, o_y, o_z)
        for ip in cellaxes(xc)
            (@index index[ip, I...]) || continue

            x = @index xc[ip, I...]
            y = @index yc[ip, I...]
            z = @index zc[ip, I...]

            if ((x - o_x)^2 + (y - o_y)^2 + (z - o_z)^2) > radius^2
                @index phases[ip, I...] = 1.0
            else
                @index phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.index, particles.coords..., origin...)
    return nothing
end

function main3D(
        igg;
        nx = 32,
        ny = 32,
        nz = 32,
        nsteps = 32,
        figdir = "ShearBands3D_DYREL_Benchmark",
        do_vtk = false,
    )
    ni = nx, ny, nz
    grid = Geometry(ni, (1.0, 1.0, 1.0))
    (; xci, xvi) = grid

    # Physical properties using GeoParams ----------------
    τ_y = 1.6
    ϕ = 30.0
    C = τ_y
    η0 = 1.0
    G0 = 1.0
    Gi = G0 / 2
    εbg = 1.0
    η_reg = 1.25e-2
    dt = η0 / G0 / 4
    dt /= 2
    el_bg = ConstantElasticity(; G = G0, ν = 0.5)
    el_inc = ConstantElasticity(; G = Gi, ν = 0.5)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        C = C,
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0.0,
    )

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    nxcell, max_xcell, min_xcell = 125, 150, 75
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    radius = 0.1
    pPhases, = init_cell_arrays(particles, Val(1))
    init_phases!(pPhases, particles, radius)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    stokes = StokesArrays(backend_JR, ni)
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (
            left = true,
            right = true,
            top = true,
            bot = true,
            front = true,
            back = true,
        ),
        no_slip = (
            left = false,
            right = false,
            top = false,
            bot = false,
            front = false,
            back = false,
        ),
    )
    stokes.V.Vx .= PTArray(backend_JR)(
        [x * εbg for x in xvi[1], _ in 1:(ny + 2), _ in 1:(nz + 2)]
    )
    stokes.V.Vz .= PTArray(backend_JR)(
        [-z * εbg for _ in 1:(nx + 2), _ in 1:(ny + 2), z in xvi[3]]
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    isdir(figdir) || mkpath(figdir)
    if do_vtk
        take(joinpath(figdir, "vtk"))
    end
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)
    Vz_c = @zeros(ni...)

    dyrel = JustRelax3D.DYREL(
        backend_JR,
        stokes,
        rheology,
        phase_ratios,
        grid.di,
        dt;
        ϵ = 1.0e-4,
        CFL = 0.99,
        γfact = 10.0,
        c_fact = 0.5,
    )

    t = 0.0
    local out
    history_step = Int[]
    history_time = Float64[]
    history_iter = Float64[]
    history_total = Float64[]
    history_velocity = Float64[]
    history_pressure = Float64[]
    iteration_file = joinpath(figdir, "DYREL_iterations.csv")
    for it in 1:nsteps
        out = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 50.0e3,
                total_iterMax = 50.0e3,

                nout = 50,
                rel_drop = 0.5,
                λ_relaxation_PH = 1.0,
                λ_relaxation_DR = 1.0,
                viscosity_relaxation = 1.0,
                linear_viscosity = true,
            ),
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        t += dt

        n = length(out.err_evo_it)
        Base.append!(history_step, fill(it, n))
        Base.append!(history_time, fill(t, n))
        Base.append!(history_iter, out.err_evo_it)
        Base.append!(history_total, out.err_evo_tot)
        Base.append!(history_velocity, out.err_evo_V)
        Base.append!(history_pressure, out.err_evo_P)
        if igg.me == 0
            open(iteration_file, "w") do io
                println(io, "step,time,iteration,residual_total,residual_velocity,residual_pressure")
                for i in eachindex(history_iter)
                    @printf(
                        io,
                        "%d,%.17g,%.0f,%.17g,%.17g,%.17g\n",
                        history_step[i],
                        history_time[i],
                        history_iter[i],
                        history_total[i],
                        history_velocity[i],
                        history_pressure[i],
                    )
                end
            end
        end

        if do_vtk
            velocity2center!(Vx_c, Vy_c, Vz_c, @velocity(stokes)...)
            data_c = (;
                τII = stokes.τ.II,
                εII = stokes.ε.II,
                εII_pl = stokes.ε_pl.II,
                P = stokes.P,
                η = stokes.viscosity.η_vep,
            )
            velocity_c = Vx_c, Vy_c, Vz_c
            save_vtk(
                joinpath(figdir, "vtk", "vtk_" * lpad("$it", 6, "0")),
                xci,
                data_c,
                velocity_c;
                t = t,
                pvd = joinpath(figdir, "vtk", "ShearBand3D_DYREL_Benchmark"),
            )
        end

        τxx_max = maximum(Array(stokes.τ.xx))
        τII_min, τII_max = extrema(Array(stokes.τ.II))
        εpl_max = maximum(Array(stokes.ε_pl.II))
        if igg.me == 0
            @printf(
                "step %d/%d, t = %.3f, residual = %.3e, max(τxx) = %.6f, τII = [%.6f, %.6f], max(εII_pl) = %.6f\n",
                it,
                nsteps,
                t,
                last(out.err_evo_tot),
                τxx_max,
                τII_min,
                τII_max,
                εpl_max,
            )
        end
    end

    return (;
        out,
        stokes,
        dyrel,
        phase_ratios,
        iteration_file,
        analytic_stress = solution(εbg, t, G0, η0),
    )
end

n = 32
nx = n
ny = n
nz = n
figdir = "ShearBands3D_DYREL_Benchmark"
do_vtk = true # set to true to generate VTK files for ParaView
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

main3D(igg; nx = nx, ny = ny, nz = nz, figdir = figdir, do_vtk = do_vtk)

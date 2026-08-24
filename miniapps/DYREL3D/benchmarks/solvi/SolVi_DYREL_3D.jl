using ParallelStencil.FiniteDifferences3D

# Benchmark reference:
#   D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
#   general shear. Geophysical Journal International, 155(1):269–288, 2003.

include("viz_SolVi_DYREL_3D.jl")

@parallel_indices (i, j, k) function init_solvi_phase!(ratios)
    @index ratios[1, i, j, k] = 1.0
    return nothing
end

# SolVi prescribes a smoothed inclusion viscosity directly on the cell centers.
function JustRelax.JustRelax3D.compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args::NamedTuple{(:T, :P, :dt, :prescribed_viscosity)},
        rheology,
        cutoff;
        kwargs...,
    )
    return nothing
end

function solvi_pureshear_bc!(stokes, xci, xvi, εbg, backend)
    xv, yv, zv = xvi
    xc, yc, zc = xci

    stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .=
        PTArray(backend)([εbg * x for x in xv, y in yc, z in zc])
    stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .=
        PTArray(backend)([εbg * y for x in xc, y in yv, z in zc])
    stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .=
        PTArray(backend)([-2 * εbg * z for x in xc, y in yc, z in zv])

    return nothing
end

@parallel function smooth!(A2::AbstractArray{T, 3}, A::AbstractArray{T, 3}, fact::T) where {T}
    @inn(A2) = @inn(A) + one(T) / 6.1 / fact * (@d2_xi(A) + @d2_yi(A) + @d2_zi(A))
    return nothing
end

@parallel_indices (ix, iy, iz) function _viscosity!(η, ηi, rc, li, di)
    lx, ly, lz = li
    dx, dy, dz = di
    if sqrt(
            ((ix - 1) * dx + 0.5 * dx - 0.5 * lx)^2 +
                ((iy - 1) * dy + 0.5 * dy - 0.5 * ly)^2 +
                ((iz - 1) * dz + 0.5 * dz - 0.5 * lz)^2,
        ) ≤ rc
        η[ix, iy, iz] = ηi
    end
    return nothing
end

function viscosity(ni, di, li, rc, η0, ηi; b_width = (1, 1, 1))
    η = @fill(η0, ni...)

    @parallel (1:ni[1], 1:ni[2], 1:ni[3]) _viscosity!(η, ηi, rc, li, di)

    # smooth viscosity field
    η2 = deepcopy(η)
    for _ in 1:10
        @hide_communication b_width begin # communication/computation overlap
            @parallel smooth!(η2, η, 1.0)
            η, η2 = η2, η
            update_halo!(η)
        end
    end

    return η
end

function solVi3D(;
        Δη = 1.0e-3,
        nx = 32 - 1,
        ny = 32 - 1,
        nz = 32 - 1,
        lx = 1.0e1,
        ly = 1.0e1,
        lz = 1.0e1,
        rc = 1.0e0,
        εbg = 1.0e0,
        init_MPI = true,
        finalize_MPI = false,
    )
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = nx, ny, nz # number of nodes in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...) # init MPI
    li = (lx, ly, lz)  # domain length in x- and y-
    origin = zero(nx), zero(ny), zero(nz)
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1 # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(backend, ni)
    (; η) = stokes.viscosity
    ## Setup-specific parameters and fields
    η0 = 1.0 # matrix viscosity
    ηi = Δη # inclusion viscosity
    dt = Inf
    η .= viscosity(ni, di, li, rc, η0, ηi)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η0),)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    for ratios in (
            phase_ratios.center, phase_ratios.vertex,
            phase_ratios.yz, phase_ratios.xz, phase_ratios.xy,
        )
        @parallel (@idx size(ratios)) init_solvi_phase!(ratios)
    end
    args = (; T = 0.0, P = stokes.P, dt = dt, prescribed_viscosity = true)

    ## Boundary conditions
    solvi_pureshear_bc!(stokes, xci, xvi, εbg, backend)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, back = true, front = true),
        no_slip = (left = false, right = false, top = false, bot = false, back = false, front = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    ## Body forces
    ρg = ntuple(_ -> @zeros(ni...), Val(3))

    dyrel = DYREL(
        backend, stokes, rheology, phase_ratios, grid.di, dt;
        ϵ = 1.0e-8, CFL = 0.99, γfact = 20.0,
    )

    ## Time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve_DYREL!(
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
                iterMax = 50.0e3,
                total_iterMax = 50.0e3,
                nout = 20,
                rel_drop = 0.1,
                verbose_PH = true,
                verbose_DR = false,
            ),
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

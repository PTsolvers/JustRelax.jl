using ParallelStencil.FiniteDifferences3D
using Statistics: mean

# benchmark reference:
#   C. Burstedde, G. Stadler, L. Alisic, L. C. Wilcox, E. Tan, M. Gurnis, and O. Ghattas.
#   Large-scale adaptive mantle convection simulation. Geophysical Journal International, 2013

include("viz_Burstedde_DYREL.jl")

@parallel_indices (i, j, k) function init_burstedde_phase!(ratios)
    @index ratios[1, i, j, k] = 1.0
    return nothing
end

# Keep the analytical fields below, just as the original Stokes benchmark does.
const BursteddeArgs = NamedTuple{(:T, :P, :dt, :prescribed_viscosity, :prescribed_body_force)}
JustRelax.JustRelax3D.compute_viscosity!(
    _stokes::JustRelax.StokesArrays, _phase_ratios, _args::BursteddeArgs, _rheology, _cutoff; _kwargs...
) = nothing
JustRelax.JustRelax3D.compute_ρg!(
    _ρg, _phase_ratios::JustPIC.PhaseRatios, _rheology, _args::BursteddeArgs
) = nothing

@parallel_indices (i, j, k) function _viscosity!(η, x, y, z, β)
    η[i, j, k] = exp(1 - β * (x[i] * (1 - x[i]) + y[j] * (1 - y[j]) + z[k] * (1 - z[k])))

    return nothing
end

function viscosity!(η, xci, β)
    @parallel (@idx size(η)) _viscosity!(η, xci[1], xci[2], xci[3], β)

    return η
end

function body_forces(xi::NTuple{3, T}, η, β) where {T}
    xx, yy, zz = xi
    x = PTArray(backend)([x for x in xx, y in yy, z in zz])
    y = PTArray(backend)([y for x in xx, y in yy, z in zz])
    z = PTArray(backend)([z for x in xx, y in yy, z in zz])

    dηdx = @. -β * (1 - 2 * x) * η
    dηdy = @. -β * (1 - 2 * y) * η
    dηdz = @. -β * (1 - 2 * z) * η

    fx = @. ((y * z + 3 * x^2 * y^3 * z) - η * (2 + 6 * x * y)) -
        dηdx * (2 + 4 * x + 2 * y + 6 * x^2 * y) - dηdy * (x + x^3 + y + 2 * x * y^2) -
        dηdz * (-3 * z - 10 * x * y * z)
    fy = @. ((x * z + 3 * x^3 * y^2 * z) - η * (2 + 2 * x^2 + 2 * y^2)) -
        dηdx * (x + x^3 + y + 2 * x * y^2) - dηdy * (2 + 2 * x + 4 * y + 4 * x^2 * y) -
        dηdz * (-3 * z - 5 * x^2 * z)
    fz = @. ((x * y + x^3 * y^3) - η * (-10 * y * z)) - dηdx * (-3 * z - 10 * x * y * z) -
        dηdy * (-3 * z - 5 * x^2 * z) - dηdz * (-4 - 6 * x - 6 * y - 10 * x^2 * y)

    return -fx, -fy, -fz
end

function velocity!(stokes, xci, xvi)
    xv, yv, zv = xvi
    di = ntuple(i -> xci[i][2] - xci[i][1], Val(3))
    xc, yc, zc = ntuple(
        i -> LinRange(xci[i][1] - di[i], xci[i][end] + di[i], length(xci[i]) + 2), Val(3)
    )
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz
    _velocity_x(x, y) = x + x^2 + x * y + x^3 * y
    _velocity_y(x, y) = y + x * y + y^2 + x^2 * y^2
    _velocity_z(x, y, z) = -2z - 3x * z - 3y * z - 5x^2 * y * z

    @parallel_indices (i, j, k) function _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
        T = eltype(Vx)
        if all((i, j, k) .≤ size(Vx))
            if (i == size(Vx, 1)) ||
                    (j == size(Vx, 2)) ||
                    (k == size(Vx, 3)) ||
                    (i == 1) ||
                    (j == 1) ||
                    (k == 1)
                Vx[i, j, k] = _velocity_x(xv[i], yc[j])
            else
                Vx[i, j, k] = zero(T)
            end
        end
        if all((i, j, k) .≤ size(Vy))
            if (i == size(Vy, 1)) ||
                    (j == size(Vy, 2)) ||
                    (k == size(Vy, 3)) ||
                    (i == 1) ||
                    (j == 1) ||
                    (k == 1)
                Vy[i, j, k] = _velocity_y(xc[i], yv[j])
            else
                Vy[i, j, k] = zero(T)
            end
        end
        if all((i, j, k) .≤ size(Vz))
            if (i == size(Vz, 1)) ||
                    (j == size(Vz, 2)) ||
                    (k == size(Vz, 3)) ||
                    (i == 1) ||
                    (j == 1) ||
                    (k == 1)
                Vz[i, j, k] = _velocity_z(xc[i], yc[j], zv[k])
            else
                Vz[i, j, k] = zero(T)
            end
        end

        return nothing
    end

    return @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
end

"""
    remove_net_flux!(stokes, ni, di)

Shift the boundary-normal velocities by a constant so that the discrete flux through the
domain boundary sums to zero. With velocities prescribed on all six faces the pressure is
only defined up to a constant, and it has a fixed point only if the prescribed data is
discretely divergence-free. Point-sampling the analytical solution at the face centers
leaves a net flux of O(di^2), which would otherwise put a floor on the divergence residual.
"""
function remove_net_flux!(stokes, ni, di)
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz
    dx, dy, dz = di
    # face areas of a single cell, normal to x, y and z
    Ax, Ay, Az = dy * dz, dx * dz, dx * dy

    flux = @views (
        (sum(Vx[end, 2:(end - 1), 2:(end - 1)]) - sum(Vx[1, 2:(end - 1), 2:(end - 1)])) * Ax +
            (sum(Vy[2:(end - 1), end, 2:(end - 1)]) - sum(Vy[2:(end - 1), 1, 2:(end - 1)])) * Ay +
            (sum(Vz[2:(end - 1), 2:(end - 1), end]) - sum(Vz[2:(end - 1), 2:(end - 1), 1])) * Az
    )
    area = 2 * (ni[2] * ni[3] * Ax + ni[1] * ni[3] * Ay + ni[1] * ni[2] * Az)
    δ = flux / area

    @views Vx[1, :, :] .+= δ
    @views Vx[end, :, :] .-= δ
    @views Vy[:, 1, :] .+= δ
    @views Vy[:, end, :] .-= δ
    @views Vz[:, :, 1] .+= δ
    @views Vz[:, :, end] .-= δ

    return nothing
end

function burstedde(; nx = 16, ny = 16, nz = 16, β = 10.0, init_MPI = true, finalize_MPI = false)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    lx = ly = lz = 1.0e0
    li = (lx, ly, lz)  # domain length in x- and y-
    origin = 0.0, 0.0, 0.0
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...) # init MPI
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1 # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(backend, ni)
    ## Setup-specific parameters and fields
    (; η) = stokes.viscosity
    viscosity!(η, xci, β)
    ρg = body_forces(xci, η, β) # => ρ*(gx, gy, gz)
    dt = Inf
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    for ratios in (
            phase_ratios.center, phase_ratios.vertex,
            phase_ratios.yz, phase_ratios.xz, phase_ratios.xy,
        )
        @parallel (@idx size(ratios)) init_burstedde_phase!(ratios)
    end
    args = (;
        T = 0.0,
        P = stokes.P,
        dt = dt,
        prescribed_viscosity = true,
        prescribed_body_force = true,
    )

    ## Boundary conditions
    # the velocity is prescribed on every face, so no face is free_slip nor no_slip and
    # `flow_bcs!` leaves the boundary values written below untouched
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (
            left = false, right = false,
            top = false, bot = false,
            back = false, front = false,
        ),

        no_slip = (
            left = false, right = false,
            top = false, bot = false,
            back = false, front = false,
        ),
    )
    # impose analytical velocity at the boundaries of the domain
    velocity!(stokes, xci, xvi)
    remove_net_flux!(stokes, ni, di)
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(
        backend, stokes, rheology, phase_ratios, grid.di, dt;
        ϵ = 1.0e-8, CFL = 0.99, γfact = 20.0,
    )

    # Physical time loop
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
                b_width = (4, 4, 4),
                verbose_PH = true,
                verbose_DR = false,
            )
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

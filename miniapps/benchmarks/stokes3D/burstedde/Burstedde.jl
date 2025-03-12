using ParallelStencil.FiniteDifferences3D

# benchmark reference:
#   C. Burstedde, G. Stadler, L. Alisic, L. C. Wilcox, E. Tan, M. Gurnis, and O. Ghattas.
#   Large-scale adaptive mantle convection simulation. Geophysical Journal International, 2013

include("vizBurstedde.jl")

@parallel_indices (i, j, k) function _viscosity!(η, x, y, z, β)
    η[i, j, k] = exp(1 - β * (x[i] * (1 - x[i]) + y[j] * (1 - y[j]) + z[k] * (1 - z[k])))

    return nothing
end

function viscosity(xi, di, β)
    ni = length.(xi)
    η = @zeros(ni...)
    @parallel (@idx ni) _viscosity!(η, xi[1], xi[2], xi[3], β)

    return η
end

function body_forces(xi::NTuple{3, T}, η, β) where {T}
    xx, yy, zz = xi
    x = PTArray([x for x in xx, y in yy, z in zz])
    y = PTArray([y for x in xx, y in yy, z in zz])
    z = PTArray([z for x in xx, y in yy, z in zz])

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

    return fx, fy, fz
end

function static(x, y, z, η, β)
    return (1, -η, (1 - 2 * x) * β * η, (1 - 2 * y) * β * η, (1 - 2 * z) * β * η)
end

function body_forces_x(x, y, z, η, β)
    fx = (
        y * z + 3 * x^2 * y^3 * z,
        2 + 6x * y,
        2 + 4x + 2y + 6x^2 * y,
        x + y + 2x * y^2 + x^3,
        -3z - 10x * y * z,
    )

    st = static(x, y, z, η, β)

    return dot(st, fx)
end

function body_forces_y(x, y, z, η, β)
    fy = (
        x * z + 3 * x^3 * y^2 * z,
        2 + 2x^2 + 2y^2,
        x + y + 2x * y^2 + x^3,
        2 + 2x + 4y + 4x^2 * y,
        -3z - 5x^2 * z,
    )

    st = static(x, y, z, η, β)

    return dot(st, fy)
end

function body_forces_z(x, y, z, η, β)
    fz = (
        x * y + x^3 * y^3,
        -10y * z,
        -3z - 10x * y * z,
        -3z - 5x^2 * z,
        -4 - 6x - 6y - 10x^2 * y,
    )

    st = static(x, y, z, η, β)

    return dot(st, fz)
end

function velocity!(stokes, xci, xvi, di)
    # xc, yc, zc = xci
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

    # @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
    return @parallel _velocity!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, xc, yc, zc, xv, yv, zv)
end

function analytical_velocity!(stokes, xci, xvi, di)
    xc, yc, zc = xci
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
        if (i ≤ size(Vx, 1)) && (j ≤ size(Vx, 2)) #&& (k ≤ size(Vx, 3))
            Vx[i, j, k] = _velocity_x(xv[i], yc[j])
        end
        if (i ≤ size(Vy, 1)) && (j ≤ size(Vy, 2)) #&& (k ≤ size(Vy, 3))
            Vy[i, j, k] = _velocity_y(xc[i], yv[j])
        end
        if (i ≤ size(Vz, 1)) && (j ≤ size(Vz, 2)) && (k ≤ size(Vz, 3))
            Vz[i, j, k] = _velocity_z(xc[i], yc[j], zv[k])
        end

        return nothing
    end

    return @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
end

function burstedde(; nx = 16, ny = 16, nz = 16, init_MPI = true, finalize_MPI = false)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    lx = ly = lz = 1.0e0
    li = (lx, ly, lz)  # domain length in x- and y-
    origin = zero(nx), zero(ny), zero(nz)
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...) # init MPI
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total siηlation time
    Δt = 1 # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(backend, ni)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di; CFL = 1 / √3)

    ## Setup-specific parameters and fields
    β = 10.0
    η = viscosity(xci, di, β) # add reference
    ρg = body_forces(xci, η, β) # => ρ*(gx, gy, gz)
    dt = Inf
    G = @fill(Inf, ni...)
    K = @fill(Inf, ni...)

    ## Boundary conditions
    flow_bcs = VelocityBoundaryConditionsionsions(;
        free_slip = (left = false, right = false, top = false, bot = false, back = false, front = false),
        no_slip = (left = false, right = false, top = false, bot = false, back = false, front = false),
    )
    # impose analytical velociity at the boundaries of the domain
    velocity!(stokes, xci, xvi, di)
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # Physical time loop
    t = 0.0

    local iters
    while t < ttot
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            G,
            K,
            dt,
            igg;
            iterMax = 10.0e3,
            nout = 1.0e3,
            b_width = (4, 4, 4),
            verbose = false,
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

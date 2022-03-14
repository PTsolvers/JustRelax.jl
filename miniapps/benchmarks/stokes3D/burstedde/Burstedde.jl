using ParallelStencil.FiniteDifferences3D

# benchmark reference:
#   C. Burstedde, G. Stadler, L. Alisic, L. C. Wilcox, E. Tan, M. Gurnis, and O. Ghattas.
#   Large-scale adaptive mantle convection simulation. Geophysical Journal International, 2013

include("vizBurstedde.jl")

@parallel_indices (ix, iy, iz) function _viscosity!(η, x, y, z, β)
    η[ix, iy, iz] = exp(
        1 - β * (x[ix] * (1 - x[ix]) + y[iy] * (1 - y[iy]) + z[iz] * (1 - z[iz]))
    )

    return nothing
end

function viscosity(xi, β)
    ni = length.(xi)
    η = @allocate ni...
    @parallel (1:ni[1], 1:ni[2], 1:ni[3]) _viscosity!(η, xi[1], xi[2], xi[3], β)

    return η
end

function body_forces(xi::NTuple{3,T}, η, β) where {T}
    xx, yy, zz = xi
    x = [x for x in xx, y in yy, z in zz]
    y = [y for x in xx, y in yy, z in zz]
    z = [z for x in xx, y in yy, z in zz]

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

function velocity!(stokes, xci, xvi)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    _velocity_x(x, y, z) = x + x^2 + x * y + x^3 * y
    _velocity_y(x, y, z) = y + x * y + y^2 + x^2 * y^2
    _velocity_z(x, y, z) = -2z - 3x * z - 3y * z - 5x^2 * y * z

    @parallel_indices (ix, iy, iz) function _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
        # Vx
        if (ix ≤ size(Vx, 1)) && (iy ≤ size(Vx, 2)) && (iz ≤ size(Vx, 3))
            if (ix == size(Vx, 1)) ||
                (iy == size(Vx, 2)) ||
                (iz == size(Vx, 3)) ||
                (ix == 1) ||
                (iy == 1) ||
                (iz == 1)
                Vx[ix, iy, iz] = _velocity_x(xv[ix], yc[iy], zc[iz])
            else
                Vx[ix, iy, iz] = zero(eltype(Vx))
            end
        end
        # Vy
        if (ix ≤ size(Vy, 1)) && (iy ≤ size(Vy, 2)) && (iz ≤ size(Vy, 3))
            if (ix == size(Vy, 1)) ||
                (iy == size(Vy, 2)) ||
                (iz == size(Vy, 3)) ||
                (ix == 1) ||
                (iy == 1) ||
                (iz == 1)
                Vy[ix, iy, iz] = _velocity_y(xc[ix], yv[iy], zc[iz])
            else
                Vy[ix, iy, iz] = zero(eltype(Vx))
            end
        end
        # Vz
        if (ix ≤ size(Vz, 1)) && (iy ≤ size(Vz, 2)) && (iz ≤ size(Vz, 3))
            if (ix == size(Vz, 1)) ||
                (iy == size(Vz, 2)) ||
                (iz == size(Vz, 3)) ||
                (ix == 1) ||
                (iy == 1) ||
                (iz == 1)
                Vz[ix, iy, iz] = _velocity_z(xc[ix], yc[iy], zv[iz])
            else
                Vz[ix, iy, iz] = zero(eltype(Vx))
            end
        end

        return nothing
    end

    @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
end

function analytical_velocity!(stokes, xci, xvi)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    _velocity_x(x, y, z) = x + x^2 + x * y + x^3 * y
    _velocity_y(x, y, z) = y + x * y + y^2 + x^2 * y^2
    _velocity_z(x, y, z) = -2z - 3x * z - 3y * z - 5x^2 * y * z

    @parallel_indices (ix, iy, iz) function _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
        if (ix ≤ size(Vx, 1)) && (iy ≤ size(Vx, 2)) && (iz ≤ size(Vx, 3))
            Vx[ix, iy, iz] = _velocity_x(xv[ix], yc[iy], zc[iz])
        end
        if (ix ≤ size(Vy, 1)) && (iy ≤ size(Vy, 2)) && (iz ≤ size(Vy, 3))
            Vy[ix, iy, iz] = _velocity_y(xc[ix], yv[iy], zc[iz])
        end
        if (ix ≤ size(Vz, 1)) && (iy ≤ size(Vz, 2)) && (iz ≤ size(Vz, 3))
            Vz[ix, iy, iz] = _velocity_z(xc[ix], yc[iy], zv[iz])
        end

        return nothing
    end

    @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
end

function burstedde(; nx=16, ny=16, nz=16, init_MPI=true, finalize_MPI=false)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    lx = ly = lz = 1e0
    li = (lx, ly, lz)  # domain length in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI
    @static if USE_GPU # select one GPU per MPI local rank (if >1 GPU per node)
        select_device()
    end
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li) # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total siηlation time
    dt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; Re=6π, CFL=0.9 / √3)

    ## Setup-specific parameters and fields
    β = 10.0
    η = viscosity(xci, β) # add reference 
    ρg = body_forces(xci, η, β) # => ρ*(gx, gy, gz)
    G = Inf

    ## Boundary conditions
    freeslip = (freeslip_x=false, freeslip_y=false, freeslip_z=false)
    # impose analytical velociity at the boundaries of the domain
    velocity!(stokes, xci, xvi)

    # Physical time loop
    t = 0.0

    local iters
    while t < ttot
        iters = solve!(
            stokes, pt_stokes, ni, di, li, max_li, freeslip, ρg, η, G, dt, igg; iterMax=10e3
        )
        t += dt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

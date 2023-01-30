using MPI

# FVCA8 benchmark for the Stokes and Navier-Stokes
#     equations with the TrioCFD code – benchmark session
#     P.-E. Angeli, M.-A. Puscas, G. Fauchet, A. Cartalade
# HAL Id: cea-02434556
# https://hal-cea.archives-ouvertes.fr/cea-02434556

include("vizTaylorGreen.jl")

function body_forces(xi::NTuple{3,T}) where {T}
    xx, yy, zz = xi
    x = PTArray([x for x in xx, y in yy, z in zz])
    y = PTArray([y for x in xx, y in yy, z in zz])
    z = PTArray([z for x in xx, y in yy, z in zz])

    fz, fy = @zeros(size(x)...), @zeros(size(x)...)
    fx = @. -36 * π^2 * cos(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)

    return fx, fy, fz
end

function velocity!(stokes, xci, xvi)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    _velocity_x(x, y, z) = -2cos(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)
    _velocity_y(x, y, z) = sin(2 * π * x) * cos(2 * π * y) * sin(2 * π * z)
    _velocity_z(x, y, z) = sin(2 * π * x) * sin(2 * π * y) * cos(2 * π * z)

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

function taylorGreen(; nx=16, ny=16, nz=16, init_MPI=true, finalize_MPI=false)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    lx = ly = lz = 1e0
    li = (lx, ly, lz)  # domain length in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=false)...) # init MPI
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    xci, xvi = lazy_grid(di, li) # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    dt = Inf   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di)

    ## Setup-specific parameters and fields
    β = 10.0
    η = @ones(ni...) # add reference 
    ρg = body_forces(xci) # => ρ*(gx, gy, gz)
    Gc = @fill(Inf, ni...) 
    K = @fill(Inf, ni...) 

    ## Boundary conditions
    freeslip = (freeslip_x=false, freeslip_y=false, freeslip_z=false)
    # impose analytical velociity at the boundaries of the domain
    velocity!(stokes, xci, xvi)

    # Physical time loop
    t = 0.0

    local iters
    while t < ttot
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            freeslip,
            ρg,
            η,
            K,
            Gc,
            dt,
            igg;
            iterMax=10e3,
            b_width=(4, 4, 4),
        )
        t += dt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

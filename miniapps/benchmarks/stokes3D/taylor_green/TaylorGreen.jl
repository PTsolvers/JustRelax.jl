using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid
using JustRelax
using Printf, LinearAlgebra, CairoMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# FVCA8 benchmark for the Stokes and Navier-Stokes
#     equations with the TrioCFD code – benchmark session
#     P.-E. Angeli, M.-A. Puscas, G. Fauchet, A. Cartalade
# HAL Id: cea-02434556
# https://hal-cea.archives-ouvertes.fr/cea-02434556
function body_forces(xi::NTuple{3, T}) where T
    xx, yy, zz = xi
    x = [x for x in xx, y in yy, z in zz]
    y = [y for x in xx, y in yy, z in zz]
    z = [z for x in xx, y in yy, z in zz]
   
    fz, fy = @zeros(size(x)...), @zeros(size(x)...)
    fx = @. -36*π^2*cos(2*π*x)*sin(2*π*y)*sin(2*π*z)
            
    return fx, fy, fz
end

function velocity!(stokes, xci, xvi)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    _velocity_x(x, y, z) = -2cos(2*π*x)*sin(2*π*y)*sin(2*π*z)
    _velocity_y(x, y, z) =   sin(2*π*x)*cos(2*π*y)*sin(2*π*z)
    _velocity_z(x, y, z) =   sin(2*π*x)*sin(2*π*y)*cos(2*π*z)

    @parallel_indices (ix, iy, iz) function _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
        # Vx
        if (ix ≤ size(Vx, 1)) && (iy ≤ size(Vx, 2)) && (iz ≤ size(Vx, 3))
            if (ix == size(Vx, 1)) || (iy == size(Vx, 2)) ||  (iz == size(Vx, 3)) || (ix == 1) || (iy == 1) ||  (iz ==1)
                Vx[ix, iy, iz] = _velocity_x(xv[ix], yc[iy], zc[iz])
            else
                Vx[ix, iy, iz] = zero(eltype(Vx))
            end
        end
        # Vy
        if (ix ≤ size(Vy, 1)) && (iy ≤ size(Vy, 2)) && (iz ≤ size(Vy, 3))
            if (ix == size(Vy, 1)) || (iy == size(Vy, 2)) ||  (iz == size(Vy, 3)) || (ix == 1) || (iy == 1) ||  (iz == 1)
                Vy[ix, iy, iz] = _velocity_y(xc[ix], yv[iy], zc[iz])
            else
                Vy[ix, iy, iz] = zero(eltype(Vx))
            end
        end
        # Vz
        if (ix ≤ size(Vz, 1)) && (iy ≤ size(Vz, 2)) && (iz ≤ size(Vz, 3))
            if (ix == size(Vz, 1)) || (iy == size(Vz, 2)) ||  (iz == size(Vz, 3)) || (ix == 1) || (iy == 1) ||  (iz ==1)
                Vz[ix, iy, iz] = _velocity_z(xc[ix], yc[iy], zv[iz])
            else
                Vz[ix, iy, iz] = zero(eltype(Vx))
            end
        end
        
        return
    end

    @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)

end

function analytical_velocity!(stokes, xci, xvi)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    _velocity_x(x, y, z) = -2cos(2*π*x)*sin(2*π*y)*sin(2*π*z)
    _velocity_y(x, y, z) =   sin(2*π*x)*cos(2*π*y)*sin(2*π*z)
    _velocity_z(x, y, z) =   sin(2*π*x)*sin(2*π*y)*cos(2*π*z)

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
        
        return
    end

    @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)

end

function analytical_velocity(xci, xvi)
    nvi = length.(xvi)
    nci = length.(xci)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx = @allocate(nvi[1], nci[2], nci[3])
    Vy = @allocate(nci[1], nvi[2], nci[3])
    Vz = @allocate(nci[1], nci[2], nvi[3])

    _velocity_x(x, y, z) = -2cos(2*π*x)*sin(2*π*y)*sin(2*π*z)
    _velocity_y(x, y, z) =   sin(2*π*x)*cos(2*π*y)*sin(2*π*z)
    _velocity_z(x, y, z) =   sin(2*π*x)*sin(2*π*y)*cos(2*π*z)

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
        
        return
    end

    @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)

    return Vx, Vy, Vz
end

function analytical_pressure(xci)
    nci = length.(xci)
    x, y, z = xci
    P = @allocate nci...

    _pressure(x, y, z) = -6*π*sin(2*π*x)*sin(2*π*y)*sin(2*π*z)

    @parallel_indices (ix, iy, iz) function pressure(P, x, y, z)
        P[ix, iy, iz] = _pressure(x[ix], y[iy], z[iz])
        return
    end

    @parallel pressure(P, x, y, z)

    return P
end

function analytical_solution(xci, xvi)
    Vx, Vy, Vz = analytical_velocity(xci, xvi)
    P = analytical_pressure(xci)

    return Vx, Vy, Vz, P
end

function TaylorGreen()
    nx=ny=nz=64
    lx=ly=lz=1e0
    
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    li = (lx, ly, lz)  # domain length in x- and y-
    igg = IGG(
        init_global_grid(nx, ny, nz)...
    ) # init MPI
    @static if USE_GPU select_device() end # select one GPU per MPI local rank (if >1 GPU per node)
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li/(nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li) # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    dt = Inf   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di, Re = 6π, CFL = 0.9/√3)

    ## Setup-specific parameters and fields
    β = 10.0
    η = @ones(ni...) # add reference 
    ρg = body_forces(xci) # => ρ*(gx, gy, gz)
    G = 1e0

    ## Boundary conditions
    freeslip = (
        freeslip_x = false,
        freeslip_y = false,
        freeslip_z = false
    )
    # impose analytical velociity at the boundaries of the domain
    velocity!(stokes, xci, xvi)
    
    # Physical time loop
    t = 0.0

    local iters
    while t < ttot
        iters = solve!(stokes, pt_stokes, ni, di, li, max_li, freeslip, ρg, η, G, dt; iterMax = 10e3)
        t += dt
    end

    finalize_global_grid()

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

function plot(stokes, geometry; cmap=:vik)
    xci, xvi = geometry.xci, geometry.xvi
    vx, vy, vz, p = analytical_solution(xci, xvi)


    islice = geometry.ni[1] ÷ 2
    f = Figure(resolution = (2600,900) )
    # Pressure
    ax = Axis(f[1,1], axis = 1)
    h = heatmap!(ax, xci[1],xci[2], stokes.P[islice, :, :], colormap=cmap)
    Colorbar(f[1,2], h);f

    ax = Axis(f[2,1], axis=1)
    h = heatmap!(ax, xci[1],xci[2], p[islice, :, :], colormap=:vik)
    Colorbar(f[2,2], h);f

    # Vx
    ax = Axis(f[1,3], axis = 1)
    h = heatmap!(ax, xvi[1], xci[2], stokes.V.Vx[islice, :, :], colormap=cmap)
    Colorbar(f[1,4], h);f

    ax = Axis(f[2,3], axis=1)
    h = heatmap!(ax, xvi[1], xci[2], vx[islice, :, :], colormap=cmap)
    Colorbar(f[2,4], h);f

    # Vy
    ax = Axis(f[1,5], axis = 1)
    h = heatmap!(ax, xvi[1], xci[2], stokes.V.Vy[islice, :, :], colormap=cmap)
    Colorbar(f[1,6], h);f

    ax = Axis(f[2,5], axis=1)
    h = heatmap!(ax, xvi[1], xci[2], vy[islice, :, :], colormap=cmap)
    Colorbar(f[2,6], h);f

    f
end

function error(stokes, geometry)
    gridsize = foldl(*, geometry.di)
    vx, vy, vz, p = analytical_solution(geometry.xci, geometry.xvi)

    order = 2
    L2_vx = norm(stokes.V.Vx .- vx, order)*gridsize
    L2_vy = norm(stokes.V.Vy .- vy, order)*gridsize
    L2_vz = norm(stokes.V.Vz .- vz, order)*gridsize
    L2_p = norm(stokes.P .- (p), order)*gridsize
end
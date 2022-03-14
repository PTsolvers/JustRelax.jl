using ParallelStencil.FiniteDifferences3D

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

function viscosity(ni, di, li, rc, η0, ηi; b_width=(16, 8, 4))
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
    Δη=1e-3,
    nx=32 - 1,
    ny=32 - 1,
    nz=32 - 1,
    lx=1e1,
    ly=1e1,
    lz=1e1,
    rc=1e0,
    εbg=1e0,
    init_MPI=true,
    finalize_MPI=false,
)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI
    @static if USE_GPU # select one GPU per MPI local rank (if >1 GPU per node)
        select_device()
    end
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li) # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    dt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; Re=6π, CFL=0.8 / √3)

    ## Setup-specific parameters and fields
    ξ = 1.0 # Maxwell relaxation time
    η0 = 1.0 # matrix viscosity
    ηi = Δη # inclusion viscosity
    G = 1.0 # elastic shear modulus
    dt = η0 / (G * ξ)
    η = viscosity(ni, di, li, rc, η0, ηi)

    ## Boundary conditions
    pureshear_bc!(stokes, di, li, εbg)
    freeslip = (freeslip_x=true, freeslip_y=true, freeslip_z=true)

    ## Body forces
    ρg = ntuple(_ -> @zeros(ni...), Val(3))

    ## Time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve!(
            stokes,
            pt_stokes,
            ni,
            di,
            li,
            max_li,
            freeslip,
            ρg,
            η,
            G,
            dt,
            igg;
            iterMax=5000,
            nout=2000,
        )
        t += dt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

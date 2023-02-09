using LinearAlgebra, GLMakie, JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# model resolution (number of gridpoints)
nx, ny, nz = 16, 16, 16

# set MPI
init_MPI = true
finalize_MPI = false
 
# model specific parameters
Δη = 1e1 # viscosity ratio between matrix and inclusion
rc = 1e0 # radius of the inclusion
εbg = 1e0 # background strain rate
lx, ly, lz = 1e1, 1e1, 1e1 # domain siye in x and y directions


@parallel function smooth!(A2::AbstractArray{T,3}, A::AbstractArray{T,3}, fact::T) where {T}
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

function viscosity(ni, di, li, rc, η0, ηi; b_width=(1, 1, 1))
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
    ni = nx, ny, nz # number of nodes in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    xci, xvi = lazy_grid(di, li) # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    dt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di)

    ## Setup-specific parameters and fields
    ξ = 1.0 # Maxwell relaxation time
    η0 = 1.0 # matrix viscosity
    ηi = Δη # inclusion viscosity
    G = 1.0 # elastic shear modulus
    # dt = η0 / (G * ξ)
    dt = Inf
    η = viscosity(ni, di, li, rc, η0, ηi)
    Gc = @fill(G, ni...) 
    K = @fill(Inf, ni...) 

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
            di,
            li,
            freeslip,
            ρg,
            η,
            K,
            Gc,
            dt,
            igg;
            iterMax=5000,
            nout=100,
        )
        t += dt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end


stokes = StokesArrays(ni, ViscoElastic)
pureshear_bc!(stokes, di, li, εbg)

ητ = deepcopy(η)
_di = inv.(di)

@parallel (1:nx, 1:ny, 1:nz) JustRelax.Elasticity3D.compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, _di...)
@parallel JustRelax.Elasticity3D.compute_P!(stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, K, dt, pt_stokes.r, pt_stokes.θ_dτ)
@parallel (1:nx+1, 1:ny+1, 1:nz+1) JustRelax.Elasticity3D.compute_strain_rate!(
    stokes.∇V, 
    stokes.ε.xx, 
    stokes.ε.yy, 
    stokes.ε.zz, 
    stokes.ε.yz, 
    stokes.ε.xz, 
    stokes.ε.xy, 
    stokes.V.Vx, 
    stokes.V.Vy, 
    stokes.V.Vz, 
    _di...
)

@parallel (1:nx+1, 1:ny+1, 1:nz+1) JustRelax.Elasticity3D.compute_τ!(
    stokes.τ.xx,
    stokes.τ.yy,
    stokes.τ.zz,
    stokes.τ.yz,
    stokes.τ.xz,
    stokes.τ.xy,
    stokes.τ_o.xx,
    stokes.τ_o.yy,
    stokes.τ_o.zz,
    stokes.τ_o.yz,
    stokes.τ_o.xz,
    stokes.τ_o.xy,
    stokes.ε.xx,
    stokes.ε.yy,
    stokes.ε.zz,
    stokes.ε.yz,
    stokes.ε.xz,
    stokes.ε.xy,
    η,
    G,
    dt,
    pt_stokes.θ_dτ,
)

heatmap(xvi[1], xvi[2], Array(stokes.τ.xx[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.τ.zz[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.τ.xy[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.τ.yz[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.τ.xz[:,8,:]); colormap=:batlow)

heatmap(xvi[1], xvi[2], Array(stokes.ε.xy[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.ε.yz[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.ε.xz[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.∇V[:,8,:]); colormap=:batlow)

@parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) JustRelax.Elasticity3D.compute_V!(
    stokes.V.Vx,
    stokes.V.Vy,
    stokes.V.Vz,
    stokes.R.Rx,
    stokes.R.Ry,
    stokes.R.Rz,
    stokes.P,
    ρg[1],
    ρg[2],
    ρg[3],
    stokes.τ.xx,
    stokes.τ.yy,
    stokes.τ.zz,
    stokes.τ.yz,
    stokes.τ.xz,
    stokes.τ.xy,
    ητ,
    pt_stokes.ηdτ,
    _di...,
)
apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)

heatmap(xvi[1], xvi[2], Array(stokes.V.Vx[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.V.Vz[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.R.Rx[:,8,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.R.Rz[:,9,:]); colormap=:batlow)
heatmap(xvi[1], xvi[2], Array(stokes.R.Ry[:,8,:]); colormap=:batlow)
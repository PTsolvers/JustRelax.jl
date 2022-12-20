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
    li = lx, ly, lz  # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
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
    # dt = η0 / (G * ξ)
    dt = Inf
    η = viscosity(ni, di, li, rc, η0, ηi)
    # Gc = @fill(G, ni...)
    Gc = @fill(Inf, ni...)
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
            Gc,
            K,
            dt,
            igg;
            iterMax=5000,
            nout=100,
            b_width=(4, 4, 4),
        )
        t += dt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

# phsysics
fx, fy, fz = ρg # gravitational forces
Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz # velocity
P, ∇V = stokes.P, stokes.∇V  # pressure and velociity divergence
εxx, εyy, εzz, εxy, εxz, εyz = JustRelax.strain(stokes)
τ, τ_o = JustRelax.stress(stokes) # stress 
τxx, τyy, τzz, τxy, τxz, τyz = τ
τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
# solver related
Rx, Ry, Rz, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.Rz, stokes.R.RP
ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
# geometry
_dx, _dy, _dz = @. 1 / di
nx, ny, nz = size(P)
nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx - 1, nx - 2, ny - 1, ny - 2, nz - 1, nz - 2
 
P_old = deepcopy(P)

# ~preconditioner
ητ = deepcopy(η)
@hide_communication b_width begin # communication/computation overlap
    @parallel JustRelax.compute_maxloc!(ητ, η)
    update_halo!(ητ)
end
apply_free_slip!((freeslip_x=true, freeslip_y=true, freeslip_z=true), ητ, ητ, ητ)
 # @parallel (1:size(ητ, 2), 1:size(ητ, 3)) free_slip_x!(ητ)
 # @parallel (1:size(ητ, 1), 1:size(ητ, 3)) free_slip_y!(ητ)
 # @parallel (1:size(ητ, 1), 1:size(ητ, 2)) free_slip_z!(ητ)

 # errors
 err = 2 * ϵ
 iter = 0
 cont = 0
 err_evo1 = Float64[]
 err_evo2 = Int64[]
 norm_Rx = Float64[]
 norm_Ry = Float64[]
 norm_Rz = Float64[]
 norm_∇V = Float64[]

for _ in 1:100
    # free slip boundary conditions
    apply_free_slip!(freeslip, Vx, Vy, Vz)

    @parallel (1:nx, 1:ny, 1:nz) JustRelax.Elasticity3D.compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)

    @parallel JustRelax.Elasticity3D.compute_strain_rate!(
        εxx, εyy, εzz, εyz, εxz, εxy, ∇V, Vx, Vy, Vz, _dx, _dy, _dz
    )

    @parallel JustRelax.Elasticity3D.compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)

    @parallel JustRelax.Elasticity3D.compute_τ!(
        τxx,
        τyy,
        τzz,
        τxy,
        τxz,
        τyz,
        τxx_o,
        τyy_o,
        τzz_o,
        τxy_o,
        τxz_o,
        τyz_o,
        εxx,
        εyy,
        εzz,
        εyz,
        εxz,
        εxy,
        η,
        G,
        dt,
        θ_dτ
    )

    @parallel JustRelax.Elasticity3D.compute_V!(
        Vx,
        Vy,
        Vz,
        P,
        fx,
        fy,
        fz,
        τxx,
        τyy,
        τzz,
        τxy,
        τxz,
        τyz,
        ητ,
        ηdτ,
        _dx,
        _dy,
        _dz,
        nx_1,
        nx_2,
        ny_1,
        ny_2,
        nz_1,
        nz_2,
    )
end

heatmap(xci[1], xci[2], Vx[8, :, :], colormap=:batlow)
heatmap(xci[1], xci[2], Vx[:, :, end-1], colormap=:batlow)
heatmap(xci[1], xci[2], Vx[8, 2:end-1, 2:end-1], colormap=:batlow)
heatmap(xci[1], xci[2], ητ[8, :, :], colormap=:batlow)

heatmap(xci[1], xci[2], τxx[:, 8, :], colormap=:batlow)
heatmap(xci[1], xci[2], εxx[:, 8, :], colormap=:batlow)

@parallel JustRelax.Elasticity3D.compute_Res!(
    Rx,
    Ry,
    Rz,
    fx,
    fy,
    fz,
    P,
    τxx,
    τyy,
    τzz,
    τxy,
    τxz,
    τyz,
    _dx,
    _dy,
    _dz,
)

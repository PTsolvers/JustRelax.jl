# include benchmark related functions
include("vizSolVi.jl")

@parallel function smooth!(
    A2::AbstractArray{T,2}, A::AbstractArray{T,2}, fact::Real
) where {T}
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

function _viscosity!(η, xci, yci, rc, ηi, cx, cy)
    for i in 1:length(xci), j in 1:length(yci)
        if rc < sqrt((xci[i] - cx)^2 + (yci[j] - cy)^2)
            η[i, j] = ηi
        end
    end
end

function solvi_viscosity(xci, ni, li, rc, η0, ηi)
    cx, cy = li ./ 2
    η = fill(η0, ni...)
    Rad2 = [(x - cx) ^ 2 + (y - cy) ^ 2 for x in xci[1], y in xci[2]]
    η[Rad2 .< rc] .= ηi
    # η2 = deepcopy(η)
    # η3 = deepcopy(η)
    # for _ in 1:10
    #     @parallel smooth!(η2, η, 1.0)
    #     η, η2 = η2, η
    # end
    # η3[Rad2 .< rc] .= η[Rad2 .< rc]
    # η = deepcopy(η3)
    # η2 = deepcopy(η)
    # for _ in 1:10
    #     @parallel smooth!(η2, η, 1.0)
    #     η, η2 = η2, η
    # end

    return η
end

function solViEl(; Δη=1e-3, nx=256 - 1, ny=256 - 1, lx=1e0, ly=1e0, rc=0.01, εbg=1e0)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    nDim = length(ni) # domain dimension
    xci = Tuple([(di[i] / 2):di[i]:(li[i] - di[i] / 2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 5 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di)

    ## Setup-specific parameters and fields
    η0 = 1e0  # matrix viscosity
    ηi = 1e-1 # inclusion viscosity
    η  = solvi_viscosity(xci, ni, li, rc, η0, ηi) # viscosity field
    ξ  = 1.0 # Maxwell relaxation time
    G  = 1.0 # elastic shear modulus
    # dt = η0 / (G * ξ)
    dt = 0.25
    Gc = @fill(G, ni...)
    K  = @fill(Inf, ni...)

    ## Boundary conditions
    pureshear_bc!(stokes, xci, xvi, εbg)
    freeslip = (freeslip_x=true, freeslip_y=true)

    # Physical time loop
    t = 0.0
    ρg = (@zeros(nx-1, ny), @zeros(nx, ny-1))
    local iters
    while t < ttot
        iters = solve!(
            stokes, pt_stokes, di, li, freeslip, ρg, η, Gc, K, dt; nout=500, iterMax=20e3, verbose=true
        )
        t += Δt
        heatmap(
            xci[1],
            xci[2],
            # stokes.V.Vy;
            stokes.ε.xy;
            colormap=:batlow,
        )
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

function multiple_solViEl(; Δη=1e-3, lx=1e1, ly=1e1, rc=1e0, εbg=1e0, nrange::UnitRange=4:8)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, iters = solVi(; Δη=Δη, nx=nx, ny=ny, lx=lx, ly=ly, rc=rc, εbg=εbg)
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes, Δη, εbg, rc; order=2)
        push!(L2_vx, L2_vxi)
        push!(L2_vy, L2_vyi)
        push!(L2_p, L2_pi)
    end

    nx = @. 2^nrange - 1
    h = @. (1 / nx)

    f = Figure(; fontsize=28)
    ax = Axis(
        f[1, 1];
        yscale=log10,
        xscale=log10,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(8),
    )
    lines!(ax, h, (L2_vx); linewidth=3, label="Vx")
    lines!(ax, h, (L2_vy); linewidth=3, label="Vy")
    lines!(ax, h, (L2_p); linewidth=3, label="P")
    axislegend(ax; position=:lt)
    ax.xlabel = "h"
    ax.ylabel = "L2 norm"
    return f
end

# unpack
_dx, _dy = inv.(di)
lx, ly = li
Vx, Vy = stokes.V.Vx, stokes.V.Vy
dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
εxx, εyy, εxy = JustRelax.strain(stokes)
τ, τ_o = JustRelax.stress(stokes)
τxx, τyy, τxy = τ
τxx_o, τyy_o, τxy_o = τ_o
P, ∇V = stokes.P, stokes.∇V
Rx, Ry, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.RP
ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ,  pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ

ρgx, ρgy = ρg
P_old = deepcopy(P)

# ~preconditioner
ητ = deepcopy(η)
@parallel JustRelax.compute_maxloc!(ητ, η)
apply_free_slip!(freeslip, ητ, ητ)

# PT numerical coefficients
# @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

_sqrt_leng_Rx = one(T) / sqrt(length(Rx))
_sqrt_leng_Ry = one(T) / sqrt(length(Ry))
_sqrt_leng_∇V = one(T) / sqrt(length(∇V))

# errors
err = 2 * ϵ
iter = 0
cont = 0
err_evo1 = Float64[]
err_evo2 = Float64[]
norm_Rx = Float64[]
norm_Ry = Float64[]
norm_∇V = Float64[]

# solver loop
wtime0 = 0.0
while iter < 2 || (err > ϵ && iter ≤ iterMax)
    wtime0 += @elapsed begin
        # free slip boundary conditions
        apply_free_slip!(freeslip, Vx, Vy)
        for _ in 1:1000
            @parallel JustRelax.Elasticity2D.compute_∇V!(∇V, Vx, Vy, _dx, _dy)
            @parallel JustRelax.Elasticity2D.compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
            @parallel JustRelax.Elasticity2D.compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel JustRelax.Elasticity2D.compute_τ!(
                τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, Gc, θ_dτ, dt
            )
            @parallel JustRelax.Elasticity2D.compute_V!(Vx, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
        end
        heatmap(xci[1], xci[2], εxy, colormap=:batlow)
        heatmap(xci[1], xci[2], τxx, colormap=:batlow)
        heatmap(xci[1], xci[2], Rx, colormap=:batlow)

    end

    iter += 1
    if iter % nout == 0 && iter > 1
        @parallel JustRelax.Elasticity2D.compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)

        push!(norm_Rx, maximum(abs.(Rx)))
        push!(norm_Ry, maximum(abs.(Ry)))
        push!(norm_∇V, maximum(abs.(RP)))
        err = max(norm_Rx[end], norm_Ry[end], norm_∇V[end])
        push!(err_evo1, err)
        push!(err_evo2, iter)

        if( verbose && (err > ϵ)) || (iter == iterMax)
            @printf(
                "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                iter,
                err,
                norm_Rx[end],
                norm_Ry[end],
                norm_∇V[end]
            )
        end
    end
end

Re=3π
CFL=0.9 / √2
CFL=1 / √2.1
r=0.7
lτ = min(li...)
Vpdτ = min(di...) * CFL
θ_dτ = lτ * (r + 2.0) / (Re * Vpdτ)
ηdτ = Vpdτ * lτ / Re

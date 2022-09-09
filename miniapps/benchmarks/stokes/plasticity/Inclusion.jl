# include benchmark related functions
using JustRelax
using Printf, LinearAlgebra #, CairoMakie
using GeoParams

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

function _viscosity!(η, xci, yci, rc, ηi, cx, cy)
    for i in 1:length(xci), j in 1:length(yci)
        if rc < sqrt((xci[i] - cx)^2 + (yci[j] - cy)^2)
            η[i, j] = ηi
        end
    end
end

function solvi_viscosity(ni, di, li, rc, η0, ηi)
    dx, dy = di
    lx, ly = li
    # cx, cy = lx/2, ly/2
    η = @fill(η0, ni...)
    # _viscosity!(η, xci, yci, rc, ηi, cx, cy)
    phase = Int.(@fill(1, ni...))
    Rad2 = [
        sqrt.(
            ((ix - 1) * dx + 0.5 * dx - 0.5 * lx)^2 +
            ((iy - 1) * dy + 0.5 * dy - 0.5 * ly)^2,
        ) for ix in 1:ni[1], iy in 1:ni[2]
    ]
    η[Rad2 .< rc] .= ηi
    phase[Rad2 .< rc] .= 2
    return η, phase
end

function fill_borders!(A)
    A[:, 1] .= A[:, 2]
    A[:, end] .= A[:, end - 1]
    A[1, :] .= A[2, :]
    return A[end, :] .= A[end - 1, :]
end

function viscoelastic_viscosity(MatParam, G, phase, dt, I::Vararg{Int64,N}) where {N}
    return 1.0 / (
        1.0 / G[I...] * dt +
        1.0 / computeViscosity_τII(MatParam, 0.0, phase[I...], (;); cutoff=(-Inf, Inf))
    )
end

@parallel_indices (i, j) function viscoelastic_viscosity!(ηve, MatParam, G, phase, dt)
    ηve[i, j] =
        inv(
            inv(G[i, j] * dt) +
            inv(computeViscosity_τII(MatParam, 0.0, phase[i, j], (;); cutoff=(-Inf, Inf)))
        )
    return nothing
end

@parallel_indices (i, j) function update_plastic_strain!(λ, P, ηve, τII, phase, MatParam)
    function _update_Ftrial!(λ, i, j)
        Ftrial = compute_yieldfunction(MatParam, phase[i, j], (P=P[i, j], τII=τII[i, j]))

        if Ftrial ≥ 0.0
            λ[i, j] = Ftrial * inv(ηve[i, j])
        else
            λ[i, j] = 0.0
        end
    end

    _update_Ftrial!(λ, i, j)

    return nothing
end

# @parallel_indices (i, j) function update_Ftrial!(Ftrial, λ, P, ηvp, τII, phase, MatParam)
#     Ftrial[i, j] =
#         compute_yieldfunction(MatParam, phase[i, j], (P=P[i, j], τII=τII[i, j])) -
#         λ[i, j] * ηvp[i, j]

#     return nothing
# end

# @parallel_indices (i, j) function update_plastic_strain!(λ, Ftrial, ηve, ηvp)
#     if Ftrial[i, j] < 0.0
#         λ[i, j] = Ftrial[i, j] * 1.0 / (ηve[i, j] + ηvp[i, j])
#     end
#     return nothing
# end

@parallel_indices (i, j) function second_invariant!(CII, Cxx, Cyy, Cxy)
    normal_component(i, j) = @inbounds (Cxx[i, j], Cyy[i, j])
    function shear_component(i, j)
        @inbounds (Cxy[i - 1, j - 1], τxy[i, j], Cxy[i - 1, j], Cxy[i, j - 1])
    end

    CII[i, j] = second_invariant_staggered(normal_component(i, j), shear_component(i, j))

    return nothing
end

# @parallel_indices (i, j) function stress_corrections!(
#     τxx, τyy, τxy, τII, εxx, εyy, εxy, λ, ηve
# )
#     av(i, j) = (τII[i - 1, j - 1] + τII[i, j] + τII[i - 1, j] + τII[i, j - 1]) * 0.25

#     τxx[i, j] += 2.0 * ηve[i, j] * (- λ[i, j] * (0.5 * τxx[i, j] / τII[i, j]))
#     τyy[i, j] += 2.0 * ηve[i, j] * (- λ[i, j] * (0.5 * τyy[i, j] / τII[i, j]))
#     τxy[i - 1, j - 1] +=
#         2.0 * ηve[i, j] * (- λ[i, j] * (0.5 * τxy[i, j] / av(i, j)))

#     # @all(τxx) =
#     #     (@all(τxx) + @all(τxx_o) * @Gr2() + T(2) * @all(Gdτ) * @all(εxx)) /
#     #     (one(T) + @all(Gdτ) / @all(η) + @Gr2())
#     # @all(τyy) =
#     #     (@all(τyy) + @all(τyy_o) * @Gr2() + T(2) * @all(Gdτ) * @all(εyy)) /
#     #     (one(T) + @all(Gdτ) / @all(η) + @Gr2())
#     # @all(τxy) =
#     #     (@all(τxy) + @all(τxy_o) * @harm_Gr2() + T(2) * @harm(Gdτ) * @all(εxy)) /
#     #     (one(T) + @harm(Gdτ) / @harm(η) + @harm_Gr2())

#     return nothing
# end

@parallel_indices (i, j) function stress_corrections!(
    τxx, τyy, τxy, τII, εxx, εyy, εxy, λ, ηve, Gdτ
)
    # av(A, i, j) = (A[i - 1, j - 1] + A[i, j] + A[i - 1, j] + A[i, j - 1]) * 0.25
    av(A, i, j) = sum(@inbounds inv(A[ii,jj]) for ii in i-1:i, jj in j-1:j) * 0.25
    function harm(A, i, j)
        # 4.0 / (
        #     1.0 / A[i - 1, j - 1] + 1.0 / A[i, j] + 1.0 / A[i - 1, j] + 1.0 / A[i, j - 1]
        # )
        4.0 / (
            sum(@inbounds inv(A[ii,jj]) for ii in i-1:i, jj in j-1:j)
        )
    end
    visc_eff(i, j) = 2.0 / (1.0 / Gdτ[i, j] + 1.0 / ηve[i, j])
    function update_normal_stress(τii, εii, i, j)
        visc_eff(i, j) * ((εii[i, j] - λ[i, j] * (0.5 * τii[i, j] / τII[i, j])))
    end

    τxx[i, j] = update_normal_stress(τxx, εxx, i, j)
    τyy[i, j] = update_normal_stress(τyy, εyy, i, j)
    τxy[i - 1, j - 1] +=
        2.0 / (1.0 / harm(Gdτ, i, j) + 1.0 / harm(ηve, i, j)) *
        (εxy[i, j] - λ[i, j] * (0.5 * τxy[i, j] / harm(τII, i, j)))

    return nothing
end

# @parallel_indices (i, j) function stress_corrections!(
#     τxx, τyy, τxy, τII, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, λ, ηve, G, Gdτ, _dt
# )
#     av(A, i, j) = (A[i - 1, j - 1] + A[i, j] + A[i - 1, j] + A[i, j - 1]) * 0.25
#     harm(A, i, j) = 4.0 / (1.0/A[i - 1, j - 1] + 1.0/A[i, j] + 1.0/A[i - 1, j] + 1.0/A[i, j - 1]) 
#     visc_eff(i, j) = 2.0 / (1.0 / Gdτ[i,j] + 1.0 / ηve[i, j])
#     update_normal_stress(τii, τii_o, εii, i,j) = visc_eff(i, j) * ((εii[i,j] - λ[i, j] * (0.5 * τii[i, j] / τII[i, j])) + _dt * 0.5 / G[i,j] * τii_o[i, j] + 0.5 * τii[i, j] / Gdτ[i,j])

#     τxx[i, j] = update_normal_stress(τxx, τxx_o, εxx, i, j)
#     τyy[i, j] = update_normal_stress(τyy, τyy_o, εyy, i, j)
#     τxy[i - 1, j - 1] +=
#         2.0 / (1.0 / harm(Gdτ, i, j) + 1.0 / harm(ηve, i,  j)) * (εxy[i,j] - λ[i, j] * (0.5 * τxy[i, j] / av(τII, i, j)) + _dt * 0.5 / harm(G,i,j) * τxy_o[i, j] + 0.5 * τxy[i, j] / harm(Gdτ,i,j))

#     # τxx[i, j] = visc_eff(i, j) * ((εxx[i,j] - λ[i, j] * (0.5 * τxx[i, j] / τII[i, j])) + _dt * 0.5 / G[i,j] * τxx_o[i, j] + 0.5 * τxx[i, j] / Gdτ[i,j])
#     # τyy[i, j] = visc_eff(i, j) * ((εyy[i,j] - λ[i, j] * (0.5 * τyy[i, j] / τII[i, j])) + _dt * 0.5 / G[i,j] * τyy_o[i, j] + 0.5 * τyy[i, j] / Gdτ[i,j])

#     return nothing
# end

# @parallel function viscoplastic_viscosity!(ηvp, τII, εII)
#     @all(ηvp) = 0.5 * @all(τII) / @all(εII)
#     return nothing
# end

@parallel function effective_strain_rate!(
    εxx_eff,
    εyy_eff,
    εxy_eff,
    τxx,
    τyy,
    τxy,
    τxx_o,
    τyy_o,
    τxy_o,
    εxx,
    εyy,
    εxy,
    G,
    Gdτ,
    _dt,
)
    @all(εxx_eff) = @all(εxx) + 0.5 * (_dt * @all(τxx_o) / @all(G) + @all(τxx) / @all(Gdτ))
    @all(εyy_eff) = @all(εyy) + 0.5 * (_dt * @all(τyy_o) / @all(G) + @all(τyy) / @all(Gdτ))
    @all(εxy_eff) =
        @all(εxy) + 0.5 * (_dt * @all(τxy_o) / @harm(G) + @all(τxy) / @harm(Gdτ))
    return nothing
end

@parallel function viscoplastic_viscosity!(ηvp, τII, εII)
    @all(ηvp) = 0.5 * @all(τII) / @all(εII)
    return nothing
end

Δη = 1e-3
nx = 32
ny = 32
lx = 1e0
ly = 1e0
rc = 1e-1
εbg = 1e0

function solVi(; Δη=1e-3, nx=256 - 1, ny=256 - 1, lx=1e1, ly=1e1, rc=1e0, εbg=1e0)
    CharDim = GEO_units(; length=10km, viscosity=1e20Pa * s)        # Characteristic dimensions
    MatParam = (
        SetMaterialParams(;
            Name="Matrix",
            Phase=1,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            CreepLaws=LinearViscous(; η=1e20Pa * s),
            Plasticity=DruckerPrager(; C=1.6NoUnits),
            CharDim=CharDim,
        ),
        SetMaterialParams(;
            Name="Inclusion",
            Phase=2,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            CreepLaws=LinearViscous(; η=1e20Pa * s),
            Plasticity=DruckerPrager(; C=1.0NoUnits),
            CharDim=CharDim,
        ),
    )

    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    xci = Tuple([(di[i] / 2):di[i]:(li[i] - di[i] / 2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    ξ = 4.0         # Maxwell relaxation time
    G = @ones(ni...)         # elastic shear modulus

    η0 = 1e0  # matrix viscosity
    ηi = 1e0 # inclusion viscosity
    η, phase = solvi_viscosity(ni, di, li, rc, η0, ηi) # viscosity field
    G[phase .== 2] .*= 0.5
    dt = η0 / (maximum(G) * ξ)

    ηvp = @fill(0.0, ni...) # viscoplastic viscosity
    ηve = @zeros(ni...) # viscoelastic viscosity
    @parallel (1:ni[1], 1:ni[2]) viscoelastic_viscosity!(ηve, MatParam, G, phase, dt)

    ## Plasticity 
    Ftrial = @zeros(ni...)
    λ = @zeros(ni...)
    τII = @ones(ni...)
    εII = @ones(ni...)

    ## Boundary conditions
    pureshear_bc!(stokes, di, li, εbg)
    stokes.V.Vx .= [εbg * x for x in xvi[1], y in xci[2]]
    stokes.V.Vy .= [-εbg * y for x in xci[1], y in xvi[2]]
    freeslip = (freeslip_x=true, freeslip_y=true)

    ###
    # unpack
    _dt = inv(dt)
    dx, dy = di
    _dx, _dy = inv.(di)
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    εxx, εyy, εxy = JustRelax.strain(stokes)
    εxx_eff, εyy_eff, εxy_eff = deepcopy(JustRelax.strain(stokes))
    τ, τ_o = JustRelax.stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ
    _sqrt_leng_Rx = inv(√(length(Rx)))
    _sqrt_leng_Ry = inv(√(length(Ry)))
    _sqrt_leng_∇V = inv(√(length(∇V)))
    ###

    # Physical time loop
    t = 0.0
    ρ = @zeros(size(stokes.P))
    local iters

    iterMax = 10e3
    ϵ = 1e-6
    nout = 100
    # for t_it in 1:10
    # iters = solve!(stokes, pt_stokes, di, li, max_li, freeslip, ρ, η; iterMax=10e3)

    # ~preconditioner
    ητ = deepcopy(ηve)
    @parallel JustRelax.compute_maxloc!(ητ, ηve)
    # PT numerical coefficients
    @parallel JustRelax.Elasticity2D.elastic_iter_params!(
        dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
    )

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    idx = (2:(ni[1] - 1), 2:(ni[2] - 1))

    # solver loop
    iter = 1
    while err > ϵ && iter ≤ iterMax
        @parallel JustRelax.Stokes2D.compute_strain_rate!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
        @parallel JustRelax.Stokes2D.compute_P!(∇V, P, εxx, εyy, Gdτ, r)
        # @parallel JustRelax.Elasticity2D.compute_τ!(
        #     τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, εxx, εyy, εxy, η, G, dt
        # )
        # # Plasticity kernels
        # # if iter > 1
        # @parallel idx second_invariant!(τII, τxx, τyy, τxy)
        # apply_free_slip!(freeslip, τII, τII)
        # @parallel idx second_invariant!(εII, εxx, εyy, εxy)
        @parallel effective_strain_rate!(
            εxx_eff,
            εyy_eff,
            εxy_eff,
            τxx,
            τyy,
            τxy,
            τxx_o,
            τyy_o,
            τxy_o,
            εxx,
            εyy,
            εxy,
            G,
            Gdτ,
            _dt,
        )
        # for f in (τxx, τyy, τxy)
        #     apply_free_slip!(freeslip, f, f)
        # end

        @parallel idx second_invariant!(εII, εxx_eff, εyy_eff, εxy_eff)
        apply_free_slip!(freeslip, εII, εII)

        #     # @parallel idx update_plastic_strain!(λ, P, ηve, τII, phase, MatParam)
        @parallel idx stress_corrections!(τxx, τyy, τxy, τII,  εxx_eff, εyy_eff, εxy_eff, λ, ηve, Gdτ)
        for f in (τxx, τyy, τxy)
            apply_free_slip!(freeslip, f, f)
        end
        #     @parallel idx second_invariant!(τII, τxx, τyy, τxy)
        #     apply_free_slip!(freeslip, τII, τII)

        #     @parallel viscoplastic_viscosity!(ηvp, τII, εII)
        # # end

        @parallel JustRelax.Elasticity2D.compute_dV_elastic!(
            dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρ, _dx, _dy
        )
        @parallel JustRelax.Stokes2D.compute_V!(Vx, Vy, dVx, dVy)

        # @parallel JustRelax.compute_maxloc!(ητ, ηvp)
        # # PT numerical coefficients
        # @parallel JustRelax.Elasticity2D.elastic_iter_params!(
        #     dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
        # )
        # free slip boundary conditions
        # apply_free_slip!(freeslip, Vx, Vy)

        # f,ax,h=heatmap(xvi[1],xvi[2], Vx)
        # f,ax,h=heatmap(xvi[1],xvi[2], λ)
        # f,ax,h=heatmap(xvi[1],xvi[2], εII)
        # f,ax,h=heatmap(xvi[1],xvi[2], τII)
        f, ax, h = heatmap(xvi[1], xvi[2], τxx)
        f, ax, h = heatmap(xvi[1], xvi[2], τyy)
        f, ax, h = heatmap(xvi[1], xvi[2], τxy)

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            norm_Rx = norm(Rx) / (Pmax - Pmin) * lx * _sqrt_leng_Rx
            norm_Ry = norm(Ry) / (Pmax - Pmin) * lx * _sqrt_leng_Ry
            norm_∇V = norm(∇V) / (Vmax - Vmin) * lx * _sqrt_leng_∇V
            err = maximum((norm_Rx, norm_Ry, norm_∇V))
            if (err < ϵ) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx,
                    norm_Ry,
                    norm_∇V,
                )
            end
        end
        iter += 1
        @show iter
    end

    t += Δt
    # end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

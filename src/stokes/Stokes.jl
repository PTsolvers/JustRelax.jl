## UTILS

stress(stokes::StokesArrays{Viscous,A,B,C,D,nDim}) where {A,B,C,D,nDim} = stress(stokes.τ)
stress(τ::SymmetricTensor{<:AbstractMatrix{T}}) where {T} = (τ.xx, τ.yy, τ.xy)
function stress(τ::SymmetricTensor{<:AbstractArray{T,3}}) where {T}
    return (τ.xx, τ.yy, τ.zz, τ.xy, τ.xz, τ.yz)
end

@parallel function smooth!(
    A2::AbstractArray{eltype(PTArray),2}, A::AbstractArray{eltype(PTArray),2}, fact::Real
)
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

## DIMENSION AGNOSTIC KERNELS
@parallel function compute_maxloc!(A::AbstractArray, B::AbstractArray)
    @inn(A) = @maxloc(B)
    return nothing
end

## 2D KERNELS

@parallel function compute_iter_params!(
    dτ_Rho::AbstractArray{eltype(PTArray),2},
    Gdτ::AbstractArray{eltype(PTArray),2},
    Musτ::AbstractArray{eltype(PTArray),2},
    Vpdτ::Real,
    Re::Real,
    r::Real,
    max_lxy::Real,
)
    @all(dτ_Rho) = Vpdτ * max_lxy / Re / @all(Musτ)
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + 2.0)
    return nothing
end

@parallel function compute_P!(
    ∇V::AbstractArray{eltype(PTArray),2},
    P::AbstractArray{eltype(PTArray),2},
    Vx::AbstractArray{eltype(PTArray),2},
    Vy::AbstractArray{eltype(PTArray),2},
    Gdτ::AbstractArray{eltype(PTArray),2},
    r::eltype(AbstractArray),
    dx::eltype(AbstractArray),
    dy::eltype(AbstractArray),
)
    @all(∇V) = @d_xa(Vx) / dx + @d_ya(Vy) / dy
    @all(P) = @all(P) - r * @all(Gdτ) * @all(∇V)
    return nothing
end

@parallel function compute_τ!(
    τxx::AbstractArray{eltype(PTArray),2},
    τyy::AbstractArray{eltype(PTArray),2},
    τxy::AbstractArray{eltype(PTArray),2},
    Vx::AbstractArray{eltype(PTArray),2},
    Vy::AbstractArray{eltype(PTArray),2},
    Mus::AbstractArray{eltype(PTArray),2},
    Gdτ::AbstractArray{eltype(PTArray),2},
    dx::Real,
    dy::Real,
)
    @all(τxx) =
        (@all(τxx) + 2.0 * @all(Gdτ) * @d_xa(Vx) / dx) / (@all(Gdτ) / @all(Mus) + 1.0)
    @all(τyy) =
        (@all(τyy) + 2.0 * @all(Gdτ) * @d_ya(Vy) / dy) / (@all(Gdτ) / @all(Mus) + 1.0)
    @all(τxy) =
        (@all(τxy) + 2.0 * @av(Gdτ) * (0.5 * (@d_yi(Vx) / dy + @d_xi(Vy) / dx))) /
        (@av(Gdτ) / @av(Mus) + 1.0)
    return nothing
end

@parallel function compute_dV!(
    Rx::AbstractArray{eltype(PTArray),2},
    Ry::AbstractArray{eltype(PTArray),2},
    dVx::AbstractArray{eltype(PTArray),2},
    dVy::AbstractArray{eltype(PTArray),2},
    P::AbstractArray{eltype(PTArray),2},
    τxx::AbstractArray{eltype(PTArray),2},
    τyy::AbstractArray{eltype(PTArray),2},
    τxy::AbstractArray{eltype(PTArray),2},
    dτ_Rho::AbstractArray{eltype(PTArray),2},
    ρg::Nothing,
    dx::Real,
    dy::Real,
)
    @all(Rx) = @d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx
    @all(Ry) = @d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy
    @all(dVx) = @av_xi(dτ_Rho) * @all(Rx)
    @all(dVy) = @av_yi(dτ_Rho) * @all(Ry)
    return nothing
end

@parallel function compute_dV!(
    Rx::AbstractArray{eltype(PTArray),2},
    Ry::AbstractArray{eltype(PTArray),2},
    dVx::AbstractArray{eltype(PTArray),2},
    dVy::AbstractArray{eltype(PTArray),2},
    P::AbstractArray{eltype(PTArray),2},
    τxx::AbstractArray{eltype(PTArray),2},
    τyy::AbstractArray{eltype(PTArray),2},
    τxy::AbstractArray{eltype(PTArray),2},
    dτ_Rho::AbstractArray{eltype(PTArray),2},
    ρg::AbstractArray{eltype(PTArray),2},
    dx::Real,
    dy::Real,
)
    @all(Rx) = @d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx
    @all(Ry) = @d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy - @av_yi(ρg)
    @all(dVx) = @av_xi(dτ_Rho) * @all(Rx)
    @all(dVy) = @av_yi(dτ_Rho) * @all(Ry)
    return nothing
end

@parallel function compute_V!(
    Vx::AbstractArray{eltype(PTArray),2},
    Vy::AbstractArray{eltype(PTArray),2},
    dVx::AbstractArray{eltype(PTArray),2},
    dVy::AbstractArray{eltype(PTArray),2},
)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return nothing
end

## BOUNDARY CONDITIONS 

function pureshear_bc!(
    stokes::StokesArrays, di::NTuple{2,T}, li::NTuple{2,T}, εbg
) where {T}
    # unpack
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dx, dy = di
    lx, ly = li
    # Velocity pure shear boundary conditions
    stokes.V.Vx .= PTArray([
        -εbg * ((ix - 1) * dx - 0.5 * lx) for ix in 1:size(Vx, 1), iy in 1:size(Vx, 2)
    ])
    return stokes.V.Vy .= PTArray([
        εbg * ((iy - 1) * dy - 0.5 * ly) for ix in 1:size(Vy, 1), iy in 1:size(Vy, 2)
    ])
end

## VISCOUS STOKES SOLVER 

function solve!(
    stokes::StokesArrays{Viscous,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    li::NTuple{2,T},
    max_li,
    freeslip,
    ρg,
    η;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    dx, dy = di
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τxx, τyy, τxy = stress(stokes)
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ
    freeslip_x, freeslip_y = freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_li)
    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    err_rms = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    while err > ϵ && iter <= iterMax
        if (iter == 11)
            global wtime0 = Base.time()
        end
        @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)

        # free slip boundary conditions
        if (freeslip_x)
            @parallel (1:size(Vx, 1)) free_slip_y!(Vx)
        end
        if (freeslip_y)
            @parallel (1:size(Vy, 2)) free_slip_x!(Vy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx / sqrt(length(Rx)))
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx / sqrt(length(Ry)))
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx / sqrt(length(∇V)))
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_∇V[cont]])
            push!(err_evo1, maximum([norm_Rx[cont], norm_Ry[cont], norm_∇V[cont]]))
            push!(err_evo2, iter)
            if (verbose || err < ϵ || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_∇V[cont]
                )
            end
        end
    end

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

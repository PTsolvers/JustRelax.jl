## KERNELS

@parallel function compute_maxloc!(A::AbstractArray, B::AbstractArray)
    @inn(A) = @maxloc(B)
    return
end

@parallel function compute_iter_params!(dτ_Rho::AbstractArray, Gdτ::AbstractArray, Musτ::AbstractArray, Vpdτ::Real, Re::Real, r::Real, max_lxy::Real)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/@all(Musτ)
    @all(Gdτ) = Vpdτ^2/@all(dτ_Rho)/(r+2.0)
    return
end

@parallel function compute_P!(∇V::AbstractArray, P::AbstractArray, Vx::AbstractArray, Vy::AbstractArray, Gdτ::AbstractArray, r::eltype(AbstractArray), dx::eltype(AbstractArray), dy::eltype(AbstractArray))
    @all(∇V) = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(P) = @all(P) - r*@all(Gdτ)*@all(∇V)
    return
end

@parallel function compute_τ!(τxx::AbstractArray, τyy::AbstractArray, τxy::AbstractArray, Vx::AbstractArray, Vy::AbstractArray, Mus::AbstractArray, Gdτ::AbstractArray, dx::Real, dy::Real)
    @all(τxx) = (@all(τxx) + 2.0*@all(Gdτ)*@d_xa(Vx)/dx)/(@all(Gdτ)/@all(Mus) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@all(Gdτ)*@d_ya(Vy)/dy)/(@all(Gdτ)/@all(Mus) + 1.0)
    @all(τxy) = (@all(τxy) + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(@av(Gdτ)/@av(Mus) + 1.0)
    return
end

@parallel function compute_dV!(Rx::AbstractArray, Ry::AbstractArray, dVx::AbstractArray, dVy::AbstractArray, P::AbstractArray, τxx::AbstractArray, τyy::AbstractArray, τxy::AbstractArray, dτ_Rho::AbstractArray, ρg::Nothing, dx::Real, dy::Real)
    @all(Rx)   = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx
    @all(Ry)   = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy
    @all(dVx)  = @av_xi(dτ_Rho)*@all(Rx)
    @all(dVy)  = @av_yi(dτ_Rho)*@all(Ry)
    return
end

@parallel function compute_dV!(Rx::AbstractArray, Ry::AbstractArray, dVx::AbstractArray, dVy::AbstractArray, P::AbstractArray, τxx::AbstractArray, τyy::AbstractArray, τxy::AbstractArray, dτ_Rho::AbstractArray, ρg::AbstractArray, dx::Real, dy::Real)
    @all(Rx)   = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx
    @all(Ry)   = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy - @av_yi(ρg)
    @all(dVx)  = @av_xi(dτ_Rho)*@all(Rx)
    @all(dVy)  = @av_yi(dτ_Rho)*@all(Ry)
    return
end

@parallel function compute_V!(Vx::AbstractArray, Vy::AbstractArray, dVx::AbstractArray, dVy::AbstractArray)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return
end

## BOUNDARY CONDITIONS 

function pureshear_bc!(stokes::StokesArrays, di::NTuple{2, T}, li::NTuple{2, T}, εbg) where T
    # unpack
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dx, dy = di 
    lx, ly = li 
    # Velocity pure shear boundary conditions
    stokes.V.Vx .= PTArray( [-εbg*((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2)] )
    stokes.V.Vy .= PTArray( [ εbg*((iy-1)*dy -0.5*ly) for ix=1:size(Vy,1), iy=1:size(Vy,2)] )
end

## UTILS

stress(stokes::StokesArrays) = stress(stokes.τ)

stress(τ::SymmetricTensor{<:AbstractMatrix{T}}) where T = (τ.xx, τ.yy, τ.xy)

stress(τ::SymmetricTensor{<:AbstractArray{T, 3}}) where T = (τ.xx, τ.yy, τ.yy, τ.xy, τ.xy, τ.yz)

@parallel function smooth!(A2::AbstractArray, A::AbstractArray, fact::Real)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel_indices (i) function smooth_boundaries_y!(A2::AbstractArray, A::AbstractArray, fact::Real)
    A2[1, i] = A[1, i] + 1.0/4.1/fact*(
        (A[1, i+1] - 2*A[1, i] +A[1, i-1]) +
        (A[3, i] - 2*A[2, i] +A[1, i]) 
    )
    A2[end, i] = A[end, i] + 1.0/4.1/fact*(
        (A[end, i+1] - 2*A[end, i] +A[end, i-1]) +
        (A[end-2, i] - 2*A[end-1, i] +A[end, i]) 
    )
    return
end

@parallel_indices (i) function smooth_boundaries_x!(A2::AbstractArray, A::AbstractArray, fact::Real)
    A2[i, 1] = A[i, 1] + 1.0/4.1/fact*(
        (A[i+1, 1] - 2*A[i, 1] +A[i-1, 1]) +
        (A[i, 3] - 2*A[i, 2] +A[i, 1]) 
    )
    A2[i, end] = A[i, end] + 1.0/4.1/fact*(
        (A[i+1, end] - 2*A[i, end] +A[i-1, end]) +
        (A[i, end-2] - 2*A[i, end-1] +A[i, end]) 
    )
    return
end

## SOLVERS 

function solve!(stokes::StokesArrays, pt_stokes::PTStokesCoeffs, di::NTuple{2, T}, li::NTuple{2, T}, max_li, freeslip, ρg, η; iterMax = 10e3, nout = 500) where T
   
    # unpack
    dx, dy = di 
    lx, ly = li 
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τxx, τyy, τxy = stress(stokes)
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ, pt_stokes.dτ_Rho, pt_stokes.ϵ,  pt_stokes.Re,  pt_stokes.r,  pt_stokes.Vpdτ
    freeslip_x, freeslip_y = freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
   
    # PT numerical coefficients
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_li)
    
    # errors
    err=2*ϵ; iter=0; err_evo1=Float64[]; err_evo2=Float64[]; err_rms = Float64[]
    
    # solver loop
    while err > ϵ && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)

        # free slip boundary conditions
        if (freeslip_x) @parallel (1:size(Vx,1)) free_slip_y!(Vx) end
        if (freeslip_y) @parallel (1:size(Vy,2)) free_slip_x!(Vy) end

        iter += 1
        if iter % nout == 0
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
            norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
            norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
            # norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
            err = maximum([norm_Rx, norm_Ry, norm_∇V])
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_∇V)
        end
    end

    return (iter= iter, err_evo1=err_evo1, err_evo2=err_evo2)
end

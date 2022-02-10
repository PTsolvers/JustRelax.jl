@parallel function compute_maxloc!(A::PTArray, B::PTArray)
    @inn(A) = @maxloc(B)
    return
end

@parallel function compute_iter_params!(dτ_Rho::PTArray, Gdτ::PTArray, Musτ::PTArray, Vpdτ::Real, Re::Real, r::Real, max_lxy::Real)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/@all(Musτ)
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2.0)
    return
end

@parallel function compute_P!(∇V::PTArray, P::PTArray, Vx::PTArray, Vy::PTArray, Gdτ::PTArray, r::eltype(PTArray), dx::eltype(PTArray), dy::eltype(PTArray))
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(P)  = @all(P) - r*@all(Gdτ)*@all(∇V)
    return
end

@parallel function compute_τ!(τxx::PTArray, τyy::PTArray, τxy::PTArray, Vx::PTArray, Vy::PTArray, Mus::PTArray, Gdτ::PTArray, dx::Real, dy::Real)
    @all(τxx) = (@all(τxx) + 2.0*@all(Gdτ)*@d_xa(Vx)/dx)/(@all(Gdτ)/@all(Mus) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@all(Gdτ)*@d_ya(Vy)/dy)/(@all(Gdτ)/@all(Mus) + 1.0)
    @all(τxy) = (@all(τxy) + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(@av(Gdτ)/@av(Mus) + 1.0)
    return
end

@parallel function compute_dV!(Rx::PTArray, Ry::PTArray, dVx::PTArray, dVy::PTArray, P::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray, dτ_Rho::PTArray, ρg::Nothing, dx::Real, dy::Real)
    @all(Rx)   = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx
    @all(Ry)   = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy
    @all(dVx)  = @av_xi(dτ_Rho)*@all(Rx)
    @all(dVy)  = @av_yi(dτ_Rho)*@all(Ry)
    return
end

@parallel function compute_dV!(Rx::PTArray, Ry::PTArray, dVx::PTArray, dVy::PTArray, P::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray, dτ_Rho::PTArray, ρg::PTArray, dx::Real, dy::Real)
    @all(Rx)   = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx
    @all(Ry)   = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy - @av_yi(ρg)
    @all(dVx)  = @av_xi(dτ_Rho)*@all(Rx)
    @all(dVy)  = @av_yi(dτ_Rho)*@all(Ry)
    return
end

@parallel function compute_V!(Vx::PTArray, Vy::PTArray, dVx::PTArray, dVy::PTArray)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return
end

@parallel_indices (iy) function free_slip_x!(A::PTArray)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function free_slip_y!(A::PTArray)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel function smooth!(A2::PTArray, A::PTArray, fact::Real)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

stress(stokes::StokesArrays) = stress(stokes.τ)

stress(τ::SymmetricTensor{Array{T, 2}}) where T = (τ.xx, τ.yy, τ.xy)

stress(τ::SymmetricTensor{Array{T, 3}}) where T = (τ.xx, τ.yy, τ.yy, τ.xy, τ.xy, τ.yz)

function solve!(stokes::StokesArrays, pt_stokes::PTStokesCoeffs, geometry::Geometry{2}, freeslip, ρg, η; iterMax = 10e3, nout = 500)
    # unpack
    dx, dy = geometry.di 
    lx, ly = geometry.li 
    (; Vx, Vy) = stokes.V
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τxx, τyy, τxy = stress(stokes)
    (; P, ∇V) = stokes
    (; Ry, Rx) = stokes.R
    (; Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ) = pt_stokes
    (;freeslip_x, freeslip_y) =freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, geometry.max_li)
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
end

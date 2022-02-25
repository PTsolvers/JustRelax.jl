## 2D elasticity kernels

@parallel function elastic_iter_params!(dτ_Rho::PTArray, Gdτ::PTArray, Musτ::PTArray, Vpdτ::Real, G::Real, dt::Real, Re::Real, r::Real, max_lxy::Real)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/(1.0/(1.0/@all(Musτ) + 1.0/(G*dt)))
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2)
    return
end

@parallel function compute_dV_elastic!(dVx::PTArray, dVy::PTArray, P::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray, dτ_Rho::PTArray, ρg::PTArray, dx::Real, dy::Real)
    @all(dVx) = (@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx)*@av_xi(dτ_Rho)
    @all(dVy) = (@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy  - @av_yi(ρg))*@av_yi(dτ_Rho)
    return
end

@parallel function compute_Res!(Rx::PTArray, Ry::PTArray, P::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray, ρg::PTArray, dx::Real, dy::Real)
    @all(Rx) = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx
    @all(Ry) = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy - @av_yi(ρg)
    return
end

@parallel function copy_τ!(τxx_o::PTArray, τyy_o::PTArray, τxy_o::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    return
end

function copy_τ!(stokes::StokesArrays{ViscoElastic, A, B, C, D, 2}) where {A, B, C, D}
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    @parallel copy_τ!(τxx_o, τyy_o, τxy_o, τxx, τyy, τxy, )
end

macro Gr() esc(:( @all(Gdτ)/(G*dt) )) end

macro av_Gr() esc(:(  @av(Gdτ)/(G*dt) )) end

@parallel function compute_τ!(τxx::PTArray, τyy::PTArray, τxy::PTArray, τxx_o::PTArray, τyy_o::PTArray, τxy_o::PTArray, Gdτ::PTArray, Vx::PTArray, Vy::PTArray, Mus::PTArray, G::Real, dt::Real, dx::Real, dy::Real)
    @all(τxx) = (@all(τxx) + @all(τxx_o)*   @Gr() + 2.0*@all(Gdτ)*(@d_xa(Vx)/dx))/(1.0 + @all(Gdτ)/@all(Mus) + @Gr())
    @all(τyy) = (@all(τyy) + @all(τyy_o)*   @Gr() + 2.0*@all(Gdτ)*(@d_ya(Vy)/dy))/(1.0 + @all(Gdτ)/@all(Mus) + @Gr())
    @all(τxy) = (@all(τxy) + @all(τxy_o)*@av_Gr() + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(1.0 + @av(Gdτ)/@av(Mus) + @av_Gr())
    return
end

stress(stokes::StokesArrays{ViscoElastic, A, B, C, D, 2}) where {A, B, C, D, T} = stress(stokes.τ), stress(stokes.τ_o)

## VISCO-ELASTIC STOKES SOLVER 

function solve!(
    stokes::StokesArrays{ViscoElastic, A, B, C, D, 2}, 
    pt_stokes::PTStokesCoeffs, di::NTuple{2,T}, 
    li::NTuple{2,T}, 
    max_li, 
    freeslip,
    ρg, 
    η,
    G,
    dt;
    iterMax = 10e3, 
    nout = 500
) where {A, B, C, D, T}
    
    # unpack
    dx, dy = di 
    lx, ly = li 
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ, pt_stokes.dτ_Rho, pt_stokes.ϵ,  pt_stokes.Re,  pt_stokes.r,  pt_stokes.Vpdτ
    freeslip_x, freeslip_y = freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

    # errors
    err=2*ϵ; iter=0; err_evo1=Float64[]; err_evo2=Float64[]; err_rms = Float64[]
    
    # solver loop
    while iter < 2 || (err > ϵ && iter <= iterMax)
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, Vx, Vy, η, G, dt, dx, dy)
        @parallel compute_dV_elastic!(dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)

        # free slip boundary conditions
        if (freeslip_x) @parallel (1:size(Vx,1)) free_slip_y!(Vx) end
        if (freeslip_y) @parallel (1:size(Vy,2)) free_slip_x!(Vy) end

        iter += 1
        if iter % nout == 0
            @parallel compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρg, dx, dy)
            Vmin, Vmax = minimum(Vy), maximum(Vy)
            Pmin, Pmax = minimum(P), maximum(P)
            norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
            norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
            norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
            err = maximum([norm_Rx, norm_Ry, norm_∇V])
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_∇V)
        end
    end

    copy_τ!(stokes)

    return (iter= iter, err_evo1=err_evo1, err_evo2=err_evo2)
end

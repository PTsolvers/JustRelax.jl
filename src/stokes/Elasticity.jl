## UTILS

stress(stokes::StokesArrays{ViscoElastic, A, B, C, D, nDim}) where {A, B, C, D, nDim} = stress(stokes.τ), stress(stokes.τ_o)

## DIMENSION AGNOSTIC ELASTIC KERNELS

@parallel function elastic_iter_params!(dτ_Rho::PTArray, Gdτ::PTArray, Musτ::PTArray, Vpdτ::Real, G::Real, dt::Real, Re::Real, r::Real, max_lxy::Real)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/(1.0/(1.0/@all(Musτ) + 1.0/(G*dt)))
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2)
    return
end

## 2D ELASTIC KERNELS

@parallel function compute_dV_elastic!(dVx::AbstractArray{eltype(PTArray),2}, dVy::AbstractArray{eltype(PTArray),2}, P::AbstractArray{eltype(PTArray),2}, τxx::AbstractArray{eltype(PTArray),2}, τyy::AbstractArray{eltype(PTArray),2}, τxy::AbstractArray{eltype(PTArray),2}, dτ_Rho::AbstractArray{eltype(PTArray),2}, ρg::AbstractArray{eltype(PTArray),2}, dx::Real, dy::Real)
    @all(dVx) = (@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx)*@av_xi(dτ_Rho)
    @all(dVy) = (@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy  - @av_yi(ρg))*@av_yi(dτ_Rho)
    return
end

@parallel function compute_Res!(Rx::AbstractArray{eltype(PTArray),2}, Ry::AbstractArray{eltype(PTArray),2}, P::AbstractArray{eltype(PTArray),2}, τxx::AbstractArray{eltype(PTArray),2}, τyy::AbstractArray{eltype(PTArray),2}, τxy::AbstractArray{eltype(PTArray),2}, ρg::AbstractArray{eltype(PTArray),2}, dx::Real, dy::Real)
    @all(Rx) = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(P)/dx
    @all(Ry) = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(P)/dy - @av_yi(ρg)
    return
end

@parallel function copy_τ!(τxx_o::AbstractArray{eltype(PTArray),2}, τyy_o::AbstractArray{eltype(PTArray),2}, τxy_o::AbstractArray{eltype(PTArray),2}, τxx::AbstractArray{eltype(PTArray),2}, τyy::AbstractArray{eltype(PTArray),2}, τxy::AbstractArray{eltype(PTArray),2})
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

@parallel function compute_τ!(τxx::AbstractArray{eltype(PTArray),2}, τyy::AbstractArray{eltype(PTArray),2}, τxy::AbstractArray{eltype(PTArray),2}, τxx_o::AbstractArray{eltype(PTArray),2}, τyy_o::AbstractArray{eltype(PTArray),2}, τxy_o::AbstractArray{eltype(PTArray),2}, Gdτ::AbstractArray{eltype(PTArray),2}, Vx::AbstractArray{eltype(PTArray),2}, Vy::AbstractArray{eltype(PTArray),2}, Mus::AbstractArray{eltype(PTArray),2}, G::Real, dt::Real, dx::Real, dy::Real)
    @all(τxx) = (@all(τxx) + @all(τxx_o)*   @Gr() + 2.0*@all(Gdτ)*(@d_xa(Vx)/dx))/(1.0 + @all(Gdτ)/@all(Mus) + @Gr())
    @all(τyy) = (@all(τyy) + @all(τyy_o)*   @Gr() + 2.0*@all(Gdτ)*(@d_ya(Vy)/dy))/(1.0 + @all(Gdτ)/@all(Mus) + @Gr())
    @all(τxy) = (@all(τxy) + @all(τxy_o)*@av_Gr() + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(1.0 + @av(Gdτ)/@av(Mus) + @av_Gr())
    return
end

## 2D VISCO-ELASTIC STOKES SOLVER 

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
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
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

## 3D ELASTICITY KERNELS

@parallel_indices (ix,iy,iz) function copy_τ!(τxx::AbstractArray{eltype(PTArray),3}, τyy::AbstractArray{eltype(PTArray),3}, τzz::AbstractArray{eltype(PTArray),3}, τxy::AbstractArray{eltype(PTArray),3},  τxz::AbstractArray{eltype(PTArray),3},  τyz::AbstractArray{eltype(PTArray),3}, τxx_o::AbstractArray{eltype(PTArray),3}, τyy_o::AbstractArray{eltype(PTArray),3}, τzz_o::AbstractArray{eltype(PTArray),3}, τxy_o::AbstractArray{eltype(PTArray),3}, τxz_o::AbstractArray{eltype(PTArray),3}, τyz_o::AbstractArray{eltype(PTArray),3})
    if (ix≤size(τxx,1) && iy≤size(τxx,2) && iz≤size(τxx,3))  
        τxx_o[ix,iy,iz] = τxx[ix,iy,iz]  
    end
    if (ix≤size(τyy,1) && iy≤size(τyy,2) && iz≤size(τyy,3))  
        τyy_o[ix,iy,iz] = τyy[ix,iy,iz]  
    end
    if (ix≤size(τzz,1) && iy≤size(τzz,2) && iz≤size(τzz,3))  
        τzz_o[ix,iy,iz] = τzz[ix,iy,iz]  
    end
    if (ix≤size(τxy,1) && iy≤size(τxy,2) && iz≤size(τxy,3))  
        τxy_o[ix,iy,iz] = τxy[ix,iy,iz]  
    end
    if (ix≤size(τxz,1) && iy≤size(τxz,2) && iz≤size(τxz,3))  
        τxz_o[ix,iy,iz] = τxz[ix,iy,iz]  
    end
    if (ix≤size(τyz,1) && iy≤size(τyz,2) && iz≤size(τyz,3))  
        τyz_o[ix,iy,iz] = τyz[ix,iy,iz]  
    end
    return
end

macro inn_yz_Gdτ(ix,iy,iz) esc(:( Gdτ[$ix  ,$iy+1,$iz+1] )) end
macro inn_xz_Gdτ(ix,iy,iz) esc(:( Gdτ[$ix+1,$iy  ,$iz+1] )) end
macro inn_xy_Gdτ(ix,iy,iz) esc(:( Gdτ[$ix+1,$iy+1,$iz  ] )) end
macro inn_yz_Mus(ix,iy,iz) esc(:( Mus[$ix  ,$iy+1,$iz+1] )) end
macro inn_xz_Mus(ix,iy,iz) esc(:( Mus[$ix+1,$iy  ,$iz+1] )) end
macro inn_xy_Mus(ix,iy,iz) esc(:( Mus[$ix+1,$iy+1,$iz  ] )) end
macro av_xyi_Gdτ(ix,iy,iz) esc(:( (Gdτ[$ix  ,$iy  ,$iz+1] + Gdτ[$ix+1,$iy  ,$iz+1] + Gdτ[$ix  ,$iy+1,$iz+1] + Gdτ[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_xzi_Gdτ(ix,iy,iz) esc(:( (Gdτ[$ix  ,$iy+1,$iz  ] + Gdτ[$ix+1,$iy+1,$iz  ] + Gdτ[$ix  ,$iy+1,$iz+1] + Gdτ[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_yzi_Gdτ(ix,iy,iz) esc(:( (Gdτ[$ix+1,$iy  ,$iz  ] + Gdτ[$ix+1,$iy+1,$iz  ] + Gdτ[$ix+1,$iy  ,$iz+1] + Gdτ[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_xyi_Mus(ix,iy,iz) esc(:( (Mus[$ix  ,$iy  ,$iz+1] + Mus[$ix+1,$iy  ,$iz+1] + Mus[$ix  ,$iy+1,$iz+1] + Mus[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_xzi_Mus(ix,iy,iz) esc(:( (Mus[$ix  ,$iy+1,$iz  ] + Mus[$ix+1,$iy+1,$iz  ] + Mus[$ix  ,$iy+1,$iz+1] + Mus[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_yzi_Mus(ix,iy,iz) esc(:( (Mus[$ix+1,$iy  ,$iz  ] + Mus[$ix+1,$iy+1,$iz  ] + Mus[$ix+1,$iy  ,$iz+1] + Mus[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro inn_yz_Gr(ix,iy,iz) esc(:( @inn_yz_Gdτ($ix,$iy,$iz)/(G*dt) )) end
macro inn_xz_Gr(ix,iy,iz) esc(:( @inn_xz_Gdτ($ix,$iy,$iz)/(G*dt) )) end
macro inn_xy_Gr(ix,iy,iz) esc(:( @inn_xy_Gdτ($ix,$iy,$iz)/(G*dt) )) end
macro av_xyi_Gr(ix,iy,iz) esc(:( @av_xyi_Gdτ($ix,$iy,$iz)/(G*dt) )) end
macro av_xzi_Gr(ix,iy,iz) esc(:( @av_xzi_Gdτ($ix,$iy,$iz)/(G*dt) )) end
macro av_yzi_Gr(ix,iy,iz) esc(:( @av_yzi_Gdτ($ix,$iy,$iz)/(G*dt) )) end

@parallel_indices (ix,iy,iz) function compute_P_τ!(P::AbstractArray{eltype(PTArray),3}, τxx::AbstractArray{eltype(PTArray),3}, τyy::AbstractArray{eltype(PTArray),3}, τzz::AbstractArray{eltype(PTArray),3}, τxy::AbstractArray{eltype(PTArray),3}, τxz::AbstractArray{eltype(PTArray),3}, τyz::AbstractArray{eltype(PTArray),3}, τxx_o::AbstractArray{eltype(PTArray),3}, τyy_o::AbstractArray{eltype(PTArray),3}, τzz_o::AbstractArray{eltype(PTArray),3}, τxy_o::AbstractArray{eltype(PTArray),3}, τxz_o::AbstractArray{eltype(PTArray),3}, τyz_o::AbstractArray{eltype(PTArray),3}, Vx::AbstractArray{eltype(PTArray),3}, Vy::AbstractArray{eltype(PTArray),3}, Vz::AbstractArray{eltype(PTArray),3}, Mus::AbstractArray{eltype(PTArray),3}, Gdτ::AbstractArray{eltype(PTArray),3}, r::Real, G::Real, dt::Real, _dx::Real, _dy::Real, _dz::Real)
    if (ix≤size(P,1)  && iy≤size(P,2)  && iz≤size(P,3))   
         P[ix,iy,iz] = P[ix,iy,iz] - r*Gdτ[ix,iy,iz]*(_dx*(Vx[ix+1,iy,iz] - Vx[ix,iy,iz]) + _dy*(Vy[ix,iy+1,iz] - Vy[ix,iy,iz]) + _dz*(Vz[ix,iy,iz+1] - Vz[ix,iy,iz]))  
    end
    if (ix≤size(τxx,1) && iy≤size(τxx,2) && iz≤size(τxx,3))  
        τxx[ix,iy,iz] = 
            ( τxx[ix,iy,iz] + τxx_o[ix,iy,iz]*@inn_yz_Gr(ix,iy,iz) + 2.0*@inn_yz_Gdτ(ix,iy,iz)*(_dx*(Vx[ix+1,iy+1,iz+1] - Vx[ix,iy+1,iz+1]))) /
            (1.0 + @inn_yz_Gdτ(ix,iy,iz)/@inn_yz_Mus(ix,iy,iz) + @inn_yz_Gr(ix,iy,iz))  
    end
    if (ix≤size(τyy,1) && iy≤size(τyy,2) && iz≤size(τyy,3))  
        τyy[ix,iy,iz] = 
            ( τyy[ix,iy,iz] + τyy_o[ix,iy,iz]*@inn_xz_Gr(ix,iy,iz) + 2.0*@inn_xz_Gdτ(ix,iy,iz)*(_dy*(Vy[ix+1,iy+1,iz+1] - Vy[ix+1,iy,iz+1])))/(1.0 + @inn_xz_Gdτ(ix,iy,iz) / 
            @inn_xz_Mus(ix,iy,iz) + @inn_xz_Gr(ix,iy,iz))  
    end
    if (ix≤size(τzz,1) && iy≤size(τzz,2) && iz≤size(τzz,3))  
        τzz[ix,iy,iz] = 
            ( τzz[ix,iy,iz] + τzz_o[ix,iy,iz]*@inn_xy_Gr(ix,iy,iz) + 2.0*@inn_xy_Gdτ(ix,iy,iz)*(_dz*(Vz[ix+1,iy+1,iz+1] - Vz[ix+1,iy+1,iz])))/(1.0 + @inn_xy_Gdτ(ix,iy,iz) / 
            @inn_xy_Mus(ix,iy,iz) + @inn_xy_Gr(ix,iy,iz))  
    end
    if (ix≤size(τxy,1) && iy≤size(τxy,2) && iz≤size(τxy,3))  
        τxy[ix,iy,iz] = 
            ( τxy[ix,iy,iz] + τxy_o[ix,iy,iz]*@av_xyi_Gr(ix,iy,iz) + 2.0*@av_xyi_Gdτ(ix,iy,iz)*(0.5*(_dy*(Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy,iz+1]) + _dx*(Vy[ix+1,iy+1,iz+1] - Vy[ix,iy+1,iz+1])))) /
            (1.0 + @av_xyi_Gdτ(ix,iy,iz)/@av_xyi_Mus(ix,iy,iz) + @av_xyi_Gr(ix,iy,iz))  
    end
    if (ix≤size(τxz,1) && iy≤size(τxz,2) && iz≤size(τxz,3))  
        τxz[ix,iy,iz] = 
            ( τxz[ix,iy,iz] + τxz_o[ix,iy,iz]*@av_xzi_Gr(ix,iy,iz) + 2.0*@av_xzi_Gdτ(ix,iy,iz)*(0.5*(_dz*(Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy+1,iz]) + _dx*(Vz[ix+1,iy+1,iz+1] - Vz[ix,iy+1,iz+1])))) /
            (1.0 + @av_xzi_Gdτ(ix,iy,iz)/@av_xzi_Mus(ix,iy,iz) + @av_xzi_Gr(ix,iy,iz))  
    end
    if (ix≤size(τyz,1) && iy≤size(τyz,2) && iz≤size(τyz,3))  
        τyz[ix,iy,iz] = 
            ( τyz[ix,iy,iz] + τyz_o[ix,iy,iz]*@av_yzi_Gr(ix,iy,iz) + 2.0*@av_yzi_Gdτ(ix,iy,iz)*(0.5*(_dz*(Vy[ix+1,iy+1,iz+1] - Vy[ix+1,iy+1,iz]) + _dy*(Vz[ix+1,iy+1,iz+1] - Vz[ix+1,iy,iz+1])))) /
            (1.0 + @av_yzi_Gdτ(ix,iy,iz)/@av_yzi_Mus(ix,iy,iz) + @av_yzi_Gr(ix,iy,iz))  
    end
    return
end

macro av_xi_dτ_Rho(ix,iy,iz)  esc(:( (dτ_Rho[$ix  ,$iy+1,$iz+1] + dτ_Rho[$ix+1,$iy+1,$iz+1])*0.5 )) end
macro av_yi_dτ_Rho(ix,iy,iz)  esc(:( (dτ_Rho[$ix+1,$iy  ,$iz+1] + dτ_Rho[$ix+1,$iy+1,$iz+1])*0.5 )) end
macro av_zi_dτ_Rho(ix,iy,iz)  esc(:( (dτ_Rho[$ix+1,$iy+1,$iz  ] + dτ_Rho[$ix+1,$iy+1,$iz+1])*0.5 )) end

@parallel_indices (ix,iy,iz) function compute_V!(Vx::AbstractArray{eltype(PTArray),3}, Vy::AbstractArray{eltype(PTArray),3}, Vz::AbstractArray{eltype(PTArray),3}, P::AbstractArray{eltype(PTArray),3}, fx::AbstractArray{eltype(PTArray),3}, fy::AbstractArray{eltype(PTArray),3}, fz::AbstractArray{eltype(PTArray),3}, τxx::AbstractArray{eltype(PTArray),3}, τyy::AbstractArray{eltype(PTArray),3}, τzz::AbstractArray{eltype(PTArray),3}, τxy::AbstractArray{eltype(PTArray),3}, τxz::AbstractArray{eltype(PTArray),3}, τyz::AbstractArray{eltype(PTArray),3}, dτ_Rho::AbstractArray{eltype(PTArray),3}, _dx::Real, _dy::Real, _dz::Real, nx_1, nx_2, ny_1, ny_2, nz_1, nz_2)
    if (ix≤nx_1 && iy≤ny_2 && iz≤nz_2)
        Vx[ix+1,iy+1,iz+1] = Vx[ix+1,iy+1,iz+1] + (_dx*(τxx[ix+1,iy,iz] - τxx[ix,iy,iz]) + _dy*(τxy[ix,iy+1,iz] - τxy[ix,iy,iz]) + _dz*(τxz[ix,iy,iz+1] - τxz[ix,iy,iz]) - _dx*(P[ix+1,iy+1,iz+1] - P[ix,iy+1,iz+1]) + fx[ix,iy+1,iz+1])*@av_xi_dτ_Rho(ix,iy,iz)  
    end
    if (ix≤nx_2 && iy≤ny_1 && iz≤nz_2)
        Vy[ix+1,iy+1,iz+1] = Vy[ix+1,iy+1,iz+1] + (_dy*(τyy[ix,iy+1,iz] - τyy[ix,iy,iz]) + _dx*(τxy[ix+1,iy,iz] - τxy[ix,iy,iz]) + _dz*(τyz[ix,iy,iz+1] - τyz[ix,iy,iz]) - _dy*(P[ix+1,iy+1,iz+1] - P[ix+1,iy,iz+1]) + fy[ix+1,iy,iz+1])*@av_yi_dτ_Rho(ix,iy,iz)  
    end
    if (ix≤nx_2 && iy≤ny_2 && iz≤nz_1)
        Vz[ix+1,iy+1,iz+1] = Vz[ix+1,iy+1,iz+1] + (_dz*(τzz[ix,iy,iz+1] - τzz[ix,iy,iz]) + _dx*(τxz[ix+1,iy,iz] - τxz[ix,iy,iz]) + _dy*(τyz[ix,iy+1,iz] - τyz[ix,iy,iz]) - _dz*(P[ix+1,iy+1,iz+1] - P[ix+1,iy+1,iz]) + fz[ix+1,iy,iz+1])*@av_zi_dτ_Rho(ix,iy,iz)  
    end
    return
end

@parallel_indices (ix,iy,iz) function compute_Res!(∇V::AbstractArray{eltype(PTArray),3}, Rx::AbstractArray{eltype(PTArray),3}, Ry::AbstractArray{eltype(PTArray),3}, Rz::AbstractArray{eltype(PTArray),3}, Vx::AbstractArray{eltype(PTArray),3}, Vy::AbstractArray{eltype(PTArray),3}, Vz::AbstractArray{eltype(PTArray),3}, P::AbstractArray{eltype(PTArray),3}, τxx::AbstractArray{eltype(PTArray),3}, τyy::AbstractArray{eltype(PTArray),3}, τzz::AbstractArray{eltype(PTArray),3}, τxy::AbstractArray{eltype(PTArray),3}, τxz::AbstractArray{eltype(PTArray),3}, τyz::AbstractArray{eltype(PTArray),3}, _dx::Real, _dy::Real, _dz::Real)
    if (ix≤size(∇V,1) && iy≤size(∇V,2) && iz≤size(∇V,3))  
        ∇V[ix,iy,iz] = _dx*(Vx[ix+1,iy,iz] - Vx[ix,iy,iz]) + _dy*(Vy[ix,iy+1,iz] - Vy[ix,iy,iz]) + _dz*(Vz[ix,iy,iz+1] - Vz[ix,iy,iz]) 
    end
    if (ix≤size(Rx,1) && iy≤size(Rx,2) && iz≤size(Rx,3))  
        Rx[ix,iy,iz] = _dx*(τxx[ix+1,iy,iz] - τxx[ix,iy,iz]) + _dy*(τxy[ix,iy+1,iz] - τxy[ix,iy,iz]) + _dz*(τxz[ix,iy,iz+1] - τxz[ix,iy,iz]) - _dx*(P[ix+1,iy+1,iz+1] - P[ix,iy+1,iz+1]) 
    end
    if (ix≤size(Ry,1) && iy≤size(Ry,2) && iz≤size(Ry,3))  
        Ry[ix,iy,iz] = _dy*(τyy[ix,iy+1,iz] - τyy[ix,iy,iz]) + _dx*(τxy[ix+1,iy,iz] - τxy[ix,iy,iz]) + _dz*(τyz[ix,iy,iz+1] - τyz[ix,iy,iz]) - _dy*(P[ix+1,iy+1,iz+1] - P[ix+1,iy,iz+1]) 
    end
    if (ix≤size(Rz,1) && iy≤size(Rz,2) && iz≤size(Rz,3))  
        Rz[ix,iy,iz] = _dz*(τzz[ix,iy,iz+1] - τzz[ix,iy,iz]) + _dx*(τxz[ix+1,iy,iz] - τxz[ix,iy,iz]) + _dy*(τyz[ix,iy+1,iz] - τyz[ix,iy,iz]) - _dz*(P[ix+1,iy+1,iz+1] - P[ix+1,iy+1,iz]) 
    end
    return
end

## 3D VISCO-ELASTIC STOKES SOLVER 

function solve!(
    stokes::StokesArrays{ViscoElastic, A, B, C, D, 3}, 
    pt_stokes::PTStokesCoeffs, 
    ni::NTuple{3,T}, 
    di::NTuple{3,T}, 
    li::NTuple{3,T}, 
    max_li, 
    freeslip,
    ρg, 
    η,
    G,
    dt;
    iterMax = 10e3, 
    nout = 500
) where {A, B, C, D, T}
    
    ## UNPACK
    # geometry
    dx, dy, dz = di 
    lx, ly, lz = li 
    nx, ny, nz = ni
    nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx-1, nx-2, ny-1, ny-2, nz-1, nz-2
    # phsysics
    fx, fy, fz = ρg # gravitational forces
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy,  stokes.V.Vz # velocity
    dVx, dVy, dVy = stokes.dV.Vx, stokes.dV.Vy,  stokes.dV.Vy # velocity derivatives
    P, ∇V = stokes.P, stokes.∇V  # pressure and velociity divergence
    τ, τ_o = stress(stokes) # stress 
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    # solver related
    Rx, Ry, Rz = stokes.R.Rx, stokes.R.Ry, stokes.R.Rz
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ, pt_stokes.dτ_Rho, pt_stokes.ϵ,  pt_stokes.Re,  pt_stokes.r,  pt_stokes.Vpdτ
   
    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

    # errors
    err=2*ϵ; iter=0; err_evo1=Float64[]; err_evo2=Float64[]; err_rms = Float64[]
    
    # solver loop
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P_τ!(P, τxx, τyy, τzz, τxy, τxz, τyz, τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, Vx, Vy, Vz, η, Gdτ, r, G, dt, 1/dx, 1/dy, 1/dz)
        @parallel compute_V!(Vx, Vy, Vz, P,  fx, fy, fz,  τxx, τyy, τzz, τxy, τxz, τyz, dτ_Rho, 1/dx, 1/dy, 1/dz, nx_1, nx_2, ny_1, ny_2, nz_1, nz_2)
        apply_free_slip!(freeslip, Vx, Vy, Vz)
                
        iter += 1
        if iter % nout == 0
            @parallel compute_Res!(∇V, Rx, Ry, Rz, Vx, Vy, Vz, P, τxx, τyy, τzz, τxy, τxz, τyz, 1/dx, 1/dy, 1/dz)
            Vmin, Vmax = minimum_g(Vx), maximum_g(Vx)
            Pmin, Pmax = minimum_g(P), maximum_g(P)
            norm_Rx    = norm_g(Rx)/(Pmax-Pmin)*lx/sqrt(len_Rx_g)
            norm_Ry    = norm_g(Ry)/(Pmax-Pmin)*lx/sqrt(len_Ry_g)
            norm_Rz    = norm_g(Rz)/(Pmax-Pmin)*lx/sqrt(len_Rz_g)
            norm_∇V    = norm_g(∇V)/(Vmax-Vmin)*lx/sqrt(len_∇V_g)
            # norm_Rx = norm_g(Rx)/len_Rx_g; norm_Ry = norm_g(Ry)/len_Ry_g; norm_Rz = norm_g(Rz)/len_Rz_g; norm_∇V = norm_g(∇V)/len_∇V_g
            err = maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])
            if isnan(err) error("NaN") end
            push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])); push!(err_evo2,iter)
            if (me==0) @printf("Step = %d, iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry, norm_Rz, norm_∇V) end
        end
    end

    copy_τ!(stokes)

    return (iter= iter, err_evo1=err_evo1, err_evo2=err_evo2)
end            
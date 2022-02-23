## 2D elasticity kernels

@parallel function elastic_iter_params!(dτ_Rho::PTArray, Gdτ::PTArray, Musτ::PTArray, Vpdτ::Real, G::Real, dt::Real, Re::Real, r::Real, max_lxy::Real)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/(1.0/(1.0/@all(Musτ) + 1.0/(G*dt)))
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2)
    return
end

@parallel function compute_dV!(dVx::PTArray, dVy::PTArray, Pt::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray, dτ_Rho::PTArray, dx::Real, dy::Real)
    @all(dVx) = (@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx)*@av_xi(dτ_Rho)
    @all(dVy) = (@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy)*@av_yi(dτ_Rho)
    return
end

@parallel function compute_Res!(Rx::PTArray, Ry::PTArray, Pt::PTArray, τxx::PTArray, τyy::PTArray, τxy::PTArray, dx::Real, dy::Real)
    @all(Rx) = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry) = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy
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
    @parallel copy_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o)
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

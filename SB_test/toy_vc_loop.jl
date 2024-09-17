using Printf
using GeoParams, Plots, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

import JustRelax.JustRelax2D as JR

const backend = CPUBackend

using ElasticArrays,Printf
# using Plots,Plots.Measures
# default(size=(800,500),framestyle=:box,label=false,grid=false,margin=3mm,lw=6,labelfontsize=11,tickfontsize=11,titlefontsize=11)

@inline amean(a,b) = 0.5*(a + b)
@inline hmean(a,b) = 2.0/(1.0/a + 1.0/b)
@inline amean4(a,b,c,d) = 0.25*(a+b+c+d)
@inline hmean4(a,b,c,d) = 4.0/(1.0/a+1.0/b+1.0/c+1.0/d)
const av  = amean
const av4 = amean4
@views amean1(A)  = 0.5.*(A[1:end-1] .+ A[2:end])
@views avx(A)     = av.(A[1:end-1,:], A[2:end,:])
@views avy(A)     = av.(A[:,1:end-1], A[:,2:end])
@views avxy(A)    = av4.(A[1:end-1,1:end-1],A[2:end,1:end-1],A[1:end-1,2:end],A[2:end,2:end])
@views ameanx(A)  = amean.(A[1:end-1,:], A[2:end,:])
@views ameany(A)  = amean.(A[:,1:end-1], A[:,2:end])
@views ameanxy(A) = amean4.(A[1:end-1,1:end-1],A[2:end,1:end-1],A[1:end-1,2:end],A[2:end,2:end])
@views hmeanx(A)  = hmean.(A[1:end-1,:], A[2:end,:])
@views hmeany(A)  = hmean.(A[:,1:end-1], A[:,2:end])
@views hmeanxy(A) = hmean4.(A[1:end-1,1:end-1],A[2:end,1:end-1],A[1:end-1,2:end],A[2:end,2:end])
@views maxloc(A)  = max.(A[1:end-2,1:end-2],A[1:end-2,2:end-1],A[1:end-2,3:end],
                         A[2:end-1,1:end-2],A[2:end-1,2:end-1],A[2:end-1,3:end],
                         A[3:end  ,1:end-2],A[3:end  ,2:end-1],A[3:end  ,3:end])
@views bc2!(A)    = begin A[[1,end],:]=A[[2,end-1],:]; A[:,[1,end]]=A[:,[2,end-1]]; end

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, radius)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius)
        x, y = xc[i], yc[j]
        if ((x)^2 + (y)^2) ≥ radius^2
            # if ((x-o_x)^2 + (y-o_y)^2) > radius^2
            JustRelax.@cell phases[1, i, j] = 1.0
            JustRelax.@cell phases[2, i, j] = 0.0

        else
            JustRelax.@cell phases[1, i, j] = 0.0
            JustRelax.@cell phases[2, i, j] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius)
    @parallel (@idx ni.+1) init_phases!(phase_ratios.vertex, xvi..., origin..., radius)
    return nothing
end

@views function update_iteration_params!((;η,ητ,η_vep))
    # ητ[2:end-1,2:end-1] .= maxloc(amean.(η,η_vep)./2)
    ητ[2:end-1,2:end-1] .= maxloc(η); bc2!(ητ)
    return
end

@views function update_stresses!((;εxx_ve,εyy_ve,εxy_ve,εII_ve,Pr,Pr_c,εxx,εyy,εxy,εxyv,dτxx,dτyy,dτxy,τxx,τyy,τxy,τxyv,τxx_old,τyy_old,τxy_old,Vx,Vy,∇V,η,ητ,G,F,λ,dQdτxx,dQdτyy,dQdτxy,τII,η_vep,dτ_r,Fchk,dPr,Pr_old),K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy)
    θ_dτ    = lτ*(r+2.0)/(re_mech*vdτ)
    dτ_r   .= 1.0./(θ_dτ .+ η./(G.*dt) .+ 1.0)
    # ∇V     .= diff(Vx[:, 2:end-1],dims=1)./dx .+ diff(Vy[2:end-1, :],dims=2)./dy
    dPr    .= .-∇V .- (Pr .- Pr_old)./K./dt
    # Pr    .+= r/θ_dτ.*η.*dPr                      # explicit
    Pr    .+= dPr./(1.0./(r/θ_dτ.*η) .+ 1.0./K./dt) # implicit
    # strain rates
    # εxx    .= diff(Vx[:, 2:end-1],dims=1)./dx .- ∇V./3.0
    # εyy    .= diff(Vy[2:end-1, :],dims=2)./dy .- ∇V./3.0
    # εxyv   .= 0.5*(diff(Vx,dims=2)./dy .+ diff(Vy,dims=1)./dx)
    εxy    .= ameanxy(εxyv)
    # visco-elastic strain rates
    εxx_ve .= εxx .+ 0.5.*τxx_old./(G.*dt)
    εyy_ve .= εyy .+ 0.5.*τyy_old./(G.*dt)
    εxy_ve .= εxy .+ 0.5.*τxy_old./(G.*dt)
    εII_ve .= sqrt.(0.5.*(εxx_ve.^2 .+ εyy_ve.^2) .+ εxy_ve.^2)
    # stress increments
    dτxx   .= (.-(τxx .- τxx_old).*η./(G.*dt) .- τxx .+ 2.0.*η.*εxx).*dτ_r
    dτyy   .= (.-(τyy .- τyy_old).*η./(G.*dt) .- τyy .+ 2.0.*η.*εyy).*dτ_r
    dτxy   .= (.-(τxy .- τxy_old).*η./(G.*dt) .- τxy .+ 2.0.*η.*εxy).*dτ_r
    τII    .= sqrt.(0.5.*((τxx.+dτxx).^2 .+ (τyy.+dτyy).^2) .+ (τxy.+dτxy).^2)
    # yield function
    F      .= τII .- τ_y .- Pr.*sinϕ
    λ      .= (1.0 .- relλ).*λ .+ relλ.*(max.(F,0.0)./(η.*dτ_r .+ η_reg .+ K.*dt.*sinϕ.*sinψ))
    dQdτxx .= 0.5.*(τxx.+dτxx)./τII
    dQdτyy .= 0.5.*(τyy.+dτyy)./τII
    dQdτxy .=      (τxy.+dτxy)./τII
    Pr_c   .= Pr .+ K.*dt.*λ.*sinψ
    τxx   .+= (.-(τxx .- τxx_old).*η./(G.*dt) .- τxx .+ 2.0.*η.*(εxx .-      λ.*dQdτxx)).*dτ_r
    τyy   .+= (.-(τyy .- τyy_old).*η./(G.*dt) .- τyy .+ 2.0.*η.*(εyy .-      λ.*dQdτyy)).*dτ_r
    τxy   .+= (.-(τxy .- τxy_old).*η./(G.*dt) .- τxy .+ 2.0.*η.*(εxy .- 0.5.*λ.*dQdτxy)).*dτ_r
    τxyv[2:end-1,2:end-1] .= ameanxy(τxy)
    τII    .= sqrt.(0.5.*(τxx.^2 .+ τyy.^2) .+ τxy.^2)
    Fchk   .= τII .- τ_y .- Pr_c.*sinϕ .- λ.*η_reg
    η_vep  .= τII ./ 2.0 ./ εII_ve
    return
end

function update_stresses_loop!(
    (;
        Pr,Pr_c,
        εxx, εyy, εxy, εxyv,
        τxx, τyy, τxy, τxyv,
        τxx_old, τyy_old, τxy_old,
        Vx, Vy, ∇V,
        η, ητ, G,
        F, λ,
        τII, η_vep, Fchk, Pr_old
    )
    ,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy
)

    for j in axes(Pr,2), i in axes(Pr,1)
        I = i,j
        θ_dτ    = lτ*(r+2.0)/(re_mech*vdτ)
        dτ_r    = 1.0/(θ_dτ + η[I...] /(G[I...]*dt) + 1.0)
        dPr     = -∇V[I...] - (Pr[I...] - Pr_old[I...])/K/dt
        Pr[I...]+= dPr/(1.0/(r/θ_dτ*η[I...]) + 1.0/K/dt) # implicit
        # strain rates
        εxy[I...] = 0.25 * (εxyv[I...] + εxyv[i+1,j] + εxyv[i,j+1] + εxyv[I.+1...])
        # visco-elastic strain rates
        εxx_ve  = εxx[I...] + 0.5*τxx_old[I...]/(G[I...]*dt)
        εyy_ve  = εyy[I...] + 0.5*τyy_old[I...]/(G[I...]*dt)
        εxy_ve  = εxy[I...] + 0.5*τxy_old[I...]/(G[I...]*dt)
        εII_ve  = √(0.5*(εxx_ve^2 + εyy_ve.^2) + εxy_ve.^2)
        # stress increments
        dτxx   = (-(τxx[I...] - τxx_old[I...]) * η[I...] / (G[I...]*dt) - τxx[I...] .+ 2.0 * η[I...] * εxx[I...]) * dτ_r
        dτyy   = (-(τyy[I...] - τyy_old[I...]) * η[I...] / (G[I...]*dt) - τyy[I...] .+ 2.0 * η[I...] * εyy[I...]) * dτ_r
        dτxy   = (-(τxy[I...] - τxy_old[I...]) * η[I...] / (G[I...]*dt) - τxy[I...] .+ 2.0 * η[I...] * εxy[I...]) * dτ_r
        τII[I...] = √(0.5*((τxx[I...] + dτxx)^2 + (τyy[I...] + dτyy)^2) + (τxy[I...] + dτxy)^2)
        # yield function
        F       = τII[I...]  - τ_y - Pr[I...] .*sinϕ
        λ[I...] = (1.0 - relλ)*λ[I...]  + relλ.*(max(F,0.0)/(η[I...] *dτ_r + η_reg + K*dt*sinϕ*sinψ))
        dQdτxx  = 0.5 *(τxx[I...] + dτxx) / τII[I...]
        dQdτyy  = 0.5 *(τyy[I...] + dτyy) / τII[I...]
        dQdτxy  =      (τxy[I...] + dτxy) / τII[I...]
        Pr_c[I...] = Pr[I...]  + K*dt*λ[I...] *sinψ
        τxx[I...] += (-(τxx[I...]  - τxx_old[I...] )*η[I...] /(G[I...] *dt) - τxx[I...]  + 2.0 * η[I...]  *(εxx[I...] -     λ[I...] *dQdτxx ))*dτ_r
        τyy[I...] += (-(τyy[I...]  - τyy_old[I...] )*η[I...] /(G[I...] *dt) - τyy[I...]  + 2.0 * η[I...]  *(εyy[I...] -     λ[I...] *dQdτyy ))*dτ_r
        τxy[I...] += (-(τxy[I...]  - τxy_old[I...] )*η[I...] /(G[I...] *dt) - τxy[I...]  + 2.0 * η[I...]  *(εxy[I...] - 0.5*λ[I...] *dQdτxy ))*dτ_r
        τII[I...] = √(0.5*((τxx[I...] )^2 + (τyy[I...] )^2) + (τxy[I...])^2)
        Fchk[I...]  = τII[I...] - τ_y - Pr_c[I...]*sinϕ - λ[I...]*η_reg
        η_vep[I...] = τII[I...] / 2.0 / εII_ve
    end
    τxyv[2:end-1,2:end-1] .= ameanxy(τxy)
    return
end

function update_stresses2!(
    (;
        Pr,Pr_c,dPr,
        εxx, εyy, εxyv,
        τxx, τyy, τxy, τxyv,
        τxx_old, τyy_old, τxy_old,
        Vx, Vy, ∇V,
        η, ητ, G,
        F, λ,
        τII, η_vep, Fchk, Pr_old
    )
    ,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy,
    rheology, phase_center
)

    θ_dτ  = lτ*(r+2.0)/(re_mech*vdτ)
    ε     = εxx, εyy, εxyv
    τ     = τxx, τyy, τxy
    τ_o   = τxx_old, τyy_old, τxy_old

    for j in axes(Pr,2), i in axes(Pr,1)
        I = i,j
        
        phase = @inbounds phase_center[I...]
        _Gdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        is_pl, τ_y, sinϕ, cosϕ, sinψ, η_reg = JR.plastic_params_phase(rheology, 0e0, phase)

        # strain rates
        εxy   = 0.25 * (εxyv[I...] + εxyv[i+1,j] + εxyv[i,j+1] + εxyv[I.+1...])
        τij   = getindex.(τ, I...)
        τij_o = getindex.(τ_o, I...)
        εij   = ε[1][I...], ε[2][I...], εxy

        ηij = η[I...]

        ###
        dτ_r        = 1.0/(θ_dτ + η[I...] * _Gdt + 1.0)
        dPr[I...]   = -∇V[I...] - (Pr[I...] - Pr_old[I...])/K/dt
        Pr[I...]   += dPr/(1.0/(r/θ_dτ*ηij) + 1.0/K/dt) # implicit
        εij_ve      = @. εij + 0.5*τij_o* _Gdt
        εII_ve      = GP.second_invariant(εij_ve)
        # stress increments
        dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        τII[I...]   = τII_ij =  GP.second_invariant(dτij .+ τij)
        # yield function
        F           = τII_ij - τ_y - Pr[I...] .*sinϕ
        if is_pl
            λ[I...]     = (1.0 - relλ)*λ[I...]  + relλ.*(max(F,0.0)/(η[I...] *dτ_r + η_reg + K*dt*sinϕ*sinψ))
            dQdτij      = @. 0.5 * (τij + dτij) / τII_ij
            dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            setindex!.(τ, dτij .+ τij, I...)

            τII[I...]   = √(0.5*((τxx[I...] )^2 + (τyy[I...] )^2) + (τxy[I...])^2)
            
            Pr_c[I...]  = Pr[I...]  + K*dt*λ[I...] *sinψ
            Fchk[I...]  = τII_ij - τ_y - Pr_c[I...]*sinϕ - λ[I...]*η_reg
            η_vep[I...] = τII_ij / 2.0 / εII_ve
        else
            setindex!.(τ, dτij .+ τij, I...)
            Fchk[I...]  = 0e0
            η_vep[I...] = η[I...]
        end
    end
    τxyv[2:end-1,2:end-1] .= ameanxy(τxy)
    return
end

function update_stresses3!(
    ε::NTuple{N, T},
    τ::NTuple{N, T},
    τ_o::NTuple{N, T},
    (;
        Pr,Pr_c,dPr,
        τxyv,
        η, ητ, G,
        F, λ,
        τII, η_vep, Fchk, Pr_old
    )
    ,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy,
    rheology, phase_center
) where {N, T}

    θ_dτ  = lτ*(r+2.0)/(re_mech*vdτ)

    for j in axes(Pr,2), i in axes(Pr,1)
        I = i,j
        
        phase = @inbounds phase_center[I...]
        _Gdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR.plastic_params_phase(rheology, 0e0, phase)

        # plastic volumetric change K * dt * sinϕ * sinψ
        K = fn_ratio(JR.get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ

        # strain rates
        τij, τij_o, εij = JR.cache_tensors(τ, τ_o, ε, i, j)
        ηij = η[I...]

        ###
        dτ_r        = 1.0/(θ_dτ + η[I...] * _Gdt + 1.0)
        εij_ve      = @. εij + 0.5*τij_o* _Gdt
        εII_ve      = GP.second_invariant(εij_ve)
        # stress increments
        dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        τII[I...]   = τII_ij =  GP.second_invariant(dτij .+ τij)
        # yield function
        F           = @. τII_ij - C  - Pr[I...] *sinϕ
        if is_pl #&& F > 0
            λ[I...]     = (1.0 - relλ)*λ[I...]  + relλ.*(max(F,0.0)/(η[I...] *dτ_r + η_reg + volume))
            dQdτij      = @. 0.5 * (τij + dτij) / τII_ij
            dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            τij         = dτij .+ τij
            setindex!.(τ, τij, I...)
            τII[I...]   = GP.second_invariant(τij)
            Pr_c[I...]  = Pr[I...] + K*dt*λ[I...] * sinψ
            Fchk[I...]  = τII_ij - τ_y - Pr_c[I...]*sinϕ - λ[I...]*η_reg
            η_vep[I...] = τII_ij / 2.0 / εII_ve
        else
            setindex!.(τ, dτij .+ τij, I...)
            Fchk[I...]  = 0e0
            η_vep[I...] = ηij
        end
        
        Pr_c[I...]  = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

    end
    
    τxyv[2:end-1,2:end-1] .= ameanxy(τ[3])

    return
end


function update_stresses4!(
    ε::NTuple{N, T},
    τ::NTuple{N, T},
    τ_o::NTuple{N, T},
    Pr,
    Pr_c,
    η,
    λ,
    τII,
    η_vep, 
    Fchk, 
    relλ,dt,re_mech,vdτ,lτ,r,
    rheology, phase_center
) where {N, T}

    θ_dτ  = lτ*(r+2.0)/(re_mech*vdτ)

    for j in axes(Pr,2), i in axes(Pr,1)
        I = i,j
        
        phase = @inbounds phase_center[I...]
        _Gdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR.plastic_params_phase(rheology, 0e0, phase)

        # plastic volumetric change K * dt * sinϕ * sinψ
        K = fn_ratio(JR.get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ

        # strain rates
        τij, τij_o, εij = JR.cache_tensors(τ, τ_o, ε, i, j)
        ηij = η[I...]

        ###
        dτ_r        = 1.0/(θ_dτ + η[I...] * _Gdt + 1.0)
        εij_ve      = @. εij + 0.5*τij_o* _Gdt
        εII_ve      = GP.second_invariant(εij_ve)
        # stress increments
        dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        τII[I...]   = τII_ij =  GP.second_invariant(dτij .+ τij)
        # yield function
        F           = @. τII_ij - C  - Pr[I...] *sinϕ
        if is_pl #&& F > 0
            λ[I...]     = (1.0 - relλ)*λ[I...]  + relλ.*(max(F,0.0)/(η[I...] *dτ_r + η_reg + volume))
            dQdτij      = @. 0.5 * (τij + dτij) / τII_ij
            dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            τij         = dτij .+ τij
            setindex!.(τ, τij, I...)
            τII[I...]   = GP.second_invariant(τij)
            Pr_c[I...]  = Pr[I...] + K*dt*λ[I...] * sinψ
            Fchk[I...]  = τII_ij - τ_y - Pr_c[I...]*sinϕ - λ[I...]*η_reg
            η_vep[I...] = τII_ij / 2.0 / εII_ve
        else
            setindex!.(τ, dτij .+ τij, I...)
            Fchk[I...]  = 0e0
            η_vep[I...] = ηij
        end
        
        Pr_c[I...]  = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

    end
    
    return
end


@views function update_old!(
    (;
        τxx_old,τyy_old,τxy_old,τxx,τyy,τxy,
        τxxv_old,τyyv_old,τxyv_old,τxxv,τyyv,τxyv,
        Pr_c, Pr, Pr_old,
        λ , λv
    )
)
    τxx_old .= τxx
    τyy_old .= τyy
    τxy_old .= τxy
    τxxv_old .= τxxv
    τyyv_old .= τyyv
    τxyv_old .= τxyv
    Pr      .= Pr_c
    Pr_old  .= Pr
    λ       .= 0.0
    λv      .= 0.0
    return
end

@views function c2v!(C, V)
    C[2:end-1, 2:end-1] .= average(V)  
    boundaries!(C)
    nothing
end

@views function boundaries!(A)
    A[1,:] .= A[2,:]
    A[end,:] .= A[end-1,:]
    A[:, 1] .= A[:, 2]
    A[:, end] .= A[:, end-1]
    nothing
end

@views  average(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])

@views function update_stresses_vc!(
    (;
        Pr,Pv,
        εxx,εyy,
        εxxv,εyyv,εxyv,
        τxxv,τyyv,τxyv,
        τxxv_old,τyyv_old,τxyv_old,
        ηv,Gv,λv,Fchk
    ), 
    K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy
)
    θ_dτ    = lτ*(r+2.0)/(re_mech*vdτ)
    dτ_r    = 1.0./(θ_dτ .+ ηv./(Gv.*dt) .+ 1.0)
    c2v!(Pv ,   Pr)
    c2v!(εxxv,   εxx)
    c2v!(εyyv,   εyy)
  
    # stress increments
    dτxx    = (.-(τxxv .- τxxv_old).*ηv./(Gv.*dt) .- τxxv .+ 2.0.*ηv.*εxxv).*dτ_r
    dτyy    = (.-(τyyv .- τyyv_old).*ηv./(Gv.*dt) .- τyyv .+ 2.0.*ηv.*εyyv).*dτ_r
    dτxy    = (.-(τxyv .- τxyv_old).*ηv./(Gv.*dt) .- τxyv .+ 2.0.*ηv.*εxyv).*dτ_r
    τII     = sqrt.(0.5.*((τxxv.+dτxx).^2 .+ (τyyv.+dτyy).^2) .+ (τxyv.+dτxy).^2)
    # yield function
    F       = τII .- τ_y .- Pv.*sinϕ
    λv     .= (1.0 .- relλ).*λv .+ relλ.*(max.(F,0.0)./(ηv.*dτ_r .+ η_reg .+ K.*dt.*sinϕ.*sinψ))
    dQdτxy  =      (τxyv.+dτxy)./τII
    τxyv  .+= (-(τxyv .- τxyv_old).*ηv./(Gv.*dt) .- τxyv .+ 2.0.*ηv.*(εxyv .- 0.5.*λv.*dQdτxy)).*dτ_r
    return
end

function update_stresses_vc_loop!(
    (;
        Pr,Pv,
        εxx, εyy, εxyv,
        τxxv,τyyv,τxyv,
        τxxv_old,τyyv_old,τxyv_old,
        ηv,λv
    ), 
    K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy, phase_vertex, rheology
)
    θ_dτ    = lτ*(r+2.0)/(re_mech*vdτ)
    c2v!(Pv ,   Pr)
    c2v!(εxxv,   εxx)
    c2v!(εyyv,   εyy)

    θ_dτ  = lτ*(r+2.0)/(re_mech*vdτ)

    for j in axes(εyyv,2), i in axes(εyyv,1)
        I     = i,j
        
        phase = @inbounds phase_vertex[I...]
        _Gdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR.plastic_params_phase(rheology, 0e0, phase)

        # plastic volumetric change K * dt * sinϕ * sinψ
        K = fn_ratio(JR.get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ

        dτ_r    = 1.0/(θ_dτ + ηv[I...] * _Gdt + 1.0)
        
        # stress increments
        dτxx    = (-(τxxv[I...] - τxxv_old[I...]) * ηv[I...] * _Gdt - τxxv[I...] + 2.0 * ηv[I...] * εxxv[I...]).*dτ_r
        dτyy    = (-(τyyv[I...] - τyyv_old[I...]) * ηv[I...] * _Gdt - τyyv[I...] + 2.0 * ηv[I...] * εyyv[I...]).*dτ_r
        dτxy    = (-(τxyv[I...] - τxyv_old[I...]) * ηv[I...] * _Gdt - τxyv[I...] + 2.0 * ηv[I...] * εxyv[I...]).*dτ_r
        τII     = √(0.5*((τxxv[I...]  + dτxx)^2 + (τyyv[I...] + dτyy)^2) + (τxyv[I...] + dτxy)^2)
        
        # yield function
        F       = τII - τ_y - Pv[I...] * sinϕ
        λv[I...]= (1.0 - relλ) * λv[I...] + relλ*(max(F, 0.0) / (ηv[I...] * dτ_r + η_reg + volume))
        dQdτxy  = 0.5 * (τxyv[I...] + dτxy) / τII
        τxyv[I...] += (-(τxyv[I...] - τxyv_old[I...] ) * ηv[I...] * _Gdt - τxyv[I...]  + 2.0 * ηv[I...]  *(εxyv[I...] - 0.5 * λv[I...] * dQdτxy)) * dτ_r
    end
    return
end

function clamped_indices(ni, i, j)
    nx, ny = ni
    i0 = clamp(i-1, 1, nx)
    ic = clamp(i, 1, nx)
    j0 = clamp(j-1, 1, ny)
    jc = clamp(j, 1, ny)
    return i0, j0, ic, jc
end

av_clamped_indices(A, i0, j0, ic, jc) = 0.25 * (A[i0, j0] + A[ic, jc] + A[i0, jc] + A[ic, j0])

function update_stresses_vc_loop2!(
    (;
        Pr,Pv,
        εxx, εyy, εxyv,
        τxx, τyy, τxyv,
        τxx_old, τyy_old, τxyv_old,
        η, λv
    ), 
    K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy, phase_vertex, rheology
)

    θ_dτ  = lτ*(r+2.0)/(re_mech*vdτ)
    ni    = size(Pr)

    for j in axes(εyyv,2), i in axes(εyyv,1)
        I       = i,j
        Ic      = clamped_indices(ni, i, j)

        # interpolate to ith vertex
        Pv_ij       = av_clamped_indices(Pr, Ic...)
        εxxv_ij     = av_clamped_indices(εxx, Ic...)
        εyyv_ij     = av_clamped_indices(εyy, Ic...)
        τxxv_ij     = av_clamped_indices(τxx, Ic...)
        τyyv_ij     = av_clamped_indices(τyy, Ic...)
        τxxv_old_ij = av_clamped_indices(τxx_old, Ic...)
        τyyv_old_ij = av_clamped_indices(τyy_old, Ic...)
        ηv_ij       = av_clamped_indices(η, Ic...)

        phase = @inbounds phase_vertex[I...]
        _Gdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR.plastic_params_phase(rheology, 0e0, phase)

        # plastic volumetric change K * dt * sinϕ * sinψ
        K = fn_ratio(JR.get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ

        dτ_r    = 1.0/(θ_dτ + ηv_ij * _Gdt + 1.0)
        
        # stress increments
        dτxx    = (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij).*dτ_r
        dτyy    = (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij).*dτ_r
        dτxy    = (-(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gdt - τxyv[I...] + 2.0 * ηv_ij * εxyv[I...]).*dτ_r
        τII     = √(0.5*((τxxv_ij  + dτxx)^2 + (τyyv_ij + dτyy)^2) + (τxyv[I...] + dτxy)^2)
        
        # yield function
        if is_pl
            F       = τII - τ_y - Pv_ij[I...] * sinϕ
            λv[I...]= (1.0 - relλ) * λv[I...] + relλ*(max(F, 0.0) / (ηv_ij * dτ_r + η_reg + volume))
            dQdτxy  = 0.5 * (τxyv[I...] + dτxy) / τII
            τxyv[I...] += (-(τxyv[I...] - τxyv_old[I...] ) * ηv_ij * _Gdt - τxyv[I...]  + 2.0 * ηv_ij  *(εxyv[I...] - 0.5 * λv[I...] * dQdτxy)) * dτ_r
        else
            τxyv[I...] += dτxy
        end

    end
    return
end

@views function update_velocities!((;Vx,Vy,Pr_c,τxx,τyy,τxyv,ητ),vdτ,lτ,re_mech,dx,dy)
    nudτ = vdτ*lτ/re_mech
    Vx[2:end-1, 2:end-1] .+= (diff(.-Pr_c.+τxx,dims=1)./dx .+ diff(τxyv[2:end-1,:],dims=2)./dy).*nudτ./avx(ητ)
    Vy[2:end-1, 2:end-1] .+= (diff(.-Pr_c.+τyy,dims=2)./dy .+ diff(τxyv[:,2:end-1],dims=1)./dx).*nudτ./avy(ητ)
    return
end

@views function update_velocities2!((;Vx,Vy,Pr_c,τxx,τyy,τxyv,ητ),vdτ,lτ,re_mech,dx,dy)
    nudτ = vdτ*lτ/re_mech
    nx = max(size.((Vx,Vy),1)...)
    ny = max(size.((Vx,Vy),2)...)

    for j in 1:ny, i in 1:nx
        if all( (1,1) .< (i,j) .< size(Vx)  )
            Vx[i, j] += ((-(Pr_c[i,j-1] - Pr_c[i-1,j-1]) + (τxx[i,j-1] - τxx[i-1,j-1]))/dx +
                (τxyv[i,j] - τxyv[i,j-1])/dy) * nudτ / ((ητ[i,j-1] + ητ[i-1,j-1]) * 0.5)
        end

        if all( (1,1) .< (i,j) .< size(Vy)  )
            Vy[i, j] += ((-(Pr_c[i-1,j] - Pr_c[i-1,j-1]) + (τyy[i-1,j] - τyy[i-1,j-1]))/dy +
                (τxyv[i,j] - τxyv[i-1,j])/dx) * nudτ / ((ητ[i-1,j] + ητ[i-1,j-1]) * 0.5)
        end
    end

    return
end

@views function compute_residuals!((;r_Vx,r_Vy,Pr_c,τxx,τyy,τxyv),dx,dy)
    r_Vx .= diff(.-Pr_c[:,2:end-1].+τxx[:,2:end-1],dims=1)./dx .+ diff(τxyv[2:end-1,2:end-1],dims=2)./dy
    r_Vy .= diff(.-Pr_c[2:end-1,:].+τyy[2:end-1,:],dims=2)./dy .+ diff(τxyv[2:end-1,2:end-1],dims=1)./dx
    return
end

function main()

    # MAIN SCRIPT --------------------------------------------------------------------
    n      = 63
    nx     = n
    ny     = n
    figdir = @__DIR__

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = @. -li / 2     # origin coordinates
    # origin       = 0e0, 0e0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt           = Inf

    # Physical properties using GeoParams ----------------
    τ_y     = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 30            # friction angle
    C       = τ_y           # Cohesion
    η0      = 1.0           # viscosity
    G0      = 1.0           # elastic shear modulus
    Gi      = G0/(6.0-4.0)  # elastic shear modulus perturbation
    εbg     = 1.0           # background strain-rate
    η_reg   = 8e-3          # regularisation "viscosity"
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    el_bg   = ConstantElasticity(; G=G0, Kb=4)
    el_inc  = ConstantElasticity(; G=Gi, Kb=4)
    visc    = LinearViscous(; η=η0)
    pl      = DruckerPrager_regularised(;  # non-regularized plasticity
        C    = C,
        ϕ    = ϕ,
        η_vp = η_reg,
        Ψ    = 5
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((visc, )),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity        = el_bg,
        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((LinearViscous(; η=0.5),)),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity        = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    radius       = 0.1
    phase_ratios = PhaseRatio(backend, ni, length(rheology))
    init_phases!(phase_ratios, xci, xvi, radius)

    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6, Re=3π, CFL = 0.9 / √2.1)
    # Buoyancy forces
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)
    # Rheology
    stokes.viscosity.η .= 1
    # Boundary conditions
    flow_bcs     = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    ρg = @zeros(ni...), @zeros(ni...)

    # physics
    lx,ly      = 1.0,1.0
    radi       = 0.01*lx
    τ_y        = 1.6
    sinϕ       = sind(30)
    sinψ       = sind(5)
    η0         = 1.0
    G0         = 1.0
    Gi         = G0/2
    K          = 4*G0
    ξ          = 4.0
    εbg        = 1.0
    dt         = η0/G0/ξ/1.5
    # numerics
    # nx,ny      = 63,63
    nt         = 15
    η_reg      = 8.0e-3
    ϵtol       = (1e-6,1e-6,1e-6)
    maxiter    = 100max(nx,ny)
    ncheck     = ceil(Int,5max(nx,ny))
    r          = 0.7
    re_mech    = 3π
    relλ       = 0.2
    # preprocessing
    dx,dy      = lx/nx,ly/ny
    xv,yv      = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1)
    xc,yc      = amean1(xv),amean1(yv)
    lτ         = min(lx,ly)
    vdτ        = 0.99*min(dx,dy)/sqrt(2.1)

    stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    θ = @zeros(ni...)
    # array allocation
    fields = (
        Vx         = stokes.V.Vx,
        Vy         = stokes.V.Vy,
        Pr         = stokes.P,
        Pr_c       = zeros(nx  ,ny  ),
        Pr_old     = stokes.P0,
        ∇V         = stokes.∇V,
        τxx        = stokes.τ.xx,
        τyy        = stokes.τ.yy,
        τxy        = stokes.τ.xy_c,
        τxyv       = stokes.τ.xy,
        τxx_old    = stokes.τ_o.xx,
        τyy_old    = stokes.τ_o.yy,
        τxy_old    = stokes.τ_o.xy_c,
        τII        = stokes.τ.II,
        Vmag       = zeros(nx  ,ny  ),
        dPr        = zeros(nx  ,ny  ),
        r_Vx       = zeros(nx-1,ny-2),
        r_Vy       = zeros(nx-2,ny-1),
        ητ         = zeros(nx  ,ny  ),
        dτ_r       = zeros(nx  ,ny  ),
        F          = zeros(nx  ,ny  ),
        λ          = zeros(nx  ,ny  ),
        dQdτxx     = zeros(nx  ,ny  ),
        dQdτyy     = zeros(nx  ,ny  ),
        dQdτxy     = zeros(nx  ,ny  ),
        Fchk       = zeros(nx  ,ny  ),
        εxx        = stokes.ε.xx,
        εyy        = stokes.ε.yy,
        εxy        = stokes.ε.xy_c,
        εxx_ve     = zeros(nx  ,ny  ),
        εyy_ve     = zeros(nx  ,ny  ),
        εxy_ve     = zeros(nx  ,ny  ),
        εII_ve     = zeros(nx  ,ny  ),
        εxyv       = stokes.ε.xy,
        dτxx       = zeros(nx  ,ny  ),
        dτyy       = zeros(nx  ,ny  ),
        dτxy       = zeros(nx  ,ny  ),
        η          = η0.*ones(nx  ,ny  ),
        G          = G0.*ones(nx  ,ny  ),
        η_vep      = η0.*ones(nx  ,ny  ),

        τyyv       = zeros(nx+1,ny+1),
        τxxv       = zeros(nx+1,ny+1),
        τxxv_old   = zeros(nx+1,ny+1),
        τyyv_old   = zeros(nx+1,ny+1),
        τxyv_old   = zeros(nx+1,ny+1),
        εxxv       = zeros(nx+1,ny+1),
        εyyv       = zeros(nx+1,ny+1),
        Pv         = zeros(nx+1,ny+1),
      
        ηv          = η0.*ones(nx+1,ny+1),
        Gv          = G0.*ones(nx+1,ny+1),
        ηv_vep      = η0.*ones(nx+1,ny+1),
        λv         = zeros(nx+1,ny+1),
    )
    # initialisation
    (Xvx,Yvx) = ([x for x=xv,y=yc], [y for x=xv,y=yc])
    (Xvy,Yvy) = ([x for x=xc,y=yv], [y for x=xc,y=yv])
    fields.G .= [x^2 + y^2 ≥ radius^2 ? G0 : Gi for x in xci[1], y in xci[2]]
    fields.Gv.= [x^2 + y^2 ≥ radius^2 ? G0 : Gi for x in xvi[1], y in xvi[2]]

    fields.Vx[:, 2:end-1]  .=   εbg.*Xvx
    fields.Vy[2:end-1, :]  .= .-εbg.*Yvy

    fields.Vx[:, 1]   .=   fields.Vx[:, 2]
    fields.Vx[:, end] .=   fields.Vx[:, end-1]
    fields.Vy[1, :]   .=   fields.Vy[2,:]
    fields.Vy[end, :] .=   fields.Vy[end-1,:]

    iter_evo = Float64[]; errs_evo = ElasticMatrix{Float64}(undef,length(ϵtol),0)
    opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:turbo, framestyle=:box)
    t = 0.0; evo_t=[]; evo_τxx=[]
    # time loop
    _di = inv.((dx,dy))
    for it = 1:nt
        @printf("it=%d\n",it)
        update_old!(fields)
        errs = 2.0.*ϵtol; iter = 1
        resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
        while any(errs .>= ϵtol) && iter <= maxiter
            update_iteration_params!(fields)
            
            @parallel (@idx ni) JR.compute_∇V!(fields.∇V, fields.Vx, fields.Vy, _di...)
            @parallel (@idx ni .+ 1) JR.compute_strain_rate!(
                fields.εxx, fields.εyy, fields.εxyv, fields.∇V, fields.Vx, fields.Vy, _di...
            )
            
            JR.compute_P!(
                fields.Pr,
                fields.Pr_old,
                fields.dPr,
                fields.∇V,
                fields.ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                lτ*(r+2.0)/(re_mech*vdτ),
                (;),
            )

            # update_stresses!(fields,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy)
            # update_stresses_loop!(fields,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy)
            # update_stresses2!(fields,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy, rheology, phase_center)

            # update_stresses3!(
            #     (fields.εxx, fields.εyy, fields.εxyv),
            #     (fields.τxx, fields.τyy, fields.τxy),
            #     (fields.τxx_old, fields.τyy_old, fields.τxy_old),
            #     fields,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy, rheology, phase_center
            # )

            update_stresses4!(
                (fields.εxx, fields.εyy, fields.εxyv),
                (fields.τxx, fields.τyy, fields.τxy),
                (fields.τxx_old, fields.τyy_old, fields.τxy_old),
                fields.Pr,
                fields.Pr_c,
                fields.η,
                fields.λ,
                fields.τII,
                fields.η_vep, 
                fields.Fchk, 
                relλ,dt,
                re_mech,vdτ,lτ,r,
                rheology, phase_center
            )
            fields.τxyv[2:end-1,2:end-1] .= ameanxy(fields.τxy)

            # update_stresses_vc!(fields,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy)

            update_stresses_vc_loop2!(
                fields, 
                K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy, phase_ratios.vertex, rheology
            )

            @parallel JR.compute_V!(
                fields.Vx, fields.Vy,
                fields.Pr_c,
                fields.τxx, fields.τyy, fields.τxyv,
                vdτ*lτ/re_mech,
                ρg...,
                fields.ητ,
                _di...,
            )  

            fields.Vx[:, 1]   .= fields.Vx[:, 2]
            fields.Vx[:, end] .= fields.Vx[:, end-1]
            fields.Vy[1, :]   .= fields.Vy[2,:]
            fields.Vy[end, :] .= fields.Vy[end-1,:]

            if iter % ncheck == 0
                # update residuals
                compute_residuals!(fields,dx,dy)
                errs = maximum.((abs.(fields.r_Vx),abs.(fields.r_Vy),abs.(fields.dPr)))
                push!(iter_evo,iter/max(nx,ny));append!(errs_evo,errs)
                @printf("  iter/nx=%.3f,errs=[ %1.3e, %1.3e, %1.3e ] (Fchk=%1.2e)\n",iter/max(nx,ny),errs...,maximum(fields.Fchk))
            end
            iter += 1
        end
        t += dt
        push!(evo_t,t); push!(evo_τxx,maximum(fields.τxx))
        # visualisation
        # fields.Vmag .= sqrt.(ameanx(fields.Vx).^2 + ameany(fields.Vy).^2)
        p1=heatmap(xc,yc,ameanx(fields.Vx[:, 2:end-1])',title="Vx";opts...)
        p2=heatmap(xc,yc,fields.η_vep',title="η_vep";opts...)
        p3=heatmap(xc,yc,fields.τII',title="τII";opts...)
        p4=plot(evo_t,evo_τxx,legend=false,xlabel="time",ylabel="max(τxx)",linewidth=0,markershape=:circle,markersize=3,framestyle=:box)
        display(plot(p1,p2,p3,p4,layout=(2,2)))
    end
    return
end

main()

update_stresses_center_vertex!(
    (fields.εxx, fields.εyy, fields.εxyv),
    (fields.τxx, fields.τyy, fields.τxy),
    (fields.τxx_old, fields.τyy_old, fields.τxy_old),
    fields.Pr,
    fields.Pr_c,
    fields.η,
    fields.λ,
    fields.τII,
    fields.η_vep, 
    fields.Fchk, 
    relλ,dt,
    re_mech,vdτ,lτ,r,
    rheology, phase_center
)


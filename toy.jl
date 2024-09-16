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

@views function update_old!((;τxx_old,τyy_old,τxy_old,Pr_old,τxx,τyy,τxy,Pr,Pr_c,λ))
    τxx_old .= τxx
    τyy_old .= τyy
    τxy_old .= τxy
    Pr      .= Pr_c
    Pr_old  .= Pr
    λ       .= 0.0
    return
end

@views function update_iteration_params!((;η,ητ,η_vep))
    # ητ[2:end-1,2:end-1] .= maxloc(amean.(η,η_vep)./2)
    ητ[2:end-1,2:end-1] .= maxloc(η); bc2!(ητ)
    return
end

@views function update_stresses!((;εxx_ve,εyy_ve,εxy_ve,εII_ve,Pr,Pr_c,εxx,εyy,εxy,εxyv,dτxx,dτyy,dτxy,τxx,τyy,τxy,τxyv,τxx_old,τyy_old,τxy_old,Vx,Vy,∇V,η,ητ,G,F,λ,dQdτxx,dQdτyy,dQdτxy,τII,η_vep,dτ_r,Fchk,dPr,Pr_old),K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy)
    θ_dτ    = lτ*(r+2.0)/(re_mech*vdτ)
    dτ_r   .= 1.0./(θ_dτ .+ η./(G.*dt) .+ 1.0)
    ∇V     .= diff(Vx,dims=1)./dx .+ diff(Vy,dims=2)./dy
    dPr    .= .-∇V .- (Pr .- Pr_old)./K./dt
    # Pr    .+= r/θ_dτ.*η.*dPr                      # explicit
    Pr    .+= dPr./(1.0./(r/θ_dτ.*η) .+ 1.0./K./dt) # implicit
    # strain rates
    εxx    .= diff(Vx,dims=1)./dx .- ∇V./3.0
    εyy    .= diff(Vy,dims=2)./dy .- ∇V./3.0
    εxyv[2:end-1,2:end-1] .= 0.5*(diff(Vx[2:end-1,:],dims=2)./dy .+ diff(Vy[:,2:end-1],dims=1)./dx)
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
    λ      .= (1.0 .- relλ).*λ .+ relλ.*(max.(F,0.0)./( .+ η_reg .+ K.*dt.*sinϕ.*sinψ))
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

@views function update_velocities!((;Vx,Vy,Pr_c,τxx,τyy,τxyv,ητ),vdτ,lτ,re_mech,dx,dy)
    nudτ = vdτ*lτ/re_mech
    Vx[2:end-1,:] .+= (diff(.-Pr_c.+τxx,dims=1)./dx .+ diff(τxyv[2:end-1,:],dims=2)./dy).*nudτ./avx(ητ)
    Vy[:,2:end-1] .+= (diff(.-Pr_c.+τyy,dims=2)./dy .+ diff(τxyv[:,2:end-1],dims=1)./dx).*nudτ./avy(ητ)
    return
end

@views function compute_residuals!((;r_Vx,r_Vy,Pr_c,τxx,τyy,τxyv),dx,dy)
    r_Vx .= diff(.-Pr_c[:,2:end-1].+τxx[:,2:end-1],dims=1)./dx .+ diff(τxyv[2:end-1,2:end-1],dims=2)./dy
    r_Vy .= diff(.-Pr_c[2:end-1,:].+τyy[2:end-1,:],dims=2)./dy .+ diff(τxyv[2:end-1,2:end-1],dims=1)./dx
    return
end

function main()
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
    nx,ny      = 63,63
    nt         = 50
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
    # array allocation
    fields = (
        Vx         = zeros(nx+1,ny  ),
        Vy         = zeros(nx  ,ny+1),
        Pr         = zeros(nx  ,ny  ),
        Pr_c       = zeros(nx  ,ny  ),
        Pr_old     = zeros(nx  ,ny  ),
        ∇V         = zeros(nx  ,ny  ),
        τxx        = zeros(nx  ,ny  ),
        τyy        = zeros(nx  ,ny  ),
        τxy        = zeros(nx  ,ny  ),
        τxyv       = zeros(nx+1,ny+1),
        τxx_old    = zeros(nx  ,ny  ),
        τyy_old    = zeros(nx  ,ny  ),
        τxy_old    = zeros(nx  ,ny  ),
        τII        = zeros(nx  ,ny  ),
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
        εxx        = zeros(nx  ,ny  ),
        εyy        = zeros(nx  ,ny  ),
        εxy        = zeros(nx  ,ny  ),
        εxx_ve     = zeros(nx  ,ny  ),
        εyy_ve     = zeros(nx  ,ny  ),
        εxy_ve     = zeros(nx  ,ny  ),
        εII_ve     = zeros(nx  ,ny  ),
        εxyv       = zeros(nx+1,ny+1),
        dτxx       = zeros(nx  ,ny  ),
        dτyy       = zeros(nx  ,ny  ),
        dτxy       = zeros(nx  ,ny  ),
        η          = η0.*ones(nx  ,ny  ),
        G          = G0.*ones(nx  ,ny  ),
        η_vep      = η0.*ones(nx  ,ny  ),
    )
    # initialisation
    (Xvx,Yvx) = ([x for x=xv,y=yc], [y for x=xv,y=yc])
    (Xvy,Yvy) = ([x for x=xc,y=yv], [y for x=xc,y=yv])
    rad       = xc.^2 .+ yc'.^2
    fields.G[rad.<radi] .= Gi
    fields.Vx   .=   εbg.*Xvx
    fields.Vy   .= .-εbg.*Yvy
    iter_evo = Float64[]; errs_evo = ElasticMatrix{Float64}(undef,length(ϵtol),0)
    opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:turbo, framestyle=:box)
    t = 0.0; evo_t=[]; evo_τxx=[]
    # time loop
    for it = 1:nt
        @printf("it=%d\n",it)
        update_old!(fields)
        errs = 2.0.*ϵtol; iter = 1
        resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
        while any(errs .>= ϵtol) && iter <= maxiter
            update_iteration_params!(fields)
            update_stresses!(fields,K,τ_y,sinϕ,sinψ,η_reg,relλ,dt,re_mech,vdτ,lτ,r,dx,dy)
            update_velocities!(fields,vdτ,lτ,re_mech,dx,dy)
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
        fields.Vmag .= sqrt.(ameanx(fields.Vx).^2 + ameany(fields.Vy).^2)
        p1=heatmap(xc,yc,ameanx(fields.Vx)',title="Vx";opts...)
        p2=heatmap(xc,yc,fields.η_vep',title="η_vep";opts...)
        p3=heatmap(xc,yc,fields.τII',title="τII";opts...)
        p4=plot(evo_t,evo_τxx,legend=false,xlabel="time",ylabel="max(τxx)",linewidth=0,markershape=:circle,markersize=3,framestyle=:box)
        display(plot(p1,p2,p3,p4,layout=(2,2)))
    end
    return
end

# main()
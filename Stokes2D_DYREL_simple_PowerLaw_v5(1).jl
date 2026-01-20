# Initialisation
using Plots, Printf, Statistics, LinearAlgebra
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

# can be replaced by AD
function Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ, Δx, Δy, ncx, ncy)
    
    Dx = zeros(ncx-1, ncy)
    Dy = zeros(ncx, ncy-1)
    λmaxVx = zeros(ncx-1, ncy)
    λmaxVy = zeros(ncx, ncy-1)
    
    # Compute Dx, Cxx, Cxy for x-component
    for j in 1:ncy
        for i in 1:ncx-1
            ηN = (j > 1) ? ηv[i+1, j] : ηv[i+1, j+1]
            ηS = (j < ncy) ? ηv[i+1, j+1] : ηv[i+1, j]
            ηW = ηc[i, j]
            ηE = ηc[i+1, j]
            ebW = γ[i, j]
            ebE = γ[i+1, j]
            
            Cxx = abs(ηN / Δy^2) + abs(ηS / Δy^2) + 
                  abs(ebE / Δx^2 + (4/3) * ηE / Δx^2) + 
                  abs(ebW / Δx^2 + (4/3) * ηW / Δx^2) + 
                  abs(-(-ηN / Δy - ηS / Δy) / Δy + (ebE / Δx + ebW / Δx) / Δx + 
                      ((4/3) * ηE / Δx + (4/3) * ηW / Δx) / Δx)
            
            Cxy = abs(ebE / (Δx * Δy) - 2/3 * ηE / (Δx * Δy) + ηN / (Δx * Δy)) + 
                  abs(ebE / (Δx * Δy) - 2/3 * ηE / (Δx * Δy) + ηS / (Δx * Δy)) + 
                  abs(ebW / (Δx * Δy) + ηN / (Δx * Δy) - 2/3 * ηW / (Δx * Δy)) + 
                  abs(ebW / (Δx * Δy) + ηS / (Δx * Δy) - 2/3 * ηW / (Δx * Δy))
            
            Dx[i, j] = -(-ηN / Δy - ηS / Δy) / Δy + (ebE / Δx + ebW / Δx) / Δx + 
                       ((4/3) * ηE / Δx + (4/3) * ηW / Δx) / Δx
            
            λmaxVx[i, j] = 1.0 / Dx[i, j] * (Cxx + Cxy)
        end
    end
    
    # Compute Dy, Cyy, Cyx for y-component
    for j in 1:ncy-1
        for i in 1:ncx
            ηE = (i < ncx) ? ηv[i+1, j+1] : ηv[i, j+1]
            ηW = (i > 1) ? ηv[i, j+1] : ηv[i+1, j+1]
            ηS = ηc[i, j]
            ηN = ηc[i, j+1]
            ebS = γ[i, j]
            ebN = γ[i, j+1]
            
            Cyy = abs(ηE / Δx^2) + abs(ηW / Δx^2) + 
                  abs(ebN / Δy^2 + (4/3) * ηN / Δy^2) + 
                  abs(ebS / Δy^2 + (4/3) * ηS / Δy^2) + 
                  abs((ebN / Δy + ebS / Δy) / Δy + ((4/3) * ηN / Δy + (4/3) * ηS / Δy) / Δy - 
                      (-ηE / Δx - ηW / Δx) / Δx)
            
            Cyx = abs(ebN / (Δx * Δy) + ηE / (Δx * Δy) - 2/3 * ηN / (Δx * Δy)) + 
                  abs(ebN / (Δx * Δy) - 2/3 * ηN / (Δx * Δy) + ηW / (Δx * Δy)) + 
                  abs(ebS / (Δx * Δy) + ηE / (Δx * Δy) - 2/3 * ηS / (Δx * Δy)) + 
                  abs(ebS / (Δx * Δy) - 2/3 * ηS / (Δx * Δy) + ηW / (Δx * Δy))
            
            Dy[i, j] = (ebN / Δy + ebS / Δy) / Δy + ((4/3) * ηN / Δy + (4/3) * ηS / Δy) / Δy - 
                       (-ηE / Δx - ηW / Δx) / Δx
            
            λmaxVy[i, j] = 1.0 / Dy[i, j] * (Cyx + Cyy)
        end
    end
    
    return Dx, Dy, λmaxVx, λmaxVy
end

function Gershgorin_Stokes2D_SchurComplement_old(ηc, ηv, γ, Δx, Δy, ncx  ,ncy)
        
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    ebW   = γ[1:end-1,:] 
    ebE   = γ[2:end-0,:] 
    Cxx   = @. abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    Cxy   = @. abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))
    Dx    = @. -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS   = ηc[:,1:end-1]
    ηN   = ηc[:,2:end-0]
    ebS  = γ[:,1:end-1] 
    ebN  = γ[:,2:end-0] 
    Cyy  = @. abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    Cyx  = @. abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))
    Dy   = @. (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./Dx .* (Cxx .+ Cxy)
    λmaxVy = 1.0./Dy .* (Cyx .+ Cyy)

    return Dx, Dy, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D_VEP(n, npwl, ηrel)
    sc = (σ=1e0, t=1e0, L=1e0)
    # Physics
    Lx, Ly   = 1e0/sc.L, 1e0/sc.L # domain size
    radi     = sqrt(0.01)/sc.L           # inclusion radius
    η0       = 1e2/sc.σ/sc.t       # viscous viscosity
    ηi       = 1e-1/sc.σ/sc.t       # min/max inclusion viscosity
    G0       = 1e30/sc.σ
    Gi       = G0/2/sc.σ/sc.t       # min/max inclusion viscosity
    C        = 1e30/cosd(30)/sc.σ
    Δt       = 1#η0/G0/4
    εbg      = 1e0*sc.t      # background strain-rate
    comp     = true                 
    K        = 5e20/sc.σ  
    ϕ        = 30.0 
    ψ        = 5.0   
    ηvp      = 1e-2/sc.σ/sc.t    
    # Numerics
    ncx, ncy = n*32, n*32   # numerical grid resolution
    nt       = 1            # time steps
    ϵ        = 1e-7         # tolerance
    iterMax  = 20000        # max number of iters
    nout     = 400          # @Albert: apparently it's better to keep it large for incompressible tests
    c_fact   = 0.5          # damping factor
    dτ_local = true         # helps a little bit sometimes, sometimes not! 
    CFL_v    = 0.99         # CFL: can't make it larger
    γfact    = 20           # penalty: multiplier to the arithmetic mean of η 
    rel_drop = 1e-5          # @Albert: large not to oversolve
    λ̇rel     = 1.00         # overrelaxation helps!
    ηrel     = 1e-2         # power law relaxation
    # Preprocessing
    Δx, Δy  = Lx/ncx, Ly/ncy
    # Array initialisation
    Pt       = zeros(ncx  ,ncy  )
    Pt0      = zeros(ncx  ,ncy  ) 
    Ptv      = zeros(ncx+1,ncy+1)
    ΔPψ      = zeros(ncx  ,ncy  )
    ∇V       = zeros(ncx  ,ncy  )
    Vx       = zeros(ncx+1,ncy+2) 
    Vy       = zeros(ncx+2,ncy+1)
    dVx      = zeros(ncx-1,ncy  )
    dVy      = zeros(ncx  ,ncy-1)
    EIIc     = zeros(ncx  ,ncy  )
    EIIv     = zeros(ncx+1,ncy+1)
    Exx      = zeros(ncx  ,ncy  )
    Eyy      = zeros(ncx  ,ncy  )
    Exy      = zeros(ncx+1,ncy+1)
    Exxv     = zeros(ncx+1,ncy+1)
    Eyyv     = zeros(ncx+1,ncy+1)
    Exyc     = zeros(ncx  ,ncy  )
    TIIc     = zeros(ncx  ,ncy  )
    TIIv     = zeros(ncx+1,ncy+1)
    Txx      = zeros(ncx  ,ncy  )
    Tyy      = zeros(ncx  ,ncy  )
    Txy      = zeros(ncx+1,ncy+1)
    Txxv     = zeros(ncx+1,ncy+1)
    Tyyv     = zeros(ncx+1,ncy+1)
    Txyc     = zeros(ncx  ,ncy  )
    Txx0     = zeros(ncx  ,ncy  )
    Tyy0     = zeros(ncx  ,ncy  )
    Txy0     = zeros(ncx+1,ncy+1)
    Txxv0    = zeros(ncx+1,ncy+1)
    Tyyv0    = zeros(ncx+1,ncy+1)
    Txy0c    = zeros(ncx  ,ncy  )
    Fc       = zeros(ncx  ,ncy  ) 
    Fv       = zeros(ncx+1,ncy+1) 
    λ̇c       = zeros(ncx  ,ncy  )
    λ̇v       = zeros(ncx+1,ncy+1)
    λ̇_true_c = zeros(ncx  ,ncy  )
    λ̇_true_v = zeros(ncx+1,ncy+1)
    Rx       = zeros(ncx-1,ncy  )
    Ry       = zeros(ncx  ,ncy-1)
    Rp       = zeros(ncx  ,ncy  )
    Rx0      = zeros(ncx-1,ncy  )
    Ry0      = zeros(ncx  ,ncy-1)
    dVxdτ    = zeros(ncx-1,ncy  )
    dVydτ    = zeros(ncx  ,ncy-1)
    βVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    βVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    cVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    cVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    αVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    αVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    ηb       = η0*ones(ncx  ,ncy  )
    ηc0      = η0*ones(ncx  ,ncy  )
    ηv0      = η0*ones(ncx+1,ncy+1)
    ηc       = η0*ones(ncx  ,ncy  )
    ηv       = η0*ones(ncx+1,ncy+1)
    ηc_true  = η0*ones(ncx  ,ncy  )
    ηv_true  = η0*ones(ncx+1,ncy+1)
    Gc       = G0*ones(ncx  ,ncy  )
    Gv       = G0*ones(ncx+1,ncy+1)
    ηve_c    = zeros(ncx  ,ncy  )
    ηve_v    = zeros(ncx+1,ncy+1)
    ηvp_c    = zeros(ncx  ,ncy  )
    ηvp_v    = zeros(ncx+1,ncy+1)
    ηvep_c   = zeros(ncx  ,ncy  )
    ηvep_v   = zeros(ncx+1,ncy+1)
    P_num    = zeros(ncx  ,ncy  )
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    # Effective viscosity
    ηv0[(xv).^2 .+ (yv').^2 .< radi^2 ] .= ηi
    # Harmonic averaging mimicking PIC interpolation
    ηc0    .= av(ηv0)
    ηv0[2:end-1,2:end-1] .= av(ηc0)
    ηv0[1,:] .=  ηv0[2,:]; ηv0[end,:] .=  ηv0[end-1,:]
    ηv0[:,1] .=  ηv0[:,2]; ηv0[:,end] .=  ηv0[:,end-1]
    # Effective viscosity
    Gv[(xv).^2 .+ (yv').^2 .< radi^2 ] .= Gi
    # Harmonic averaging mimicking PIC interpolation
    Gc    .= av4_harm(Gv)
    Gv[2:end-1,2:end-1] .= av4_harm(Gc)
    Gv[1,:] .=  Gv[2,:]; Gv[end,:] .=  Gv[end-1,:]
    Gv[:,1] .=  Gv[:,2]; Gv[:,end] .=  Gv[:,end-1]
    # Visco-elastic viscosity
    ηc .= ηc0
    ηv .= ηv0
    ηve_c .= (1 ./ ηc .+ 1 ./ (Gc*Δt)).^-1
    ηve_v .= (1 ./ ηv .+ 1 ./ (Gv*Δt)).^-1
    # Bulk viscosity
    ηb    .= K .* Δt
    # Select γ
    γi   = γfact*mean(ηc).*ones(size(ηc))
    # (Pseudo-)compressibility
    γ_eff = zeros(size(ηb)) 
    if comp
        γ_num = γi.*ones(size(ηb))
        γ_phy = ηb
        γ_eff = ((γ_phy.*γ_num)./(γ_phy.+γ_num))
    else
        γ_eff .= γi
        γ_eff .= γ_eff
    end
    # Optimal pseudo-time steps - can be replaced by AD
    Dx, Dy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηve_c, ηve_v, γ_eff, Δx, Δy, ncx ,ncy)
    # Select dτ
    if dτ_local
        dτVx =  2.0./sqrt.(λmaxVx)*CFL_v
        dτVy =  2.0./sqrt.(λmaxVy)*CFL_v
    else
        dτVx =  2.0./sqrt.(maximum(λmaxVx))*CFL_v 
        dτVy =  2.0./sqrt.(maximum(λmaxVy))*CFL_v
    end
    βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
    βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
    αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
    αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
    # Initial condition
    Vx     .=   εbg.*xv .+    0*yce'
    Vy     .=     0*xce .- εbg.*yv'
    # Vx[2:end-1,:] .= 0 # ensure non zero initial pressure residual
    # Vy[:,2:end-1] .= 0 # ensure non zero initial pressure residual
    Vx[2:end-1,:] .+= 1e-5*rand(size(   Vx[2:end-1,:])...) # ensure non zero initial pressure residual
    Vy[:,2:end-1] .+= 1e-5*rand(size(   Vy[:,2:end-1])...) # ensure non zero initial pressure residual

    # Time
    Tii_evo = zeros(nt) 
    it_evo  = zeros(nt)
    itg = 0; kiter = 0
    err_evo_it, err_evo_V, err_evo_P = zeros(iterMax), zeros(iterMax), zeros(iterMax)
    for it=1:nt
        Txx0 .= Txx; Tyy0 .= Tyy; Txy0 .= Txy; Txy0c .= Txyc;  Txxv0 .= Txxv; Tyyv0 .= Tyyv; Pt0 .= Pt
        # Iteration loop
        errVx0 = 1.0;  errVy0 = 1.0;  errPt0 = 1.0 
        errVx00= 1.0;  errVy00= 1.0; 
        iter=0;  kiter = 0; err=2*ϵ; err_evo_it .= 0.; err_evo_V .= 0.; err_evo_P .= 0.;
        @time for itPH = 1:1000
            # Boundaries
            Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
            Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
            # Divergence
            ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
            # Pressure on vertices
            Ptv[2:end-1,2:end-1] .= av(Pt)
            Ptv[1,:] .=  Ptv[2,:]; Ptv[end,:] .=  Ptv[end-1,:]
            Ptv[:,1] .=  Ptv[:,2]; Ptv[:,end] .=  Ptv[:,end-1]
            # Deviatoric strain rate
            Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
            Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
            Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
            Exxv[2:end-1,2:end-1] .= av(Exx); Exxv[[1 end], :] .= Exxv[[2 end-1], :]; Exxv[:, [1 end]] .= Exxv[:, [2 end-1]]
            Eyyv[2:end-1,2:end-1] .= av(Eyy); Eyyv[[1 end], :] .= Eyyv[[2 end-1], :]; Eyyv[:, [1 end]] .= Eyyv[:, [2 end-1]]
            Exyc  .= av(Exy)
            EIIc  .= sqrt.(0.5.*((Exx  .+ Txx0 ./(2*Gc*Δt)).^2 .+ (Eyy  .+ Tyy0 ./(2*Gc*Δt)).^2 .+ (.-(Exx  .+ Txx0 ./(2*Gc*Δt)).-(Eyy  .+ Tyy0 ./(2*Gc*Δt))).^2) .+ (Exyc .+ Txy0c./(2*Gc*Δt)).^2 )
            EIIv  .= sqrt.(0.5.*((Exxv .+ Txxv0./(2*Gv*Δt)).^2 .+ (Eyyv .+ Tyyv0./(2*Gv*Δt)).^2 .+ (.-(Exxv .+ Txxv0./(2*Gv*Δt)).-(Eyyv .+ Tyyv0./(2*Gv*Δt))).^2) .+ (Exy  .+ Txy0 ./(2*Gv*Δt)).^2 )
            
            # Visco-elastic viscosity
            # ηc_true .= ηc0 .* EIIc.^(1/npwl-1) 
            # ηv_true .= ηv0 .* EIIv.^(1/npwl-1)
            # ηc      .= exp.(ηrel*log.(ηc_true) .+ (1-ηrel).*log.(ηc))
            # ηv      .= exp.(ηrel*log.(ηv_true) .+ (1-ηrel).*log.(ηv))
            # ηve_c .= (1 ./ ηc .+ 1 ./ (Gc*Δt)).^-1
            # ηve_v .= (1 ./ ηv .+ 1 ./ (Gv*Δt)).^-1
            
            # Deviatoric stress
            Txx   .= 2.0.*ηve_c.*(Exx  .+ Txx0 ./(2*Gc*Δt))
            Tyy   .= 2.0.*ηve_c.*(Eyy  .+ Tyy0 ./(2*Gc*Δt))
            Txy   .= 2.0.*ηve_v.*(Exy  .+ Txy0 ./(2*Gv*Δt))
            Txxv  .= 2.0.*ηve_v.*(Exxv .+ Txxv0./(2*Gv*Δt))
            Tyyv  .= 2.0.*ηve_v.*(Eyyv .+ Tyyv0./(2*Gv*Δt))
            Txyc  .= 2.0.*ηve_c.*(Exyc .+ Txy0c./(2*Gc*Δt))
            TIIc  .= sqrt.(0.5.*(Txx.^2  .+ Tyy.^2  .+ (.-Txx.-Tyy).^2)   .+ Txyc.^2 )
            TIIv  .= sqrt.(0.5.*(Txxv.^2 .+ Tyyv.^2 .+ (.-Txxv.-Tyyv).^2) .+ Txy.^2 )
            
            ηc_true .= @. 2^(npwl-1) * ηc0^npwl * TIIc^(1 - npwl)
            ηv_true .= @. 2^(npwl-1) * ηv0^npwl * TIIv^(1 - npwl)
            ηc      .= exp.(ηrel*log.(ηc_true) .+ (1-ηrel).*log.(ηc))
            ηv      .= exp.(ηrel*log.(ηv_true) .+ (1-ηrel).*log.(ηv))
            ηve_c   .= ηc
            ηve_v   .= ηv

            
            # # Plasticity
            # λ̇c            .= 0.
            # λ̇v            .= 0.
            # Fc            .= TIIc .- C.*cosd(ϕ) .- Pt .*sind(ϕ)
            # Fv            .= TIIv .- C.*cosd(ϕ) .- Ptv.*sind(ϕ)
            # λ̇c[Fc.>0]     .= Fc[Fc.>0]./(ηve_c[Fc.>0] .+ ηvp .+ K.*Δt.*sind(ϕ).*sind.(ψ))      
            # λ̇v[Fv.>0]     .= Fv[Fv.>0]./(ηve_v[Fv.>0] .+ ηvp .+ K.*Δt.*sind(ϕ).*sind.(ψ))      
            # ηvep_c        .= ηve_c
            # ηvep_v        .= ηve_v
            # ηvp_c .= (TIIc.-λ̇c.*ηve_c) ./ (2 .* EIIc)
            # ηvp_v .= (TIIv.-λ̇v.*ηve_v) ./ (2 .* EIIv)
            # ηvep_c[Fc.>0] .= ηvp_c[Fc.>0]
            # ηvep_v[Fv.>0] .= ηvp_v[Fv.>0]
            # Txx   .= 2.0.*ηvep_c.*(Exx .+ Txx0./(2*Gc*Δt))
            # Tyy   .= 2.0.*ηvep_c.*(Eyy .+ Tyy0./(2*Gc*Δt))
            # Txy   .= 2.0.*ηvep_v.*(Exy .+ Txy0./(2*Gv*Δt))
            # Txxv  .= 2.0.*ηvep_v.*(Exxv .+ Txxv0./(2*Gv*Δt))
            # Tyyv  .= 2.0.*ηvep_v.*(Eyyv .+ Tyyv0./(2*Gv*Δt))
            # Txyc  .= 2.0.*ηvep_c.*(Exyc .+ Txy0c./(2*Gc*Δt))
            # ΔPψ   .= λ̇c.*sind(ψ).*K.*Δt
            # Check
            TIIc  .= sqrt.(0.5.*(Txx.^2  .+ Tyy.^2  .+ (.-Txx.-Tyy).^2)   .+ Txyc.^2 )
            TIIv  .= sqrt.(0.5.*(Txxv.^2 .+ Tyyv.^2 .+ (.-Txxv.-Tyyv).^2) .+ Txy.^2 )
            Fc    .= TIIc .- C.*cosd(ϕ) .- (Pt .+ λ̇c.*sind(ψ).*K.*Δt).*sind(ϕ)  .- ηvp.*λ̇c
            Fv    .= TIIv .- C.*cosd(ϕ) .- (Ptv.+ λ̇v.*sind(ψ).*K.*Δt).*sind(ϕ)  .- ηvp.*λ̇v
            # Residuals
            Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .- (ΔPψ[2:end,:] .- ΔPψ[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
            Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .- (ΔPψ[:,2:end] .- ΔPψ[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
            Rp    .= .-∇V .- comp*(Pt.-Pt0)./ηb 
            # Residual check
            errVx = norm(Rx) / sqrt(length(Rx)) 
            errVy = norm(Ry) / sqrt(length(Ry)) 
            errPt = norm(Rp) / sqrt(length(Rp)) 
            if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; end
            err = maximum((min(errVx/errVx0, errVx), min(errVy/errVy0, errVy), min(errPt/(errPt0+eps()), errPt)))
            # err = maximum([min(errVx/errVx0, errVx), min(errVy/errVy0, errVy)]) #, min(errPt/errPt0, errPt)
            kiter += 1
            err_evo_V[kiter] = errVx/errVx0; err_evo_P[kiter] = errPt/errPt0; err_evo_it[kiter] =  iter
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[Rx=%1.3e %1.3e, Ry=%1.3e %1.3e, Rp=%1.3e %1.3e] \n", itPH, iter, iter/ncx, err, errVx, errVx/errVx0, errVy, errVy/errVy0, errPt, errPt/errPt0)
            isnan(err) && error("NaNs")
            if (err<ϵ) break end
            # Set tolerance of velocity solve proportional to residual
            ϵ_vel = err*rel_drop
            itPT  = 0.;
            λ̇rel = 1e-0
            while (err>ϵ_vel && itPT<=iterMax)
                iter   += 1 
                itPT   += 1
                itg    += 1
                # Pseudo-old dudes 
                Rx0   .= Rx
                Ry0   .= Ry
                # Boundaries
                Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
                Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
                # Divergence 
                ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
                Rp    .= .-∇V .- comp*(Pt.-Pt0)./ηb 
                # Deviatoric strain rate
                Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
                Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
                Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
                Exxv[2:end-1,2:end-1] .= av(Exx); Exxv[[1 end], :] .= Exxv[[2 end-1], :]; Exxv[:, [1 end]] .= Exxv[:, [2 end-1]]
                Eyyv[2:end-1,2:end-1] .= av(Eyy); Eyyv[[1 end], :] .= Eyyv[[2 end-1], :]; Eyyv[:, [1 end]] .= Eyyv[:, [2 end-1]]
                Exyc  .= av(Exy)
                EIIc  .= sqrt.(0.5.*((Exx  .+ Txx0 ./(2*Gc*Δt)).^2 .+ (Eyy  .+ Tyy0 ./(2*Gc*Δt)).^2 .+ (.-(Exx  .+ Txx0 ./(2*Gc*Δt)).-(Eyy  .+ Tyy0 ./(2*Gc*Δt))).^2) .+ (Exyc .+ Txy0c./(2*Gc*Δt)).^2 )
                EIIv  .= sqrt.(0.5.*((Exxv .+ Txxv0./(2*Gv*Δt)).^2 .+ (Eyyv .+ Tyyv0./(2*Gv*Δt)).^2 .+ (.-(Exxv .+ Txxv0./(2*Gv*Δt)).-(Eyyv .+ Tyyv0./(2*Gv*Δt))).^2) .+ (Exy  .+ Txy0 ./(2*Gv*Δt)).^2 )
               
                # Visco-elastic viscosity
                # ηc_true .= ηc0 .* EIIc.^(1/npwl-1) 
                # ηv_true .= ηv0 .* EIIv.^(1/npwl-1)
                # ηc      .= exp.(ηrel*log.(ηc_true) .+ (1-ηrel).*log.(ηc))
                # ηv      .= exp.(ηrel*log.(ηv_true) .+ (1-ηrel).*log.(ηv))
                # ηve_c .= (1 ./ ηc_true .+ 1 ./ (Gc*Δt)).^-1
                # ηve_v .= (1 ./ ηv_true .+ 1 ./ (Gv*Δt)).^-1
               
                # Deviatoric stress
                Txx   .= 2.0.*ηve_c.*(Exx  .+ Txx0 ./(2*Gc*Δt)) 
                Tyy   .= 2.0.*ηve_c.*(Eyy  .+ Tyy0 ./(2*Gc*Δt)) 
                Txy   .= 2.0.*ηve_v.*(Exy  .+ Txy0 ./(2*Gv*Δt))
                Txxv  .= 2.0.*ηve_v.*(Exxv .+ Txxv0./(2*Gv*Δt))
                Tyyv  .= 2.0.*ηve_v.*(Eyyv .+ Tyyv0./(2*Gv*Δt))
                Txyc  .= 2.0.*ηve_c.*(Exyc .+ Txy0c./(2*Gc*Δt))
                TIIc  .= sqrt.(0.5.*(Txx.^2  .+ Tyy.^2  .+ (.-Txx.-Tyy).^2)   .+ Txyc.^2 )
                TIIv  .= sqrt.(0.5.*(Txxv.^2 .+ Tyyv.^2 .+ (.-Txxv.-Tyyv).^2) .+ Txy.^2 )
                
                ηc_true .= @. 2^(npwl-1) * ηc0^npwl * TIIc^(1 - npwl)
                ηv_true .= @. 2^(npwl-1) * ηv0^npwl * TIIv^(1 - npwl)
                ηc      .= exp.(ηrel*log.(ηc_true) .+ (1-ηrel).*log.(ηc))
                ηv      .= exp.(ηrel*log.(ηv_true) .+ (1-ηrel).*log.(ηv))
                ηve_c   .= ηc
                ηve_v   .= ηv

                # # Plasticity
                # Fc              .= TIIc .- C.*cosd(ϕ) .- Pt .*sind(ϕ)
                # Fv              .= TIIv .- C.*cosd(ϕ) .- Ptv.*sind(ϕ)
                # λ̇_true_c        .= 0.
                # λ̇_true_v        .= 0.
                # λ̇_true_c[Fc.>0] .= Fc[Fc.>0]./(ηve_c[Fc.>0] .+ ηvp .+ K.*Δt.*sind(ϕ).*sind.(ψ))      
                # λ̇_true_v[Fv.>0] .= Fv[Fv.>0]./(ηve_v[Fv.>0] .+ ηvp .+ K.*Δt.*sind(ϕ).*sind.(ψ))
                # λ̇c              .= λ̇rel*λ̇_true_c .+ (1-λ̇rel).*λ̇c
                # λ̇v              .= λ̇rel*λ̇_true_v .+ (1-λ̇rel).*λ̇v 
                # ηvep_c .= ηve_c
                # ηvep_v .= ηve_v
                # ηvp_c  .= (TIIc.-λ̇c.*ηve_c) ./ (2 .* EIIc)
                # ηvp_v  .= (TIIv.-λ̇v.*ηve_v) ./ (2 .* EIIv)
                # ηvep_c[Fc.>0] .= ηvp_c[Fc.>0]
                # ηvep_v[Fv.>0] .= ηvp_v[Fv.>0] 
                # Txx    .= 2.0.*ηvep_c.*(Exx  .+ Txx0 ./(2*Gc*Δt)) 
                # Tyy    .= 2.0.*ηvep_c.*(Eyy  .+ Tyy0 ./(2*Gc*Δt)) 
                # Txy    .= 2.0.*ηvep_v.*(Exy  .+ Txy0 ./(2*Gv*Δt))
                # ΔPψ    .= λ̇c.*sind(ψ).*K.*Δt
                # Residuals
                # Rp    .= .-∇V .- comp*(Pt.-Pt0)./ηb 
                P_num  .= γ_eff .* Rp
                Rx     .= (1.0./Dx).*(.-(P_num[2:end,:] .- P_num[1:end-1,:])./Δx .- (Pt[2:end,:] .- Pt[1:end-1,:])./Δx .- (ΔPψ[2:end,:] .- ΔPψ[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
                Ry     .= (1.0./Dy).*(.-(P_num[:,2:end] .- P_num[:,1:end-1])./Δy .- (Pt[:,2:end] .- Pt[:,1:end-1])./Δy .- (ΔPψ[:,2:end] .- ΔPψ[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
                # Damping-pong
                dVxdτ  .= αVx.*dVxdτ .+ Rx
                dVydτ  .= αVy.*dVydτ .+ Ry
                # PT updates
                Vx[2:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx 
                Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
                # Residual check
                if mod(iter, nout)==0
                    kiter += 1
                    errVx = norm(Dx.*Rx) / sqrt(length(Rx)) 
                    errVy = norm(Dy.*Ry) / sqrt(length(Ry)) 
                    if iter==nout errVx00=errVx; errVy00=errVy; end
                    err = maximum((errVx./errVx00, errVy./errVy00))
                    err_evo_V[kiter] = errVx/errVx00; err_evo_P[kiter] = errPt/errPt0; err_evo_it[kiter] =  iter
                    dVx .= dVxdτ.*βVx.*dτVx
                    dVy .= dVydτ.*βVy.*dτVy
                    f              = TIIc .- ηve_c.*λ̇c   .- C.*cosd(ϕ) .- (Pt .+ ΔPψ).*sind(ϕ) .- ηvp.*λ̇c
                    @printf("it = %d, iter = %d, err = %1.3e - max(f) = %1.3e \n", it, iter, err, maximum(f))
                    # λminV  = abs.((sum(dVx.*(Rx .- Rx0))) + abs.((sum(dVy.*(Ry .- Ry0))) )/ ( sum(dVx.*dVx)) + sum(dVy.*dVy) ) 
                    λminV  = abs(  sum(dVx.*(Rx .- Rx0)) + sum(dVy.*(Ry .- Ry0))  ) / (sum(dVx.*dVx) .+ sum(dVy.*dVy))
                    cVx .= 2*sqrt.(λminV)*c_fact
                    cVy .= 2*sqrt.(λminV)*c_fact
                    # Optimal pseudo-time steps - can be replaced by AD
                    Dx, Dy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηve_c, ηve_v, γ_eff, Δx, Δy, ncx ,ncy)
                    # Select dτ
                    if dτ_local
                        dτVx =  2.0./sqrt.(λmaxVx)*CFL_v
                        dτVy =  2.0./sqrt.(λmaxVy)*CFL_v
                    else
                        dτVx =  2.0./sqrt.(maximum(λmaxVx))*CFL_v 
                        dτVy =  2.0./sqrt.(maximum(λmaxVy))*CFL_v
                    end
                    βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
                    βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
                    αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
                    αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
                end
            end
            Pt .+= γ_eff.*Rp
        end
        Tii_evo[it] = maximum(TIIc)
        it_evo[it]  = iter/ncx
        # @show err_evo_P[1:kiter]
        # @show err_evo_P[1]
        # Plotting
        EIIc  .= sqrt.(0.5.*((Exx).^2 .+ (Eyy).^2 .+ (.-(Exx).-(Eyy)).^2) .+ (Exyc).^2 )
        p1 = heatmap(xc, yc, log10.(EIIc'./sc.t) , aspect_ratio=1, c=:coolwarm, title="EII", xlims=(-Lx/2,Lx/2), xlabel="x",ylabel="y", clim=(-0.4, 0.4))
        # p2 = heatmap(xc, yc, TIIc'.*sc.σ , aspect_ratio=1, c=:turbo, title="τII", xlims=(-Lx/2,Lx/2), xlabel="x",ylabel="y", clim=(0, 250))
        p2 = heatmap(xc, yc, Txx'.*sc.σ , aspect_ratio=1, c=:turbo, title="τxx", xlims=(-Lx/2,Lx/2), xlabel="x",ylabel="y")
        p3 = plot(title="Convergence", xlabel="DR iterations / nx",ylabel="errors") 
        p3 = plot!((err_evo_it[1:kiter])/ncx, log10.(err_evo_V[1:kiter]), label="V")
        p3 = plot!((err_evo_it[1:kiter])/ncx, log10.(err_evo_P[1:kiter]), label="P")
        # p3 = plot!((err_evo_it.-err_evo_it[1])/ncx, log10.(ϵ.*ones(size(err_evo_it))), label="tol")  
        p4 = plot(1:it, Tii_evo[1:it]*sc.σ, xlabel="time",ylabel="mean dev. stress", label=:none)
        p4 = heatmap(xc, yc, log10.(ηc)' , aspect_ratio=1, c=:inferno, title="ηc", xlims=(-Lx/2,Lx/2))
        display(plot(p1, p2, p3, p4))
        @show iter/ncx
        @show itg

        # filename = @sprintf("./results/VEP%03d.jld2", it)
        # save(filename, "err_evo_it", err_evo_it, "ncx", ncx, "kiter", kiter, "err_evo_V", err_evo_V, "err_evo_P", err_evo_P, "it_evo", it_evo, "Tii_evo", Tii_evo,  "xc", xc, "yc", yc, "∇V", ∇V, "EIIc", EIIc, "Pt", Pt )

    end
    n   = length(ηc)
    # @show η_h = 1.0 / sum(1.0/n ./ηc)
    # @show η_g = exp( sum( 1.0/n*log.(ηc)))
    # @show η_a = mean(ηc)
    return
end

# let 
    npwl = 3.0
    ηrel = 1.0
    nres = 4
    # Stokes2D_VEP(nres, npwl, ηrel)
# end

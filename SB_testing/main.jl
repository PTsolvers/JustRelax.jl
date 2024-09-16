using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

import JustRelax.JustRelax2D as JR

const backend = CPUBackend

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

@views  average(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])

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

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, radius)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius)
        x, y = xc[i], yc[j]
        if ((x-o_x)^2 + (y-o_y)^2) > radius^2
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

# MAIN SCRIPT --------------------------------------------------------------------
n      = 63
nx     = n
ny     = n
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

function main(igg)

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
# origin       = @. -li / 2     # origin coordinates
origin       = 0e0, 0e0     # origin coordinates
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
    Ψ    = 0
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
# STOKES ---------------------------------------------
# Allocate arrays needed for every Stokes problem
stokes    = StokesArrays(backend, ni)
pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.9 / √2.1)
# Buoyancy forces
ρg        = @zeros(ni...), @zeros(ni...)
args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)
# Rheology
compute_viscosity!(
    stokes, phase_ratios, args, rheology, (-Inf, Inf)
)
# Boundary conditions
flow_bcs     = VelocityBoundaryConditions(;
    free_slip = (left = true, right = true, top = true, bot = true),
    no_slip   = (left = false, right = false, top = false, bot=false),
)
stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
flow_bcs!(stokes, flow_bcs) # apply boundary conditions
update_halo!(@velocity(stokes)...)
# IO -------------------------------------------------
take(figdir)
# Time loop
t, it      = 0.0, 0
tmax       = 5
τII        = Float64[]
sol        = Float64[]
ttot       = Float64[]
_di = inv.(di)
(; ϵ, r, θ_dτ, ηdτ) = pt_stokes
(; η, η_vep) = stokes.viscosity
ni = size(stokes.P)

# ~preconditioner
ητ = deepcopy(η)
viscosity_cutoff = (-Inf, Inf)
iterMax = 15e3
nout = 15e3

 # vertex arrays

 τxyv       = zeros(nx+1,ny+1)
 τxxv       = zeros(nx+1,ny+1)
 τyyv       = zeros(nx+1,ny+1)
 τxxv_old   = zeros(nx+1,ny+1)
 τyyv_old   = zeros(nx+1,ny+1)
 τxyv_old   = zeros(nx+1,ny+1)
 τIIv       = zeros(nx+1,ny+1)
 Fv         = zeros(nx+1,ny+1)
 λv         = zeros(nx+1,ny+1)
 dQdτxxv    = zeros(nx+1,ny+1)
 dQdτyyv    = zeros(nx+1,ny+1)
 dQdτxyv    = zeros(nx+1,ny+1)
 εxxv_ve    = zeros(nx+1,ny+1)
 εyyv_ve    = zeros(nx+1,ny+1)
 εxyv_ve    = zeros(nx+1,ny+1)
 dτxxv      = zeros(nx+1,ny+1)
 dτyyv      = zeros(nx+1,ny+1)
 dτxyv      = zeros(nx+1,ny+1)
 Gv         = G0.*ones(nx+1,ny+1)
 G          = G0.*ones(nx,ny)

 Xvx,Yvx   = ([x for x=xvi[1],y=xci[2]], [y for x=xvi[1],y=xci[2]])
 Xvy,Yvy   = ([x for x=xci[1],y=xvi[2]], [y for x=xci[1],y=xvi[2]])
 rad       = (xci[1].-0.5).^2 .+ (xci[2].-0.5)'.^2
 G[rad.<radius.^2] .= Gi
 rad       = (xvi[1].-0.5).^2 .+ (xvi[2].-0.5)'.^2
 Gv[rad.<radius.^2] .= Gi

 Pv = @zeros(ni.+1...)
 εxxv = @zeros(ni.+1...)
 εyyv = @zeros(ni.+1...)
 Gdτv = @zeros(ni.+1...)
 ηv_veτ = @zeros(ni.+1...)
 ηv = @ones(ni.+1...)
 lτ    = 1
 Gdτ   = ητ.*(pt_stokes.Re/lτ*pt_stokes.Vpdτ/(pt_stokes.r+2.0))
η_veτ   = @. 1.0/(1.0/Gdτ + 1.0./1e0 + 1.0./(G*dt))
@. ηv_veτ .= 1.0/(1.0/Gdτv + 1.0./1e0 + 1.0./(Gv*dt))

while t < tmax
    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]
    sizehint!(norm_Rx, Int(iterMax))
    sizehint!(norm_Ry, Int(iterMax))
    sizehint!(norm_∇V, Int(iterMax))
    sizehint!(err_evo1, Int(iterMax))
    sizehint!(err_evo2, Int(iterMax))

    # solver loop
    @copy stokes.P0 stokes.P
    θ = @zeros(ni...)
    λ = @zeros(ni...)
    λv= @zeros(ni.+1...)
    η0 = deepcopy(η)
    do_visc = true

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    # displacement2velocity!(stokes, dt, flow_bcs)

    # K = @zeros(size(stokes.P)...)
    # isfirsttimestep = all(iszero, stokes.ε.xx)
    while iter ≤ iterMax
        10 < iter && err < ϵ && break

        # wtime0 += @elapsed begin
            JR.compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) JR.compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            JR.compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
                args,
            )

            JR.update_ρg!(ρg[2], phase_ratios, rheology, args)
            @parallel (@idx ni .+ 1) JR.compute_strain_rate!(

                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            @parallel (@idx ni) JR.compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor_center(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                phase_ratios.center,
                rheology, # needs to be a tuple
                dt,
                θ_dτ,
                args,
            )

            # vertex2center!(stokes.ε.xy_c, stokes.ε.xy)

            # # visco-elastic strain rates
            # εxx_ve  = @. stokes.ε.xx + 0.5 * stokes.τ_o.xx./(G.*dt)
            # εyy_ve  = @. stokes.ε.yy + 0.5 * stokes.τ_o.yy./(G.*dt)
            # εxy_ve  = @. stokes.ε.xy_c + 0.5 * stokes.τ_o.xy_c./(G.*dt)
            # εII_ve  = sqrt.(0.5.*(εxx_ve.^2 .+ εyy_ve.^2) .+ εxy_ve.^2)
            # # stress increments
            # dτxx    = (.-(stokes.τ.xx .- stokes.τ_o.xx)./(G.*dt) .- stokes.τ.xx./η .+ 2.0.*stokes.ε.xx).*η_veτ
            # dτyy    = (.-(stokes.τ.yy .- stokes.τ_o.yy)./(G.*dt) .- stokes.τ.yy./η .+ 2.0.*stokes.ε.yy).*η_veτ
            # dτxy    = (.-(stokes.τ.xy_c .- stokes.τ_o.xy_c)./(G.*dt) .- stokes.τ.xy_c./η .+ 2.0.*stokes.ε.xy_c).*η_veτ
            # stokes.τ.II .= sqrt.(0.5.*((stokes.τ.xx.+dτxx).^2 .+ (stokes.τ.yy.+dτyy).^2) .+ (stokes.τ.xy_c.+dτxy).^2)
            # # yield function
            # F       = stokes.τ.II .- τ_y .- stokes.P.*sin(ϕ)
            # λ       = max.(F,0.0)./(η_veτ .+ η_reg)
            # dQdτxx  = 0.5.*(stokes.τ.xx.+dτxx)./stokes.τ.II
            # dQdτyy  = 0.5.*(stokes.τ.yy.+dτyy)./stokes.τ.II
            # dQdτxy  =      (stokes.τ.xy_c.+dτxy)./stokes.τ.II
            # stokes.τ.xx   .+= (.-(stokes.τ.xx .- stokes.τ_o.xx)./(G.*dt) .- stokes.τ.xx./η .+ 2.0.*(stokes.ε.xx .-      λ.*dQdτxx)).*η_veτ
            # stokes.τ.yy   .+= (.-(stokes.τ.yy .- stokes.τ_o.yy)./(G.*dt) .- stokes.τ.yy./η .+ 2.0.*(stokes.ε.yy .-      λ.*dQdτyy)).*η_veτ
            # stokes.τ.xy_c .+= (.-(stokes.τ.xy_c .- stokes.τ_o.xy_c)./(G.*dt) .- stokes.τ.xy_c./η .+ 2.0.*(stokes.ε.xy_c .- 0.5.*λ.*dQdτxy)).*η_veτ
            # # τxyv[2:end-1,2:end-1] .= ameanxy(τxy)
            # stokes.τ.II   .= sqrt.(0.5.*(stokes.τ.xx.^2 .+ stokes.τ.yy.^2) .+ stokes.τ.xy_c.^2)
            # # η_vep .= τII ./ 2.0 ./ εII_ve

            center2vertex!(stokes.τ.xy, stokes.τ.xy_c)

            # c2v!(Pv,   stokes.P)
            # c2v!(εxxv, stokes.ε.xx)
            # c2v!(εyyv, stokes.ε.yy)
            # c2v!(Gdτv, Gdτ)


            # # visco-elastic strain rates
            # εxxv_ve .= εxxv .+ 0.5.*τxxv_old./(Gv.*dt)
            # εyyv_ve .= εyyv .+ 0.5.*τyyv_old./(Gv.*dt)
            # εxyv_ve .= stokes.ε.xy .+ 0.5.*τxyv_old./(Gv.*dt)
            # # stress increments
            # dτxxv   .= (.-(τxxv .- τxxv_old)./(Gv.*dt) .- τxxv./ηv .+ 2.0.*εxxv).*ηv_veτ
            # dτyyv   .= (.-(τyyv .- τyyv_old)./(Gv.*dt) .- τyyv./ηv .+ 2.0.*εyyv).*ηv_veτ
            # dτxyv   .= (.-(τxyv .- τxyv_old)./(Gv.*dt) .- τxyv./ηv .+ 2.0.*stokes.ε.xy).*ηv_veτ
            # τIIv    .= sqrt.(0.5.*((τxxv.+dτxxv).^2 .+ (τyyv.+dτyyv).^2) .+ (τxyv.+dτxyv).^2)
            # # yield function
            # Fv      .= τIIv .- τ_y .- Pv.*sin(ϕ)
            # λv      .= max.(Fv,0.0)./(ηv_veτ .+ η_reg)
            # dQdτxyv .=  (τxyv.+dτxyv)./τIIv
            # stokes.τ.xy .+= (.-(τxyv .- τxyv_old)./(Gv.*dt) .- τxyv./ηv .+ 2.0.*(stokes.ε.xy .- 0.5.*λv.*dQdτxyv)).*ηv_veτ

            @parallel JR.compute_V!(
                @velocity(stokes)...,
                stokes.P,
                @stress(stokes)...,
                ηdτ,
                ρg...,
                ητ,
                _di...,
            )
            flow_bcs!(stokes, flow_bcs)

        iter += 1

        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) JR.compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
            )
            errs = (
                JR.norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) / length(stokes.R.Rx),
                JR.norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) / length(stokes.R.Ry),
                JR.norm_mpi(stokes.R.RP) / length(stokes.R.RP),
            )
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = JR.maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)

            if igg.me == 0 #&& ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ && iter ≥ 20000
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    @parallel (@idx ni .+ 1) JR.multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni)  JR.multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    # heatmap(stokes.viscosity.η_vep, colormap=:lipari)
    # heatmap(stokes.τ.xy, colormap=:lipari)
    # heatmap(stokes.τ.xy_c, colormap=:lipari)
    # heatmap(stokes.τ_o.xy_c, colormap=:lipari)
    # heatmap(stokes.EII_pl, colormap=:lipari)

    tensor_invariant!(stokes.ε)
    push!(τII, maximum(stokes.τ.xx))
    it += 1
    t  += dt
    # push!(sol, solution(εbg, t, G0, η0))
    push!(ttot, t)
    println("it = $it; t = $t \n")
    # visualisation
    th    = 0:pi/50:3*pi;
    xunit = @. radius * cos(th) + 0.5;
    yunit = @. radius * sin(th) + 0.5;
    fig   = Figure(size = (1600, 1600), title = "t = $t")
    ax1   = Axis(fig[1,1], aspect = 1, title = L"\tau_{II}", titlesize=35)
    # ax2   = Axis(fig[2,1], aspect = 1, title = "η_vep")
    ax2   = Axis(fig[2,1], aspect = 1, title = L"E_{II}", titlesize=35)
    ax3   = Axis(fig[1,2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize=35)
    ax4   = Axis(fig[2,2], aspect = 1)
    # heatmap!(ax1, xci..., Array(stokes.τ.II) , colormap=:lipari)
    heatmap!(ax1, xci..., Array(stokes.τ.xy) , colormap=:lipari)
    # heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
    heatmap!(ax2, xci..., Array(log10.(stokes.EII_pl)) , colormap=:lipari)
    heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)) , colormap=:lipari)
    lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
    lines!(ax4, ttot, τII, color = :black)
    # lines!(ax4, ttot, sol, color = :red)
    hidexdecorations!(ax1)
    hidexdecorations!(ax3)
    save(joinpath(@__DIR__, "$(it).png"), fig)
end

end

main(igg)
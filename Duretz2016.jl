using Printf
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax2D as JR
const backend_JR = CPUBackend

using JustPIC, JustPIC._2D
const backend = JustPIC.CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

# Load script dependencies
using LinearAlgebra, GeoParams, GLMakie

include("mask.jl")
include("MiniKernels.jl")

# Velocity helper grids for the particle advection
function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end
    @parallel f_x(A, B)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

topo_fun(x, r) =  -√(r^2 - x^2)

function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        r=0.5

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x     = @index px[ip, i, j]
            depth = (@index py[ip, i, j])

            h =  -√(r^2 - x^2)
            if depth ≤ h
                @index phases[ip, i, j] = 2.0
            else 
                @index phases[ip, i, j] = 1.0
            end


        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n        = 101
nx       = n
ny       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
# function main(igg, nx, ny)

    # Physical domain ------------------------------------
    thick_air    = 0             # thickness of sticky air layer
    ly           = 0.5           # domain length in y
    lx           = 0.5           # domain length in x
    ni           = nx, ny        # number of cells
    li           = lx, ly        # domain length in x- and y-
    di           = @. li / ni    # grid step in x- and -y
    origin       = -0.25, -0.75  # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    L = 0.5
    H = 0.5
    r = 0.5
    
    # Physical properties using GeoParams ----------------
    rheology     = rheology = (
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=1),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1),)),
            Gravity           = ConstantGravity(; g=1),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ=1),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1),)),
            Gravity           = ConstantGravity(; g=1),
        ),
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 15
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    init_phases!(pPhases, particles)
    phase_ratios  = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # RockRatios
    air_phase = 1
    ϕ         = RockRatio(ni...)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.95 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend_JR, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg               = @zeros(ni...), @zeros(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    # @parallel init_P!(stokes.P, ρg[2], xci[2])
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right = true, top = true, bot = true),
        free_surface = true
    )

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    figdir = "Duretz2016"
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    dt = 1e3 * (3600 * 24 * 365.25)
    
    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)
    iterMax          =        15e3
    nout             =         1e3
    viscosity_cutoff = (-Inf, Inf)
    free_surface     =       false
    ητ = @zeros(ni...)
    # while it < 1

        ## variational solver
       
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
        wtime0 = 0.0
        relλ = 0.2
        θ = deepcopy(stokes.P)
        λ = @zeros(ni...)
        λv = @zeros(ni .+ 1...)
        η0 = deepcopy(η)
        do_visc = true

        for Aij in @tensor_center(stokes.ε_pl)
            Aij .= 0.0
        end
        # Vx_on_Vy = @zeros(size(stokes.V.Vy))

        # compute buoyancy forces and viscosity
        compute_ρg!(ρg[end], phase_ratios, rheology, args)
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

        while iter ≤ iterMax
            err < ϵ && break
                # for _ in 1:100
                JR.compute_maxloc!(ητ, η; window=(1, 1))
                # update_halo!(ητ)
        
                @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di)
        
                @parallel (@idx ni)  compute_P!(
                    θ, 
                    stokes.P0, 
                    stokes.R.RP, 
                    stokes.∇V, 
                    ητ, 
                    rheology,
                    phase_ratios.center,
                    ϕ,
                    dt,
                    pt_stokes.r,
                    pt_stokes.θ_dτ
                )
        
                JR.update_ρg!(ρg[2], phase_ratios, rheology, args)
        
                @parallel (@idx ni .+ 1) JR.compute_strain_rate!(
                    @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
                )
        
                # if rem(iter, nout) == 0
                #     @copy η0 η
                # end
                # if do_visc
                # update_viscosity!(
                #     stokes,
                #     phase_ratios,
                #     args,
                #     rheology,
                #     viscosity_cutoff;
                #     relaxation=viscosity_relaxation,
                # )
                # end
        
                @parallel (@idx ni .+ 1) update_stresses_center_vertex_ps!(
                    @strain(stokes),
                    @tensor_center(stokes.ε_pl),
                    stokes.EII_pl,
                    @tensor_center(stokes.τ),
                    (stokes.τ.xy,),
                    @tensor_center(stokes.τ_o),
                    (stokes.τ_o.xy,),
                    θ,
                    stokes.P,
                    stokes.viscosity.η,
                    λ,
                    λv,
                    stokes.τ.II,
                    stokes.viscosity.η_vep,
                    relλ,
                    dt,
                    θ_dτ,
                    rheology,
                    phase_ratios.center,
                    phase_ratios.vertex,
                    ϕ,
                )
                # update_halo!(stokes.τ.xy)
        
                # @parallel (1:(size(stokes.V.Vy, 1) - 2), 1:size(stokes.V.Vy, 2)) JR.interp_Vx∂ρ∂x_on_Vy!(
                #     Vx_on_Vy, stokes.V.Vx, ρg[2], _di[1]
                # )
        
                # @hide_communication b_width begin # communication/computation overlap
                    @parallel (@idx ni.+1) compute_V!(
                        @velocity(stokes)...,
                        stokes.R.Rx,
                        stokes.R.Ry,
                        stokes.P,
                        @stress(stokes)...,
                        ηdτ,
                        ρg...,
                        ητ,
                        ϕ,
                        # ϕ.Vx,
                        # ϕ.Vy,
                        _di...,
                    )
                    # apply boundary conditions
                    # velocity2displacement!(stokes, dt)
                    # JR.free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
                    flow_bcs!(stokes, flow_bcs)
                # end
                    # f,ax,h=heatmap(stokes.V.Vy)
                    # # f,ax,h=heatmap(stokes.V.Vx)
                    # Colorbar(f[1,2], h, label="Vy"); f
                #     update_halo!(@velocity(stokes)...)
                # end
        
            iter += 1
        
            if iter % nout == 0 && iter > 1
                # er_η = norm_mpi(@.(log10(η) - log10(η0)))
                # er_η < 1e-3 && (do_visc = false)
                # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
                errs = (
                    norm(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) / length(stokes.R.Rx),
                    norm(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) / length(stokes.R.Ry),
                    norm(stokes.R.RP) / length(stokes.R.RP),
                )
                push!(norm_Rx, errs[1])
                push!(norm_Ry, errs[2])
                push!(norm_∇V, errs[3])
                err = maximum(errs)
                push!(err_evo1, err)
                push!(err_evo2, iter)
        
                # if igg.me == 0 #&& ((verbose && err > ϵ) || iter == iterMax)
                    @printf(
                        "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                        iter,
                        err,
                        norm_Rx[end],
                        norm_Ry[end],
                        norm_∇V[end]
                    )
                # end
                isnan(err) && error("NaN(s)")
                isinf(err) && error("Inf(s)")
            end
        end
        
        dt = compute_dt(stokes, di) / 2
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ, phase_ratios, air_phase)

        @show it += 1
        t        += dt

        if it == 1 || rem(it, 1) == 0
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 5
            fig = Figure(size = (900, 900), title = "t = $t")
            ax  = Axis(fig[1,1], aspect = 1, title = " t=$(round.(t/(1e3 * 3600 * 24 *365.25); digits=3)) Kyrs")
            heatmap!(ax, xci[1].*1e-3, xci[2].*1e-3, Array([argmax(p) for p in phase_ratios.vertex]), colormap = :grayC)
            # arrows!(
            #     ax,
            #     xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))...,
            #     lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
            #     color = :red,
            # )
            fig
            save(joinpath(figdir, "$(it).png"), fig)

        end
    # end
    return nothing
# end
# ## END OF MAIN SCRIPT ----------------------------------------------------------------
main(igg, nx, ny)

# # # (Path)/folder where output data and figures are stored
# # n        = 100
# # nx       = n
# # ny       = n
# # igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
# #     IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
# # else
# #     igg
# # end


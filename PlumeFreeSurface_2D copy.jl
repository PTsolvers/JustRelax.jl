using JustRelax, JustRelax.DataIO
import JustRelax.@cell

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

using JustPIC, JustPIC._2D
const backend = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays

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

@parallel function smooth!(
    A2::AbstractArray{T,2}, A::AbstractArray{T,2}, fact::Real
) where {T}
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

# Initial pressure profile - not accurate
# @parallel function init_P!(P, ρg, z)
#     @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
#     return nothing
# end

@parallel_indices (i, j) function init_P!(P, ρg, z)
    P[i, j] = sum(abs(ρg[i, jj] * z[jj]) * z[jj] < 0.0 for jj in j:size(P, 2))
    return nothing
end

function init_phases!(phases, particles)
    ni = size(phases)
    
    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        r = 25e3
        
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j]) 
            
            if 0e0 ≤ depth ≤ 50e3
                @cell phases[ip, i, j] = 1.0

            else 
                @cell phases[ip, i, j] = 2.0
                
                if ((x - 125e3)^2 + (depth - 100e3)^2 ≤ r^2)
                    @cell phases[ip, i, j] = 3.0
                end
            end

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny)

    # Physical domain ------------------------------------
    thick_air    = 50e3             # thickness of sticky air layer
    ly           = 200e3 + thick_air # domain length in y
    lx           = 250e3             # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = rheology = (
        # Name              = "Air",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=1e1),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e17),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ=3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Plume",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e18),)),
            Gravity           = ConstantGravity(; g=9.81),
        )
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 15
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly 
    init_phases!(pPhases, particles)
    phase_ratios  = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 1 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    # ----------------------------------------------------
   
    # Buoyancy forces & rheology
    ρg               = @zeros(ni...), @zeros(ni...)
    η                = @ones(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = 1e3)
    @parallel (@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (1e16, Inf)
    )
    η_vep            = copy(η)
    
    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(; 
        free_slip    = (left = true, right=true, top=true, bot=true),
        # no_slip      = (left = false, right=false, top=false, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    # for _ in 1:10
    #     @parallel smooth!(η_vep, η, 50)
    #     η_vep, η = η, η_vep
    #     @views η[1,:]    .= η[2,:]
    #     @views η[end,:]  .= η[end-1,:]
    #     @views η[:, 1]   .= η[:, 2]
    #     @views η[:, end] .= η[:, end-1]
    # end

    # Plot initial T and η profiles
    let
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(ρg[2][:]./9.81), Y./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        # scatter!(ax2, Array(stokes.P[:]), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        fig
    end

    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    figdir = "FreeSurfacePlume"
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    dt = dt_max = 10e3 * (3600 * 24 *365.25)
    while it < 1 # run only for 5 Myrs

        # Stokes solver ----------------
        args = (; T = thermal.Tc, P = stokes.P,  dt=dt)
        
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
        )        

        # for _ in 1:50
        #     @parallel smooth!(η_vep, η, 2)
        #     η_vep, η = η, η_vep
        #     @views η[1,:]    .= η[2,:]
        #     @views η[end,:]  .= η[end-1,:]
        #     @views η[:, 1]   .= η[:, 2]
        #     @views η[:, end] .= η[:, end-1]
        # end

        η .= mean(η)
        # η .= 1e21

        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            iterMax = 70e3,
            nout=1e3,
            viscosity_cutoff=(-Inf, Inf)
        )
        dt = min(compute_dt(stokes, di), dt_max) #/2
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)        
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        
        @show it += 1
        t        += dt

        if it == 1 || rem(it, 1) == 0
            JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 2
            fig = Figure(size = (900, 900), title = "t = $t")
            ax  = Axis(fig[1,1], aspect = 1, title = " t=$(t/(1e3 * 3600 * 24 *365.25)) Kyrs")
            heatmap!(ax, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η)), colormap = :grayC)
            arrows!(
                ax,
                xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))..., 
                lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
                color = :red,
            )
            fig
            save(joinpath(figdir, "$(it).png"), fig)

            let
                Y   = [y for x in xci[1], y in xci[2]][:]
                fig = Figure(size = (1200, 900))
                ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
                ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
                scatter!(ax1, Array(ρg[2][:]./9.81), Y./1e3)
                scatter!(ax2, Array(log10.(η[:])), Y./1e3)
                # scatter!(ax2, Array(stokes.P[:]), Y./1e3)
                ylims!(ax1, minimum(xvi[2])./1e3, 0)
                ylims!(ax2, minimum(xvi[2])./1e3, 0)
                hideydecorations!(ax2)
                save(joinpath(figdir, "profile_$(it).png"), fig)
                fig
            end

        end
    end
    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n        = 100
nx       = n
ny       = n
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

main(igg, nx, ny)

# @inline function f1(x_new::T, x_old::T, ν) where {T}
#     x_cont = exp((1 - ν) * log(x_old) + ν * log(x_new))
#     return isnan(x_cont) ? 0.0 : x_cont
# end
# @inline f2(x_new, x_old, ν) = (1 - ν) * x_old + ν * x_new

# n=100
# x0 = vcat(zeros(n>>>1), fill(100,n>>>1))
# x1 = LinRange(0, 100, n)
# x2 = LinRange(0, 100, n)
# lines(x0)
# lines!(f1.(x0, x1, 1e-1))
# lines!(f2.(x0, x1, 1e-1))
# xn = x1
# # for i in 1:10
# #     # xn = f1.(x0, xn, 0.5)
# #     xn = f2.(x0, xn, 0.5)
# #     lines!(xn)
# # end

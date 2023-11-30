using CUDA
CUDA.allowscalar(false)

using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model  = PS_Setup(:CUDA, Float64, 2)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
# Initialize phases on the particles
function init_phases!(phase_ratios, x_inc, y_inc, r_inc, phase_inc)
    ni      = size(phase_ratios.center)
    
    @parallel_indices (i, j) function init_phases!(phases, xc, yc, x_inc, y_inc, r_inc, phase_inc)
        x, y = xc[i], yc[j]

        inlcusion_id = isin_inclusion(x, y, x_inc, y_inc, r_inc)

        if inlcusion_id == 0
            JustRelax.@cell phases[1, i, j] = 1.0 # matrix
            JustRelax.@cell phases[2, i, j] = 0.0
            JustRelax.@cell phases[3, i, j] = 0.0
        
        else

            p = phase_inc[inlcusion_id]
            if p == 1
                JustRelax.@cell phases[1, i, j] = 0.0
                JustRelax.@cell phases[2, i, j] = 1.0
                JustRelax.@cell phases[3, i, j] = 0.0
            elseif p == 2
                JustRelax.@cell phases[1, i, j] = 0.0
                JustRelax.@cell phases[2, i, j] = 0.0
                JustRelax.@cell phases[3, i, j] = 1.0
            end

        end

        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phase_ratios.center, xci..., x_inc, y_inc, r_inc, phase_inc)
end

function isin_inclusion(x, y, x_inc, y_inc, r_inc)
    for inc in eachindex(x_inc)
        if (x-x_inc[inc])^2 + (y-y_inc[inc])^2 < r_inc[inc]^2
            return inc
        end
    end
    return 0
end

@parallel function smooth!(
    A2::AbstractArray{T,2}, A::AbstractArray{T,2}, fact::Real
) where {T}
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

function main(igg; nx=64, ny=64, figdir="model_figs")

    # Physical domain ------------------------------------
    ly       = 1e0          # domain length in y
    lx       = ly           # domain length in x
    ni       = nx, ny       # number of cells
    li       = lx, ly       # domain length in x- and y-
    di       = @. li / ni   # grid step in x- and -y
    origin   = 0.0, 0.0     # origin coordinates
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    dt       = Inf 

    # Physical properties using GeoParams ----------------
    εbg     = 1.0 # background strain-rate
    dt      = 1.0
    rheology = (
        # matrix
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1), )),
        ),
        # strong balls
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e4), )),
        ),
        # weak balls
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e-4), )),
        ),
    )

    # ηi    = (s=1e4, w=1e-4) 
    x_inc = @. (0.0 ,  0.2, -0.3, -0.4,  0.0, -0.3, 0.4 , 0.3 , 0.35, -0.1) * 1 + lx/2
    y_inc = @. (0.0 ,  0.4,  0.4, -0.3, -0.2,  0.2, -0.2, -0.4, 0.2 , -0.4) * 1 + ly/2
    r_inc = @. (0.08, 0.09, 0.05, 0.08, 0.08,  0.1, 0.07, 0.08, 0.07, 0.07) * 1
    # η_inc = (ηi.s, ηi.w, ηi.w, ηi.s, ηi.w, ηi.s, ηi.w, ηi.s, ηi.s, ηi.w)
    phase_inc = (1, 2, 2, 1, 2, 1, 2, 1, 1, 2)

    # Initialize phase ratios -------------------------------
    radius       = 0.1
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, x_inc, y_inc, r_inc, phase_inc)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.95 / √2.1)

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)
    
    # Rheology
    η         = @ones(ni...)
    η_vep     = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args, rheology, (-Inf, Inf)
    )

    for ism in 1:1 # *********** nsm --> 1
        @parallel smooth!(η_vep, η, 1.0)
        η_vep, η =  η, η_vep
    end

    # Boundary conditions
    flow_bcs     = FlowBoundaryConditions(; 
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    stokes.V.Vx .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    
    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    take(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it      = 0.0, 0
    tmax       = 3.5
    τII        = Float64[]
    sol        = Float64[]
    ttot       = Float64[]

    while t < tmax

        # Stokes solver ----------------
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
            Inf,
            igg;
            verbose          = true,
            iterMax          = 150e3,
            nout             = 1e3,
            viscosity_cutoff = (-Inf, Inf)
        )
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # visualisation
        th    = 0:pi/50:3*pi;
        xunit = @. radius * cos(th) + 0.5;
        yunit = @. radius * sin(th) + 0.5;

        fig   = Figure(resolution = (1600, 1600), title = "t = $t")
        ax1   = Axis(fig[1,1], aspect = 1, title = "τII")
        ax2   = Axis(fig[2,1], aspect = 1, title = "η_vep")
        ax3   = Axis(fig[1,2], aspect = 1, title = "log10(εII)")
        ax4   = Axis(fig[2,2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(log10.(η_vep)) , colormap=:batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)) , colormap=:batlow)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black) 
        lines!(ax4, ttot, sol, color = :red) 
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)

    end

    return nothing
end

N      = 128
n      = N + 2
nx     = n - 2
ny     = n - 2
figdir = "ShearBands2D"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
else
    igg
end
# main(igg; figdir = figdir, nx = nx, ny = ny);


# p = [argmax(p)   for p in phase_ratios.center]
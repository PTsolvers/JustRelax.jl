using CUDA
CUDA.allowscalar(false)

using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model  = PS_Setup(:CUDA, Float64, 2)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
# Initialize phases on the particles
function init_phases!(phase_ratios, xci, x_inc, y_inc, r_inc, phase_inc)
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
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e2), )),
        ),
        # weak balls
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e-2), )),
        ),
    )

    # ηi    = (s=1e4, w=1e-4) 
    x_inc = @. (0.0 ,  0.2, -0.3, -0.4,  0.0, -0.3, 0.4 , 0.3 , 0.35, -0.1) * 1 + lx/2
    y_inc = @. (0.0 ,  0.4,  0.4, -0.3, -0.2,  0.2, -0.2, -0.4, 0.2 , -0.4) * 1 + ly/2
    r_inc = @. (0.08, 0.09, 0.05, 0.08, 0.08,  0.1, 0.07, 0.08, 0.07, 0.07) * 1
    phase_inc = (1, 2, 2, 1, 2, 1, 2, 1, 1, 2)

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, xci, x_inc, y_inc, r_inc, phase_inc)
    
    phase_ratios_v = PhaseRatio(ni.+1, length(rheology))
    init_phases!(phase_ratios_v, xvi, x_inc, y_inc, r_inc, phase_inc)
    pv = [argmax(p) for p in Array(phase_ratios_v.center)]
    ηv = @ones(ni .+ 1)
    @views ηv[pv .== 2] .= 1e2
    @views ηv[pv .== 3] .= 1e-2

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-7,  CFL = 0.9 / √2.1)

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)
    
    # Rheology
    η         = @ones(ni...)
    # @parallel (@idx ni) compute_viscosity!(
    #     η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args, rheology, (-Inf, Inf)
    # )
    η_vep     = similar(η) # effective visco-elasto-plastic viscosity
    @parallel vertex2center!(η, ηv)


    # Boundary conditions
    flow_bcs     = FlowBoundaryConditions(; 
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    stokes.V.Vx .= PTArray([-(x-lx/2)*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([ (y-ly/2)*εbg for _ in 1:nx+2, y in xvi[2]])
    
    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    take(figdir)
    # ----------------------------------------------------

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
        iterMax          = 500e3,
        nout             = 1e3,
        viscosity_cutoff = (-Inf, Inf)
    )
    @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)

    # visualisation
    fig   = Figure(resolution = (1600, 900))
    ax1   = Axis(fig[1,1], aspect = 1, title = "Pressure")
    ax2   = Axis(fig[1,3], aspect = 1, title = "η")
    h1 = heatmap!(ax1, xci..., Array(stokes.P)  , colormap=:inferno, colorrange=(-3, 3))
    h2 = heatmap!(ax2, xci..., Array(log10.(η)) , colormap=:grayC)
    Colorbar(fig[1, 2], h1, height = Relative(0.75))
    Colorbar(fig[1, 4], h2, height = Relative(0.75))
    fig
    # save(joinpath(figdir, "$(it).png"), fig)

end

n      = 160
nx     = n
ny     = n
figdir = "Inclusions2D"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
else
    igg
end
fig = main(igg; figdir = figdir, nx = nx, ny = ny)


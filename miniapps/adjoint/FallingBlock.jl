#using Pkg
#Pkg.offline()
#Pkg.activate("FallingBlock")
using Enzyme

const isCUDA = false
#const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.DataIO
# using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using JustRelax.JustRelax2D_AD

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, CellArrays, CairoMakie

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, xc, yc, r)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            y = @index py[ip, i, j]
            depth = -(@index py[ip, i, j])
            # plume - rectangular

            @index phases[ip, i, j] = if ((x -xc)^2 ≤ r^2) && ((depth - yc)^2 ≤ r^2)
                2.0
            elseif (y >= 0.3)
                3.0
            else
                1.0
            end


        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, xc, yc, r)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg)*abs(@all_j(z))
    return nothing
end

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function sinking_block2D(igg; ar=2, ny=8, nx=ny*4, figdir="figs2D")

    # Physical domain ------------------------------------
    ly           = 1.0
    lx           = 2.0
    origin       = -1.0, -0.5                         # origin coordinates
    ni           = nx, ny                           # number of cells
    li           = lx, ly                           # domain length in x- and y-
    di           = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = (
        SetMaterialParams(;
            Name              = "Matrix",
            Phase             = 1,
            Density           = ConstantDensity(; ρ=1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), ConstantElasticity(G=1.0, ν=0.5) )),
            Gravity           = ConstantGravity(; g=1.0),
        ),
        SetMaterialParams(;
            Name              = "Block",
            Phase             = 2,
            Density           = ConstantDensity(; ρ=1.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1000.0), ConstantElasticity(G=1.0, ν=0.5) )),
            Gravity           = ConstantGravity(; g=1.0),
        ),
        SetMaterialParams(;
            Name              = "StickyAir",
            Phase             = 3,
            Density           = ConstantDensity(; ρ=0.1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 0.1), ConstantElasticity(G=1.0, ν=0.5) )),
            Gravity           = ConstantGravity(; g=1.0),
    )
)
    # heat diffusivity
    dt = 1
    # ----------------------------------------------------
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 12
        particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    # temperature
    pPhases,      = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases, )
    # Rectangular density anomaly
    xc_anomaly   =  0.0  # x origin of block
    yc_anomaly   =  0.0  # y origin of block
    r_anomaly    =  0.1  # half-width of block
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    #phase_ratios_vertex = PhaseRatio(backend, ni.+1, length(rheology))
    init_phases!(pPhases, particles, xc_anomaly, abs(yc_anomaly), r_anomaly)
    phase_ratios_center!(phase_ratios, particles, xci, pPhases)
    phase_ratios_vertex!(phase_ratios, particles, xvi, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    stokesAD  = StokesArraysAdjoint(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; Re=0.5, ϵ=1e-5,  CFL = 0.95 / √2.1)
    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=@ones(ni...), P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # ----------------------------------------------------

    # Viscosity
    args     = (; dt = dt, ΔTc = @zeros(ni...))
    η_cutoff = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # ----------------------------------------------------

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip    = (left =  true, right =  true, top =  true, bot =  true),
        no_slip      = (left =  false, right =  false, top =  false, bot =  false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    plottingInt = 1  # plotting interval
    it = 1

    # IO ------------------------------------------------
    take(figdir)
    # ----------------------------------------------------
    # Stokes solver ----------------
    args = (; T = @ones(ni...), P = stokes.P, dt=dt, ΔTc = @zeros(ni...))
    test = adjoint_solve!(
        stokes,
        stokesAD,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratios,
        rheology,
        args,
        dt,
        it, #Glit
        igg;
        kwargs = (
            grid,
            origin,
            li,
            iterMax=150e3,
            nout=1e3,
            viscosity_cutoff = η_cutoff,
            verbose = false,
            ADout=plottingInt
        )
    );

    Vx_v    = @zeros(ni.+1...)
    Vy_v    = @zeros(ni.+1...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    velocity = @. √(Vx_v^2 + Vy_v^2 )
    (; η_vep, η) = stokes.viscosity

    Xc, Yc = meshgrid(xci[1], xci[2])


    ind = findall(xci[2] .≤ 0.29)
    #test.ηb[ind] = NaN
    #test.ρb[ind] = NaN

    # Plotting ---------------------
    ar  = 2
    fig = Figure(size = (1200, 900), title = "Falling Block")
    ax1 = Axis(fig[1,1], aspect = ar, title = "Density")
    ax2 = Axis(fig[1,2], aspect = ar, title = "Viscosity")
    ax3 = Axis(fig[2,1],aspect = ar, title = "Vx")
    ax4 = Axis(fig[2,2],aspect = ar,  title = "Vy")
    ax5 = Axis(fig[3,1],aspect = ar,  title = "sensitivity eta")
    ax6 = Axis(fig[3,2],aspect = ar,  title = "sensitivity rho")
    ax7 = Axis(fig[4,1],aspect = ar,  title = "sensitivity G")
    ax8 = Axis(fig[4,2],aspect = ar,  title = "sensitivity fr")

    h1  = heatmap!(ax1, xci[1], xci[2], Array(ρg[2]))
    h2  = heatmap!(ax2, xci[1], xci[2], Array(log.(η)))
    h3  = heatmap!(ax3, xvi[1], xvi[2], Array(Vx_v), colormap=:vikO)
    h4  = heatmap!(ax4, xvi[1], xvi[2], Array(Vy_v), colormap=:vikO)
    h5  = heatmap!(ax5, xci[1], xci[2][ind], Array(test.ηb)[:,ind])
    h6  = heatmap!(ax6, xci[1], xci[2][ind], Array(test.ρb)[:,ind])
    h7  = heatmap!(ax7, xci[1], xci[2][ind], Array(stokesAD.G)[:,ind])
    h8  = heatmap!(ax8, xci[1], xci[2][ind], Array(stokesAD.fr)[:,ind])
    hidexdecorations!(ax1)
    hidexdecorations!(ax2)
    hidexdecorations!(ax3)
    hidexdecorations!(ax4)
    hidexdecorations!(ax5)
    hidexdecorations!(ax6)
    hidexdecorations!(ax7)
    hidexdecorations!(ax8)
    Colorbar(fig[1,1][1,2], h1)
    Colorbar(fig[1,2][1,2], h2)
    Colorbar(fig[2,1][1,2], h3)
    Colorbar(fig[2,2][1,2], h4)
    Colorbar(fig[3,1][1,2], h5)
    Colorbar(fig[3,2][1,2], h6)
    Colorbar(fig[4,1][1,2], h7)
    Colorbar(fig[4,2][1,2], h8)
    linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)
    #CUDA.allowscalar() do
        scatter!(ax2, Xc[16:end-15,end-8], Yc[16:end-15,end-8], color=:red, markersize=10)
    #end
    #display(fig)
    fig
    #checkpoint = joinpath(figdir, "checkpoint")
    save(joinpath(figdir, "AdjointOutput.png"), fig)
    # ------------------------------

#=
    if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, basename(@__FILE__))
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
    end
=#
return stokes, stokesAD, xci, phase_ratios

end



figdir = "FallingBlock_Adjoint"
ar     = 2 # aspect ratio
n      = 32
nx     = n*ar - 2
ny     = n - 2
igg    = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end
stokes,stokesAD,xci = sinking_block2D(igg; ar=ar, nx=nx, ny=ny, figdir=figdir);

#fig = Figure(size = (1200, 900));
#ax1 = Axis(fig[1,1])
#h1 = heatmap!(ax1, xci[1], xci[2], Array(eta))
#Colorbar(fig[1,2], h1)
#fig

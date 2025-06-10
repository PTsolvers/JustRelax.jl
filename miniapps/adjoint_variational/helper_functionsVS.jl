const isCUDA = false
@static if isCUDA
    using CUDA
end
using JustRelax, JustRelax.JustRelax2D_AD, JustRelax.DataIO
using GeoParams, CairoMakie, CellArrays, JLD2
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
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using Enzyme, KahanSummation, LinearAlgebra

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function init_phasesFD!(phase_ratios, xci, xvi, radius, FDxmin, FDxmax, FDymin, FDymax,di)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius, FDxmin, FDxmax, FDymin, FDymax,di)
        x, y = xc[i], yc[j]

            
        if (x <= FDxmax) && (x >= FDxmin) && (y <= FDymax) && (y >= FDymin) && ((x-o_x)^2 ≤ radius^2) && ((y-o_y)^2 ≤ radius^2)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 0.0
            @index phases[4, i, j] = 1.0
        elseif (x <= FDxmax) && (x >= FDxmin) && (y <= FDymax) && (y >= FDymin) 
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 1.0
            @index phases[4, i, j] = 0.0   
        elseif ((x-o_x)^2 ≤ radius^2) && ((y-o_y)^2 ≤ radius^2) #((x-o_x)^2 + (y-o_y)^2) ≤ radius^2
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0    # inclusion
            @index phases[3, i, j] = 0.0
            @index phases[4, i, j] = 0.0  
        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 0.0
            @index phases[4, i, j] = 0.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius, FDxmin, FDxmax, FDymin, FDymax,di)
    @parallel (@idx ni.+1) init_phases!(phase_ratios.vertex, xvi..., origin..., radius, FDxmin, FDxmax, FDymin, FDymax,di)
    return nothing
end

function plot_forward_solve(stokes,xci,ρg,par)
            # visualisation
            fig   = Figure(size = (1600, 1600))
            ax1   = Axis(fig[1,1], aspect = 1, title = L"η", titlesize=35)
            ax2   = Axis(fig[1,2], aspect = 1, title = L"ρ", titlesize=35)
            ax3   = Axis(fig[2,1], aspect = 1, title = L"τ_{II}", titlesize=35)
            ax4   = Axis(fig[2,2], aspect = 1, title = L"ε_{II}", titlesize=35)
            ax5   = Axis(fig[3,1], aspect = 1, title = L"Vx", titlesize=35)
            ax6   = Axis(fig[3,2], aspect = 1, title = L"Vy", titlesize=35)
            h1 = heatmap!(ax1, xci..., Array(stokes.viscosity.η) , colormap=:batlow)
            h2 = heatmap!(ax2, xci..., Array(ρg[2]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci..., Array(stokes.τ.II) , colormap=:batlow)
            h4 = heatmap!(ax4, xci..., Array(log10.(stokes.ε.II)) , colormap=:batlow)
            h5 = heatmap!(ax5, xci..., Array(stokes.V.Vx) , colormap=:batlow)
            h6 = heatmap!(ax6, xci..., Array(stokes.V.Vy) , colormap=:batlow)
            #CUDA.allowscalar() do
            #scatter!(ax2, vec(Xc[SensInd[1].+1,SensInd[2]]), vec(Yc[SensInd[1].#+1,SensInd[2]]), color=:red, markersize=10)
            #end
            Colorbar(fig[1,1][1,2], h1)
            Colorbar(fig[1,2][1,2], h2)
            Colorbar(fig[2,1][1,2], h3)
            Colorbar(fig[2,2][1,2], h4)
            Colorbar(fig[3,1][1,2], h5)
            Colorbar(fig[3,2][1,2], h6)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            hidexdecorations!(ax4)
            save(joinpath(figdir, "$(par).png"), fig)
end

#### Plotting Comparison ####
function plot_FD_vs_AD(refcost,cost,dp,Sens,nx,ny,ηref,ρref,stokesAD,figdir,f, Adjoint, Ref, run_param)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cellsvertices of the cells
    Xc, Yc = meshgrid(xci[1], xci[2])

    #ind = findall(xci[2] .≤ 0.29)

    o_x   =  0.5  # x origin of block
    o_y   =  0.5  # y origin of block
    r    =  1*di[1]*f  # half-width of block

    if maximum(cost.-refcost) == 0.0
        @warn("FD Sensitivites are zero")
        end
    
    if run_param
        sol_FD = Array(@zeros(nx,ny))
        sol_FD .= ((cost .- refcost)./dp)
    else
        sol_FD = Array(@zeros(nx,ny))
        sol_FD .= cost
    end
    #AD = deepcopy(stokesAD.G)
    AD = Sens# ./ refcost .* (di[1] * di[2]) 
    #AD_G .*= 2.0
    #AD_G[ind_block] ./= 2.0
    #AD_G[ind_block] .*= 0.5
    AD .= AD# ./ abs(refcost) #./(di[1] * di[2]) 

    #sol_FD .= sol_FD .* ηref ./refcost
    #sol_FD .= sol_FD .* ρref  ./refcost


    # scale the Sensitivities
    AD_norm = Array(AD .- (minimum(AD)))
    AD_norm .= AD_norm ./ maximum(AD_norm)

    sol_FD_norm = Array(sol_FD .- (minimum(sol_FD)))
    sol_FD_norm .= sol_FD_norm ./ maximum(sol_FD_norm)

    error = Array((sol_FD .- AD ./ sol_FD))
    error .= log10.(abs.(error))

    sumλVx = round(sum(abs.(stokesAD.VA.Vx)),digits=10)
    sumλVy = round(sum(abs.(stokesAD.VA.Vy)),digits=10)
    sumFD = round(sum(abs.(sol_FD)),digits=8)
    sumAD = round(sum(abs.(AD)),digits=8)
    fig = Figure(size = (720, 1000), title = "Compare Adjoint Sensitivities with Finite Difference Sensitivities",fontsize=16)
    ax1   = Axis(fig[1,1], aspect = 1, title = L"\tau_{II}")
    ax2   = Axis(fig[1,2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})")
    ax3   = Axis(fig[2,1], aspect = 1, title = L"Vx")
    ax4   = Axis(fig[2,2], aspect = 1, title = L"Vy")
    ax5   = Axis(fig[3,1], aspect = 1, title = "λ Vx sum(abs)=$sumλVx")
    ax6   = Axis(fig[3,2], aspect = 1, title = "λ Vy sum(abs)=$sumλVy")
    ax7   = Axis(fig[4,1], aspect = 1, title = "FD sum(abs)=$sumFD",titlesize=16)
    ax8   = Axis(fig[4,2], aspect = 1, title = "AD sum(abs)=$sumAD",titlesize=16)
    ax9   = Axis(fig[5,1], aspect = 1, title = "Error wrt FD",titlesize=16)
    ax10  = Axis(fig[5,2], aspect = 1, title = "Log10 Error",titlesize=16)
    h1 = heatmap!(ax1, xci..., Array(Ref.τ.II) , colormap=:managua)
    h2 = heatmap!(ax2, xci..., Array((Ref.ε.II)) , colormap=:managua)
    #h1 = heatmap!(ax1, xci..., Array(ρref) , colormap=:managua)
    #h2 = heatmap!(ax2, xci..., Array(log10.(ηref)) , colormap=:managua)
    #Vx_range = maximum(abs.(Ref.V.Vx))
    #Vy_range = maximum(abs.(Ref.V.Vy))
    h3  = heatmap!(ax3, xci[1], xci[2], Array(Ref.V.Vx),colormap=:roma)#,colorrange=(-Vx_range,Vx_range))
    h4  = heatmap!(ax4, xci[1], xci[2], Array(Ref.V.Vy),colormap=:roma)#,colorrange=(-Vy_range,Vy_range))
    #λVx_range = maximum(abs.(stokesAD.VA.Vx))
    #λVy_range = maximum(abs.(stokesAD.VA.Vy))
    h5  = heatmap!(ax5, xci[1], xci[2], Array(stokesAD.VA.Vx),colormap=:lipari)#, colorrange=(-λVx_range,λVx_range))
    h6  = heatmap!(ax6, xci[1], xci[2], Array(stokesAD.VA.Vy),colormap=:lipari)#, colorrange=(-λVy_range,λVy_range))
    h7  = heatmap!(ax7, xci[1], xci[2], Array(sol_FD),colormap=:lipari)
    h8  = heatmap!(ax8, xci[1], xci[2], Array(AD),colormap=:lipari)
    #h7  = heatmap!(ax7, xci[1], xci[2], Array(sol_FD_norm),colormap=:lipari)
    #h8  = heatmap!(ax8, xci[1], xci[2], Array(AD_norm),colormap=:lipari)
    h9  = heatmap!(ax9, xci[1], xci[2], Array(error),colormap=:jet)
    h10 = heatmap!(ax10, xci[1], xci[2], Array((log10.(abs.(Array(error))))),colormap=:jet)
    hidexdecorations!(ax1);hidexdecorations!(ax2);hidexdecorations!(ax3);hidexdecorations!(ax4);hidexdecorations!(ax5);hidexdecorations!(ax6);hidexdecorations!(ax7);hidexdecorations!(ax8) 
    hidexdecorations!(ax6);hideydecorations!(ax2);hideydecorations!(ax4);hideydecorations!(ax6);hideydecorations!(ax8)

    Colorbar(fig[1,1][1,2], h1, height=Relative(0.8))
    Colorbar(fig[1,2][1,2], h2, height=Relative(0.8))
    Colorbar(fig[2,1][1,2], h3, height=Relative(0.8))
    Colorbar(fig[2,2][1,2], h4, height=Relative(0.8))
    Colorbar(fig[3,1][1,2], h5, height=Relative(0.8))
    Colorbar(fig[3,2][1,2], h6, height=Relative(0.8))
    Colorbar(fig[4,1][1,2], h7, height=Relative(0.8))
    Colorbar(fig[4,2][1,2], h8, height=Relative(0.8))
    Colorbar(fig[5,1][1,2], h9, height=Relative(0.8))
    Colorbar(fig[5,2][1,2], h10, height=Relative(0.8))
    colsize!(fig.layout, 1, Aspect(1, 1.4))
    colsize!(fig.layout, 2, Aspect(1, 1.4))
    colgap!(fig.layout, 8)
    #rowgap!(fig.layout, 4.0)
    #linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)    
    save(joinpath(figdir, "Comparison$nx.png"), fig)
    sol_FD_cpu = Array(sol_FD)
    jldsave(joinpath(figdir, "FD_solution.jld2"),sol_FD_cpu=sol_FD_cpu)

    return sol_FD, AD
end

function hockeystick(refcost,refcostdot,plot_sens,dp,dM,FD,iter)

    #### Dot product test ####
    dirFD  = (refcostdot-refcost)/dp
    dirAD  = (sum(plot_sens .* dM)) # AD
    #FD1    = sum(FD[1] .* dM) # FD


    #hockeystick test
    epsilons  = collect(LinRange(-2.0, 12.0, iter))
    epsilons  = 10.0 .^ -epsilons
    FDs       = zeros(Float64,length(epsilons))
    run_param = false
    run_ref   = false
    error     = zeros(Float64,length(epsilons))
    #errorFD   = zeros(Float64,length(epsilons))

    j=1
    for (j,i) in enumerate(epsilons)
        dp=i
        refcostNot, cost, dpNot, Adjoint, ηref, ρref, stokesAD, stokesRef, ρg, refcostdot = main(igg; figdir = figdir, nx = nx, ny = ny,f,run_param, run_ref, dp, dM);
        FDs[j]     = (refcostdot-refcost)/dp
        #error[j]   = abs((dirAD-FDs[j]))
        error[j]   = abs(((FDs[j]/dirAD)-1.0))
        print(error[j],"##########################\n")
        print(j,"\n")
        print(error[j],"\n")
        print(error[j],"##########################\n")
        #errorFD[j] = abs((FD1-FDs[j])/dirAD)
        #errorFD[j]   = abs(((FDs[j]/FD1)-1.0))

    end

    eps = epsilons[1:round(Int,end/2)]
    fig = Figure(size = (1000, 1000));
    ax1 = Axis(fig[1,1], aspect = 1, title = "Dot Product Test", titlesize=34,xlabel = "ϵ",ylabel="error")
    l1 = lines!(ax1, log10.(epsilons), log10.(error), color = :blue, linewidth = 2, label = "Adjoint")
    #l2 = lines!(ax1, log10.(epsilons), log10.(errorFD), color = :red, linewidth = 2, label = "FD")
    lines!(ax1,log10.(eps),log10.(eps), color = :black, linewidth = 2, linestyle = :dot, label = "convergence order")
    #l3 = lines!(ax1, log10.(epsilons), log10.(FDs), color = :green, linewidth = 2, linestyle = :dot, label = "directional deriv.")
    #l4 = lines!(ax1, log10.(epsilons), log10.(AD), color = :black, linewidth = 2, linestyle = :dot, label = "Adjoint")
    axislegend(position = :rt)
    fig
    save(joinpath(figdir, "HockeyStick.png"), fig)
    return error, FDs
end

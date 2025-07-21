using CairoMakie

grey_theme = Theme(
    Axis = (
        backgroundcolor = :white,
        leftspinecolor = :gray32,
        rightspinecolor = :gray32,
        bottomspinecolor = :gray32,
        topspinecolor = :gray32,
        xgridcolor = :gray80,
        ygridcolor = :gray80,
        xticklabelcolor=:gray32,
        ylabelcolor=:gray32,
        xlabelcolor=:gray32,
        yticklabelcolor=:gray32,
        xtickcolor=:gray32,
        ytickcolor=:gray32,
    ),
    Colorbar =(
        labelcolor = :gray32,
        tickcolor = :gray32,
        ticklabelcolor = :gray32,
        leftspinecolor = :gray32,
        rightspinecolor = :gray32,
        bottomspinecolor = :gray32,
        topspinecolor = :gray32,

    ),
    Legend =(
        framecolor = :gray32,
        labelcolor = :gray32,
        tickcolor = :gray32,
        rowgap = 20,

    )
)

function make_final_plot(refcost,cost,dp,nx,ny,stokesAD,figdir, Ref,epsilon,error_η,error_ρ,error_G,mach_ϵ)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cellsvertices of the cells


    Vy_cen = (stokesRef.V.Vy[2:end-1,1:end-1].+stokesRef.V.Vy[2:end-1,2:end])./2
    function plot_fig()
    #fig = Figure(size = (720, 1000), title = "Viscoelastic Falling Block Setup",fontsize=16)
    fig = Figure(size = (720, 1000), fontsize = 14, padding = (10, 20, 10, 10))  # top, right, bottom, left

    ticksD = ([0.25,0.5,0.75,0.8],[0.25,0.5,0.75,0.8])

    ticks = ([1e-10,1e-8, 1e-6, 1e-4, 1e-2, 1e-0],
    [L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}",L"10^{0}"])
    yticks = ([1e-8, 1e-6, 1e-4, 1e-2],
    [L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}"])
    ax1   = Axis(fig[1,1], aspect = 1, title = L"V_y",titlesize=20)
    ax2   = Axis(fig[1,2], aspect = 1, title = L"\tau_{II}",titlesize=20)
    ax3   = Axis(fig[2,1], aspect = 1, title = L"$\eta$ - Sensitivity",titlesize=20)
    #ax4   = Axis(fig[2,2], aspect = 1, title = "Gradient Test η",titlesize=20)
    ax4   = Axis(fig[2,2], aspect = 1, title = L"Gradient Test $\eta$", titlesize=20,xlabel = "α",ylabelsize=18,ylabel=L"\mathcal{D}_{\alpha}",xscale = log10, yscale = log10,xticks = ticks)
    ax5   = Axis(fig[3,1], aspect = 1, title = L"$G$ - Sensitivity",titlesize=20)
    #ax6   = Axis(fig[3,2], aspect = 1, title = "Gradient Test G",titlesize=20)
    ax6   = Axis(fig[3,2], aspect = 1, title = L"Gradient Test $G$", titlesize=20,xlabel = "α",ylabelsize=18,ylabel=L"\mathcal{D}_{\alpha}",xscale = log10, yscale = log10,xticks = ticks)
    ax7   = Axis(fig[4,1], aspect = 1, title = L"$K$ - Sensitivity",titlesize=20)
    #ax8   = Axis(fig[4,2], aspect = 1, title = "Gradient Test ρ",titlesize=20)
    ax8   = Axis(fig[4,2], aspect = 1, title = L"Gradient Test $K$", titlesize=20,xlabel = L"\alpha",ylabelsize=18,xlabelsize=18,ylabel=L"\mathcal{D}_{\alpha}",xscale = log10, yscale = log10,xticks = ticks)


    ex_η = maximum(abs.(extrema(stokesAD.η)))
    ex_G = maximum(abs.(extrema(stokesAD.G)))
    ex_ρ = maximum(abs.(extrema(stokesAD.ρ)))

    eps  = 10 .^ range(-10, stop=2, length=10)
    eps2 = eps.^2
    mach_ϵs = zeros(size(epsilon))
    mach_ϵs .= mach_ϵ
    h1 = heatmap!(ax1, xci..., Array(Vy_cen),colormap=:roma)
    h2 = heatmap!(ax2, xci..., Array((Ref.τ.II)),colormap=:roma)
    #h1 = contour!(ax1, xci..., Array(Vy_cen),colormap=:roma,colorrange=[0.0,1.0],levels = 11)
    #h2 = contourf!(ax2, xci..., Array((Ref.τ.II)),colormap=:roma,levels = 11)
    h3 = contourf!(ax3, xci..., Array(stokesAD.η) , colormap=:vik,levels = -ex_η:(2*ex_η)/11:ex_η)
    #h4 = heatmap!(ax4, xci..., Array(stokesAD.η) , colormap=:lipari)
    h4 = scatter!(ax4, (epsilon), (error_η), color = :blue, markersize = 8)
    lines!(ax4,eps,((eps2)*(10^-3.55)), color = :black, linewidth = 4,alpha=0.4, label = "convergence order")
    #lines!(ax4,(epsilon), mach_ϵs, color = :black, linewidth = 4,alpha=0.4, label = "convergence order")
    h5 = contourf!(ax5, xci..., Array(stokesAD.G) , colormap=:vik,levels=-ex_G:(2*ex_G)/11:ex_G)
    #h6 = heatmap!(ax6, xci..., Array(stokesAD.G) , colormap=:lipari)
    h6 = scatter!(ax6, (epsilon), (error_G), color = :blue, markersize = 8)
    lines!(ax6,(eps),(eps2).*(10^-3.25), color = :black, linewidth = 4, alpha=0.4, label = "convergence order")
    h7 = contourf!(ax7, xci..., Array(stokesAD.ρ) , colormap=:vik,levels= -ex_ρ:(2*ex_ρ)/11:ex_ρ)
    #scatter!(ax7, xci[1][indx],xci[2][indy], color = :yellow, markersize = 8)
    #scatter!(ax5, xci[1][indx],xci[2][indy], color = :yellow, markersize = 8)
    #scatter!(ax3, xci[1][indx],xci[2][indy], color = :yellow, markersize = 8)
    #h8 = heatmap!(ax8, xci..., Array(stokesAD.ρ) , colormap=:lipari)
    h8 = scatter!(ax8, (epsilon), (error_ρ), color = :blue, markersize = 8)
    #lines!(ax8,eps,((eps2)*(10^-3.0)), color = :black, linewidth = 4,alpha=0.4, label = "convergence order")

    max_sens = max(maximum(error_η), maximum(error_G), maximum(error_ρ))
    min_sens = min(minimum(error_η), minimum(error_G), minimum(error_ρ))
    #min_sens= 1e-18
    ylims!(ax4, min_sens*0.1, max_sens*10)
    ylims!(ax6, min_sens*0.1, max_sens*10)
    ylims!(ax8, min_sens*0.1, max_sens*10)
    #xlims!(ax4, 6e-9, 2e0)
    #xlims!(ax6, 6e-9, 2e0)
    #xlims!(ax8, 6e-9, 2e0)
    xlims!(ax4, 6e-7, 2e0)
    xlims!(ax6, 6e-7, 2e0)
    xlims!(ax8, 6e-7, 2e0)
    hidexdecorations!(ax3);hidexdecorations!(ax4);hidexdecorations!(ax5);hidexdecorations!(ax6);
    hidexdecorations!(ax6);hideydecorations!(ax2);

    ax4.xgridvisible = true
    ax6.xgridvisible = true
    ax3.xgridvisible = true
    ax5.xgridvisible = true
    tightlimits!(ax3)
    tightlimits!(ax5)
    tightlimits!(ax7)

    Colorbar(fig[1,1][1,2], h1, height=Relative(0.8))
    Colorbar(fig[1,2][1,2], h2, height=Relative(0.8))
    Colorbar(fig[2,1][1,2], h3, height=Relative(0.8))
    #Colorbar(fig[2,2][1,2], h4, height=Relative(0.8))
    Colorbar(fig[3,1][1,2], h5, height=Relative(0.8))
    #Colorbar(fig[3,2][1,2], h6, height=Relative(0.8))
    Colorbar(fig[4,1][1,2], h7, height=Relative(0.8))
    #Colorbar(fig[4,2][1,2], h8, height=Relative(0.8))
    #=
    colsize!(fig.layout, 1, Aspect(1, 1.4))
    colsize!(fig.layout, 2, Aspect(1, 1.4))
    colgap!(fig.layout, 2)
    rowgap!(fig.layout, 8)
    fig
    =#

    
        colsize!(fig.layout, 1, Aspect(1, 1.2))  # tighter layout
        colsize!(fig.layout, 2, Aspect(1, 1.2))
        colgap!(fig.layout, 4)   # smaller horizontal gap
        rowgap!(fig.layout, 14)   # smaller vertical gap
        fig
    end

    figfinal = with_theme(plot_fig, grey_theme)
    #save(joinpath(figdir, "GradientTest$nx.png"), figfinal)
    save(joinpath(figdir, "GradientTestAdlow$nx.pdf"), figfinal)
end


function make_tolerance_test(refcost,cost,dp,nx,ny,stokesAD,figdir, Ref,epsilon,error_η,error_ρ,error_G,mach_ϵ)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cellsvertices of the cells


    Vy_cen = (stokesRef.V.Vy[2:end-1,1:end-1].+stokesRef.V.Vy[2:end-1,2:end])./2
    function plot_fig()

    fig = Figure(size = (720, 1000), fontsize = 14, padding = (10, 20, 10, 10))  # top, right, bottom, left

    ticksD = ([0.25,0.5,0.75,0.8],[0.25,0.5,0.75,0.8])
    ticks = ([1e-10,1e-8, 1e-6, 1e-4, 1e-2, 1e-0],
    [L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}",L"10^{0}"])
    yticks = ([1e-8, 1e-6, 1e-4, 1e-2],
    [L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}"])
    ax1   = Axis(fig[1,1], aspect = 1, title = L"Forward Tolerance Test",titlesize=20)

    eps  = 10 .^ range(-10, stop=2, length=10)
    eps2 = eps.^2
    mach_ϵs = zeros(size(epsilon))
    mach_ϵs .= mach_ϵ
    h1 = heatmap!(ax1, xci..., Array(Vy_cen),colormap=:roma)
    h2 = heatmap!(ax2, xci..., Array((Ref.τ.II)),colormap=:roma)
    #h1 = contour!(ax1, xci..., Array(Vy_cen),colormap=:roma,colorrange=[0.0,1.0],levels = 11)
    #h2 = contourf!(ax2, xci..., Array((Ref.τ.II)),colormap=:roma,levels = 11)
    h3 = contourf!(ax3, xci..., Array(stokesAD.η) , colormap=:vik,levels = -ex_η:(2*ex_η)/11:ex_η)
    #h4 = heatmap!(ax4, xci..., Array(stokesAD.η) , colormap=:lipari)
    h4 = scatter!(ax4, (epsilon), (error_η), color = :blue, markersize = 8)
    lines!(ax4,eps,((eps2)*(10^-3.55)), color = :black, linewidth = 4,alpha=0.4, label = "convergence order")
    #lines!(ax4,(epsilon), mach_ϵs, color = :black, linewidth = 4,alpha=0.4, label = "convergence order")
    h5 = contourf!(ax5, xci..., Array(stokesAD.G) , colormap=:vik,levels=-ex_G:(2*ex_G)/11:ex_G)
    #h6 = heatmap!(ax6, xci..., Array(stokesAD.G) , colormap=:lipari)
    h6 = scatter!(ax6, (epsilon), (error_G), color = :blue, markersize = 8)
    lines!(ax6,(eps),(eps2).*(10^-3.25), color = :black, linewidth = 4, alpha=0.4, label = "convergence order")
    h7 = contourf!(ax7, xci..., Array(stokesAD.ρ) , colormap=:vik,levels= -ex_ρ:(2*ex_ρ)/11:ex_ρ)

    #h8 = heatmap!(ax8, xci..., Array(stokesAD.ρ) , colormap=:lipari)
    h8 = scatter!(ax8, (epsilon), (error_ρ), color = :blue, markersize = 8)
    #lines!(ax8,eps,((eps2)*(10^-3.0)), color = :black, linewidth = 4,alpha=0.4, label = "convergence order")

    max_sens = max(maximum(error_η), maximum(error_G), maximum(error_ρ))
    min_sens = min(minimum(error_η), minimum(error_G), minimum(error_ρ))
    #min_sens= 1e-18
    ylims!(ax4, min_sens*0.1, max_sens*10)
    ylims!(ax6, min_sens*0.1, max_sens*10)
    ylims!(ax8, min_sens*0.1, max_sens*10)

    xlims!(ax4, 6e-7, 2e0)
    xlims!(ax6, 6e-7, 2e0)
    xlims!(ax8, 6e-7, 2e0)
    hidexdecorations!(ax3);hidexdecorations!(ax4);hidexdecorations!(ax5);hidexdecorations!(ax6);
    hidexdecorations!(ax6);hideydecorations!(ax2);

    ax4.xgridvisible = true
    ax6.xgridvisible = true
    ax3.xgridvisible = true
    ax5.xgridvisible = true
    tightlimits!(ax3)
    tightlimits!(ax5)
    tightlimits!(ax7)

    Colorbar(fig[1,1][1,2], h1, height=Relative(0.8))
    Colorbar(fig[1,2][1,2], h2, height=Relative(0.8))
    Colorbar(fig[2,1][1,2], h3, height=Relative(0.8))
    #Colorbar(fig[2,2][1,2], h4, height=Relative(0.8))
    Colorbar(fig[3,1][1,2], h5, height=Relative(0.8))
    #Colorbar(fig[3,2][1,2], h6, height=Relative(0.8))
    Colorbar(fig[4,1][1,2], h7, height=Relative(0.8))
    #Colorbar(fig[4,2][1,2], h8, height=Relative(0.8))

    colsize!(fig.layout, 1, Aspect(1, 1.2))  # tighter layout
    colsize!(fig.layout, 2, Aspect(1, 1.2))
    colgap!(fig.layout, 4)   # smaller horizontal gap
    rowgap!(fig.layout, 14)   # smaller vertical gap
    fig
    end

    figfinal = with_theme(plot_fig, grey_theme)
    #save(joinpath(figdir, "GradientTest$nx.png"), figfinal)
    save(joinpath(figdir, "GradientTestAdlow$nx.pdf"), figfinal)
end

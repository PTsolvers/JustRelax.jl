using ParallelStencil.FiniteDifferences2D

include("SolKz_solution.jl")

function solKz_viscosity(geometry; B=log(1e6))
    xc, yc = geometry.xci
    y = [yci for _ in xc, yci in yc]
    η = PTArray(zeros(geometry.ni))

    _viscosity(y, B) = exp(B*y)

    @parallel function viscosity(η, y, B) 
        @all(η) =  _viscosity(@all(y), B)
        return
    end

    @parallel viscosity(η, y, B) 

    return η

end

function solKz_density(geometry)
    xc, yc = geometry.xci
    x = [xci for xci in xc, _ in yc]
    y = [yci for _ in xc, yci in yc]
    ρ = PTArray(zeros(geometry.ni))

    _density(x, y) = -sin(2*y)*cos(3*π*x)

    @parallel function density(ρ, x, y) 
        @all(ρ) = _density(@all(x), @all(y))
        return
    end

    @parallel density(ρ, x, y)

    return ρ

end

function solkz_solution(geometry::Geometry)
    # element center
    xci, yci = geometry.xci
    xc = [xc for xc in xci,  _ in yci]
    yc = [yc for  _ in xci, yc in yci]
    # element vertices
    xvi, yvi = geometry.xvi
    xv_x = [xc for xc in xvi,  _ in yci] # for vx
    yv_x = [yc for  _ in xvi, yc in yci] # for vx

    xv_y = [xc for xc in xci,  _ in yvi] # for vy
    yv_y = [yc for  _ in xci, yc in yvi] # for vy

    # analytical solution 
    ps = similar(xc) # @ centers
    vxs = similar(xv_x) # @ vertices
    vys = similar(xv_y) # @ vertices
    Threads.@threads for i in eachindex(xc)
        @inbounds _, _, ps[i] = _solkz_solution(xc[i], yc[i])
    end
    Threads.@threads for i in eachindex(xv_x)
        @inbounds vxs[i], = _solkz_solution(xv_x[i], yv_x[i])
    end
    Threads.@threads for i in eachindex(xv_y)
        @inbounds vys[i], = _solkz_solution(xv_y[i], yv_y[i])
    end

    return (vx=vxs, vy=vys, p=ps)
end

function err2(A::AbstractArray, B::AbstractArray)
    err = similar(A)
    Threads.@threads for i in eachindex(err)
        @inbounds err[i] = √(((A[i]-B[i])^2))
    end
    err
end

function Li_error(geometry::Geometry, stokes::StokesArrays; order = 2)
    solk = solkz_solution(geometry)

    Li(A, B; order = 2) = norm(A.-B, order)/length(A)
    
    # L2_vx = Li(stokes.V.Vx[2:end-1,2:end-1], solk.vx[2:end-1,2:end-1], order=order)
    # L2_vy = Li(stokes.V.Vy[2:end-1,2:end-1], solk.vy[2:end-1,2:end-1], order=order)
    L2_vx = Li(stokes.V.Vx, solk.vx, order=order)
    L2_vy = Li(stokes.V.Vy, solk.vy, order=order)
    L2_p = Li(stokes.P, solk.p, order=order)

    return L2_vx, L2_vy, L2_p
end

function plot_solkz(geometry::Geometry, stokes::StokesArrays; cmap = :vik)
    f=Figure(resolution=(3000, 1800), fontsize=28)
    
    #Ddensity
    ax1= Axis(f[1, 1], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2], ρ, colormap=cmap)
   
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,2], h1, label="density")

    # Pressure
    ax1= Axis(f[1, 3], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2], stokes.P, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,4], h1, label="P")

    # Velocity-x
    ax1= Axis(f[2, 1], aspect=1)
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], stokes.V.Vx, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 2], h1, label="Vx")
    
    # Velocity-y
    ax1= Axis(f[2, 3], aspect=1)
    h1=heatmap!(ax1,  geometry.xvi[2], geometry.xci[1], stokes.V.Vy, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 4], h1, label="Vy")

    f
end

function plot_solkz_error(geometry::Geometry, stokes::StokesArrays; cmap = :vik)
    
    solk = solkz_solution(geometry)
    
    # Plot
    f=Figure(resolution=(3000, 1800), fontsize=28)
    
    # Density
    ax1= Axis(f[1, 1], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2], stokes.P, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,2], h1, label="density")

    # Pressure
    ax1= Axis(f[1, 3], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2],  log10.(err2(stokes.P, solk.p)), colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,4], h1, label="error P")

    # Velocity-x
    ax1= Axis(f[2, 1], aspect=1)
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], log10.(err2(stokes.V.Vx, solk.vx)), colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 2], h1, label="Vx")
   
    # Velocity-y
    ax1= Axis(f[2, 3], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xvi[2], log10.(err2(stokes.V.Vy, solk.vy)), colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 4], h1, label="Vy")

    f
end

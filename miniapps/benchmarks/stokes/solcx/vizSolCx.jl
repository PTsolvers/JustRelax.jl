using ParallelStencil.FiniteDifferences2D

include("SolCx_solution.jl")

# function solCx_viscosity(geometry; Δη = 1e6)
#     xc, yc = geometry.xci
#     x = [xci for xci in xc, yci in yc]
#     η = PTArray(zeros(geometry.ni))

#     _viscosity(x, Δη) = ifelse( x ≤ 0.5, 1e0, Δη )

#     @parallel function viscosity(η, x) 
#         @all(η) =  _viscosity(@all(x), Δη)
#         return
#     end

#     @parallel viscosity(η, x) 

#     return η

# end

# function solCx_density(geometry)
#     xc, yc = geometry.xci
#     x = [xci for xci in xc, _ in yc]
#     y = [yci for _ in xc, yci in yc]
#     ρ = PTArray(zeros(geometry.ni))

#     _density(x, y) = -sin(π*y)*cos(π*x)

#     @parallel function density(ρ, x, y) 
#         @all(ρ) = _density(@all(x), @all(y))
#         return
#     end

#     @parallel density(ρ, x, y)

#     return ρ

# end

function solCx_solution(geometry; η_left=1,  η_right=1e6)
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
        @inbounds _, _, ps[i] = _solCx_solution(xc[i], yc[i], η_left, η_right)
    end
    Threads.@threads for i in eachindex(xv_x)
        @inbounds vxs[i], = _solCx_solution(xv_x[i], yv_x[i], η_left, η_right)
    end
    Threads.@threads for i in eachindex(xv_y)
        @inbounds _, vys[i], = _solCx_solution(xv_y[i], yv_y[i], η_left, η_right)
    end

    return (vx=vxs, vy=vys, p=ps)
end

innfunloc(A, i, j, f)  = f(
        A[i,   j],
        A[i-1, j],
        A[i+1, j],
        A[i, j-1],
        A[i, j+1],
    )

function compute_funloc!(A, B, f::Function)

    Threads.@threads for i in 2:size(A,1)-1
        for j in 2:size(A,2)-1
            @inbounds  A[i, j] = innfunloc(A, i, j, f)
        end 
    end

    Threads.@threads for i in 2:size(A, 2)-1
        @inbounds B[i, 1] = f(
            A[i,     1],
            A[i+1,   1],
            A[i-1,   1],
            A[i,   1+1],
        )
        @inbounds B[i, size(A, 2)] = f(
            A[i,   size(A, 2)  ],
            A[i+1, size(A, 2)  ],
            A[i-1, size(A, 2)  ],
            A[i,   size(A, 2)-1],
        )
    end

end

mean(args...) = sum(args)/length(args)
harmmean(args...) = length(args)/sum(1.0./args)
geometricmean(args...) = reduce(*, args)^(1/length(args))

@parallel_indices (iy) function smooth_boundaries_x!(A::PTArray)
    A[iy, 1  ] = A[iy, 20    ]
    A[iy, 2  ] = A[iy, 20    ]
    A[iy, 3  ] = A[iy, 20    ]
    A[iy, 4  ] = A[iy, 20    ]
    A[iy, 5  ] = A[iy, 20    ]
    A[iy, 6  ] = A[iy, 20    ]
    A[iy, end] = A[iy, end-20]
    A[iy, end-1] = A[iy, end-20]
    A[iy, end-2] = A[iy, end-20]
    A[iy, end-3] = A[iy, end-20]
    A[iy, end-4] = A[iy, end-20]
    A[iy, end-5] = A[iy, end-20]
    return
end

function err2(A::AbstractArray, B::AbstractArray)
    err = similar(A)
    Threads.@threads for i in eachindex(err)
        @inbounds err[i] = √(((A[i]-B[i])^2))
    end
    err
end

function solve2!(stokes::StokesArrays, pt_stokes::PTStokesCoeffs, geometry::Geometry{2}, freeslip, ρg, η; iterMax = 10e3, nout = 500)
    # unpack
    dx, dy = geometry.di 
    lx, ly = geometry.li 
    (; Vx, Vy) = stokes.V
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τxx, τyy, τxy = stress(stokes)
    (; P, ∇V) = stokes
    (; Ry, Rx) = stokes.R
    (; Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ) = pt_stokes
    (;freeslip_x, freeslip_y) =freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    # @parallel compute_maxloc!(ητ, η)
    compute_funloc!(ητ, η, geometricmean)
    # PT numerical coefficients
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, geometry.max_li)
    # errors
    err=2*ϵ; iter=0; err_evo1=Float64[]; err_evo2=Float64[]; err_rms = Float64[]
    
    # solver loop
    # Gdτ *= 1e-1
    # dτ_Rho *= 1e-3
    while err > ϵ && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)

        # free slip boundary conditions
        if (freeslip_x) @parallel (1:size(Vx,1)) free_slip_y!(Vx) end
        if (freeslip_y) @parallel (1:size(Vy,2)) free_slip_x!(Vy) end

        iter += 1
        if (iter > 1) && (iter % nout == 0)
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
            norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
            norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
            # norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
            err = maximum([norm_Rx, norm_Ry, norm_∇V])
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_∇V)
        end
    end
end

function solcx_error(geometry, stokes::StokesArrays; order = 2)
    solk = solCx_solution(geometry)

    gridsize = reduce(*, geometry.di)
    Li(A, B; order = 2) = norm(A.-B, order)
    
    # L2_vx = Li(stokes.V.Vx[2:end-1,2:end-1], solk.vx[2:end-1,2:end-1], order=order)
    # L2_vy = Li(stokes.V.Vy[2:end-1,2:end-1], solk.vy[2:end-1,2:end-1], order=order)
    L2_vx = Li(stokes.V.Vx, solk.vx, order=order)*gridsize
    L2_vy = Li(stokes.V.Vy, solk.vy, order=order)*gridsize
    L2_p = Li(stokes.P, solk.p, order=order)*gridsize

    return L2_vx, L2_vy, L2_p
end

function plot_solCx(geometry::Geometry, stokes::StokesArrays, ρ; cmap = :vik, fun = heatmap!)
    f=Figure(resolution=(3000, 1800), fontsize=28)
    
    #Ddensity
    ax1= Axis(f[1, 1], aspect=1)
    h1=fun(ax1, geometry.xci[1], geometry.xci[2], ρ, colormap=cmap)
   
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,2], h1, label="density")

    # Pressure
    ax1= Axis(f[1, 3], aspect=1)
    h1=fun(ax1, geometry.xci[1], geometry.xci[2], stokes.P, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,4], h1, label="P")

    # Velocity-x
    ax1= Axis(f[2, 1], aspect=1)
    h1=fun(ax1, geometry.xvi[1], geometry.xci[2], stokes.V.Vx, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 2], h1, label="Vx")
    
    # Velocity-y
    ax1= Axis(f[2, 3], aspect=1)
    h1=fun(ax1,  geometry.xci[1], geometry.xvi[2], stokes.V.Vy, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 4], h1, label="Vy")

    f
end

function plot_solCx_error(geometry::Geometry, stokes::StokesArrays, Δη; cmap = :vik)
    
    solc = solCx_solution(geometry, η_right = Δη)
    
    # Plot
    f=Figure(resolution=(2200, 1800), fontsize=28)
    
    # ROW 1: PRESSURE
    # Numerical pressure
    ax1= Axis(f[1, 1], aspect=1, title="numerical")
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2], stokes.P, colormap=cmap, colorrange = extrema(stokes.P))
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    # Colorbar(f[1,2], h1, label="Pressure")

    # Analytical pressure
    ax1= Axis(f[1, 2], aspect=1, title="analytical")
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2],  solc.p, colormap=cmap, colorrange = extrema(stokes.P))
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,3], h1, label="P", width = 20, tellheight=true)

    # Pressure
    ax1= Axis(f[1, 4], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2],  (err1(stokes.P, solc.p)), colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[1,5], h1, label="error P")
    # rowsize!(f.layout, 1, ax1.scene.px_area[].widths[2])

    # ROW 2: Velocity-x
    # Numerical
    ax1= Axis(f[2, 1], aspect=1, title= "Numerical")
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], stokes.V.Vx, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))

    ax1= Axis(f[2, 2], aspect=1)
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], solc.vx, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 3], h1, label="Vx", width = 20, tellheight=true)

    ax1= Axis(f[2, 4], aspect=1)
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], (err1(stokes.V.Vx, solc.vx)), colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[2, 5], h1, label="error Vx", width = 20, tellheight=true)
    # rowsize!(f.layout, 1, ax1.scene.px_area[].widths[2])

    # ROW 3: Velocity-y
    # Numerical
    ax1= Axis(f[3, 1], aspect=1, title= "Numerical")
    h1=heatmap!(ax1, geometry.xci[1], geometry.xvi[2], stokes.V.Vy, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))

    ax1= Axis(f[3, 2], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xvi[2], solc.vy, colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[3, 3], h1, label="Vy", width = 20, tellheight=true)

    ax1= Axis(f[3, 4], aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xvi[2], (err1(stokes.V.Vy, solc.vy)), colormap=cmap)
    xlims!(ax1, (0,1))
    ylims!(ax1, (0,1))
    Colorbar(f[3, 5], h1, label="error Vy", width = 20, tellheight=true)
    # rowsize!(f.layout, 1, ax1.scene.px_area[].widths[2])

    f
end

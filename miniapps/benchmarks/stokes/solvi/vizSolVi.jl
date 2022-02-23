using ParallelStencil.FiniteDifferences2D
# Analytical solution found in:
#     D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
#     general shear. Geophysical Journal International, 155(1):269–288, 2003.
function _solvi_solution(X, Y;
    ε = 1,
    ηm = 1,
    ηc = 1e-3,
    rc = 1,
    )

    # geometry
    Z = @. X+im*Y
    A = ηm*(ηc-ηm)/(ηc+ηm)
    
    # pressure
    P = @. -2*A*real((rc^2)/(Z^2))*(2*ε)
    idx = @.(sqrt(X^2 + Y^2)) .< rc # nodes within circular inclusion
    P[idx] .= 0

    # velocity function
    vel(ϕ, dϕ, ψ, Z, η) = (ϕ - Z*conj(dϕ) - conj(ψ))/2/η

    # velocity in matrix
    ϕm = @. -(0+2*ε)*A*rc*rc/Z
    dϕm = @. (0+2*ε)*A*rc*rc/(Z^2)
    ψm = @.  (0-2*ε)*ηm*Z - (0+2*ε)*ηm*ηc/(ηm+ηc)*A*(rc^4)/Z/Z/Z
    vm = vel.(ϕm, dϕm, ψm, Z, ηm)

    # velocity in clast
    ϕc = 0.0
    dϕc = 0.0
    ψc = @. Z*2*(0-2*ε)*ηm*ηc/(ηm+ηc)
    vc = vel.(ϕc, dϕc, ψc, Z, ηc)

    # domain velocity
    v = vm
    v[idx] .= vc[idx]

    return P, v
end

function solvi_solution(geometry, η0, ηi, εbg, rc)
    # element center
    xci, yci = geometry.xci
    xc = [xc for xc in xci,  _ in yci] .- (xci[end]-xci[1])/2
    yc = [yc for  _ in xci, yc in yci] .- (yci[end]-yci[1])/2
    # element vertices
    xvi, yvi = geometry.xvi
    xv_x = [xc for xc in xvi,  _ in yci] .- (xvi[end]-xvi[1])/2 # for vx
    yv_x = [yc for  _ in xvi, yc in yci] .- (yci[end]-yci[1])/2 # for vx

    xv_y = [xc for xc in xci,  _ in yvi] .- (xci[end]-xci[1])/2 # for vy
    yv_y = [yc for  _ in xci, yc in yvi] .- (yvi[end]-yvi[1])/2 # for vy

    # pressure analytical solution 
    ps, = _solvi_solution(xc, yc;  
        ηm = η0,
        ηc = ηi,
        ε = εbg,
        rc = rc
    )
    # x-velocity analytical solution 
    _, va = _solvi_solution(xv_x, yv_x;  
        ηm = η0,
        ηc = ηi,
        ε = εbg,
        rc = rc
    )
    vxs = real.(va)
    # y-velocity analytical solution 
    _, va = _solvi_solution(xv_y, yv_y;  
        ηm = η0,
        ηc = ηi,
        ε = εbg,
        rc = rc
    )
    vys = imag.(va)

    return (p=ps, vx=-vxs, vy=-vys)
end

function Li_error(geometry, stokes::StokesArrays, Δη, εbg, rc, ; order = 2)
    
    # analytical solution
    sol =  solvi_solution(geometry, 1, Δη, εbg, rc)
    gridsize = reduce(*, geometry.di)

    Li(A, B; order = 2) = norm(A.-B, order)
    
    L2_vx = Li(stokes.V.Vx, PTArray(sol.vx), order=order)*gridsize
    L2_vy = Li(stokes.V.Vy, PTArray(sol.vy), order=order)*gridsize
    L2_p = Li(stokes.P, PTArray(sol.p), order=order)*gridsize

    return L2_vx, L2_vy, L2_p
end

function plot_solVi_error(geometry, stokes::StokesArrays, Δη, εbg, rc)

    # analytical solution
    sol =  solvi_solution(geometry, 1, Δη, εbg, rc)

    cx, cy = (geometry.xvi[1][end]- geometry.xvi[1][1])/2, (geometry.xvi[2][end]- geometry.xvi[2][1])/2
    θ = LinRange(0, 2π, 100)
    ix, iy = @.(rc*cos(θ)+cx), @.(rc*sin(θ)+cy)

    f=Figure(resolution=(1200, 1200), fontsize=20)

    # Pressure plots
    ax1= Axis(f[1, 1], title="P numeric", aspect=1)
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2], stokes.P, 
        colorrange = extrema(stokes.P), colormap=:romaO)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)

    hidexdecorations!(ax1)
    
    ax1= Axis(f[1, 2], title="P analytical", aspect=1)
    h=heatmap!(ax1, geometry.xci[1], geometry.xci[2], sol.p',
        colorrange = extrema(stokes.P), colormap=:romaO)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)
    Colorbar(f[1, 3], h, height = 300)
    
    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    ax1= Axis(f[1, 4], title="P error", aspect=1)
    h=heatmap!(ax1, geometry.xci[1], geometry.xci[2],  @.(log10(sqrt((stokes.P' - sol.p)^2))), colormap=Reverse(:batlow), colorrange = (-7, 1))
    lines!(ax1, ix, iy, linewidth = 3, color = :black)
    Colorbar(f[1, 5], h)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # Velocity-x plots
    ax1= Axis(f[2, 1], title="Vx numeric", aspect=1)
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], stokes.V.Vx,
        colorrange = extrema(stokes.V.Vx), colormap=:romaO)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)

    hidexdecorations!(ax1)

    ax1= Axis(f[2, 2], title="Vx analytical", aspect=1)
    h=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], sol.vx,
        colorrange = extrema(stokes.V.Vx), colormap=:romaO)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)
    Colorbar(f[2, 3], h, height = 300)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    ax1= Axis(f[2, 4], title="Vx error", aspect=1)
    h=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], @.(log10(sqrt((stokes.V.Vx - sol.vx)^2))), colormap=:batlow)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)
    Colorbar(f[2, 5], h)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # Velocity-z plots
    ax1= Axis(f[3, 1], title="Vy numeric", aspect=1)
    h1=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], stokes.V.Vy,
        colorrange = extrema(stokes.V.Vy), colormap=:romaO)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)

    ax1= Axis(f[3, 2], title="Vy analytical", aspect=1)
    h=heatmap!(ax1, geometry.xvi[1], geometry.xci[2], sol.vy, colormap=:romaO)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)
    Colorbar(f[3, 3], h, height = 300)

    hideydecorations!(ax1)
    
    ax1= Axis(f[3, 4], title="Vy error", aspect=1)
    h=heatmap!(ax1, geometry.xvi[1], geometry.xci[2],  @.(log10(sqrt((stokes.V.Vy - sol.vy)^2))), colormap=:batlow)
    lines!(ax1, ix, iy, linewidth = 3, color = :black)
    Colorbar(f[3, 5], h)

    hideydecorations!(ax1)
    
    f
end

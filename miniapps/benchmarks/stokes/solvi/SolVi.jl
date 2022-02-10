"""
Analytical solution found in:
    D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
    general shear. Geophysical Journal International, 155(1):269–288, 2003.
""" 
function _solvi_solution(
    X, Y;
    ε = 1,
    ηm = 1e1,
    ηc = 1,
    rc = 1,
    N = 1000
    )

    # geometry
    Z = @. X+im*Y
    A = ηm*(ηc-ηm)/(ηc+ηm)
    # pressure
    P = @. -2*A*real((rc^2)/(Z^2))*(2*ε)
    P[abs.(Z).<rc] .= 0
    # velocity in matrix
    vm = @. ε*A*(rc^2)/ηm*(-1/Z + Z/conj(Z^2) - 1/conj(Z^3) - conj(Z)*ηm/A/(rc^2))
    # velocity in clast
    vc = @. -4*ε/2/ηc * (ηc-ηm)/(ηc+ηm) * conj(Z)
    v = vm
    v[abs.(Z).<rc] .= vc[abs.(Z).<rc]

    return P, v
end

function solvi_viscosity(geometry, η0, ηi)
    dx, dy = geometry.di
    lx, ly = geometry.li
    Rad2 = [sqrt.(((ix-1)*dx +0.5*dx -0.5*lx)^2 + ((iy-1)*dy +0.5*dy -0.5*ly)^2) for ix=1:ni[1], iy=1:ni[2]]
    η       = η0*ones(ni...)
    η[Rad2.<rc] .= ηi
    η2 = deepcopy(η)
    for ism=1:10
        @parallel smooth!(η2, η, 1.0)
        η, η2 = η2, η
    end
    η
end

function solvi_solution(geometry::Geometry, η0, ηi, εbg, rc)
    yv = geometry.xci[1]
    lx, ly = geometry.li
    # get analytical solution
    X = PTArray( [yv[ix]-lx/2 for ix=1:size(yv,1), iy=1:size(yv,1)] )
    Y = PTArray( [yv[iy]-ly/2 for ix=1:size(yv,1), iy=1:size(yv,1)] )
    Pa, va = _solvi_solution(
        X, Y;  
        ηm = η0,
        ηc = ηi,
        ε = εbg,
        rc = rc
    )
    vxa = real.(va)
    vya = imag.(va)
    return Pa, vxa, vya
end

function plot_solvi(geometry::Geometry, stokes::StokesArrays, Psolvi)
    f=Figure(resolution=(3000, 1800))
    ax1= Axis(f[1, 1], title="P numeric")
    h1=heatmap!(ax1, geometry.xci[1], geometry.xci[2], stokes.P', colormap=:vik)
    xlims!(ax1, (5,8))
    ylims!(ax1, (5,8))
    Colorbar(f[1,2], h1)

    ax1= Axis(f[1, 3], title="P analytical")
    h=heatmap!(ax1, geometry.xci[1], geometry.xci[2], Psolvi, colormap=:vik)
    xlims!(ax1, (5,8))
    ylims!(ax1, (5,8))
    Colorbar(f[1,4], h)

    ax1= Axis(f[2, 1], title="P error")
    h=heatmap!(ax1, geometry.xci[1], geometry.xci[2],  @.(log10(sqrt((stokes.P' -  Psolvi')^2))), colormap=Reverse(:batlow), colorrange = (-7, 1))
    Colorbar(f[2, 2], h)
    xlims!(ax1, (5,10))
    ylims!(ax1, (5,10))

    f
end
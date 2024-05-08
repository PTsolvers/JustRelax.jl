
compute_vorticity!(stokes, di) = compute_vorticity!(backend(stokes), stokes, di)

function compute_vorticity!(::CPUBackendTrait, stokes, di)
    ω_xy = stokes.ω.xy
    ni = size(ω_xy)
    @parallel (@idx ni) compute_vorticity!(ω_xy, @velocity(stokes)..., inv.(di)...)
    
    return nothing
end

@parallel_indices (i, j) function compute_vorticity!(ω_xy, Vx, Vy, _dx, _dy)
    dx(A) = _d_xa(A, i, j, _dx)
    dy(A) = _d_ya(A, i, j, _dy)

    ω_xy[i, j] = 0.5 * (dx(Vy) - dy(Vx))

    return nothing
end
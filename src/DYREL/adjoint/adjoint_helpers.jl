function initialize_adjoint_iteration!(adjoint, ni)
    @parallel (@idx ni .+ 2) _initialize_adjoint_iteration!(
        adjoint.P,
        (adjoint.V.Vx, adjoint.V.Vy),
        (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy),
        (adjoint.R.Rx, adjoint.R.Ry, adjoint.R.RP),
        (adjoint.λV.Vx, adjoint.λV.Vy),
        adjoint.λP,
    )
    return nothing
end

@parallel_indices (i, j) function _initialize_adjoint_iteration!(
        P, V, ε, τ, R, λV, λP
    )
    if i ≤ size(P, 1) && j ≤ size(P, 2)
        @inbounds begin
            P[i, j] = 0.0
            ε[1][i, j] = 0.0
            ε[2][i, j] = 0.0
            τ[1][i, j] = 0.0
            τ[2][i, j] = 0.0
            R[3][i, j] = λP[i, j]
        end
    end
    if i ≤ size(V[1], 1) && j ≤ size(V[1], 2)
        @inbounds V[1][i, j] = 0.0
    end
    if i ≤ size(V[2], 1) && j ≤ size(V[2], 2)
        @inbounds V[2][i, j] = 0.0
    end
    if i ≤ size(ε[3], 1) && j ≤ size(ε[3], 2)
        @inbounds begin
            ε[3][i, j] = 0.0
            τ[3][i, j] = 0.0
        end
    end
    if i ≤ size(R[1], 1) && j ≤ size(R[1], 2)
        @inbounds R[1][i, j] = λV[1][i + 1, j + 1]
    end
    if i ≤ size(R[2], 1) && j ≤ size(R[2], 2)
        @inbounds R[2][i, j] = λV[2][i + 1, j + 1]
    end
    return nothing
end

# Apply the adjoint Schur-complement correction and diagonal preconditioner,
# then advance the damped velocity iterate. Keep the unpreconditioned corrected
# residual in Rx/Ry so the caller can evaluate the same norm as before fusion.
@parallel_indices (i, j) function update_adjoint_V_damping_DR!(
        Vx,
        Vy,
        dVxdτ,
        dVydτ,
        Rx,
        Ry,
        P,
        γ_eff,
        Dx,
        Dy,
        αVx,
        αVy,
        βVx,
        βVy,
        dτVx,
        dτVy,
        _di_center,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            iE = wrap_next(i, size(γ_eff, 1))
            Rx_ij = Rx[i + 1, j + 1] -
                (γ_eff[i, j] * P[i, j] - γ_eff[iE, j] * P[iE, j]) * _di_center[1]
            Rx[i + 1, j + 1] = Rx_ij
            dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij / Dx[i, j], αVx[i, j], βVx[i, j], dτVx[i, j])
            dVxdτ[i, j] = dVx_new
            Vx[i + 1, j + 1] += ΔVx
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            jN = wrap_next(j, size(γ_eff, 2))
            Ry_ij = Ry[i + 1, j + 1] -
                (γ_eff[i, j] * P[i, j] - γ_eff[i, jN] * P[i, jN]) * _di_center[2]
            Ry[i + 1, j + 1] = Ry_ij
            dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij / Dy[i, j], αVy[i, j], βVy[i, j], dτVy[i, j])
            dVydτ[i, j] = dVy_new
            Vy[i + 1, j + 1] += ΔVy
        end
    end
    return nothing
end

"""
    enzyme_reverse_rowwise!(f, n, args...)

Reverse-mode differentiate the point function `f(args..., i, j)` over the 2D index range
`1:n[1] × 1:n[2]`, with `args` given as Enzyme annotations.

Differentiating a whole `@parallel_indices` kernel makes Enzyme differentiate its threaded loop,
which stores every point's intermediate values on a heap tape and is ~100× slower than the
forward kernel. Differentiating one point at a time keeps the tape on the stack. The shadow
accumulation into neighbouring points is then parallelized by row coloring, which requires `f`
to touch at most two consecutive rows `j + o`, `j + o + 1` of each array from row `j` (a fixed
offset `o` per array, e.g. centers `j-1:j` and vertices `j:j+1` for the stress kernels). Rows of
equal parity are then disjoint and run in parallel. The first and last row can wrap across a
periodic seam, so they run on a single thread before and after. CPU only.
"""
function enzyme_reverse_rowwise!(f::F, n::NTuple{2, Integer}, args::Vararg{Any, N}) where {F, N}
    nx, ny = n
    # the serial first row also compiles the reverse pass before any thread needs it
    _enzyme_reverse_row!(f, 1, nx, args...)
    for first_row in (2, 3)
        Threads.@threads for j in first_row:2:(ny - 1)
            _enzyme_reverse_row!(f, j, nx, args...)
        end
    end
    ny > 1 && _enzyme_reverse_row!(f, ny, nx, args...)
    return nothing
end

function _enzyme_reverse_row!(f::F, j, nx, args::Vararg{Any, N}) where {F, N}
    for i in 1:nx
        Enzyme.autodiff(
            Enzyme.Reverse, Enzyme.Const(f), Enzyme.Const, args..., Enzyme.Const(i), Enzyme.Const(j)
        )
    end
    return nothing
end

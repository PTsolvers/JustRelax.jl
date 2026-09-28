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
    enzyme_reverse_pointwise!(f, n, args)

Reverse-mode differentiate the point function `f(primals..., i, j)` over the 2D index range
`1:n[1] × 1:n[2]`. `args` lists the arguments of `f` as pairs `x => dx`: `dx` is the shadow
(adjoint) of `x`, and `x => nothing` keeps `x` constant. The seeds in the shadows of the outputs
are consumed and the input shadows accumulate, as for the reverse of the whole kernel.

Differentiating a whole `@parallel_indices` kernel makes Enzyme differentiate its threaded
loop, which stores every point's intermediate values on a heap tape and is ~100× slower than
the forward kernel. Here Enzyme differentiates one point at a time inside ParallelStencil
kernels, so the tape stays on the stack. The pairs are split on the host into a tuple of primals
and a tuple of shadows, plain arrays that ParallelStencil hands to any backend, and the Enzyme
annotations are built inside the kernel (see `annotate`).

The reverse of point `(i, j)` accumulates into the adjoints of its neighbours, and the point
functions touch at most two consecutive rows `j + o`, `j + o + 1` of each array (a fixed offset
`o` per array). Rows two apart therefore never collide: each kernel index reverses one whole row
in order (contiguous access, and no race across a periodic seam in x), with all even and then
all odd interior rows in parallel. The first and last row can wrap across a periodic seam in y
and run in launches of their own.
"""
function enzyme_reverse_pointwise!(f::F, n::NTuple{2, Integer}, args::Tuple) where {F}
    primals, shadows = map(first, args), map(last, args)
    reverse_colored!(ReversePoint(f), n, primals, shadows, nothing)
    return nothing
end

# Annotation of one argument of a point function: differentiated, constant, or (for the
# material-parameter sensitivities) active with its derivative returned.
struct ActiveArgument end
const ACTIVE = ActiveArgument()

@inline annotate(x, dx) = Enzyme.DuplicatedNoNeed(x, dx)
@inline annotate(x, ::Nothing) = Enzyme.Const(x)
@inline annotate(x, ::ActiveArgument) = Enzyme.Active(x)

# Reverse of the point function `f` at one point
struct ReversePoint{F}
    f::F
end

@inline function apply_point!(op::ReversePoint, primals, shadows, extra, i, j)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(op.f), Enzyme.Const,
        map(annotate, primals, shadows)..., Enzyme.Const(i), Enzyme.Const(j),
    )
    return nothing
end

# Run `apply_point!(op, …, i, j)` over `1:n[1] × 1:n[2]` in the race-free order of
# `enzyme_reverse_pointwise!`. `extra` carries further arrays `op` writes to (or `nothing`).
function reverse_colored!(op, n::NTuple{2, Integer}, primals, shadows, extra)
    for first_row in (2, 3)
        rows = length(first_row:2:(n[2] - 1))
        rows > 0 && @parallel (1:1, 1:rows) reverse_rows_kernel!(op, primals, shadows, extra, n[1], first_row)
    end
    for row in (1, n[2])
        @parallel (1:1, 1:1) reverse_rows_kernel!(op, primals, shadows, extra, n[1], row)
    end
    return nothing
end

# reverse rows `first_row`, `first_row + 2`, … (one per index), each point in order
@parallel_indices (I...) function reverse_rows_kernel!(op, primals, shadows, extra, nx, first_row)
    for i in 1:nx
        apply_point!(op, primals, shadows, extra, i, first_row + 2 * (I[2] - 1))
    end
    return nothing
end

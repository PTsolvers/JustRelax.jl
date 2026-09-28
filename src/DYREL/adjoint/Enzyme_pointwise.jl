using Enzyme

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

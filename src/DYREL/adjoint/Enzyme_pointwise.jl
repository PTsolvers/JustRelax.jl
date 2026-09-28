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
    reverse_colored!(f, n, primals, shadows, nothing)
    return nothing
end

@inline annotate(x, dx) = Enzyme.DuplicatedNoNeed(x, dx)
@inline annotate(x, ::Nothing) = Enzyme.Const(x)
@inline annotate(x, ::typeof(Enzyme.Active)) = Enzyme.Active(x)

@inline function apply_point!(f::F, primals, shadows, extra, i, j) where {F <: Function}
    Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(f), Enzyme.Const,
        map(annotate, primals, shadows)..., Enzyme.Const(i), Enzyme.Const(j),
    )
    return nothing
end

function reverse_colored!(op, n::NTuple{2, Integer}, primals, shadows, extra)
    nx, ny = n

    # Non-adjacent interior rows can run concurrently.
    for first_row in (2, 3)
        number_of_rows = length(first_row:2:(ny - 1))
        iszero(number_of_rows) && continue
        @parallel (1:number_of_rows) reverse_row_group!(
            op, primals, shadows, extra, nx, first_row
        )
    end

    # Boundary rows run separately because periodic stencils can connect them.
    for row in (1, ny)
        @parallel (1:1) reverse_row_group!(op, primals, shadows, extra, nx, row)
    end

    return nothing
end

@parallel_indices (row_in_group) function reverse_row_group!(
        op, primals, shadows, extra, nx, first_row
    )
    row = first_row + 2 * (row_in_group - 1)

    # Adjacent columns can share shadow entries, so process one row sequentially.
    for column in 1:nx
        apply_point!(op, primals, shadows, extra, column, row)
    end

    return nothing
end

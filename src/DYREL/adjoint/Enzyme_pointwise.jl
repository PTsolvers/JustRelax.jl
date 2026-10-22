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
function enzyme_reverse_pointwise!(f, n::NTuple{2, Integer}, args::Tuple)
    primals, shadows = map(first, args), map(last, args)
    reverse_colored!(f, n, primals, shadows, nothing)
    return nothing
end

@inline annotate(x, dx) = Enzyme.DuplicatedNoNeed(x, dx)
@inline annotate(x, ::Nothing) = Enzyme.Const(x)

# Temporary fix for an Enzyme bug (0.13.199–0.13.216): a read-only input annotated
# `DuplicatedNoNeed` crashes reverse mode when the kept primal write reads it, e.g. a tuple
# output such as `ρg`. Wrap the shadow of such inputs to annotate them `Duplicated`; remove
# once Enzyme is fixed (reproducer: test/mwe_enzyme_reverse_duplicatednoneed.jl).
struct ReadOnlyShadow{A}
    dx::A
end
Adapt.@adapt_structure ReadOnlyShadow

@inline annotate(x, s::ReadOnlyShadow) = Enzyme.Duplicated(x, s.dx)
@inline annotate(x, ::typeof(Enzyme.Active)) = Enzyme.Active(x)

@inline function apply_point!(f, primals, shadows, context, i, j)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(f), Enzyme.Const,
        map(annotate, primals, shadows)..., Enzyme.Const(i), Enzyme.Const(j),
    )
    return nothing
end

# Reverse one point update with the rheology argument active and store its material derivatives.
# `store_rheology_gradients!` is defined with the material-gradient helpers included below.
struct PointwiseMaterialGradient{F, rheology_argument}
    update!::F
end

@inline function apply_point!(
        op::PointwiseMaterialGradient{F, rheology_argument}, primals, shadows, context, i, j,
    ) where {F, rheology_argument}
    (; gradient_fields, parameter_paths_by_phase) = context
    rheology = primals[rheology_argument]

    # Array fields paired with adjoint arrays become `DuplicatedNoNeed` arguments. Enzyme
    # therefore accumulates their reverse derivatives directly into the supplied adjoint
    # arrays, for example `adjoint.ε`, `adjoint.P`, and `adjoint.viscosity.η`.
    # The rheology is different: it is an `Active` argument without a shadow object, so its
    # complete nested derivative is returned by `autodiff_deferred` instead.
    annotations = map(annotate, primals, shadows)
    rheology_derivative = Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(op.update!), Enzyme.Const, annotations...,
        Enzyme.Const(i), Enzyme.Const(j),
    )[1][rheology_argument]

    # Select only the material-parameter paths requested in `gradients` from the returned
    # rheology derivative and accumulate them in the phase- and point-resolved fields.
    store_rheology_gradients!(
        gradient_fields, parameter_paths_by_phase, rheology, rheology_derivative, (i, j)
    )
    return nothing
end

function reverse_colored!(op, n::NTuple{2, Integer}, primals, shadows, context)
    nx, ny = n

    # Non-adjacent interior rows can run concurrently.
    for first_row in (2, 3)
        number_of_rows = length(first_row:2:(ny - 1))
        iszero(number_of_rows) && continue
        @parallel (1:number_of_rows) reverse_row_group!(
            op, primals, shadows, context, nx, first_row
        )
    end

    # Boundary rows run separately because periodic stencils can connect them.
    for row in (1, ny)
        @parallel (1:1) reverse_row_group!(op, primals, shadows, context, nx, row)
    end

    return nothing
end

@parallel_indices (row_in_group) function reverse_row_group!(
        op, primals, shadows, context, nx, first_row
    )
    row = first_row + 2 * (row_in_group - 1)

    # Adjacent columns can share shadow entries, so process one row sequentially.
    for column in 1:nx
        apply_point!(op, primals, shadows, context, column, row)
    end

    return nothing
end

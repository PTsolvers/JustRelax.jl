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
functions touch at most two consecutive indices in each dimension (with a fixed offset per
array). Points two apart in both dimensions therefore never collide. Even-sized dimensions
use two colors; odd-sized dimensions use a third color for the final index so the first and
last points cannot collide across a periodic seam.
"""
function enzyme_reverse_pointwise!(f, n::NTuple{2, Integer}, args::Tuple)
    primals, shadows = map(first, args), map(kernel_shadow, map(last, args))
    reverse_colored!(f, n, primals, shadows, nothing)
    return nothing
end

# A type object cannot be passed as GPU kernel data. Construct Active on the device.
@inline kernel_shadow(dx) = dx === Enzyme.Active ? Val(:active) : dx

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
# @inline annotate(x, ::typeof(Enzyme.Active)) = Enzyme.Active(x)
@inline annotate(x, ::Val{:active}) = Enzyme.Active(x)

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
    for (first_column, column_stride, number_of_columns) in color_groups(nx)
        iszero(number_of_columns) && continue
        for (first_row, row_stride, number_of_rows) in color_groups(ny)
            iszero(number_of_rows) && continue
            @parallel (1:number_of_columns, 1:number_of_rows) reverse_point_group!(
                op, primals, shadows, context,
                first_column, column_stride, first_row, row_stride,
            )
        end
    end
    return nothing
end

function color_groups(n::Integer)
    alternating = n - isodd(n)
    groups = (
        (1, 2, cld(alternating, 2)),
        (2, 2, fld(alternating, 2)),
    )
    return isodd(n) ? (groups..., (n, 0, 1)) : groups
end

@parallel_indices (column_in_group, row_in_group) function reverse_point_group!(
        op, primals, shadows, context,
        first_column, column_stride, first_row, row_stride,
    )
    column = first_column + column_stride * (column_in_group - 1)
    row = first_row + row_stride * (row_in_group - 1)
    apply_point!(op, primals, shadows, context, column, row)

    return nothing
end

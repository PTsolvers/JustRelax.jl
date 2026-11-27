"""
    AbstractBoundaryConditions

Supertype for all boundary condition types, e.g. [`TemperatureBoundaryConditions`](@ref).
"""
abstract type AbstractBoundaryConditions end

"""
    AbstractFlowBoundaryConditions

Supertype for velocity/displacement boundary condition types
([`VelocityBoundaryConditions`](@ref), [`DisplacementBoundaryConditions`](@ref)).
"""
abstract type AbstractFlowBoundaryConditions <: AbstractBoundaryConditions end

@inline _bc_value(bc, key::Symbol) = hasproperty(bc, key) ? getproperty(bc, key) : false
@inline function _thermal_bc_tuple(bc, ::Val{2})
    return (
        left = _bc_value(bc, :left),
        right = _bc_value(bc, :right),
        top = _bc_value(bc, :top),
        bot = _bc_value(bc, :bot),
    )
end
@inline function _thermal_bc_tuple(bc, ::Val{3})
    return (
        left = _bc_value(bc, :left),
        right = _bc_value(bc, :right),
        front = _bc_value(bc, :front),
        back = _bc_value(bc, :back),
        top = _bc_value(bc, :top),
        bot = _bc_value(bc, :bot),
    )
end

"""
    TemperatureBoundaryConditions(; no_flux, constant_flux, constant_value, periodic, dirichlet)

Create thermal boundary conditions for 2D or 3D temperature fields.

Boundary tuples use `left`, `right`, `top`, and `bot` in 2D. In 3D they also use
`front` and `back`. Omitted faces are filled with `false`, and the dimensionality
is inferred from the longest boundary tuple that is passed: pass a complete
six-face tuple for a 3D set, since the defaults are four-face 2D tuples. A longest
tuple with any other number of faces, or a face name that is not valid for the
inferred dimension, is rejected.

The face values have the following meaning:

- `no_flux`: `true` copies the adjacent interior temperature into the ghost layer.
- `constant_value`: numeric values prescribe the boundary temperature through the
  ghost value `Tghost = 2 * value - Tinterior`.
- `constant_flux`: numeric values prescribe heat fluxes in the pseudo-transient
  diffusion flux kernels.
- `periodic`: `true` copies the opposite interior temperature into the ghost layer.
- `false`: leaves that boundary inactive for the corresponding condition.

Periodic faces must be enabled in pairs (`left`/`right`, `front`/`back`, or
`bot`/`top`) and cannot also use `no_flux`, `constant_flux`, or
`constant_value`.

`dirichlet` accepts the mask-based Dirichlet forms supported by `Dirichlet`, for
example `(; constant = value, mask = mask)`.

# Examples

```julia
TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, top = false, bot = false),
    constant_value = (top = 273.0, bot = 1573.0),
)

TemperatureBoundaryConditions(;
    no_flux = (left = false, right = false, top = false, bot = false),
    constant_value = (top = 273.0, bot = 1573.0),
    periodic = (left = true, right = true, top = false, bot = false),
)
```
"""
struct TemperatureBoundaryConditions{T1, T2, T3, T4, D, nD} <: AbstractBoundaryConditions
    no_flux::T1
    constant_flux::T2
    constant_value::T3
    periodic::T4
    dirichlet::D
    function TemperatureBoundaryConditions(;
            no_flux::T1 = (left = true, right = false, top = false, bot = false),
            constant_flux::T2 = (left = false, right = false, top = false, bot = false),
            constant_value::T3 = (left = false, right = false, top = false, bot = false),
            periodic::T4 = (left = false, right = false, top = false, bot = false),
            dirichlet = (; constant = nothing, mask = nothing),
        ) where {T1, T2, T3, T4}

        D = Dirichlet(dirichlet)
        nfaces = maximum(length, (no_flux, constant_flux, constant_value, periodic))
        nD = if nfaces == 4
            2
        elseif nfaces == 6
            3
        else
            throw(ArgumentError("thermal boundary conditions must use 4 (2D) or 6 (3D) faces"))
        end

        check_thermal_bc_faces(nD, no_flux, constant_flux, constant_value, periodic)

        # fill omitted faces with `false` and expand to 3D
        dummy = (; front = false, back = false)

        no_flux_exp = merge(dummy, _thermal_bc_tuple(no_flux, Val(nD)))
        constant_flux_exp = merge(dummy, _thermal_bc_tuple(constant_flux, Val(nD)))
        constant_value_exp = merge(dummy, _thermal_bc_tuple(constant_value, Val(nD)))
        periodic_exp = merge(dummy, _thermal_bc_tuple(periodic, Val(nD)))

        check_periodic_pairs(periodic_exp, nD)
        check_periodic_conflicts(
            periodic_exp, no_flux_exp, constant_flux_exp, constant_value_exp
        )

        return new{typeof(no_flux_exp), typeof(constant_flux_exp), typeof(constant_value_exp), typeof(periodic_exp), typeof(D), nD}(
            no_flux_exp, constant_flux_exp, constant_value_exp, periodic_exp, D
        )
    end
end

"""
    DisplacementBoundaryConditions(; no_slip, free_slip, periodic, free_surface=false)

Define 2D or 3D boundary conditions for the displacement field. Each face is
controlled independently with a named tuple. Use exactly four faces in 2D
(`left`, `right`, `top`, `bot`) or six in 3D (also `front`, `back`).

A face can carry at most one of `no_slip`, `free_slip`, and `periodic`; a face
where all three are `false` is left for the caller to prescribe explicitly.
Periodic faces must be enabled in pairs (`left`/`right`, `front`/`back`, or
`bot`/`top`), and a periodic top face is incompatible with `free_surface=true`.
"""
struct DisplacementBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    periodic::T
    free_surface::Bool

    function DisplacementBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            periodic::T = map(_ -> false, no_slip),
            free_surface::Bool = false,
        ) where {T}
        @assert length(no_slip) === length(free_slip) === length(periodic)
        check_flow_bcs(no_slip, free_slip, periodic, free_surface)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T, nD}(no_slip, free_slip, periodic, free_surface)
    end
end

"""
    VelocityBoundaryConditions(; no_slip, free_slip, periodic, free_surface=false, dirichlet=(;))

Define 2D or 3D boundary conditions for the velocity field. Face names are
`left`, `right`, `top`, and `bot` in 2D, with `front` and `back` added in 3D.

`no_slip`, `free_slip`, and `periodic` are mutually exclusive on each face.
Faces where all three are `false` are not modified by `flow_bcs!`, which is how
a prescribed velocity field is imposed. Periodic faces must be enabled in pairs
(`left`/`right`, `front`/`back`, or `bot`/`top`), and a periodic top face is
incompatible with `free_surface=true`.

`dirichlet` prescribes the velocity inside the domain, for example a region moving
at a fixed plate velocity. It takes one entry per velocity component, `Vx` and `Vy`,
each in any form accepted by `Dirichlet`: `(; constant = v, mask = mask)` for a
uniform value (zero included), or a `DirichletBoundaryCondition(value, Mask(mask))`
for a value that varies in space. Masks and values are sized like the velocity
component they constrain, ghost nodes included (`(nx+1, ny+2)` for `Vx`,
`(nx+2, ny+1)` for `Vy`). Omitted components are unconstrained. A
`DirichletBoundaryCondition` built from a single array infers its mask from the
non-zero entries, so it cannot prescribe a velocity of exactly zero. Interior Dirichlet
velocities are enforced by the 2D `solve_DYREL!` and
`solve_VariationalDYREL!` solvers.

# Examples

```julia
mask = zeros(nx + 1, ny + 2)
mask[10:20, 5:8] .= 1
VelocityBoundaryConditions(;
    free_slip = (left = true, right = true, top = true, bot = true),
    dirichlet = (; Vx = (; constant = 1.0e-9, mask = mask)),
)
```
"""
struct VelocityBoundaryConditions{T, nD, D} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    periodic::T
    free_surface::Bool
    dirichlet::D

    function VelocityBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            periodic::T = map(_ -> false, no_slip),
            free_surface::Bool = false,
            dirichlet::NamedTuple = (;),
        ) where {T}
        @assert length(no_slip) === length(free_slip) === length(periodic)
        check_flow_bcs(no_slip, free_slip, periodic, free_surface)

        nD = length(no_slip) == 4 ? 2 : 3
        D = parse_velocity_dirichlet(dirichlet, Val(nD))
        return new{T, nD, typeof(D)}(no_slip, free_slip, periodic, free_surface, D)
    end
end

function parse_velocity_dirichlet(dirichlet::NamedTuple, ::Val{nD}) where {nD}
    components = nD == 2 ? (:Vx, :Vy) : (:Vx, :Vy, :Vz)
    for (k, v) in pairs(dirichlet)
        k in components || throw(
            ArgumentError("unknown velocity component `$k` in `dirichlet`; valid components are $components")
        )
        v isa NamedTuple && !haskey(v, :constant) && throw(
            ArgumentError("`dirichlet.$k` needs a `constant`; use `DirichletBoundaryCondition(value, Mask(mask))` for a varying value")
        )
    end
    return NamedTuple{components}(map(k -> Dirichlet(get(dirichlet, k, (;))), components))
end

@inline has_dirichlet(::AbstractDirichletBoundaryCondition{Nothing, Nothing}) = false
@inline has_dirichlet(::AbstractDirichletBoundaryCondition) = true

# Interior nodes only: the solver never updates wall-normal velocities, so a value written
# there would persist as in- or outflow through the wall.
function apply_interior_dirichlet!(V, bc)
    has_dirichlet(bc) || return nothing
    I = ntuple(d -> 2:(size(V, d) - 1), ndims(V))
    value = bc.value isa ConstantArray ? bc.value : view(bc.value, I...)
    apply_mask!(view(V, I...), value, Mask(view(bc.mask.mask, I...)))
    return nothing
end

"""
    velocity_dirichlet(bcs)

Return the interior Dirichlet velocity conditions of `bcs` as a NamedTuple with one
condition per velocity component (`Vx`, `Vy`[, `Vz`]). Flow boundary conditions without
interior conditions, such as `DisplacementBoundaryConditions`, return empty conditions.
"""
@inline velocity_dirichlet(bcs::VelocityBoundaryConditions) = bcs.dirichlet
@inline velocity_dirichlet(::DisplacementBoundaryConditions{T, nD}) where {T, nD} =
    parse_velocity_dirichlet((;), Val(nD))

"""
    check_velocity_dirichlet(bcs, V)

Check that the interior Dirichlet velocity conditions of `bcs` can be imposed on the
velocity components `V`: each mask must be sized like its component, and interior
conditions are only supported by the 2D DYREL solvers.
"""
function check_velocity_dirichlet(bcs::AbstractFlowBoundaryConditions, V::NTuple{N, AbstractArray}) where {N}
    D = velocity_dirichlet(bcs)
    N == 3 && any(has_dirichlet, D) &&
        throw(ArgumentError("interior Dirichlet velocities are only supported by the 2D DYREL solvers"))
    foreach(check_dirichlet_size, D, V, keys(D))
    return nothing
end

function check_dirichlet_size(bc, V, name)
    has_dirichlet(bc) || return nothing
    size(bc.mask.mask) == size(V) || throw(
        DimensionMismatch(
            "the `dirichlet` mask for $name has size $(size(bc.mask.mask)), but $name has size $(size(V))"
        )
    )
    return nothing
end

"""
    check_flow_bcs(no_slip, free_slip, periodic, free_surface)

Throw if flow boundary conditions conflict or if a periodic direction is not paired.
A boundary flagged as neither `no_slip`, `free_slip`, nor `periodic` is left untouched
by `flow_bcs!`, which is how a prescribed velocity field is imposed: the caller writes
the boundary and ghost values itself.
"""
function check_flow_bcs(no_slip::T, free_slip::T, periodic::T, free_surface) where {T}
    nD = length(periodic) == 4 ? 2 : 3
    check_periodic_pairs(periodic, nD)
    for (v1, v2, vp, k) in
        zip(values(no_slip), values(free_slip), values(periodic), keys(no_slip))
        if count(==(true), (v1, v2, vp)) > 1
            error(
                "Incompatible boundary conditions on the $k boundary",
            )
        end
    end
    free_surface && periodic.top &&
        error("Incompatible boundary conditions: the top can't be both periodic and free_surface")
    return
end

function check_periodic_pairs(periodic, nD)
    pairs = if nD == 2
        ((:left, :right), (:bot, :top))
    else
        ((:left, :right), (:front, :back), (:bot, :top))
    end
    for (a, b) in pairs
        getproperty(periodic, a) == getproperty(periodic, b) ||
            error("Periodic boundary conditions must be paired: $a and $b")
    end
    return
end

function check_thermal_bc_faces(nD, conditions...)
    faces = nD == 2 ? (:left, :right, :top, :bot) : (:left, :right, :front, :back, :top, :bot)
    for condition in conditions, k in keys(condition)
        k in faces || throw(
            ArgumentError("unknown $(nD)D thermal boundary face `$k`; valid faces are $faces")
        )
    end
    return
end

function check_periodic_conflicts(periodic, conditions...)
    for k in keys(periodic)
        getproperty(periodic, k) || continue
        any(condition -> getproperty(condition, k) !== false, conditions) &&
            error("Incompatible boundary conditions on the $k boundary")
    end
    return
end

"""
    periodic_dims(bcs::AbstractFlowBoundaryConditions)
    periodic_dims(stokes::StokesArrays)

Periodicity of each spatial direction, as an `N`-tuple of `Bool`s ordered `(x, y[, z])`.

A boundary-condition object reports what the caller asked for. A `StokesArrays` reports what its
momentum residuals were allocated for: in a periodic direction the two coincident boundary faces
are a single unknown, so that direction carries one residual row per cell rather than one per
interior face. `check_periodic_bcs` requires the two to agree.
"""
function periodic_dims end

@inline periodic_dims(bcs::VelocityBoundaryConditions{T, nD}) where {T, nD} =
    _periodic_dims(bcs.periodic, Val(nD))
@inline periodic_dims(bcs::DisplacementBoundaryConditions{T, nD}) where {T, nD} =
    _periodic_dims(bcs.periodic, Val(nD))

# `check_periodic_pairs` requires both faces of a direction to agree, so the lower face decides.
@inline _periodic_dims(periodic, ::Val{2}) = (periodic.left, periodic.bot)
@inline _periodic_dims(periodic, ::Val{3}) = (periodic.left, periodic.front, periodic.bot)

@inline momentum_residuals(R::Residual, ::Val{2}) = (R.Rx, R.Ry)
@inline momentum_residuals(R::Residual, ::Val{3}) = (R.Rx, R.Ry, R.Rz)

@inline function periodic_dims(stokes::StokesArrays)
    dim = static_dims(stokes)
    R = momentum_residuals(stokes.R, dim)
    return ntuple(d -> size(R[d], d) == size(stokes.P, d), dim)
end

"""
    check_periodic_bcs(stokes, bcs, igg, di_center)

Throw unless the periodic directions of `bcs` are ones the solver can actually solve.

The momentum row of a periodic seam only exists if `stokes` was allocated for it, which is what
`StokesArrays(backend, ni, bcs)` does; a `stokes` built without the boundary conditions leaves that
row out and the seam velocity is then frozen at its initial value. The other two conditions mark
combinations that are not implemented rather than ones that are wrong in principle.
"""
function check_periodic_bcs(stokes, bcs::AbstractFlowBoundaryConditions, igg, di_center)
    bc_periodic = periodic_dims(bcs)
    any(bc_periodic) || return nothing

    array_periodic = periodic_dims(stokes)
    bc_periodic == array_periodic || error(
        "Periodic directions $(bc_periodic) do not match the StokesArrays allocation \
        $(array_periodic). Build the containers from the boundary conditions, \
        `StokesArrays(backend, ni, bcs)`, so the periodic seams get a momentum row."
    )

    igg.nprocs == 1 || error(
        "Periodic boundary conditions are only supported on a single MPI rank. \
        `init_global_grid` with `periodx`/`periody` rescales the global grid against the \
        overlap, which does not match the `Geometry` the solver is given."
    )

    for (d, isperiodic) in enumerate(bc_periodic)
        isperiodic && di_center[d] isa AbstractVector && error(
            "Periodic boundary conditions require uniform grid spacing in the periodic \
            direction; direction $d has a variable cell-center spacing. The seam face spans the \
            wrap, so a single spacing value has to describe it."
        )
    end
    return nothing
end

# The flow boundary conditions sit at a different position in each `solve!` signature, so the
# guards below pick them out of the argument tuple by type. `nothing` means the call has none,
# which is not this check's business.
@inline flow_bcs_of(args::Tuple) = _flow_bcs_of(args...)
@inline _flow_bcs_of(bcs::AbstractFlowBoundaryConditions, rest...) = bcs
@inline _flow_bcs_of(_, rest...) = _flow_bcs_of(rest...)
@inline _flow_bcs_of() = nothing

"""
    reject_periodic_bcs(bcs, solver)

Throw if `bcs` asks for a periodic direction, naming the `solver` that does not implement it.

Only the DYREL solvers write the momentum row for a periodic seam: `solve_DYREL!` in 2D and 3D,
and `solve_VariationalDYREL!` in 2D. Every other path would leave that row unsolved and silently
return a field pinned to its initial guess at the seam.
"""
function reject_periodic_bcs(bcs::AbstractFlowBoundaryConditions, solver)
    any(periodic_dims(bcs)) && error(
        "Periodic boundary conditions are not implemented for $solver: the momentum row of the \
        periodic seam would be left unsolved. Use `solve_DYREL!`, or \
        `solve_VariationalDYREL!` in 2D."
    )
    return nothing
end

reject_periodic_bcs(::Nothing, solver) = nothing

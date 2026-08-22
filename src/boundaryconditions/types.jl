abstract type AbstractBoundaryConditions end
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
`front` and `back`. Omitted faces are filled with `false`, and the dimensionality is
inferred from the longest boundary tuple that is passed.

The face values have the following meaning:

- `no_flux`: `true` copies the adjacent interior temperature into the ghost layer.
- `constant_value`: numeric values prescribe the boundary temperature through the
  ghost value `Tghost = 2 * value - Tinterior`.
- `constant_flux`: numeric values prescribe heat fluxes in the pseudo-transient
  diffusion flux kernels.
- `periodic`: `true` copies the opposite interior temperature into the ghost layer.
- `false`: leaves that boundary inactive for the corresponding condition.

`dirichlet` accepts the mask-based Dirichlet forms supported by `Dirichlet`, for
example `(; constant = value, mask = mask)`.

# Examples

```julia
TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, top = false, bot = false),
    constant_value = (top = 273.0, bot = 1573.0),
)

TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, front = true, back = true, top = false, bot = false),
    constant_flux = (top = 0.0, bot = 0.03),
    periodic = (left = false, right = false, front = false, back = false, top = false, bot = false),
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
        nD = maximum(length, (no_flux, constant_flux, constant_value, periodic)) == 4 ? 2 : 3

        # expand to 3D
        dummy = (; front = false, back = false)

        no_flux_exp = merge(dummy, no_flux)
        constant_flux_exp = merge(dummy, constant_flux)
        constant_value_exp = merge(dummy, constant_value)
        periodic_exp = merge(dummy, periodic)

        check_periodic_pairs(periodic_exp, nD)
        check_periodic_conflicts(
            periodic_exp, no_flux_exp, constant_flux_exp, constant_value_exp
        )

        return new{typeof(no_flux_exp), typeof(constant_flux_exp), typeof(constant_value_exp), typeof(periodic_exp), typeof(D), nD}(
            no_flux_exp, constant_flux_exp, constant_value_exp, periodic_exp, D
        )
    end
end

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
struct VelocityBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    periodic::T
    free_surface::Bool

    function VelocityBoundaryConditions(;
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

function check_periodic_conflicts(periodic, conditions...)
    for k in keys(periodic)
        getproperty(periodic, k) || continue
        any(condition -> getproperty(condition, k) !== false, conditions) &&
            error("Incompatible boundary conditions on the $k boundary")
    end
    return
end

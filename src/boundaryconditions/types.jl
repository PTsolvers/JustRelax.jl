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
        nD = maximum(length, (no_flux, constant_flux, constant_value, periodic)) > 4 ? 3 : 2
        no_flux = _thermal_bc_tuple(no_flux, Val(nD))
        constant_flux = _thermal_bc_tuple(constant_flux, Val(nD))
        constant_value = _thermal_bc_tuple(constant_value, Val(nD))
        periodic = _thermal_bc_tuple(periodic, Val(nD))
        return new{typeof(no_flux), typeof(constant_flux), typeof(constant_value), typeof(periodic), typeof(D), nD}(
            no_flux, constant_flux, constant_value, periodic, D
        )
    end
end

struct DisplacementBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool

    function DisplacementBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            free_surface::Bool = false,
        ) where {T}
        @assert length(no_slip) === length(free_slip)
        check_flow_bcs(no_slip, free_slip)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T, nD}(no_slip, free_slip, free_surface)
    end
end
struct VelocityBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool

    function VelocityBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            free_surface::Bool = false,
        ) where {T}
        @assert length(no_slip) === length(free_slip)
        check_flow_bcs(no_slip, free_slip)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T, nD}(no_slip, free_slip, free_surface)
    end
end

function check_flow_bcs(no_slip::T, free_slip::T) where {T}
    v1 = values(no_slip)
    v2 = values(free_slip)
    k = keys(no_slip)
    for (v1, v2, k) in zip(v1, v2, k)
        if v1 == v2
            error(
                "Incompatible boundary conditions. The $k boundary condition can't be the same for no_slip and free_slip",
            )
        end
    end
    return
end

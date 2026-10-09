"""
    observation_mask(stokes_ad, grid, observation)

Resolve the objective `J` of the adjoint solve. `observation` is either

  - a box, `(; field, center, half_width)`: `J = Σ field` over the nodes of `field` whose
    coordinates lie within `half_width` of `center`, or
  - a weighted field, `(; field, weights)`: `J = Σ weights · field`, with `weights` an array of
    the size of the observed field, i.e. `∂J/∂field`. Weights can follow the material, e.g. a
    phase fraction, to track a body as it moves.

`field` is `:Vx`, `:Vy` or `:P`. Returns `(; field, target, i, j, weights)`, where `target` is the
adjoint array the objective seeds; `i, j` are the box indices (`nothing` for weights) and
`weights` is `nothing` for a box.
"""
function observation_mask(stokes_ad, grid, observation)
    field = observation.field

    target, x, y, offset = if field === :Vx
        stokes_ad.V.Vx, grid.xvi[1], grid.xci[2], (0, 1)
    elseif field === :Vy
        stokes_ad.V.Vy, grid.xci[1], grid.xvi[2], (1, 0)
    elseif field === :P
        stokes_ad.P, grid.xci[1], grid.xci[2], (0, 0)
    else
        throw(ArgumentError("observation field must be :Vx, :Vy, or :P"))
    end

    if haskey(observation, :weights)
        weights = observation.weights
        size(weights) == size(target) || throw(
            DimensionMismatch("observation weights must have the size of the $field field, $(size(target))")
        )
        any(!iszero, weights) || throw(ArgumentError("the observation weights are zero everywhere"))
        return (; field, target, i = nothing, j = nothing, weights)
    end

    (; center, half_width) = observation
    i = findall(xi -> abs(xi - center[1]) ≤ half_width[1], x) .+ offset[1]
    j = findall(yj -> abs(yj - center[2]) ≤ half_width[2], y) .+ offset[2]
    (isempty(i) || isempty(j)) && throw(ArgumentError("the observation region does not contain any $field nodes"))
    return (; field, target, i, j, weights = nothing)
end

# Seed the adjoint of the observed field with -∂J/∂field: -1 on every node of a box, -weights
# for a weighted objective. `target` has just been zeroed by `initialize_adjoint_iteration!`.
function seed_observation!(observation)
    (; target, i, j, weights) = observation
    if isnothing(weights)
        target[i, j] .= -1.0
    else
        target .= .-weights
    end
    return nothing
end

# The objective must only read unknowns of the reduced system: an observed velocity in an
# eliminated (air) row is not a degree of freedom of the forward solve.
function check_observation_mask(observation, maskV, maskP)
    (; field, i, j, weights) = observation
    mask, offset = if field === :Vx
        Array(maskV[1]), 1
    elseif field === :Vy
        Array(maskV[2]), 1
    else
        Array(maskP), 0
    end
    valid(a, b) = checkbounds(Bool, mask, a - offset, b - offset) && mask[a - offset, b - offset]
    # a weighted objective only reads the nodes it weights
    nodes = isnothing(weights) ? ((a, b) for a in i, b in j) : (Tuple(I) for I in findall(!iszero, Array(weights)))
    all(n -> valid(n...), nodes) || throw(
        ArgumentError("the observation region must lie in the rock part of the domain; some $field nodes are masked out")
    )
    return nothing
end

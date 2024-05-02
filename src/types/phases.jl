struct PhaseRatio{T}
    vertex::T
    center::T

    PhaseRatio(vertex::T, center::T) where {T<:AbstractArray} = new{T}(vertex, center)
end

@inline PhaseRatio(::Type{CPUBackend}, ni, num_phases) = PhaseRatio(ni, num_phases)

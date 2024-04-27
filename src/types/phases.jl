struct PhaseRatio{T}
    vertex::T
    center::T

    PhaseRatio(vertex::T, center::T) where {T<:AbstractArray} = new{T}(vertex, center)
end

# function PhaseRatio(ni::NTuple{N,Integer}, num_phases::Integer) where {N}
#     center = @fill(0.0, ni..., celldims = (num_phases,))
#     vertex = @fill(0.0, ni .+ 1..., celldims = (num_phases,))
#     # T = typeof(center)
#     return PhaseRatio(vertex, center)
# end

@inline PhaseRatio(::Type{CPUBackend}, ni, num_phases) = PhaseRatio(ni, num_phases)

# backend trait
# @inline backend(x::PhaseRatio) = backend(x.center.data)

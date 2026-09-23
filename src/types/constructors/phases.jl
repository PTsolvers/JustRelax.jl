function PhaseRatio(
        ::Type{CPUBackend}, ni::NTuple{N, Integer}, num_phases::Integer
    ) where {N}
    return PhaseRatio(ni, num_phases)
end

function PhaseRatio(ni::NTuple{N, Integer}, num_phases::Integer) where {N}
    center = @fill(0.0, ni..., celldims = (num_phases,))
    vertex = @fill(0.0, ni .+ 1..., celldims = (num_phases,))
    # T = typeof(center)
    return JustRelax.PhaseRatio(vertex, center)
end

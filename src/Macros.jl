export @allocate, @fill

# Memory allocators
macro allocate(ni...)
    return esc(:(PTArray(undef, $(ni...))))
end

macro fill(A, ni...)
    return esc(:(PTArray(fill(eltype(PTArray)($A), $(ni...)))))
end

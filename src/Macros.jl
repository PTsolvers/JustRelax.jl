export @allocate, @fill

# Memory allocators
macro allocate(ni...)
    return esc(:(PTArray(undef, $(ni...))))
end
macro allocate(nx, ny)
    return esc(:(PTArray(undef, $nx, $ny)))
end
macro allocate(nx, ny, nz)
    return esc(:(PTArray(undef, $nx, $ny, $nz)))
end

macro fill(A, ni...)
    return esc(:(fill(eltype(PTArray)($A), $(ni...))))
end
macro fill(A, nx)
    return esc(:(fill(eltype(PTArray)($A), $nx)))
end
macro fill(A, nx, ny)
    return esc(:(fill(eltype(PTArray)($A), $nx, $ny)))
end
macro fill(A, nx, ny, nz)
    return esc(:(fill(eltype(PTArray)($A), $nx, $ny, $nz)))
end

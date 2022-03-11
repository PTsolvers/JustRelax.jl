export @allocate, @fill

# Memory allocators
macro allocate(ni...)       esc(:(PTArray(undef, $(ni...)))) end
macro allocate(nx, ny)      esc(:(PTArray(undef, $nx, $ny))) end
macro allocate(nx, ny, nz)  esc(:(PTArray(undef, $nx, $ny, $nz))) end

macro fill(A, ni...)      esc(:(fill(eltype(PTArray)($A), $(ni...)))) end
macro fill(A, nx)         esc(:(fill(eltype(PTArray)($A), $nx))) end
macro fill(A, nx, ny)     esc(:(fill(eltype(PTArray)($A), $nx, $ny))) end
macro fill(A, nx, ny, nz) esc(:(fill(eltype(PTArray)($A), $nx, $ny, $nz))) end
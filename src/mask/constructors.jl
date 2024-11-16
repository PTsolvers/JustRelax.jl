function Mask(nx, ny, inds::Vararg{UnitRange, 2})
    mask = @zeros(nx, ny)
    @views mask[inds...] .= 1
    return JustRelax.Mask(mask)
end

function Mask(nx, ny, nz, inds::Vararg{UnitRange,3})
    mask = @zeros(nx, ny, nz)
    @views mask[inds...] .= 1
    return JustRelax.Mask(mask)
end

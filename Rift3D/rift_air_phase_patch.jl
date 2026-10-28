# Local workaround for unsupported particle interpolation in full, unevenly
# populated air cells. Do not infer rock mixtures from absent particle support.
# Works in 2D and 3D; `offsets[d]` is true when the ratios are staggered along `d`.
function patch_unsupported_air_ratios!(phase_ratios, air_phase)
    for (ratios, offsets) in staggered_ratios(phase_ratios)
        @parallel (@idx size(ratios)) patch_unsupported_air_ratio!(
            ratios, phase_ratios.center, air_phase, offsets
        )
    end
    return nothing
end

staggered_ratios(phase_ratios::JustPIC.PhaseRatios{B, T}) where {B, T <: AbstractMatrix} = (
    (phase_ratios.Vx, (true, false)),
    (phase_ratios.Vy, (false, true)),
    (phase_ratios.vertex, (true, true)),
)

staggered_ratios(phase_ratios::JustPIC.PhaseRatios) = (
    (phase_ratios.Vx, (true, false, false)),
    (phase_ratios.Vy, (false, true, false)),
    (phase_ratios.Vz, (false, false, true)),
    (phase_ratios.vertex, (true, true, true)),
    (phase_ratios.yz, (false, true, true)),
    (phase_ratios.xz, (true, false, true)),
    (phase_ratios.xy, (true, true, false)),
)

@parallel_indices (I...) function patch_unsupported_air_ratio!(ratios, centers, air_phase, offsets)
    mixture = ratios[I...]
    if any(x -> !isfinite(x), mixture)
        ni = size(centers)
        hi = clamp.(I, 1, ni)
        lo = ifelse.(offsets, clamp.(I .- 1, 1, ni), hi)
        pure_air = true
        for J in CartesianIndices(UnitRange.(lo, hi))
            center = centers[Tuple(J)...]
            pure_air &= all(isfinite, center) && isone(center[air_phase])
        end
        if pure_air
            ratios[I...] = Base.setindex(zero(mixture), one(eltype(mixture)), air_phase)
        end
    end
    return nothing
end

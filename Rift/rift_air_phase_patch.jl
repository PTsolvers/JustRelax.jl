# Local workaround for unsupported particle interpolation in full, unevenly
# populated air cells. Do not infer rock mixtures from absent particle support.
function patch_unsupported_air_ratios!(phase_ratios, air_phase)
    for (ratios, offsets) in (
            (phase_ratios.Vx, (true, false)),
            (phase_ratios.Vy, (false, true)),
            (phase_ratios.vertex, (true, true)),
        )
        @parallel (@idx size(ratios)) patch_unsupported_air_ratio!(
            ratios, phase_ratios.center, air_phase, offsets
        )
    end
    return nothing
end

@parallel_indices (i, j) function patch_unsupported_air_ratio!(ratios, centers, air_phase, offsets)
    mixture = ratios[i, j]
    if any(x -> !isfinite(x), mixture)
        nx, ny = size(centers)
        iR, jR = clamp(i, 1, nx), clamp(j, 1, ny)
        iL = offsets[1] ? clamp(i - 1, 1, nx) : iR
        jL = offsets[2] ? clamp(j - 1, 1, ny) : jR
        pure_air = true
        for jj in jL:jR, ii in iL:iR
            center = centers[ii, jj]
            pure_air &= all(isfinite, center) && isone(center[air_phase])
        end
        if pure_air
            ratios[i, j] = Base.setindex(zero(mixture), one(eltype(mixture)), air_phase)
        end
    end
    return nothing
end

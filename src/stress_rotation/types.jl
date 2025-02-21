struct StressParticles{backend, nNormal, nShear, T}
    τ_normal::NTuple{nNormal, T}
    τ_shear::NTuple{nShear, T}
    ω::NTuple{nShear, T}

    function StressParticles(
            backend, τ_normal::NTuple{nNormal, T}, τ_shear::NTuple{nShear, T}, ω::NTuple{nShear, T}
        ) where {nNormal, nShear, T}
        return new{backend, nNormal, nShear, T}(τ_normal, τ_shear, ω)
    end
end

@inline unwrap(x::StressParticles) = tuple(x.τ_normal..., x.τ_shear..., x.ω...)
@inline normal_stress(x::StressParticles) = x.τ_normal
@inline shear_stress(x::StressParticles) = x.τ_shear
@inline shear_vorticity(x::StressParticles) = x.ω

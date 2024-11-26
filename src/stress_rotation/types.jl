struct StressParticles{backend, nNormal, nShear, T}
    τ_normal::NTuple{nNormal, T}
    τ_shear::NTuple{nShear, T}
    ω::NTuple{nShear, T}

    function StressParticles(backend, τ_normal::NTuple{nNormal, T}, τ_shear::NTuple{nShear, T}, ω::NTuple{nShear, T}) where {nNormal, nShear, T}
        new{backend,nNormal, nShear, T}(τ_normal, τ_shear, ω)
    end
end

@inline normal_stress(x::StressParticles) = x.τ_normal
@inline shear_stress(x::StressParticles) = x.τ_shear
@inline shear_vorticity(x::StressParticles) = x.ω
"""
    StressParticles{backend, nNormal, nShear, T}

Particle-borne deviatoric stress and vorticity: the normal components
`τ_normal`, the shear components `τ_shear`, and the vorticity components `ω`, each a
tuple of particle cell arrays. Carrying the old stress on the particles instead of on
the grid keeps it attached to the material as it advects and rotates.

Build one from the particles it follows with `StressParticles(particles)`, advance it with
`rotate_stress!`, and write it back onto `stokes.τ_o` with `stress2grid!`.
"""
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

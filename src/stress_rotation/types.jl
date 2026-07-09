"""
    StressParticles{backend, nNormal, nShear, T}

Per-particle stress carrier used to rotate the deviatoric stress tensor along the flow as
particles advect. It stores the normal stress components `τ_normal`, the shear stress
components `τ_shear`, and the vorticity `ω` as tuples of particle cell arrays (`nNormal` and
`nShear` components, respectively: `(2, 1)` in 2D and `(3, 3)` in 3D).

Construct one from a set of `JustPIC` particles with [`StressParticles(particles)`](@ref).
Interpolate between the particle stresses and the grid with [`stress2grid!`](@ref) and
[`rotate_stress!`](@ref).
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

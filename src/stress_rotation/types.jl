"""
    StressParticles{backend, nNormal, nShear, T, G}

Particle-borne deviatoric stress and vorticity: the normal components
`τ_normal`, the shear components `τ_shear`, and the vorticity components `ω`, each a
tuple of particle cell arrays. Carrying the old stress on the particles instead of on
the grid keeps it attached to the material as it advects and rotates.

`τ_grid` holds the grid stress that `stress2grid!` last produced from the particles,
on the grid locations `rotate_stress!` reads `stokes.τ` from: cell centers for the
normal components, vertices for the 2D shear component and cell centers for the 3D
shear components. `rotate_stress!` adds `stokes.τ - τ_grid` to the particles.

Build one from the particles it follows with `StressParticles(particles)`, advance it with
`rotate_stress!`, and write it back onto `stokes.τ_o` with `stress2grid!`.
"""
struct StressParticles{backend, nNormal, nShear, T, G}
    τ_normal::NTuple{nNormal, T}
    τ_shear::NTuple{nShear, T}
    ω::NTuple{nShear, T}
    τ_grid::G

    function StressParticles(
            backend, τ_normal::NTuple{nNormal, T}, τ_shear::NTuple{nShear, T}, ω::NTuple{nShear, T},
            τ_grid::NTuple{N, AbstractArray},
        ) where {nNormal, nShear, T, N}
        N == nNormal + nShear || throw(
            ArgumentError("`τ_grid` must hold $(nNormal + nShear) stress components, got $N")
        )
        return new{backend, nNormal, nShear, T, typeof(τ_grid)}(τ_normal, τ_shear, ω, τ_grid)
    end
end

"""
    unwrap(x::StressParticles)

Flatten `x` into a single tuple `(τ_normal..., τ_shear..., ω...)` of its underlying
particle cell arrays.
"""
@inline unwrap(x::StressParticles) = tuple(x.τ_normal..., x.τ_shear..., x.ω...)
@inline normal_stress(x::StressParticles) = x.τ_normal
@inline shear_stress(x::StressParticles) = x.τ_shear
@inline shear_vorticity(x::StressParticles) = x.ω
@inline grid_stress(x::StressParticles) = x.τ_grid

"""
    stress_fields(stokes, x::StressParticles)

Grid fields of `stokes` that pair entry by entry with [`unwrap(x)`](@ref unwrap), for
`inject_particles_phase!` to initialize newly injected particles:

```julia
inject_particles_phase!(
    particles, pPhases, (pT, unwrap(pτ)...), (thermal.T, stress_fields(stokes, pτ)...)
)
```

In 2D the normal stresses are read at the vertices (`τ.xx_v`, `τ.yy_v`), so refresh them
with `center2vertex!` first if the solver does not update them. In 3D every component is
read at the cell centers.
"""
@inline stress_fields(stokes, ::StressParticles{B, 2}) where {B} =
    (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.ω.xy)
@inline stress_fields(stokes, ::StressParticles{B, 3}) where {B} = (
    stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c,
    stokes.ω.yz_c, stokes.ω.xz_c, stokes.ω.xy_c,
)

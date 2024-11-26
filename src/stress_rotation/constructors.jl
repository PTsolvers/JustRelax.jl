function StressParticles(particles::JustRelax.Particles{backend, 2}) where {backend}
    τ_normal = init_cell_arrays(particles, Val(2)) # normal stress
    τ_shear = init_cell_arrays(particles, Val(1)) # normal stress
    ω = init_cell_arrays(particles, Val(1)) # vorticity

    return JustRelax.StressParticles(backend, τ_normal, τ_shear, ω)
end

function StressParticles(particles::JustRelax.Particles{backend, 3}) where {backend}
    τ_normal = init_cell_arrays(particles, Val(3)) # normal stress
    τ_shear = init_cell_arrays(particles, Val(3)) # normal stress
    ω = init_cell_arrays(particles, Val(3)) # vorticity

    return JustRelax.StressParticles(backend, τ_normal, τ_shear, ω)
end
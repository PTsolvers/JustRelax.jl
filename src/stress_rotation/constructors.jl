"""
    StressParticles(particles::Particles)

Allocate the stress and vorticity cell arrays that follow `particles`, on the same
backend and with the same per-cell capacity. Two normal and one shear component in 2-D,
three of each in 3-D. The grid reference stress `τ_grid` starts at zero, consistent with
the zero particle stress.
"""
function StressParticles(particles::JustRelax.Particles{backend, 2}) where {backend}
    τ_normal = init_cell_arrays(particles, Val(2)) # normal stress
    τ_shear = init_cell_arrays(particles, Val(1)) # normal stress
    ω = init_cell_arrays(particles, Val(1)) # vorticity

    for field in (τ_normal..., τ_shear..., ω...)
        fill!(field, zero(eltype(field)))
    end

    # particle cells carry one ghost cell on each side of the physical grid
    ni = size(particles.index) .- 2
    τ_grid = (grid_zeros(τ_normal[1], ni), grid_zeros(τ_normal[1], ni), grid_zeros(τ_normal[1], ni .+ 1))

    return JustRelax.StressParticles(backend, τ_normal, τ_shear, ω, τ_grid)
end

function StressParticles(particles::JustRelax.Particles{backend, 3}) where {backend}
    τ_normal = init_cell_arrays(particles, Val(3)) # normal stress
    τ_shear = init_cell_arrays(particles, Val(3)) # normal stress
    ω = init_cell_arrays(particles, Val(3)) # vorticity

    for field in (τ_normal..., τ_shear..., ω...)
        fill!(field, zero(eltype(field)))
    end

    ni = size(particles.index) .- 2
    τ_grid = ntuple(_ -> grid_zeros(τ_normal[1], ni), Val(6))

    return JustRelax.StressParticles(backend, τ_normal, τ_shear, ω, τ_grid)
end

# zero grid array on the device and with the element type of the particle field `A`
function grid_zeros(A, dims)
    B = similar(A.data, eltype(A.data), dims)
    fill!(B, zero(eltype(B)))
    return B
end

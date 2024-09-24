# import JustRelax.compute_ρCp

function subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀,
    phases::JustRelax.PhaseRatio,
    rheology,
    thermal::JustRelax.ThermalArrays,
    stokes::JustRelax.StokesArrays,
    xci,
    di,
)
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases.center, rheology, thermal.Tc, stokes.P, di
    )
    return nothing
end

function subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀,
    phases::AbstractArray{Int,N},
    rheology,
    thermal::JustRelax.ThermalArrays,
    stokes::JustRelax.StokesArrays,
    xci,
    di,
) where {N}
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases, rheology, thermal.Tc, stokes.P, di
    )
    return nothing
end

@parallel_indices (I...) function subgrid_characteristic_time!(
    dt₀, phase_ratios, rheology, T, P, di
)
    Pᵢ, Tᵢ = P[I...], T[I...]
    argsᵢ = (; P=Pᵢ, T=Tᵢ)
    phaseᵢ = phase_ratios[I...]

    # Compute the characteristic timescale `dt₀` of the local cell
    ρCp = compute_ρCp(rheology, phaseᵢ, argsᵢ)
    K = compute_conductivity(rheology, phaseᵢ, argsᵢ)
    sum_dxi = mapreduce(x -> inv(x)^2, +, di)
    dt₀[I...] = ρCp / (2 * K * sum_dxi)

    return nothing
end

# struct SubgridDiffusionArrays{T, CA}
#     pT0::CA # CellArray
#     pΔT::CA # CellArray
#     ΔT::T # Array

#     function SubgridDiffusionArrays(particles, ni)
#         pΔT, pT0 = init_cell_arrays(particles, Val(2))
#         ΔT = @zeros(ni.+1)
#         CA, T = typeof(pΔT), typeof(ΔT)
#         new{T, CA}(pT0, pΔT, ΔT)
#     end
# end

# @inline function init_cell_arrays(particles, ::Val{N}) where {N}
#     return ntuple(
#         _ -> @fill(
#             0.0, size(particles.coords[1])..., celldims = (cellsize(particles.index))
#         ),
#         Val(N),
#     )
# end

# function subgrid_diffusion!(
#     pT, thermal, subgrid_arrays, pPhases, rheology, stokes, particles, T_buffer, xvi,  di, dt
# )
#     (; pT0, pΔT) = subgrid_arrays
#     # ni = size(pT)

#     @copy pT0.data pT.data
#     grid2particle!(pT, xvi, T_buffer, particles)

#     @parallel (@idx ni) subgrid_diffusion!(pT, pT0, pΔT, pPhases, rheology, stokes.P, particles.index, di, dt)
#     particle2grid!(subgrid_arrays.ΔT, pΔT, xvi, particles)

#     @parallel (@idx size(subgrid_arrays.ΔT)) update_ΔT_subgrid!(subgrid_arrays.ΔT, thermal.ΔT)
#     grid2particle!(pΔT, xvi, subgrid_arrays.ΔT, particles)
#     @. pT.data = pT0.data + pΔT.data
#     return nothing
# end

# @parallel_indices (i, j) function update_ΔT_subgrid!(ΔTsubgrid::_T, ΔT::_T) where _T<:AbstractMatrix
#     ΔTsubgrid[i, j] = ΔT[i+1, j] - ΔTsubgrid[i, j]
#     return nothing
# end

# function subgrid_diffusion!(pT, pT0, pΔT, pPhases, rheology, stokes, particles, di, dt)
#     ni = size(pT)
#     @parallel (@idx ni) subgrid_diffusion!(pT, pT0, pΔT, pPhases, rheology, stokes.P, particles.index, di, dt)
# end

# @parallel_indices (I...) function subgrid_diffusion!(pT, pT0, pΔT, pPhases, rheology, P, index, di, dt)

#     P_cell = P[I...]

#     for ip in JustRelax.cellaxes(pT)
#         # early escape if there is no particle in this memory locations
#         doskip(index, ip, I...) && continue

#         pT0ᵢ = @cell pT0[ip, I...]
#         pTᵢ = @cell pT[ip, I...]
#         phase = Int(@cell(pPhases[ip, I...]))
#         argsᵢ = (; T = pTᵢ, P = P_cell)
#         # dimensionless numerical diffusion coefficient (0 ≤ d ≤ 1)
#         d = 1
#         # Compute the characteristic timescale `dt₀` of the local cell
#         ρCp = compute_ρCp(rheology, phase, argsᵢ)
#         K = compute_conductivity(rheology, phase, argsᵢ)
#         sum_dxi = mapreduce(x-> inv(x)^2, +, di)
#         dt₀ = ρCp / (2 * K * sum_dxi)
#         # subgrid diffusion of the i-th particle
#         pΔTᵢ = (pTᵢ - pT0ᵢ) * (1-exp(-d * dt / dt₀))
#         @cell pT0[ip, I...] = pT0ᵢ + pΔTᵢ
#         @cell pΔT[ip, I...] = pΔTᵢ
#     end

#     return nothing
# end

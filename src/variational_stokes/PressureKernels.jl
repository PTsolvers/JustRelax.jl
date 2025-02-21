## The following code is not used as the pressure does not need to be masked as it is
## already masked when calculating the velocity fields and the stress fields.

# function compute_P!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     Q,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio::JustPIC.PhaseRatios,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ;
#     ΔTc=nothing,
#     melt_fraction=nothing,
#     kwargs...,
# ) where {N}
#     ni = size(P)
#     @parallel (@idx ni) compute_P_kernel!(
#         P, P0, RP, ∇V, Q, η, rheology, phase_ratio.center, ϕ, dt, r, θ_dτ, ΔTc, melt_fraction
#     )
#     return nothing
# end

# @parallel_indices (I...) function compute_P_kernel!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     Q,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ,
#     ::Nothing,
#     ::Nothing,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], η[I...], K, dt, r, θ_dτ
#         )
#     else
#         RP[I...] = P[I...] = zero(eltype(P))
#     end
#     return nothing
# end

# @parallel_indices (I...) function compute_P_kernel!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     Q,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ,
#     ::Nothing,
#     melt_fraction,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], η[I...], K, dt, r, θ_dτ
#         )
#     else
#         RP[I...] = P[I...] = zero(eltype(P))
#     end
#     return nothing
# end

# @parallel_indices (I...) function compute_P_kernel!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     Q,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ,
#     ΔTc,
#     ::Nothing,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         phase_ratio_I = phase_ratio[I...]
#         K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
#         α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, η[I...], K, dt, r, θ_dτ
#         )
#     else
#         RP[I...] = P[I...] = zero(eltype(P))
#     end
#     return nothing
# end

# @parallel_indices (I...) function compute_P_kernel!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     Q,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ,
#     ΔTc,
#     ::Nothing,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         phase_ratio_I = phase_ratio[I...]
#         K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
#         α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, η[I...], K, dt, r, θ_dτ
#         )
#     else
#         RP[I...] = P[I...] = zero(eltype(P))
#     end
#     return nothing
# end

# @parallel_indices (I...) function compute_P_kernel!(
#     P,
#     P0,
#     RP,
#     ∇V,
#     Q,
#     η,
#     rheology::NTuple{N,MaterialParams},
#     phase_ratio,
#     ϕ::JustRelax.RockRatio,
#     dt,
#     r,
#     θ_dτ,
#     ΔTc,
#     melt_fraction,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
#         α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, η[I...], K, dt, r, θ_dτ
#         )
#     else
#         RP[I...] = P[I...] = zero(eltype(P))
#     end
#     return nothing
# end

# # @parallel_indices (I...) function compute_P_kernel!(
# #     P,
# #     P0,
# #     RP,
# #     ∇V,
# #     η,
# #     rheology::NTuple{N,MaterialParams},
# #     phase_ratio,
# #     ϕ::JustRelax.RockRatio,
# #     dt,
# #     r,
# #     θ_dτ,
# #     ΔTc,
# #     ::Nothing,
# # ) where {N,C<:JustRelax.CellArray}
# #     if isvalid_c(ϕ, I...)
# #         phase_ratio_I = phase_ratio[I...]
# #         K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
# #         α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
# #         RP[I...], P[I...] = _compute_P!(
# #             P[I...], P0[I...], ∇V[I...], ΔTc[I...], α, η[I...], K, dt, r, θ_dτ
# #         )
# #     else
# #         RP[I...] = P[I...] = zero(eltype(P))
# #     end
# #     return nothing
# # end

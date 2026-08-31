## The active variational pressure dispatch below preserves the generic material
## formulas while applying the variational null-space rule.

"""
    compute_variational_P!(P, P0, RP, ∇V, Q, η, rheology, phase_ratios, ϕ,
                           dt, r, θ_dτ, args)

Update cell-centred pressure and pressure residuals in the reduced liquid
system. A cell is updated only when isvalid_c(ϕ, i, j) is true. Otherwise its
pressure degree of freedom is eliminated:

    P[i,j] = RP[i,j] = 0

ϕ.center supplies the liquid pressure weight and the phase ratios select the
compressible, thermal, or melt-dependent pressure law.
"""

@parallel_indices (I...) function mask_variational_pressure!(P, RP, ϕ::JustRelax.RockRatio)
    if !isvalid_c(ϕ, I...)
        @inbounds P[I...] = RP[I...] = zero(eltype(P))
    end
    return nothing
end

function compute_variational_P!(
        P,
        P0,
        RP,
        ∇V,
        Q,
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratios,
        ϕ::JustRelax.RockRatio,
        dt,
        r,
        θ_dτ,
        args::NamedTuple,
    ) where {N}
    ΔT = get(args, :ΔT, nothing)
    melt_fraction = get(args, :melt_fraction, nothing)
    @parallel (@idx size(P)) compute_variational_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q,
        η,
        rheology,
        phase_ratios.center,
        ϕ,
        dt,
        r,
        θ_dτ,
        ΔT,
        melt_fraction,
    )
    return nothing
end

@parallel_indices (I...) function compute_variational_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q,
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        ϕ::JustRelax.RockRatio,
        dt,
        r,
        θ_dτ,
        ::Nothing,
        ::Nothing,
    ) where {N}
    if isvalid_c(ϕ, I...)
        K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
        G = fn_ratio(get_shear_modulus, rheology, @cell(phase_ratio[I...]))
        @inbounds RP[I...], P[I...] = _compute_P!(
            P[I...], P0[I...], ∇V[I...] * ϕ.center[I...], Q[I...], η[I...], K, G, dt, r, θ_dτ
        )
    else
        @inbounds RP[I...] = P[I...] = zero(eltype(P))
    end
    return nothing
end

@parallel_indices (I...) function compute_variational_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q,
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        ϕ::JustRelax.RockRatio,
        dt,
        r,
        θ_dτ,
        ΔT,
        ::Nothing,
    ) where {N}
    if isvalid_c(ϕ, I...)
        phase_ratio_I = phase_ratio[I...]
        K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
        G = fn_ratio(get_shear_modulus, rheology, phase_ratio_I)
        α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
        @inbounds RP[I...], P[I...] = _compute_P!(
            P[I...], P0[I...], ∇V[I...] * ϕ.center[I...], Q[I...], ΔT[I...], α, η[I...], K, G, dt, r, θ_dτ
        )
    else
        @inbounds RP[I...] = P[I...] = zero(eltype(P))
    end
    return nothing
end

@parallel_indices (I...) function compute_variational_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q,
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        ϕ::JustRelax.RockRatio,
        dt,
        r,
        θ_dτ,
        ΔT,
        melt_fraction,
    ) where {N}
    if isvalid_c(ϕ, I...)
        phase_ratio_I = phase_ratio[I...]
        K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
        G = fn_ratio(get_shear_modulus, rheology, phase_ratio_I)
        α = fn_ratio(
            get_thermal_expansion,
            rheology,
            phase_ratio_I,
            (; ϕ = melt_fraction[I...]),
        )
        @inbounds RP[I...], P[I...] = _compute_P!(
            P[I...], P0[I...], ∇V[I...] * ϕ.center[I...], Q[I...], ΔT[I...], α, η[I...], K, G, dt, r, θ_dτ
        )
    else
        @inbounds RP[I...] = P[I...] = zero(eltype(P))
    end
    return nothing
end

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
#     ΔT=nothing,
#     melt_fraction=nothing,
#     kwargs...,
# ) where {N}
#     ni = size(P)
#     @parallel (@idx ni) compute_P_kernel!(
#         P, P0, RP, ∇V, Q, η, rheology, phase_ratio.center, ϕ, dt, r, θ_dτ, ΔT, melt_fraction
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
#     ΔT,
#     ::Nothing,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         phase_ratio_I = phase_ratio[I...]
#         K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
#         α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I...], α, η[I...], K, dt, r, θ_dτ
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
#     ΔT,
#     ::Nothing,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         phase_ratio_I = phase_ratio[I...]
#         K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
#         α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I...], α, η[I...], K, dt, r, θ_dτ
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
#     ΔT,
#     melt_fraction,
# ) where {N}
#     if isvalid_c(ϕ, I...)
#         K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
#         α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
#         RP[I...], P[I...] = _compute_P!(
#             P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I...], α, η[I...], K, dt, r, θ_dτ
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
# #     ΔT,
# #     ::Nothing,
# # ) where {N,C<:JustRelax.CellArray}
# #     if isvalid_c(ϕ, I...)
# #         phase_ratio_I = phase_ratio[I...]
# #         K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
# #         α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
# #         RP[I...], P[I...] = _compute_P!(
# #             P[I...], P0[I...], ∇V[I...], ΔT[I...], α, η[I...], K, dt, r, θ_dτ
# #         )
# #     else
# #         RP[I...] = P[I...] = zero(eltype(P))
# #     end
# #     return nothing
# # end

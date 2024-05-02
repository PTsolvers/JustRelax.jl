## 2D KERNELS
function compute_viscosity!(
    stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation=1e0
)
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff)
end

function compute_viscosity!(::CPUBackendTrait, stokes, ν, args, rheology, cutoff)
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
end

function _compute_viscosity!(stokes::JustRelax.StokesArrays, ν, args, rheology, cutoff)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η, ν, @strain(stokes)..., args, rheology, cutoff
    )
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
    η, ν, εxx, εyy, εxyv, args, rheology, cutoff
)

    # convenience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        ε = εxx[I...], εyy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(ε...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εij = εII_0 + ε[1], -εII_0 + ε[1], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity!(η, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(η, ν, εII, args, rheology, cutoff)
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
    η, ν, εII, args, rheology, cutoff
)
    @inbounds begin
        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εII_ij = εII[I...]

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII_ij, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity!(
    stokes::JustRelax.StokesArrays, phase_ratios, args, rheology, cutoff; relaxation=1e0
)
    return compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, cutoff
    )
end

function compute_viscosity!(
    ::CPUBackendTrait,
    stokes::JustRelax.StokesArrays,
    ν,
    phase_ratios,
    args,
    rheology,
    cutoff,
)
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
end

function _compute_viscosity!(
    stokes::JustRelax.StokesArrays,
    ν,
    phase_ratios::JustRelax.PhaseRatio,
    args,
    rheology,
    cutoff,
)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        phase_ratios.center,
        @strain(stokes)...,
        args,
        rheology,
        cutoff,
    )
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
    η, ν, ratios_center, εxx, εyy, εxyv, args, rheology, cutoff
)

    # convenience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        ε = εxx[I...], εyy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(ε...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = ratios_center[I...]

        # compute second invariant of strain rate tensor
        εij = εII_0 + ε[1], -εII_0 + ε[1], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity_εII(rheology, ratio_ij, εII, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## 3D KERNELS

@parallel_indices (I...) function compute_viscosity_kernel!(
    η, ν, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, rheology, cutoff
)

    # convenience closures
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        εij_normal = εxx[I...], εyy[I...], εzz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(εij_normal...) * eps()

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εij_normal = εij_normal .+ (εII_0, -εII_0 * 0.5, -εII_0 * 0.5)
        εij_shear = gather_yz(εyzv), gather_xz(εxzv), gather_xy(εxyv)
        εij = (εij_normal..., εij_shear...)
        εII = second_invariant(εij...)

        # update stress and effective viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ijk)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
    η, ν, ratios_center, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, rheology, cutoff
)

    # convenience closures
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        εij_normal = εxx[I...], εyy[I...], εzz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(εij_normal...) * eps()

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ijk = ratios_center[I...]

        # compute second invariant of strain rate tensor
        εij_normal = εij_normal .+ (εII_0, -εII_0 * 0.5, -εII_0 * 0.5)
        εij_shear = gather_yz(εyzv), gather_xz(εxzv), gather_xy(εxyv)
        εij = (εij_normal..., εij_shear...)
        εII = second_invariant(εij...)

        # update stress and effective viscosity
        ηi = compute_phase_viscosity_εII(rheology, ratio_ijk, εII, args_ijk)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## HELPER FUNCTIONS

@inline function local_viscosity_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt=args.dt, τII_old=0.0)
    return local_args
end

@inline function local_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)...)
    return local_args
end

@generated function compute_phase_viscosity_εII(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, εII, args
) where {N}
    quote
        Base.@_inline_meta
        η = 0.0
        Base.@nexprs $N i -> (
            η += if iszero(ratio[i])
                0.0
            else
                inv(compute_viscosity_εII(rheology[i].CompositeRheology[1], εII, args)) * ratio[i]
            end
        )
        inv(η)
    end
end

# @generated function compute_phase_viscosity_εII(
#     rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, εII, args
# ) where {N}
#     quote
#         Base.@_inline_meta
#         η = 0.0
#         Base.@nexprs $N i -> (
#             η += if iszero(ratio[i])
#                 0.0
#             else
#                 compute_viscosity_εII(rheology[i].CompositeRheology[1], εII, args) * ratio[i]
#             end
#         )
#         η
#     end
# end

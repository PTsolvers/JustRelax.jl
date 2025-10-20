# # Traits

# without phase ratios
@inline function update_viscosity!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        islinear(rheology), stokes, args, rheology, cutoff; relaxation = relaxation
    )
    return nothing
end

@inline function update_viscosity!(
        ::LinearRheologyTrait,
        stokes::JustRelax.StokesArrays,
        args,
        rheology,
        cutoff;
        relaxation = 1.0e0,
    )
    return nothing
end

@inline function update_viscosity!(
        ::NonLinearRheologyTrait,
        stokes::JustRelax.StokesArrays,
        args,
        rheology,
        cutoff;
        relaxation = 1.0e0,
    )
    compute_viscosity!(stokes, args, rheology, cutoff; relaxation = relaxation)
    return nothing
end

# with phase ratios
@inline function update_viscosity!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    update_viscosity!(
        islinear(rheology),
        stokes,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff;
        relaxation = relaxation,
    )
    return nothing
end

@inline function update_viscosity!(
        ::LinearRheologyTrait,
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff;
        relaxation = 1.0e0,
    )
    return nothing
end

@inline function update_viscosity!(
        ::NonLinearRheologyTrait,
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff;
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, cutoff; relaxation = relaxation, air_phase = air_phase
    )
    return nothing
end

## 2D KERNELS
function compute_viscosity!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
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
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        ε = εxx[I...], εyy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(ε...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εij = εII_0 + ε[1], -εII_0 + ε[2], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
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
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    return compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff
    )
end

function compute_viscosity!(
        ::CPUBackendTrait,
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff,
    )
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, air_phase, cutoff)
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios::JustPIC.PhaseRatios,
        args,
        rheology,
        air_phase,
        cutoff
    )
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        phase_ratios.center,
        @strain(stokes)...,
        args,
        rheology,
        air_phase,
        cutoff,
    )
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, ratios_center, εxx, εyy, εxyv, args, rheology, air_phase::Integer, cutoff
    )

    # convenience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        ε = εxx[I...], εyy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(ε...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ij = correct_phase_ratio(air_phase, ratio_ij)

        # compute second invariant of strain rate tensor
        εij = εII_0 + ε[1], -εII_0 + ε[2], gather(εxyv)
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
    Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
    Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
    Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

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
        η,
        ν,
        ratios_center,
        εxx,
        εyy,
        εzz,
        εyzv,
        εxzv,
        εxyv,
        args,
        rheology,
        air_phase::Integer,
        cutoff,
    )

    # convenience closures
    Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
    Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
    Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        εij_normal = εxx[I...], εyy[I...], εzz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(εij_normal...) * eps()

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ijk = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ijk = correct_phase_ratio(air_phase, ratio_ijk)

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

@inline function local_viscosity_args(args, I::Vararg{Integer, N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt = args.dt, τII_old = 0.0)
    return local_args
end

@inline function local_args(args, I::Vararg{Integer, N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)...)
    return local_args
end

@generated function compute_phase_viscosity_εII(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, εII, args
    ) where {N}
    return quote
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
#         rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, εII::T, args
#     ) where {N, T}
#     return quote
#         Base.@_inline_meta
#         η = zero(T)
#         Base.@nexprs $N i -> (
#             η += if iszero(ratio[i])
#                 zero(T)
#             else
#                 compute_viscosity_εII(rheology[i].CompositeRheology[1], εII, args) * ratio[i]
#             end
#         )
#         return η
#     end
# end

function correct_phase_ratio(air_phase, ratio::SVector{N, T}) where {N, T}
    if iszero(air_phase)
        return ratio
    elseif ratio[air_phase] ≈ 1
        return SVector{N, T}(zero(T) for _ in 1:N)
    else
        mask = ntuple(i -> (i !== air_phase), Val(N))
        # set air phase ratio to zero
        corrected_ratio = ratio .* mask
        # normalize phase ratios without air
        return corrected_ratio ./ sum(corrected_ratio)
    end
end

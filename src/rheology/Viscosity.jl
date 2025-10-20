# # Traits

# without phase ratios
@inline function update_viscosity_εII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        islinear(rheology), stokes, args, rheology, cutoff, compute_viscosity_εII; relaxation = relaxation
    )
    return nothing
end

@inline function update_viscosity_τII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        islinear(rheology), stokes, args, rheology, cutoff, compute_viscosity_τII; relaxation = relaxation
    )
    return nothing
end

@inline update_viscosity!(::LinearRheologyTrait, args::Vararg{Any, N}; relaxation = 1.0e0) where {N} = nothing

@inline function update_viscosity!(
        ::NonLinearRheologyTrait,
        stokes::JustRelax.StokesArrays,
        args,
        rheology,
        cutoff,
        fn_viscosity::F;
        relaxation = 1.0e0,
    ) where {F}

    fn = get_viscosity_fn(fn_viscosity)

    fn(stokes, args, rheology, cutoff, fn_viscosity; relaxation = relaxation)

    return nothing
end

@inline get_viscosity_fn(::typeof(compute_viscosity_εII)) = compute_viscosity_εII!
@inline get_viscosity_fn(::typeof(compute_viscosity_τII)) = compute_viscosity_τII!

# with phase ratios

@inline function update_viscosity_εII!(
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
        cutoff,
        compute_viscosity_εII;
        relaxation = relaxation,
    )
    return nothing
end

@inline function update_viscosity_τII!(
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
        cutoff,
        compute_viscosity_τII;
        relaxation = relaxation,
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
        cutoff,
        fn_viscosity::F;
        relaxation = 1.0e0,
    ) where {F}

    fn = get_viscosity_fn(fn_viscosity)

    fn(
        stokes, phase_ratios, args, rheology, cutoff; relaxation = relaxation, air_phase = air_phase
    )
    return nothing
end

## 2D KERNELS

function compute_viscosity_τII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff, compute_viscosity_τII)
end

function compute_viscosity_εII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff, compute_viscosity_εII)
end

# generic fallback
function compute_viscosity!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    compute_viscosity_εII!(stokes, args, rheology, cutoff; relaxation = relaxation)
    return nothing
end

function compute_viscosity!(::CPUBackendTrait, stokes, ν, args, rheology, cutoff, fn_viscosity::F) where {F}
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff, fn_viscosity)
end

function _compute_viscosity!(stokes::JustRelax.StokesArrays, ν, args, rheology, cutoff, fn_viscosity::F) where {F}
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η, ν, @strain(stokes)..., args, rheology, cutoff, fn_viscosity
    )
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, Axx, Ayy, Axyv, args, rheology, cutoff, fn_viscosity::F
    ) where {F}

    # convenience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        A = Axx[I...], Ayy[I...], Axyv[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(A...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        Aij = AII_0 + A[1], -AII_0 + A[2], gather(Axyv)
        AII = second_invariant(Aij...)

        # compute and update stress viscosity
        ηi = fn_viscosity(rheology, AII, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity_εII!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(η, ν, εII, args, rheology, cutoff, compute_viscosity_εII)
    return nothing
end

function compute_viscosity_τII!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(η, ν, εII, args, rheology, cutoff, compute_viscosity_τII)
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, AII, args, rheology, cutoff, fn_viscosity::F
    ) where {F}
    @inbounds begin
        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        AII_ij = AII[I...]

        # compute and update stress viscosity
        ηi = fn_viscosity(rheology, AII_ij, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity_τII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff, compute_viscosity_τII
    )
    return nothing
end

function compute_viscosity_εII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff, compute_viscosity_εII
    )
    return nothing
end

# fallback

function compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff, compute_viscosity_εII
    )
    return nothing
end


function compute_viscosity!(
        ::CPUBackendTrait,
        stokes::JustRelax.StokesArrays,
        ν,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    _compute_viscosity!(stokes, ν, args, rheology, air_phase, cutoff, fn_viscosity)

    return nothing
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
        fn_viscosity::F
    ) where {F}
    _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, air_phase, cutoff, fn_viscosity)

    return nothing
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios::JustPIC.PhaseRatios,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    ni = size(stokes.viscosity.η)

    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        phase_ratios.center,
        select_tensor(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity
    )
    return nothing
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    ni = size(stokes.viscosity.η)

    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        select_tensor(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity
    )
    return nothing
end


@inline select_tensor(stokes, ::typeof(compute_viscosity_εII)) = @strain(stokes)
@inline select_tensor(stokes, ::typeof(compute_viscosity_τII)) = @stress(stokes)

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, ratios_center, Axx, Ayy, Axyv, args, rheology, air_phase::Integer, cutoff, fn_viscosity
    )

    # convenience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        A = Axx[I...], Ayy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(A...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ij = correct_phase_ratio(air_phase, ratio_ij)

        # compute second invariant of strain rate tensor
        Aij = AII_0 + A[1], -AII_0 + A[2], gather(Axyv)
        AII = second_invariant(Aij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## 3D KERNELS

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, Axx, Ayy, Azz, Ayzv, Axzv, Axyv, args, rheology, cutoff, fn_viscosity::F
    ) where {F}

    # convenience closures
    Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
    Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
    Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        Aij_normal = Axx[I...], Ayy[I...], Azz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(Aij_normal...) * eps()

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        Aij_normal = Aij_normal .+ (AII_0, -AII_0 * 0.5, -AII_0 * 0.5)
        Aij_shear = gather_yz(Ayzv), gather_xz(Axzv), gather_xy(Axyv)
        Aij = (Aij_normal..., Aij_shear...)
        AII = second_invariant(Aij...)

        # update stress and effective viscosity
        ηi = fn_viscosity(rheology, AII, args_ijk)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η,
        ν,
        ratios_center,
        Axx,
        Ayy,
        Azz,
        Ayzv,
        Axzv,
        Axyv,
        args,
        rheology,
        air_phase::Integer,
        cutoff,
        fn_viscosity::F
    ) where {F}

    # convenience closures
    Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
    Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
    Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        Aij_normal = Axx[I...], Ayy[I...], Azz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(Aij_normal...) * eps()

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ijk = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ijk = correct_phase_ratio(air_phase, ratio_ijk)

        # compute second invariant of strain rate tensor
        Aij_normal = Aij_normal .+ (AII_0, -AII_0 * 0.5, -AII_0 * 0.5)
        Aij_shear = gather_yz(Ayzv), gather_xz(Axzv), gather_xy(Axyv)
        Aij = (Aij_normal..., Aij_shear...)
        AII = second_invariant(Aij...)

        # update stress and effective viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ijk, AII, fn_viscosity, args_ijk)
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

@generated function compute_phase_viscosity(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, AII, fn_viscosity::F, args
    ) where {N, F}
    return quote
        Base.@_inline_meta
        η = 0.0
        Base.@nexprs $N i -> begin
            ηo = fn_viscosity(rheology[i].CompositeRheology[1], AII, args)
            η += iszero(ratio[i]) ?
                0.0 :
                inv(fn_viscosity(rheology[i].CompositeRheology[1], AII, args)) * ratio[i]
        end
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

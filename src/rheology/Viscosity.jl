# # Traits

# without phase ratios
@inline function update_viscosity_εII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        stokes, args, rheology, cutoff, compute_viscosity_εII; relaxation = relaxation
    )
    return nothing
end

@inline function update_viscosity_τII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        stokes, args, rheology, cutoff, compute_viscosity_τII; relaxation = relaxation
    )
    return nothing
end

# @inline update_viscosity!(::LinearRheologyTrait, args::Vararg{Any, N}; relaxation = 1.0e0) where {N} = nothing

@inline function update_viscosity!(
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

"""
    compute_viscosity_τII!(stokes::StokesArrays, [phase_ratios,] args, rheology, cutoff; air_phase=0, relaxation=1.0)

Update `stokes.viscosity.η` in place from the second invariant of the **deviatoric
stress** (`τII`), evaluating `rheology` (a single `GeoParams.MaterialParams`, or one per
phase when `phase_ratios` is given) at each cell and relaxing towards the new value with
factor `relaxation` (`1.0` = no damping). `cutoff = (ηmin, ηmax)` clamps the result.
`air_phase` (multi-phase form only) excludes that phase from the update.

See also [`compute_viscosity_εII!`](@ref) for the strain-rate-invariant convention, and
[`compute_viscosity!`](@ref) for the rheology-driven default (εII).
"""
function compute_viscosity_τII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff, compute_viscosity_τII)
end

"""
    compute_viscosity_εII!(stokes::StokesArrays, [phase_ratios,] args, rheology, cutoff; air_phase=0, relaxation=1.0)

Update `stokes.viscosity.η` in place from the second invariant of the **strain rate**
(`εII`); otherwise identical to [`compute_viscosity_τII!`](@ref).
"""
function compute_viscosity_εII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff, compute_viscosity_εII)
end

"""
    compute_viscosity!(stokes::StokesArrays, [phase_ratios,] args, rheology, cutoff; air_phase=0, relaxation=1.0)

Update `stokes.viscosity.η` in place by evaluating `rheology` at the strain-rate invariant
(equivalent to [`compute_viscosity_εII!`](@ref); see there for the arguments, and
[`compute_viscosity_τII!`](@ref) for the stress-invariant alternative).
"""
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
        AII = second_invariant(AII_0 + A[1], -AII_0 + A[2], A[3])

        # compute and update stress viscosity
        ηi = fn_viscosity(rheology, AII, args_ij)
        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity_εII!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(η)
    @parallel (@idx ni) compute_viscosity_kernel!(η, ν, εII, args, rheology, cutoff, compute_viscosity_εII)
    return nothing
end

function compute_viscosity_τII!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(η)
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

        ηi = continuation_linear(ηi, η[I...], ν)
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
    # centered viscosity
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        phase_ratios.center,
        select_tensor_center(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args,
    )
    # vertex viscosity
    # skip for 3D for now, may change in the future
    if length(ni) == 2
        @parallel (@idx ni .+ 1) compute_viscosity_kernel!(
            stokes.viscosity.ηv,
            ν,
            phase_ratios.vertex,
            select_tensor_vertex(stokes, fn_viscosity)...,
            args,
            rheology,
            air_phase,
            cutoff,
            fn_viscosity,
            local_viscosity_args_vertex,
        )
    end
    return nothing
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F,
        # do_vertices
    ) where {F}
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        select_tensor_center(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args,
    )
    # skip for 3D for now, may change in the future
    length(ni) == 3 && return

    @parallel (@idx ni .+ 1) compute_viscosity_kernel!(
        stokes.viscosity.ηv,
        ν,
        select_tensor_vertex(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args_vertex,
    )
    return nothing
end

for fn in (:select_tensor_center, :select_tensor_vertex)
    @eval @inline $fn(stokes, fn_viscosity) = $fn(stokes, fn_viscosity, JustRelax.static_dims(stokes))
end

# for 2D, we compute viscosity using the tensor defined at the cell centers or vertices, depending on the viscosity function
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_εII), ::Val{2}) = @strain_center(stokes)
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_τII), ::Val{2}) = @stress_center(stokes)
# in 3D we still do some interpolations
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_εII), ::Val{3}) = @strain(stokes)
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_τII), ::Val{3}) = @stress(stokes)

# for 2D, we compute viscosity using the tensor defined at the cell centers or vertices, depending on the viscosity function
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_εII), ::Val{2}) = @tensor_vertex(stokes.ε)
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_τII), ::Val{2}) = @tensor_vertex(stokes.τ)
# in 3D we still do some interpolations
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_εII), ::Val{3}) = @strain(stokes.ε)
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_τII), ::Val{3}) = @stress(stokes.τ)

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, ratios_center, Axx, Ayy, Axyv, args, rheology, air_phase::Integer, cutoff, fn_viscosity::F1, fn_args::F2
    ) where {F1, F2}

    # convenience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        A = Axx[I...], Ayy[I...], Axyv[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(A...) * eps()

        # argument fields at local index
        args_ij = fn_args(args, I...)
        # args_ij = local_viscosity_args(args, I...)

        # local phase ratio, with the air dropped if requested
        ratio_ij = viscosity_phase_ratio(air_phase, @cell(ratios_center[I...]))

        # compute second invariant of strain rate tensor
        Aij = AII_0 + A[1], -AII_0 + A[2], A[3]
        AII = second_invariant(Aij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)
        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## ϕ-aware (variational, DYREL) 2D KERNELS
#
# `η` and `ηv` are computed at every cell and vertex, including those a `ϕ::RockRatio` masks out
# of the momentum equations: the vertex stress interpolates the four surrounding centers with
# `harm_clamped`, and at a free surface a vertex `isvalid_v` accepts routinely has masked-out
# centers among them. A harmonic mean is set by its smallest entry, so a stale value at one of
# those centers would set the interface viscosity.
#
# What these kernels do differ in is the store: a degenerate phase-ratio sample (no particles in
# the cell) makes `compute_phase_viscosity` non-finite — `NaN` once `correct_phase_ratio`
# normalizes an all-zero ratio, `Inf` from the harmonic phase mean when `air_phase` is unset —
# and DYREL's ϕ-aware Gershgorin preconditioner reads `η`/`ηv` before its own validity check, so
# either would reach an unguarded arithmetic sum there and spread across the preconditioner.
# Keeping only finite results confines a degenerate sample to one stale cell.
function compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        ϕ::JustRelax.RockRatio,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, ϕ, args, rheology, air_phase, cutoff, compute_viscosity_εII
    )
    return nothing
end

function update_viscosity_τII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        ϕ::JustRelax.RockRatio,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, ϕ, args, rheology, air_phase, cutoff, compute_viscosity_τII
    )
    return nothing
end

function compute_viscosity!(
        ::CPUBackendTrait,
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios,
        ϕ::JustRelax.RockRatio,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    _compute_viscosity!(stokes, ν, phase_ratios, ϕ, args, rheology, air_phase, cutoff, fn_viscosity)
    return nothing
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    ni = size(stokes.viscosity.η)
    # centered viscosity
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        phase_ratios.center,
        select_tensor_center(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args,
        ϕ,
    )
    # vertex viscosity (DYREL is 2D-only)
    @parallel (@idx ni .+ 1) compute_viscosity_kernel!(
        stokes.viscosity.ηv,
        ν,
        phase_ratios.vertex,
        select_tensor_vertex(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args_vertex,
        ϕ,
    )
    return nothing
end

# `ϕ` selects this method over the unmasked kernel above; the value itself is not read, because
# the viscosity is needed at every DOF (see the note above this block).
@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, ratios_center, Axx, Ayy, Axyv, args, rheology, air_phase::Integer, cutoff, fn_viscosity::F1, fn_args::F2,
        ϕ::JustRelax.RockRatio,
    ) where {F1, F2}

    @inbounds begin
        A = Axx[I...], Ayy[I...], Axyv[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(A...) * eps()

        # argument fields at local index
        args_ij = fn_args(args, I...)

        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        if air_phase > 0
            ratio_ij = correct_phase_ratio(air_phase, ratio_ij)
        end

        # compute second invariant of strain rate tensor
        Aij = AII_0 + A[1], -AII_0 + A[2], A[3]
        AII = second_invariant(Aij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)
        ηi = clamp(continuation_linear(ηi, η[I...], ν), cutoff...)
        isfinite(ηi) && (η[I...] = ηi)
    end

    return nothing
end

## 3D KERNELS
# @parallel_indices (I...) function compute_viscosity_kernel!(
#         η, ν, Axx, Ayy, Azz, Ayzv, Axzv, Axyv, args, rheology, cutoff, fn_viscosity::F1, fn_args::F2
#     ) where {F1, F2}

#     # convenience closures
#     Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
#     Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
#     Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

#     @inbounds begin
#         Aij_normal = Axx[I...], Ayy[I...], Azz[I...]

#         # we need strain rate not to be zero, otherwise we get NaNs
#         AII_0 = allzero(Aij_normal...) * eps()

#         # # argument fields at local index
#         args_ijk = fn_args(args, I...)

#         # compute second invariant of strain rate tensor
#         Aij_normal = Aij_normal .+ (AII_0, -AII_0 * 0.5, -AII_0 * 0.5)
#         Aij_shear = gather_yz(Ayzv), gather_xz(Axzv), gather_xy(Axyv)
#         Aij = (Aij_normal..., Aij_shear...)
#         AII = second_invariant(Aij...)

#         # update stress and effective viscosity
#         ηi = fn_viscosity(rheology, AII, args_ijk)
#         ηi = continuation_linear(ηi, η[I...], ν)
#         η[I...] = clamp(ηi, cutoff...)
#     end

#     return nothing
# end

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
        fn_viscosity::F1,
        fn_args::F2
    ) where {F1, F2}

    # convenience closures
    Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
    Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
    Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        Aij_normal = Axx[I...], Ayy[I...], Azz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(Aij_normal...) * eps()

        # # argument fields at local index
        args_ijk = fn_args(args, I...)

        # local phase ratio, with the air dropped if requested
        ratio_ijk = viscosity_phase_ratio(air_phase, @cell(ratios_center[I...]))

        # compute second invariant of strain rate tensor
        Aij_normal = Aij_normal .+ (AII_0, -AII_0 * 0.5, -AII_0 * 0.5)
        Aij_shear = gather_yz(Ayzv), gather_xz(Axzv), gather_xy(Axyv)
        Aij = (Aij_normal..., Aij_shear...)
        AII = second_invariant(Aij...)

        # update stress and effective viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ijk, AII, fn_viscosity, args_ijk)
        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## HELPER FUNCTIONS
getindex_or_scalar(A::AbstractArray, I::Vararg{Integer, N}) where {N} = A[I...]
getindex_or_scalar(A::Number, I::Vararg{Integer, N}) where {N} = A

@inline local_viscosity_args(args, I::Vararg{Integer, N}) where {N} = local_viscosity_args(I...; args...)

@inline function local_viscosity_args(I::Vararg{Integer, N}; T = 0.0e0, args0...) where {N}
    args = (; args0...)
    v = getindex_or_scalar.(values(args), I...)
    T_ijk = getindex_or_scalar(T, I .+ 1...)
    # local_args = (; T=T_ijk, zip(keys(args), v)..., dt = Inf, τII_old = 0.0)
    local_args = merge(
        (; zip(keys(args), v)...),
        (; T = T_ijk, dt = Inf, τII_old = 0.0)
    )
    return local_args
end

@inline local_viscosity_args_vertex(args, I::Vararg{Integer, N}) where {N} = local_viscosity_args_vertex(I...; args...)

@inline function local_viscosity_args_vertex(i, j; T = 0.0e0, args0...)
    args = (; args0...)
    # clamp indices
    nx, ny = size(args[1])
    il = max(i - 1, 1)  # left
    ir = min(i, nx)     # right
    jb = max(j - 1, 1)  # bottom
    jt = min(j, ny)     # top
    # average values at cell centers surrounding vertex
    v11 = getindex_or_scalar.(values(args), il, jb)
    v12 = getindex_or_scalar.(values(args), ir, jb)
    v21 = getindex_or_scalar.(values(args), il, jt)
    v22 = getindex_or_scalar.(values(args), ir, jt)
    v = @. 0.25 * (v11 + v12 + v21 + v22)
    # average T from surrounding cell centers
    T_vertex = average_or_scalar(T, i, j)
    # create local args
    local_args = merge(
        (; zip(keys(args), v)...),
        (; T = T_vertex, dt = Inf, τII_old = 0.0)
    )
    return local_args
end

@inline function local_viscosity_args_vertex(i, j, k; T = 0.0e0, args0...)
    args = (; args0...)
    # clamp indices
    nx, ny, nz = size(args[1])
    il = max(i - 1, 1)  # left
    ir = min(i, nx)     # right
    jb = max(j - 1, 1)  # bottom
    jt = min(j, ny)     # top
    kf = max(k - 1, 1)  # front
    kb = min(k, nz)   # back
    # average values at cell centers surrounding vertex
    v111 = getindex_or_scalar.(values(args), il, jb, kf)
    v121 = getindex_or_scalar.(values(args), ir, jb, kf)
    v211 = getindex_or_scalar.(values(args), il, jt, kf)
    v221 = getindex_or_scalar.(values(args), ir, jt, kf)
    v112 = getindex_or_scalar.(values(args), il, jb, kb)
    v122 = getindex_or_scalar.(values(args), ir, jb, kb)
    v212 = getindex_or_scalar.(values(args), il, jt, kb)
    v222 = getindex_or_scalar.(values(args), ir, jt, kb)
    v = @. 0.125 * (v111 + v121 + v211 + v221 + v112 + v122 + v212 + v222)
    # create local args
    T_vertex = average_or_scalar(T, i, j, k)
    local_args = merge(
        (; zip(keys(args), v)...),
        (; T = T_vertex, dt = Inf, τII_old = 0.0)
    )
    return local_args
end

@inline function average_or_scalar(A::AbstractArray, i, j)
    return 0.25 * (A[i, j] + A[i + 1, j] + A[i, j + 1] + A[i + 1, j + 1])
end

@inline function average_or_scalar(A::AbstractArray, i, j, k)
    return 0.125 * (
        A[i, j, k] + A[i + 1, j, k] + A[i, j + 1, k] + A[i + 1, j + 1, k] +
            A[i, j, k + 1] + A[i + 1, j, k + 1] + A[i, j + 1, k + 1] + A[i + 1, j + 1, k + 1]
    )
end

@inline average_or_scalar(A::Number, I::Vararg{Integer, N}) where {N} = A


@inline function local_args(args, I::Vararg{Integer, N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt = Inf, τII_old = 0.0)
    return local_args
end

@generated function compute_phase_viscosity(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, AII, fn_viscosity::F, args
    ) where {N, F}
    return quote
        @inline
        # Early exit: if single phase dominates (ratio ≈ 1), skip harmonic mean
        Base.@nexprs $N i -> begin
            if ratio[i] > 0.999  # faster than ≈ comparison
                return fn_viscosity(rheology[i].CompositeRheology[1], AII, args)
            end
        end

        η = 0.0
        Base.@nexprs $N i -> begin
            if !iszero(ratio[i])
                η += inv(fn_viscosity(rheology[i].CompositeRheology[1], AII, args)) * ratio[i]
            end
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

"""
    viscosity_phase_ratio(air_phase, ratio)

Phase ratio to average viscosity over, with `air_phase` dropped and the remaining
phases renormalized. A cell holding nothing but air keeps its own ratio: averaging
over no phase at all would make the harmonic mean `Inf`, which then spreads through
`ητ` into neighbouring cells that do carry rock.
"""
@inline function viscosity_phase_ratio(air_phase, ratio::SVector{N, T}) where {N, T}
    air_phase > 0 || return ratio
    corrected = correct_phase_ratio(air_phase, ratio)
    return iszero(sum(corrected)) ? ratio : corrected
end

function correct_phase_ratio(air_phase, ratio::SVector{N, T}) where {N, T}
    if iszero(air_phase)
        return ratio
    elseif ratio[air_phase] ≈ 1
        # No rock phase in the local sample: return the raw, air-inclusive ratio. An all-zero
        # ratio would make `compute_phase_viscosity`'s harmonic mean +Inf, and this branch is
        # reachable at nodes a variational RockRatio still marks valid — ϕ and the
        # particle-sampled ratio are independent discretizations and disagree at the surface.
        return ratio
    else
        mask = ntuple(i -> (i !== air_phase), Val(N))
        # set air phase ratio to zero
        corrected_ratio = ratio .* mask
        # normalize phase ratios without air
        total = sum(corrected_ratio)
        return iszero(total) ? zeros(SVector{N, T}) : corrected_ratio ./ total
    end
end

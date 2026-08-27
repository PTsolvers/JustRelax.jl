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

compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation, dt) =
    compute_stress_DRYEL!(Val(ndims(stokes.P)), stokes, rheology, phase_ratios, λ_relaxation, dt)

@inline dyrel_vertex_λ(stokes, ::Val{2}) = stokes.λv
@inline dyrel_vertex_λ(stokes, ::Val{3}) = (stokes.λv_yz, stokes.λv_xz, stokes.λv_xy)
@inline reset_dyrel_vertex_λ!(λv) = λv .= 0.0
@inline reset_dyrel_vertex_λ!(λv::Tuple) = foreach(A -> A .= 0.0, λv)

function compute_stress_DRYEL!(
    ::Val{2},
    stokes,
    rheology,
    phase_ratios,
    λ_relaxation,
    dt,
    )

    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy),   # centers
        stokes.EII_pl,                                      # accumulated plastic strain rate @ centers
        stokes.ε_vol_pl,                                    # volumetric plastic strain rate @ centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt
    )
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        EII_pl,
        ε_vol_pl,
        P,
        λ,
        λv,
        η,
        ηv,
        η_vep,
        ΔPψ,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    ## VERTEX CALCULATION
    @inbounds begin
        Ic = clamped_indices(ni, I...)
        τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
        εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
        λvij = λv[I...]
        # ηij   = harm_clamped(η, Ic...)
        ηij = ηv[I...]
        Pij = av_clamped(P, Ic...)
        EIIv = av_clamped(EII_pl, Ic...)
        ratio = phase_ratios_vertex[I...]
        # compute local stress
        τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, _, λ_I, = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv)

        # update arrays
        τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
        ε_pl[3][I...] = εxy_pl
        λv[I...] = λ_I

        ## CENTER CALCULATION
        if all(I .≤ ni)
            τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
            εij = ε[1][I...], ε[2][I...], av(ε[3])
            λij = λ[I...]
            ηij = η[I...]
            Pij = P[I...]
            EII = EII_pl[I...]
            ratio = phase_ratios_center[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, ε_vol_pl_I =
                compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EII)
            # update arrays
            τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[1][I...], ε_pl[2][I...] = εxx_pl, εyy_pl
            ε_vol_pl[I...] = ε_vol_pl_I
            τII[I...] = τII_I
            η_vep[I...] = ηvep_I
            λ[I...] = λ_I
            ΔPψ[I...] = ΔPψ_I
        end
    end

    return nothing
end

## FUSED STRESS + τII-VISCOSITY UPDATE
# Same as `compute_stress_DRYEL!` but, unless `linear_viscosity`, also refreshes the creep
# viscosities η (center) / ηv (vertex) from the freshly-computed stress, reusing the in-register
# τ instead of relaunching a kernel that reads the stress tensor back. The τII-viscosity update
# is purely local (same cell), so this is race-free and needs no halo.
function compute_stress_viscosity_DRYEL!(
        stokes, θc, γ_eff, rheology, phase_ratios, λ_relaxation, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity
    )
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_viscosity_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy),   # centers
        stokes.EII_pl,                                      # accumulated plastic strain rate @ centers
        stokes.ε_vol_pl,                                    # volumetric plastic strain rate @ centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        θc, stokes.R.RP, γ_eff,                             # small pressure correction θc = γ_eff·RP + ΔPψ
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt,
        viscosity_relaxation, args, viscosity_cutoff, linear_viscosity,
    )
    return nothing
end

# local τII-based creep viscosity, mirroring compute_viscosity_kernel! (τII path, air_phase = 0)
@inline function _update_τII_viscosity(τxx, τyy, τxy, ratio, rheology, args_ij, η_old, ν, cutoff)
    AII_0 = allzero(τxx, τyy, τxy) * eps()
    τII = second_invariant(τxx + AII_0, τyy - AII_0, τxy)
    ηi = compute_phase_viscosity(rheology, ratio, τII, compute_viscosity_τII, args_ij)
    ηi = continuation_linear(ηi, η_old, ν)
    return clamp(ηi, cutoff...)
end

@parallel_indices (I...) function compute_stress_viscosity_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        EII_pl,
        ε_vol_pl,
        P,
        λ,
        λv,
        η,
        ηv,
        η_vep,
        ΔPψ,
        θc, RP, γ_eff,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt,
        ν, visc_args, cutoff, linear_viscosity
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    ## VERTEX CALCULATION
    @inbounds begin
        Ic = clamped_indices(ni, I...)
        τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
        εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
        λvij = λv[I...]
        ηij = ηv[I...]
        Pij = av_clamped(P, Ic...)
        EIIv = av_clamped(EII_pl, Ic...)
        ratio = phase_ratios_vertex[I...]
        # compute local stress
        τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, _, λ_I, = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv)

        # update arrays
        τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
        ε_pl[3][I...] = εxy_pl
        λv[I...] = λ_I

        # fused τII-viscosity update at the vertex (reuses in-register stress)
        if !linear_viscosity
            ηv[I...] = _update_τII_viscosity(τxx_I, τyy_I, τxy_I, ratio, rheology, local_viscosity_args_vertex(visc_args, I...), ηij, ν, cutoff)
        end

        ## CENTER CALCULATION
        if all(I .≤ ni)
            τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
            εij = ε[1][I...], ε[2][I...], av(ε[3])
            λij = λ[I...]
            ηij = η[I...]
            Pij = P[I...]
            EII = EII_pl[I...]
            ratio = phase_ratios_center[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, ε_vol_pl_I =
                compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EII)
            # update arrays
            τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[1][I...], ε_pl[2][I...] = εxx_pl, εyy_pl
            ε_vol_pl[I...] = ε_vol_pl_I
            τII[I...] = τII_I
            η_vep[I...] = ηvep_I
            λ[I...] = λ_I
            ΔPψ[I...] = ΔPψ_I

            # small pressure correction θc = P_num + ΔPψ = γ_eff·RP + ΔPψ (ΔPψ_I in-register). Both
            # terms are small corrections of similar magnitude, so summing them is precision-safe;
            # the large hydrostatic P is kept separate in the momentum kernel. Collapses the momentum
            # stencil from three pressure reads (P, P_num, ΔPψ) to two (P, θc).
            θc[I...] = γ_eff[I...] * RP[I...] + ΔPψ_I

            # fused τII-viscosity update at the center (reuses in-register stress)
            if !linear_viscosity
                η[I...] = _update_τII_viscosity(τxx_I, τyy_I, τxy_I, ratio, rheology, local_viscosity_args(visc_args, I...), ηij, ν, cutoff)
            end
        end
    end

    return nothing
end

# 3D Kernel
function compute_stress_DRYEL!(
    ::Val{3},
    stokes,
    rheology,
    phase_ratios,
    λ_relaxation,
    dt,
    )

    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c),  # centers
        (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy), # shear stresses
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.zz_v), # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.zz, stokes.τ_o.yz_c, stokes.τ_o.xz_c, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),  # shear stresses
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.zz_v), # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz),
        (stokes.ε.yz, stokes.ε.xz, stokes.ε.xy),
        (stokes.ε.yz_c, stokes.ε.xz_c, stokes.ε.xy_c),
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.zz),
        (stokes.ε_pl.yz, stokes.ε_pl.xz, stokes.ε_pl.xy),
        (stokes.ε_pl.yz_c, stokes.ε_pl.xz_c, stokes.ε_pl.xy_c),
        stokes.EII_pl,                                      # accumulated plastic strain rate @ centers
        stokes.ε_vol_pl,                                    # volumetric plastic strain rate @ centers
        stokes.P,
        stokes.λ,
        dyrel_vertex_λ(stokes, Val(3)),
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        phase_ratios.center,
        phase_ratios.vertex,
        phase_ratios.yz,
        phase_ratios.xz,
        phase_ratios.xy,
        rheology,
        λ_relaxation,
        dt
    )
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ_c,
        τ_shear,
        τ_v,
        τ_o_c,
        τ_o_shear,
        τ_o_v,
        τII,
        εii,
        εij,
        εij_c,
        εii_pl,
        εij_pl,
        εij_pl_c,
        EII_pl,
        ε_vol_pl,
        P,
        λ,
        λv,
        η,
        ηv,
        η_vep,
        ΔPψ,
        phase_ratios_center,
        phase_ratios_vertex,
        phase_yz,
        phase_xz,
        phase_xy,
        rheology,
        λ_relaxation,
        dt
    )

    ni = size(P)
    Ic = clamped_indices(ni, I...)

    ## yz
    if I[1] ≤ size(εij[1], 1) && (1 < I[2] < size(εij[1], 2)) && (1 < I[3] < size(εij[1], 3))
        # interpolate to ith vertex
        ηij   = harm_clamped_yz(η, Ic...)
        Pij   = av_clamped_yz(P, Ic...)
        EIIv_ij = av_clamped_yz(EII_pl, Ic...)
        εxxv_ij = av_clamped_yz(εii[1], Ic...)
        εyyv_ij = av_clamped_yz(εii[2], Ic...)
        εzzv_ij = av_clamped_yz(εii[3], Ic...)
        εyzv_ij = εij[1][I...]
        εxzv_ij = av_clamped_yz_y(εij[2], Ic...)
        εxyv_ij = av_clamped_yz_z(εij[3], Ic...)
        ε = (εxxv_ij, εyyv_ij, εzzv_ij, εyzv_ij, εxzv_ij, εxyv_ij)

        τxxv_old_ij = av_clamped_yz(τ_o_c[1], Ic...)
        τyyv_old_ij = av_clamped_yz(τ_o_c[2], Ic...)
        τzzv_old_ij = av_clamped_yz(τ_o_c[3], Ic...)
        τyzv_old_ij = τ_o_shear[1][I...]
        τxzv_old_ij = av_clamped_yz_y(τ_o_shear[2], Ic...)
        τxyv_old_ij = av_clamped_yz_z(τ_o_shear[3], Ic...)
        τij_o = (τxxv_old_ij, τyyv_old_ij, τzzv_old_ij, τyzv_old_ij, τxzv_old_ij, τxyv_old_ij)

        λvij = λv[1][I...]
        ratio = @inbounds phase_yz[I...]

        # compute local stress
        _, _, _, τyz_I, _, _, _, _, _, εyz_pl, _, _, _, λ_I, = compute_local_stress(ε, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv_ij)

        # update arrays
        τ_shear[1][I...] = τyz_I
        εij_pl[1][I...] = εyz_pl
        λv[1][I...] = λ_I

    elseif all(I .≤ size(τ_shear[1]))
        τ_shear[1][I...] = zero(eltype(τ_shear[1]))
    end

    ## xz
    if (1 < I[1] < size(εij[2], 1)) && I[2] ≤ size(εij[2], 2) && (1 < I[3] < size(εij[2], 3))
        ηij = harm_clamped_xz(η, Ic...)
        Pij = av_clamped_xz(P, Ic...)
        EIIv_ij = av_clamped_xz(EII_pl, Ic...)
        εxxv_ij = av_clamped_xz(εii[1], Ic...)
        εyyv_ij = av_clamped_xz(εii[2], Ic...)
        εzzv_ij = av_clamped_xz(εii[3], Ic...)
        εyzv_ij = av_clamped_xz_x(εij[1], Ic...)
        εxzv_ij = εij[2][I...]
        εxyv_ij = av_clamped_xz_z(εij[3], Ic...)
        ε = (εxxv_ij, εyyv_ij, εzzv_ij, εyzv_ij, εxzv_ij, εxyv_ij)

        τxxv_old_ij = av_clamped_xz(τ_o_c[1], Ic...)
        τyyv_old_ij = av_clamped_xz(τ_o_c[2], Ic...)
        τzzv_old_ij = av_clamped_xz(τ_o_c[3], Ic...)
        τyzv_old_ij = av_clamped_xz_x(τ_o_shear[1], Ic...)
        τxzv_old_ij = τ_o_shear[2][I...]
        τxyv_old_ij = av_clamped_xz_z(τ_o_shear[3], Ic...)
        τij_o = (τxxv_old_ij, τyyv_old_ij, τzzv_old_ij, τyzv_old_ij, τxzv_old_ij, τxyv_old_ij)

        λvij = λv[2][I...]
        ratio = @inbounds phase_xz[I...]

        _, _, _, _, τxz_I, _, _, _, _, _, εxz_pl, _, _, λ_I, = compute_local_stress(ε, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv_ij)

        τ_shear[2][I...] = τxz_I
        εij_pl[2][I...] = εxz_pl
        λv[2][I...] = λ_I

    elseif all(I .≤ size(τ_shear[2]))
        τ_shear[2][I...] = zero(eltype(τ_shear[2]))
    end

    ## xy
    if (1 < I[1] < size(εij[3], 1)) && (1 < I[2] < size(εij[3], 2)) && I[3] ≤ size(εij[3], 3)
        ηij = harm_clamped_xy(η, Ic...)
        Pij = av_clamped_xy(P, Ic...)
        EIIv_ij = av_clamped_xy(EII_pl, Ic...)
        εxxv_ij = av_clamped_xy(εii[1], Ic...)
        εyyv_ij = av_clamped_xy(εii[2], Ic...)
        εzzv_ij = av_clamped_xy(εii[3], Ic...)
        εyzv_ij = av_clamped_xy_x(εij[1], Ic...)
        εxzv_ij = av_clamped_xy_y(εij[2], Ic...)
        εxyv_ij = εij[3][I...]
        ε = (εxxv_ij, εyyv_ij, εzzv_ij, εyzv_ij, εxzv_ij, εxyv_ij)

        τxxv_old_ij = av_clamped_xy(τ_o_c[1], Ic...)
        τyyv_old_ij = av_clamped_xy(τ_o_c[2], Ic...)
        τzzv_old_ij = av_clamped_xy(τ_o_c[3], Ic...)
        τyzv_old_ij = av_clamped_xy_x(τ_o_shear[1], Ic...)
        τxzv_old_ij = av_clamped_xy_y(τ_o_shear[2], Ic...)
        τxyv_old_ij = τ_o_shear[3][I...]
        τij_o = (τxxv_old_ij, τyyv_old_ij, τzzv_old_ij, τyzv_old_ij, τxzv_old_ij, τxyv_old_ij)

        λvij = λv[3][I...]
        ratio = @inbounds phase_xy[I...]

        _, _, _, _, _, τxy_I, _, _, _, _, _, εxy_pl, _, λ_I, = compute_local_stress(ε, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv_ij)

        τ_shear[3][I...] = τxy_I
        εij_pl[3][I...] = εxy_pl
        λv[3][I...] = λ_I

    elseif all(I .≤ size(τ_shear[3]))
        τ_shear[3][I...] = zero(eltype(τ_shear[3]))
    end

    ## center
    if all(I .≤ ni)
        ε = (
            εii[1][I...],
            εii[2][I...],
            εii[3][I...],
            εij_c[1][I...],
            εij_c[2][I...],
            εij_c[3][I...],
        )
        τij_o = (
            τ_o_c[1][I...],
            τ_o_c[2][I...],
            τ_o_c[3][I...],
            τ_o_c[4][I...],
            τ_o_c[5][I...],
            τ_o_c[6][I...],
        )
        λij = λ[I...]
        ηij = η[I...]
        Pij = P[I...]
        EII = EII_pl[I...]
        ratio = @inbounds phase_ratios_center[I...]

        τxx_I, τyy_I, τzz_I, τyz_I, τxz_I, τxy_I, εxx_pl, εyy_pl, εzz_pl, εyz_pl, εxz_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, ε_vol_pl_I =
            compute_local_stress(ε, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EII)

        τ_c[1][I...], τ_c[2][I...], τ_c[3][I...] = τxx_I, τyy_I, τzz_I
        τ_c[4][I...], τ_c[5][I...], τ_c[6][I...] = τyz_I, τxz_I, τxy_I
        εii_pl[1][I...], εii_pl[2][I...], εii_pl[3][I...] = εxx_pl, εyy_pl, εzz_pl
        εij_pl_c[1][I...], εij_pl_c[2][I...], εij_pl_c[3][I...] = εyz_pl, εxz_pl, εxy_pl
        ε_vol_pl[I...] = ε_vol_pl_I
        τII[I...] = τII_I
        η_vep[I...] = ηvep_I
        λ[I...] = λ_I
        ΔPψ[I...] = ΔPψ_I
    end

    return nothing
end

@generated function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio::SVector{N}, dt, EII) where {N}
    return quote
        @inline
        # iterate over phases
        v_phases = Base.@ntuple $N phase -> begin
            # get phase ratio
            ratio_I = phase_ratio[phase]
            v = if iszero(ratio_I) # this phase does not contribute
                empty_stress_solution(εij)

            else
                # get rheological properties for this phase
                G = get_shear_modulus(rheology, phase)
                Kb = get_bulk_modulus(rheology, phase)
                ratio_I .* _compute_local_stress(
                    εij, τij_o, η, P, G, Kb, λ, λ_relaxation, rheology[phase], dt, EII
                )
            end
        end
        # sum contributions from all phases
        v = reduce(.+, v_phases)
        return v # this returns (τ_ij...), (εij_pl...), τII, λ, ΔPψ, ηvep
    end
end

@inline function _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, rheology, dt, EII)
    # Only is_pl and η_reg are still needed locally; F + gradients come from GeoParams.
    # EII=0 here matches the existing DYREL pattern (no EII-softening in the inner loop).
    ispl, _, _, _, _, η_reg = plastic_params(rheology, EII)

    # viscoelastic viscosity
    η_ve = isinf(G) ?
        inv(inv(η) + inv(G * dt)) :
        (η * G * dt) / (η + G * dt) # more efficient than inv(inv(η) + inv(G * dt))
    # effective strain rate
    inv_2Gdt = inv(2 * G * dt)
    εij_eff = @. εij + τij_o * inv_2Gdt

    εII = second_invariant(εij_eff)

    # early return if there is no deformation
    iszero(εII) && return (zero_tuple(εij)..., zero_tuple(εij)..., 0.0, 0.0, 0.0, η, 0.0)

    # trial stress
    τij = @. 2 * η_ve * εij_eff
    τII = second_invariant(τij)

    # F + gradients at trial stress via GeoParams (DP cone, DPCap cone+cap, ...)
    elements = rheology.CompositeRheology[1].elements
    args = (; P = P, τII = τII, EII = EII)
    F = _yieldfunction_elements(elements, args)
    dQdτ, dQdP, dFdP = _plastic_grad_elements(elements, τij, args)
    # dQdτ is already in tensor convention (shear slots halved inside _plastic_grad_primitive)

    λ, ε_vol_pl = if ispl && F ≥ 0
        λ_new = F / (η_ve + η_reg + Kb * dt * dFdP * dQdP)
        λ = λ_relaxation * λ_new + (1 - λ_relaxation) * λ
        # Volumetric plastic strain rate
        ε_vol_pl = -λ * dQdP
        λ, ε_vol_pl
    else
        0.0, 0.0
    end

    # Update stress and plastic strain rate
    τij, τII, εij_pl, ΔPψ = if λ > 0
        εij_pl = @. λ * dQdτ
        τij = @. τij - 2.0 * η_ve * εij_pl
        τII = second_invariant(τij)
        # Pressure correction from volumetric flow
        ΔPψ = iszero(dQdP) ? 0.0 : -λ * dQdP * Kb * dt
        τij, τII, εij_pl, ΔPψ
    else
        εij_pl = zero_tuple(εij)
        ΔPψ = 0.0
        τij, τII, εij_pl, ΔPψ
    end

    # Effective viscoelastic-plastic viscosity
    η_vep = τII * 0.5 * inv(second_invariant(εij))

    return τij..., εij_pl..., τII, λ, ΔPψ, η_vep, ε_vol_pl
end

# this returns zero for: τxx, τyy, τxy, εxx_pl, εyy_pl, εxy_pl, τII, λ, ΔPψ, ηvep, ε_vol_pl
@inline empty_stress_solution(::NTuple{3, T}) where {T} = zero_tuple(T, Val(11))
# 3D placeholder: τxx, τyy, τzz, τyz, τxz, τxy, εxx_pl..εxy_pl, τII, λ, ΔPψ, ηvep, ε_vol_pl
@inline empty_stress_solution(::NTuple{6, T}) where {T} = zero_tuple(T, Val(17))

@inline zero_tuple(::Type{T}, ::Val{N}) where {T, N} = ntuple(_ -> zero(T), Val(N))
@inline zero_tuple(::NTuple{N, T}) where {T, N} = zero_tuple(T, Val(N))


## VARIATIONAL STOKES STRESS KERNELS

function compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy), # centers
        stokes.EII_pl,                                      # accumulated plastic strain rate @ centers
        stokes.ε_vol_pl,                                    # volumetric plastic strain rate @ centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt
    )
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        EII_pl,
        ε_vol_pl,
        P,
        λ,
        λv,
        η,
        η_vep,
        ΔPψ,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    @inbounds begin
        ## VERTEX CALCULATION
        @inbounds if isvalid_v(ϕ, I...)
            Ic = clamped_indices(ni, I...)
            τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
            εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
            λvij = λv[I...]
            ηij = harm_clamped(η, Ic...)
            Pij = av_clamped(P, Ic...)
            EIIvij = av_clamped(EII_pl, Ic...)
            ratio = phase_ratios_vertex[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, _, λ_I, _, _, _ = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIvij)

            # update arrays
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[3][I...] = εxy_pl
            λv[I...] = λ_I

        else
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = 0.0e0, 0.0e0, 0.0e0
            λv[I...] = 0.0e0

        end

        ## CENTER CALCULATION
        if all(I .≤ ni)
            @inbounds if isvalid_c(ϕ, I...)
                τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
                εij = ε[1][I...], ε[2][I...], av(ε[3])
                λij = λ[I...]
                ηij = η[I...]
                Pij = P[I...]
                EIIij = EII_pl[I...]
                ratio = phase_ratios_center[I...]

                # compute local stress
                τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, ε_vol_pl_I =
                    compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EIIij)
                # update arrays
                τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
                ε_pl[1][I...], ε_pl[2][I...] = εxx_pl, εyy_pl
                ε_vol_pl[I...] = ε_vol_pl_I
                τII[I...] = τII_I
                η_vep[I...] = ηvep_I
                λ[I...] = λ_I
                ΔPψ[I...] = ΔPψ_I

            else
                τ[1][I...], τ[2][I...], τ[3][I...] = 0.0e0, 0.0e0, 0.0e0
                ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = 0.0e0, 0.0e0, 0.0e0
                ε_vol_pl[I...] = 0.0e0
                τII[I...] = 0.0e0
                η_vep[I...] = 0.0e0
                λ[I...] = 0.0e0
                ΔPψ[I...] = 0.0e0

            end
        end
    end

    return nothing
end

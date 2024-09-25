function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    thermal::ThermalArrays,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    phase_ratios::PhaseRatio,
    rheology,
    args,
    dt,
    igg::IGG;
    viscosity_cutoff=(1e16, 1e24),
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window=(1, 1))
    update_halo!(ητ)
    # end

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]
    sizehint!(norm_Rx, Int(iterMax))
    sizehint!(norm_Ry, Int(iterMax))
    sizehint!(norm_∇V, Int(iterMax))
    sizehint!(err_evo1, Int(iterMax))
    sizehint!(err_evo2, Int(iterMax))

    # solver loop
    @copy stokes.P0 stokes.P
    wtime0 = 0.0
    θ = @zeros(ni...)
    λ0 = @zeros(ni...)
    η0 = deepcopy(η)
    do_visc = true
    # GC.enable(false)

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end
    ρ_old = copy(ρg[2])

    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) JustRelax.Stokes2D.compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            @parallel (@idx ni) JustRelax.Stokes2D.compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
            )

            if rem(iter, 5) == 0
                @parallel (@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
                # @. ρg[2] = ρ_old * exp(-stokes.∇V*dt)
            end

            @parallel (@idx ni .+ 1) JustRelax.Stokes2D.compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            if rem(iter, nout) == 0
                @copy η0 η
            end
            if do_visc
                ν = 1e-2
                @parallel (@idx ni) compute_viscosity!(
                    η,
                    ν,
                    phase_ratios.center,
                    @strain(stokes)...,
                    args,
                    rheology,
                    viscosity_cutoff,
                )
            end

            @parallel (@idx ni) JustRelax.Stokes2D.compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor_center(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                phase_ratios.center,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            update_halo!(stokes.τ.xy)

            @hide_communication b_width begin # communication/computation overlap
                @parallel JustRelax.Stokes2D.compute_V!(
                    @velocity(stokes)..., θ, @stress(stokes)..., ηdτ, ρg..., ητ, _di...
                )
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            er_η = norm_mpi(@.(log10(η) - log10(η0)))
            er_η < 1e-3 && (do_visc = false)
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, θ, @stress(stokes)..., ρg..., _di...
            )
            # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            errs = (
                norm_mpi(stokes.R.Rx) / length(stokes.R.Rx),
                norm_mpi(stokes.R.Ry) / length(stokes.R.Ry),
                norm_mpi(stokes.R.RP) / length(stokes.R.RP),
            )
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    stokes.P .= θ

    # accumulate plastic strain tensor
    @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end



# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (I...) function compute_τ_nonlinear!(
    τ,      # @ centers
    τII,    # @ centers
    τ_old,  # @ centers
    ε,      # @ vertices
    ε_pl,   # @ centers
    EII,    # accumulated plastic strain rate @ centers
    P,
    θ,
    η,
    η_vep,
    λ,
    phase_center,
    rheology,
    dt,
    θ_dτ,
)
    # numerics
    ηij = @inbounds η[I...]
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(JustRelax.Stokes2D.get_shear_modulus, rheology, phase) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic parameters (if any...)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)

    # plastic volumetric change K * dt * sinϕ * sinψ
    Kb = fn_ratio(JustRelax.Stokes2D.get_bulk_modulus, rheology, phase)
    volume = isinf(Kb) ? 0.0 : Kb * dt * sinϕ * sinψ
    plastic_parameters = (; is_pl, C, sinϕ, cosϕ, η_reg, volume)

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, ε_pl, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )
    # augmented pressure with plastic volumetric strain over pressure
    @inbounds θ[I...] = P[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

    return nothing
end

###

# inner kernel to compute the plastic stress update within Pseudo-Transient stress continuation
function _compute_τ_nonlinear!(
    τ::NTuple{N1,T},
    τII,
    τ_old::NTuple{N1,T},
    ε::NTuple{N1,T},
    ε_pl::NTuple{N1,T},
    P,
    ηij,
    η_vep,
    λ,
    dτ_r,
    _Gdt,
    plastic_parameters,
    idx::Vararg{Integer,N2},
) where {N1,N2,T}

    # cache tensors
    τij, τij_o, εij = JustRelax.cache_tensors(τ, τ_old, ε, idx...)

    # Stress increment and trial stress
    dτij, τII_trial = JustRelax.compute_stress_increment_and_trial(τij, τij_o, ηij, εij, _Gdt, dτ_r)

    # visco-elastic strain rates
    εij_ve = ntuple(Val(N1)) do i
        Base.@_inline_meta
        fma(0.5 * τij_o[i], _Gdt, εij[i])
    end
    # get plastic parameters (if any...)
    (; is_pl, C, sinϕ, cosϕ, η_reg, volume) = plastic_parameters

    # yield stess (GeoParams could be used here...)
    τy = max(C * cosϕ + P[idx...] * sinϕ, 0.0)

    # check if yielding; if so, compute plastic strain rate (λdQdτ),
    # plastic stress increment (dτ_pl), and update the plastic
    # multiplier (λ)
    dτij, λdQdτ = if JustRelax.isyielding(is_pl, τII_trial, τy)
        # derivatives plastic stress correction
        dτ_pl, λ[idx...], λdQdτ = compute_dτ_pl(
            τij, dτij, τy, τII_trial, ηij, λ[idx...], η_reg, dτ_r, volume
        )
        dτ_pl, λdQdτ

    else
        # in this case the plastic strain rate is a tuples of zeros
        dτij, ntuple(_ -> zero(eltype(T)), Val(N1))
    end

    # fill plastic strain rate tensor
    update_plastic_strain_rate!(ε_pl, λdQdτ, idx)
    # update and correct stress
    correct_stress!(τ, τij .+ dτij, idx...)

    τII[idx...] = τII_ij = second_invariant(τij...)
    η_vep[idx...] = τII_ij * 0.5 * inv(second_invariant(εij_ve...))

    return nothing
end

# fill plastic strain rate tensor
@generated function update_plastic_strain_rate!(ε_pl::NTuple{N,T}, λdQdτ, idx) where {N,T}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> ε_pl[i][idx...] = !isinf(λdQdτ[i]) * λdQdτ[i]
    end
end

# check if plasticity is active
@inline isyielding(is_pl, τII_trial, τy) = is_pl * (τII_trial > τy)

@inline compute_dτ_r(θ_dτ, ηij, _Gdt) = inv(θ_dτ + fma(ηij, _Gdt, 1.0))

function compute_stress_increment_and_trial(
    τij::NTuple{N,T}, τij_o::NTuple{N,T}, ηij, εij::NTuple{N,T}, _Gdt, dτ_r
) where {N,T}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        dτ_r * fma(2.0 * ηij, εij[i], fma(-((τij[i] - τij_o[i])) * ηij, _Gdt, -τij[i]))
    end
    return dτij, second_invariant((τij .+ dτij)...)
end

function compute_dτ_pl(
    τij::NTuple{N,T}, dτij, τy, τII_trial, ηij, λ0, η_reg, dτ_r, volume
) where {N,T}
    # yield function
    F = τII_trial - τy
    # Plastic multiplier
    ν = 0.5
    λ = ν * λ0 + (1 - ν) * (F > 0.0) * F * inv(ηij * dτ_r + η_reg + volume)
    λ_τII = λ * 0.5 * inv(τII_trial)

    λdQdτ = ntuple(Val(N)) do i
        Base.@_inline_meta
        # derivatives of the plastic potential
        (τij[i] + dτij[i]) * λ_τII
    end

    dτ_pl = ntuple(Val(N)) do i
        Base.@_inline_meta
        # corrected stress
        fma(-dτ_r * 2.0, ηij * λdQdτ[i], dτij[i])
    end
    return dτ_pl, λ, λdQdτ
end

# update the global arrays τ::NTuple{N, AbstractArray} with the local τij::NTuple{3, Float64} at indices idx::Vararg{Integer, N}
@generated function correct_stress!(
    τ, τij::NTuple{N1,T}, idx::Vararg{Integer,N2}
) where {N1,N2,T}
    quote
        Base.@_inline_meta
        Base.@nexprs $N1 i -> τ[i][idx...] = τij[i]
    end
end

@inline function correct_stress!(τxx, τyy, τxy, τij, i, j)
    return correct_stress!((τxx, τyy, τxy), τij, i, j)
end

@inline function correct_stress!(τxx, τyy, τzz, τyz, τxz, τxy, τij, i, j, k)
    return correct_stress!((τxx, τyy, τzz, τyz, τxz, τxy), τij, i, j, k)
end

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false

@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)
@inline plastic_params(v, EII) = plastic_params(v.CompositeRheology[1].elements, EII)

@generated function plastic_params(v::NTuple{N,Any}, EII) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> begin
            vᵢ = v[i]
            if isplastic(vᵢ)
                C = soften_cohesion(vᵢ, EII)
                sinϕ, cosϕ = soften_friction_angle(vᵢ, EII)
                return (true, C, sinϕ, cosϕ, vᵢ.sinΨ.val, vᵢ.η_vp.val)
            end
        end
        (false, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function plastic_params_phase(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, EII, ratio
) where {N}
    data = _plastic_params_phase(rheology, EII, ratio)
    # average over phases
    is_pl = false
    C = sinϕ = cosϕ = sinψ = η_reg = 0.0
    for n in 1:N
        ratio_n = ratio[n]
        data[n][1] && (is_pl = true)
        C += data[n][2] * ratio_n
        sinϕ += data[n][3] * ratio_n
        cosϕ += data[n][4] * ratio_n
        sinψ += data[n][5] * ratio_n
        η_reg += data[n][6] * ratio_n
    end
    return is_pl, C, sinϕ, cosϕ, sinψ, η_reg
end

@generated function _plastic_params_phase(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, EII, ratio
) where {N}
    quote
        Base.@_inline_meta
        empty_args = false, 0.0, 0.0, 0.0, 0.0, 0.0
        Base.@nexprs $N i ->
            a_i = ratio[i] == 0 ? empty_args : plastic_params(rheology[i], EII)
        Base.@ncall $N tuple a
    end
end

# cache tensors
function cache_tensors(
    τ::NTuple{3,Any}, τ_old::NTuple{3,Any}, ε::NTuple{3,Any}, idx::Vararg{Integer,2}
)
    @inline av_shear(A) = 0.25 * sum(_gather(A, idx...))

    εij = ε[1][idx...], ε[2][idx...], av_shear(ε[3])
    τij = getindex.(τ, idx...)
    τij_o = getindex.(τ_old, idx...)

    return τij, τij_o, εij
end

function cache_tensors(
    τ::NTuple{6,Any}, τ_old::NTuple{6,Any}, ε::NTuple{6,Any}, idx::Vararg{Integer,3}
)
    @inline av_yz(A) = _av_yz(A, idx...)
    @inline av_xz(A) = _av_xz(A, idx...)
    @inline av_xy(A) = _av_xy(A, idx...)

    # normal components of the strain rate and old-stress tensors
    ε_normal = ntuple(i -> ε[i][idx...], Val(3))
    # shear components of the strain rate and old-stress tensors
    ε_shear = av_yz(ε[4]), av_xz(ε[5]), av_xy(ε[6])
    # cache ij-th components of the tensors into a tuple in Voigt notation
    εij = (ε_normal..., ε_shear...)
    τij_o = getindex.(τ_old, idx...)
    τij = getindex.(τ, idx...)

    return τij, τij_o, εij
end

## softening kernels

@inline function soften_cohesion(
    v::DruckerPrager{T,U,U1,S1,NoSoftening}, ::T
) where {T,U,U1,S1}
    return v.C.val
end

@inline function soften_cohesion(
    v::DruckerPrager_regularised{T,U,U1,U2,S1,NoSoftening}, ::T
) where {T,U,U1,U2,S1}
    return v.C.val
end

@inline function soften_cohesion(
    v::DruckerPrager{T,U,U1,S1,S2}, EII::T
) where {T,U,U1,S1,S2}
    return v.softening_C(EII, v.C.val)
end

@inline function soften_cohesion(
    v::DruckerPrager_regularised{T,U,U1,U2,S1,S2}, EII::T
) where {T,U,U1,U2,S1,S2}
    return v.softening_C(EII, v.C.val)
end

@inline function soften_friction_angle(
    v::DruckerPrager{T,U,U1,NoSoftening,S2}, ::T
) where {T,U,U1,S2}
    return (v.sinϕ.val, v.cosϕ.val)
end

@inline function soften_friction_angle(
    v::DruckerPrager_regularised{T,U,U1,U2,NoSoftening,S2}, ::T
) where {T,U,U1,U2,S2}
    return (v.sinϕ.val, v.cosϕ.val)
end

@inline function soften_friction_angle(
    v::DruckerPrager{T,U,U1,S1,S2}, EII::T
) where {T,U,U1,S1,S2}
    ϕ = v.softening_ϕ(EII, v.ϕ.val)
    return sincosd(ϕ)
end

@inline function soften_friction_angle(
    v::DruckerPrager_regularised{T,U,U1,U2,S1,S2}, EII::T
) where {T,U,U1,U2,S1,S2}
    ϕ = v.softening_ϕ(EII, v.ϕ.val)
    return sincosd(ϕ)
end

function cache_tensors(
    τ::NTuple{3,Any}, τ_old::NTuple{3,Any}, ε::NTuple{3,Any}, idx::Vararg{Integer,2}
)
    @inline av_shear(A) = 0.25 * sum(_gather(A, idx...))

    εij = ε[1][idx...], ε[2][idx...], av_shear(ε[3])
    τij = getindex.(τ, idx...)
    τij_o = getindex.(τ_old, idx...)

    return τij, τij_o, εij
end
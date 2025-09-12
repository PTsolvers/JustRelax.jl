# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_psAD!(
        ε::NTuple{3},         # normal components @ centers; shear components @ vertices
        ε_pl::NTuple{3},      # whole Voigt tensor @ centers
        EII,                  # accumulated plastic strain rate @ centers
        τ::NTuple{3},         # whole Voigt tensor @ centers
        τshear_v::NTuple{1},  # shear tensor components @ vertices
        τ_o::NTuple{3},
        τshear_ov::NTuple{1}, # shear tensor components @ vertices
        Pr,
        Pr_c,
        η,
        λ,
        λv,
        τII,
        η_vep,
        relλ,
        dt,
        θ_dτ,
        rheology,
        phase_center,
        phase_vertex,
        phase_xy,
        phase_yz,
        phase_xz,
    )
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij = av_clamped(Pr, Ic...)
    εxxv_ij = av_clamped(ε[1], Ic...)
    εyyv_ij = av_clamped(ε[2], Ic...)
    τxxv_ij = av_clamped(τ[1], Ic...)
    τyyv_ij = av_clamped(τ[2], Ic...)
    τxxv_old_ij = av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = av_clamped(τ_o[2], Ic...)
    EIIv_ij = av_clamped(EII, Ic...)

    ## vertex
    phase = @inbounds phase_vertex[I...]
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    #volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
    if isinf(Kv)
        volumev = 0.0
    else
        volumev = Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
    end
    ηv_ij = av_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
    dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
    dτxyv = compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
    )
    #τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
    τIIv_ij = av_clamped(τII, Ic...)
    
    # yield function @ center
    Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
    if is_pl && !iszero(τIIv_ij) && Fv > 0
        # stress correction @ vertex
        λv[I...] =
            (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
        τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[I...] * dQdτxy * dτ_rv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
    end
    
    ## center
    if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        K = fn_ratio(get_bulk_modulus, rheology, phase)
        #volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        if isinf(K)
            volume= 0.0
        else 
            volume = K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        end
        ηij = η[I...]
        dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        #εij_ve = @. εij + 0.5 * τij_o * _Gdt
        #εII_ve = GeoParams.second_invariant(εij_ve)
        # stress increments @ center
        #dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        dτij = @. (-((τij - τij_o) * _Gdt) - (τij ./ (2.0 * ηij)) .+ εij) * dτ_r
        #dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
        τII_ij = τII[I...]
        #τII_ij = GeoParams.second_invariant(dτij .+ τij)
        # yield function @ center
        
        F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

        if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            τII[I...] = GeoParams.second_invariant(τij)
            # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
            #η_vep[I...] = 0.5 * τII_ij / εII_ve
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            #η_vep[I...] = ηij
            #τII[I...] = τII_ij
        end

    #    Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end


# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_psADSens!(
    ε::NTuple{3},         # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{3},      # whole Voigt tensor @ centers
    EII,                  # accumulated plastic strain rate @ centers
    τ::NTuple{3},         # whole Voigt tensor @ centers
    τshear_v::NTuple{1},  # shear tensor components @ vertices
    τ_o::NTuple{3},
    τshear_ov::NTuple{1}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep,
    relλ,
    dt,
    θ_dτ,
    rheology,
    phase_center,
    phase_vertex,
    phase_xy,
    phase_yz,
    phase_xz,
    Sens,
)
τxyv = τshear_v[1]
τxyv_old = τshear_ov[1]
ni = size(Pr)
Ic = clamped_indices(ni, I...)

# interpolate to ith vertex
Pv_ij = av_clamped(Pr, Ic...)
εxxv_ij = av_clamped(ε[1], Ic...)
εyyv_ij = av_clamped(ε[2], Ic...)
τxxv_ij = av_clamped(τ[1], Ic...)
τyyv_ij = av_clamped(τ[2], Ic...)
τxxv_old_ij = av_clamped(τ_o[1], Ic...)
τyyv_old_ij = av_clamped(τ_o[2], Ic...)
EIIv_ij = av_clamped(EII, Ic...)

## vertex
phase = @inbounds phase_vertex[I...]
#is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
#Gv_ij       = @inbounds Sens[4][I...]
ϕv_ij       = av_clamped(Sens[2],Ic...)
Cv_ij       = av_clamped(Sens[3],Ic...)
is_pl, CvNot, sinϕvNot, cosϕvNot, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
#_Gvdt = inv(Gv_ij * dt)
sinϕv = sind(ϕv_ij)
cosϕv = cosd(ϕv_ij)
Cv    = Cv_ij

Kv = fn_ratio(get_bulk_modulus, rheology, phase)
#volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
if isinf(Kv)
    volumev = 0.0
else
    volumev = Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
end
ηv_ij = av_clamped(η, Ic...)
dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

# stress increments @ vertex
#dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
#dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
dτxyv = compute_stress_increment(
    τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
)
#τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
τIIv_ij = av_clamped(τII, Ic...)

# yield function @ center
Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
if is_pl && !iszero(τIIv_ij) && Fv > 0
    # stress correction @ vertex
    λv[I...] =
        (1.0 - relλ) * λv[I...] +
        relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
    dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
    τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[I...] * dQdτxy * dτ_rv
else
    # stress correction @ vertex
    τxyv[I...] += dτxyv
end

## center
if all(I .≤ ni)
    # Material properties
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    #is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII#[I...], phase)
    #Gc_ij       = Sens[1][I...]
    ϕc_ij       = Sens[2][I...]
    Cc_ij       = Sens[3][I...]
    is_pl, CNot, sinϕNot, cosϕNot, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
    #_Gdt = inv(Gc_ij * dt)
    sinϕ = sind(ϕc_ij)
    cosϕ = cosd(ϕc_ij)
    C    = Cc_ij

    K = fn_ratio(get_bulk_modulus, rheology, phase)
    #volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
    if isinf(K)
        volume= 0.0
    else 
        volume = K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
    end
    ηij = η[I...]
    dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

    # cache strain rates for center calculations
    τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

    # visco-elastic strain rates @ center
    #εij_ve = @. εij + 0.5 * τij_o * _Gdt
    #εII_ve = GeoParams.second_invariant(εij_ve)
    # stress increments @ center
    # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
    dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
    τII_ij = τII[I...]
    #τII_ij = GeoParams.second_invariant(dτij .+ τij)

    
    # yield function @ center
    F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

    if is_pl && !iszero(τII_ij) && F > 0
        # stress correction @ center
        λ[I...] =
            (1.0 - relλ) * λ[I...] +
            relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
        dQdτij = @. 0.5 * (τij + dτij) / τII_ij
        # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
        εij_pl = λ[I...] .* dQdτij
        dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
        τij = dτij .+ τij
        setindex!.(τ, τij, I...)
        setindex!.(ε_pl, εij_pl, I...)
        #τII[I...] = GeoParams.second_invariant(τij)
        # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
        #η_vep[I...] = 0.5 * τII_ij / εII_ve
    else
        # stress correction @ center
        setindex!.(τ, dτij .+ τij, I...)
        #η_vep[I...] = ηij
        #τII[I...] = τII_ij
    end

#    Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
end

return nothing
end

#####################################################
######## get Material Parameters Dislocation ########
#####################################################


abstract type RheologyTrait end
struct DislocationTrait <: RheologyTrait end
struct NonDislocationTrait <: RheologyTrait end

@inline isdislocation(::DislocationCreep)   = DislocationTrait()
@inline isdislocation(::AbstractConstitutiveLaw) = NonDislocationTrait()
@inline isdislocation(c::CompositeRheology) = isdislocation(c.elements)

# compares two rheologies and return linear trait only and if only both are linear
@inline isdislocation(::RheologyTrait, ::DislocationTrait) = DislocationTrait()
@inline isdislocation(::DislocationTrait, ::RheologyTrait) = DislocationTrait()
@inline isdislocation(::DislocationTrait, ::DislocationTrait) = DislocationTrait()
@inline isdislocation(::RheologyTrait, ::RheologyTrait) = NonDislocationTrait()

# compares three rheologies and return linear trait only and if only both are linear
@inline isdislocation(::RheologyTrait, ::RheologyTrait, ::DislocationTrait) = DislocationTrait()
@inline isdislocation(::DislocationTrait, ::RheologyTrait, ::RheologyTrait) = DislocationTrait()
@inline isdislocation(::RheologyTrait, ::DislocationTrait, ::RheologyTrait) = DislocationTrait()
@inline isdislocation(::DislocationTrait, ::DislocationTrait, ::RheologyTrait) = DislocationTrait()
@inline isdislocation(::DislocationTrait, ::RheologyTrait, ::DislocationTrait) = DislocationTrait()
@inline isdislocation(::RheologyTrait, ::DislocationTrait, ::DislocationTrait) = DislocationTrait()
@inline isdislocation(::RheologyTrait, ::RheologyTrait, ::RheologyTrait) = NonDislocationTrait()
#@inline isdislocation(v1::Union{AbstractConstitutiveLaw, AbstractPlasticity}, v2::Union{AbstractConstitutiveLaw, AbstractPlasticity}) = isdislocation(isdislocation(v1), isdislocation(v2))

# traits for MaterialParams
@inline isdislocation(r::MaterialParams) = isdislocation(r.CompositeRheology...)

# recursively (pairwise, right-to-left) compare rheology traits of a composite or tuple of material params
@inline isdislocation(r::NTuple{N, Union{AbstractConstitutiveLaw, AbstractPlasticity, MaterialParams}}) where {N} = isdislocation(isdislocation(first(r)), isdislocation(Base.tail(r)))
@inline isdislocation(v::NTuple{1, Union{AbstractConstitutiveLaw, AbstractPlasticity, MaterialParams}}) = isdislocation(v...)

function get_Adis(args::Vararg{Any, N}) where {N}
    Param = getdis_A(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_Edis(args::Vararg{Any, N}) where {N}
    Param = getdis_E(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_rdis(args::Vararg{Any, N}) where {N}
    Param = getdis_r(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_ndis(args::Vararg{Any, N}) where {N}
    Param = getdis_n(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_Vdis(args::Vararg{Any, N}) where {N}
    Param = getdis_V(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

for modulus in (:A, :E, :r, :n, :V)
    fun = Symbol("getdis_$(string(modulus))")
    @eval begin
        @inline $(fun)(a::DislocationCreep) = a.$(modulus).val
        @inline $(fun)(c::CompositeRheology) = $(fun)(isdislocation(c), c)
        @inline $(fun)(::DislocationTrait, c::CompositeRheology) = mapreduce(x -> $(fun)(x), +, c.elements)
        @inline $(fun)(r::AbstractMaterialParamsStruct) = $(fun)(r.CompositeRheology[1])
        @inline $(fun)(a::NTuple{N, AbstractMaterialParamsStruct}, phase) where {N} = nphase($(fun), phase, a)
        @inline $(fun)(::NonDislocationTrait, c::CompositeRheology) = 0
        @inline $(fun)(::Union{NonDislocationTrait, AbstractPlasticity, AbstractElasticity, DiffusionCreep}) = 0
    end
end

#####################################################
######## get Material Parameters Diffusion ##########
#####################################################

struct DiffusionTrait <: RheologyTrait end
struct NonDiffusionTrait <: RheologyTrait end

@inline isdiffusion(::DiffusionCreep)   = DiffusionTrait()
@inline isdiffusion(::AbstractConstitutiveLaw) = NonDiffusionTrait()
@inline isdiffusion(c::CompositeRheology) = isdiffusion(c.elements)

# compares two rheologies and return linear trait only and if only both are linear
@inline isdiffusion(::RheologyTrait, ::DiffusionTrait) = DiffusionTrait()
@inline isdiffusion(::DiffusionTrait, ::RheologyTrait) = DiffusionTrait()
@inline isdiffusion(::DiffusionTrait, ::DiffusionTrait) = DiffusionTrait()
@inline isdiffusion(::RheologyTrait, ::RheologyTrait) = NonDiffusionTrait()

# compares three rheologies and return linear trait only and if only both are linear
@inline isdiffusion(::RheologyTrait, ::RheologyTrait, ::DiffusionTrait) = DiffusionTrait()
@inline isdiffusion(::DiffusionTrait, ::RheologyTrait, ::RheologyTrait) = DiffusionTrait()
@inline isdiffusion(::RheologyTrait, ::DiffusionTrait, ::RheologyTrait) = DiffusionTrait()
@inline isdiffusion(::DiffusionTrait, ::DiffusionTrait, ::RheologyTrait) = DiffusionTrait()
@inline isdiffusion(::DiffusionTrait, ::RheologyTrait, ::DiffusionTrait) = DiffusionTrait()
@inline isdiffusion(::RheologyTrait, ::DiffusionTrait, ::DiffusionTrait) = DiffusionTrait()
@inline isdiffusion(::RheologyTrait, ::RheologyTrait, ::RheologyTrait) = NonDiffusionTrait()
#@inline isdiffusion(v1::Union{AbstractConstitutiveLaw, AbstractPlasticity}, v2::Union{AbstractConstitutiveLaw, AbstractPlasticity}) = isdiffusion(isdiffusion(v1), isdiffusion(v2))

# traits for MaterialParams
@inline isdiffusion(r::MaterialParams) = isdiffusion(r.CompositeRheology...)

# recursively (pairwise, right-to-left) compare rheology traits of a composite or tuple of material params
@inline isdiffusion(r::NTuple{N, Union{AbstractConstitutiveLaw, AbstractPlasticity, MaterialParams}}) where {N} = isdiffusion(isdiffusion(first(r)), isdiffusion(Base.tail(r)))
@inline isdiffusion(v::NTuple{1, Union{AbstractConstitutiveLaw, AbstractPlasticity, MaterialParams}}) = isdiffusion(v...)

function get_Adif(args::Vararg{Any, N}) where {N}
    Param = getdif_A(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_Edif(args::Vararg{Any, N}) where {N}
    Param = getdif_E(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_rdif(args::Vararg{Any, N}) where {N}
    Param = getdif_r(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_pdif(args::Vararg{Any, N}) where {N}
    Param = getdif_p(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

function get_Vdif(args::Vararg{Any, N}) where {N}
    Param = getdif_V(args...)
    if isnan(Param) || iszero(Param)
        return Inf
    end
    return Param
end

for modulus in (:A, :E, :r, :p, :V)
    fun = Symbol("getdif_$(string(modulus))")
    @eval begin
        @inline $(fun)(a::DiffusionCreep) = a.$(modulus).val
        @inline $(fun)(c::CompositeRheology) = $(fun)(isdiffusion(c), c)
        @inline $(fun)(::DiffusionTrait, c::CompositeRheology) = mapreduce(x -> $(fun)(x), +, c.elements)
        @inline $(fun)(r::AbstractMaterialParamsStruct) = $(fun)(r.CompositeRheology[1])
        @inline $(fun)(a::NTuple{N, AbstractMaterialParamsStruct}, phase) where {N} = nphase($(fun), phase, a)
        @inline $(fun)(::NonDiffusionTrait, c::CompositeRheology) = 0
        @inline $(fun)(::Union{NonDiffusionTrait, AbstractPlasticity, AbstractElasticity, DislocationCreep}) = 0
    end
end


@parallel_indices (I...) function assemble_parameter_matrices!(
    EII,
    Gc,
    ϕc,
    Cc,
    Kc,
    Gv,
    ϕv,
    Cv,
    Kv,
    Adis,
    ndis,
    rdis,
    Edis,
    Vdis,
    Adif,
    pdif,
    rdif,
    Edif,
    Vdif,
    rheology, 
    phase_center,
    phase_vertex,

)

    ni = size(Gc)

    Ic       = clamped_indices(ni, I...)
    EIIv_ij  = av_clamped(EII, Ic...)
    phase    = @inbounds phase_vertex[I...]
    Gv[I...] = fn_ratio(get_shear_modulus, rheology, phase)
    Kv[I...] = fn_ratio(get_bulk_modulus, rheology, phase)

    is_pl, Cvi, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    #ϕv[I...] = sinϕv #asind(sinϕv)
    #Cv[I...] = Cvi

    if all(I .≤ ni)
        phase = @inbounds phase_center[I...]
        Gc[I...] = fn_ratio(get_shear_modulus, rheology, phase)
        Kc[I...] = fn_ratio(get_bulk_modulus, rheology, phase)

        is_pl, Cci, sinϕc, cosϕc, sinψc, η_regc = plastic_params_phase(rheology, EII[I...], phase)
        ϕc[I...] = asind(sinϕc) # sinϕc
        Cc[I...] = Cci

        # get dislocation and diffusion parameters
        Adis[I...] = fn_ratio(get_Adis, rheology, phase)
        Edis[I...] = fn_ratio(get_Edis, rheology, phase)
        rdis[I...] = fn_ratio(get_rdis, rheology, phase)
        ndis[I...] = fn_ratio(get_ndis, rheology, phase)
        Vdis[I...] = fn_ratio(get_Vdis, rheology, phase)

        Adif[I...] = fn_ratio(get_Adif, rheology, phase)
        Edif[I...] = fn_ratio(get_Edif, rheology, phase)
        rdif[I...] = fn_ratio(get_rdif, rheology, phase)
        pdif[I...] = fn_ratio(get_pdif, rheology, phase)
        Vdif[I...] = fn_ratio(get_Vdif, rheology, phase)

    end

return nothing

end
















#### OLD ####
#=
@parallel_indices (I...)  function update_stresses_center_vertex_psAD!(
    ε::NTuple{3},           # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{3},       # whole Voigt tensor @ centers
    EII,                      # accumulated plastic strain rate @ centers
    τ::NTuple{3} ,         # whole Voigt tensor @ centers
    τshear_v::NTuple{1},  # shear tensor components @ vertices
    τ_o::NTuple{3},
    τshear_ov::NTuple{1}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep, 
    relλ, 
    dt, 
    θ_dτ,
    rheology, 
    phase_center,
    phase_vertex,
    phase_xy,
    phase_yz,
    phase_xz,
)

τxyv = τshear_v[1]
τxyv_old = τshear_ov[1]
ni = size(Pr)
Ic = clamped_indices(ni, I...)

# interpolate to ith vertex
Pv_ij = av_clamped(Pr, Ic...)
εxxv_ij = av_clamped(ε[1], Ic...)
εyyv_ij = av_clamped(ε[2], Ic...)
τxxv_ij = av_clamped(τ[1], Ic...)
τyyv_ij = av_clamped(τ[2], Ic...)
τxxv_old_ij = av_clamped(τ_o[1], Ic...)
τyyv_old_ij = av_clamped(τ_o[2], Ic...)
EIIv_ij = av_clamped(EII, Ic...)

## vertex
phase = @inbounds phase_vertex[I...]
is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
Kv = fn_ratio(get_bulk_modulus, rheology, phase)
#volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric #change K * dt * sinϕ * sinψ
volumev=0.0
ηv_ij = av_clamped(η, Ic...)
dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

# stress increments @ vertex
dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
dτxyv = compute_stress_increment(
    τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
)
#τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)

#τIIv_ij= av_clamped(τII, Ic...)

#=
# yield function @ center
Fv = τIIv_ij - Cv - Pv_ij * sinϕv
if is_pl && !iszero(τIIv_ij) && Fv > 0
    # stress correction @ vertex
    λv[I...] =
        (1.0 - relλ) * λv[I...] +
        relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
    dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
    τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λv[I...] * dQdτxy * dτ_rv
else
=#    # stress correction @ vertex
    τxyv[I...] += dτxyv
#end

## center
if all(I .≤ ni)
    # Material properties
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
    K = fn_ratio(get_bulk_modulus, rheology, phase)
    #volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change #K * dt * sinϕ * sinψ
    volume=0.0
    ηij = η[I...]
    dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

    # cache strain rates for center calculations
    τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

    # visco-elastic strain rates @ center
    εij_ve = @. εij + 0.5 * τij_o * _Gdt
    εII_ve = GeoParams.second_invariant(εij_ve)
    # stress increments @ center
    # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
    dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
    #τII_ij = GeoParams.second_invariant(dτij .+ τij)

    #τII_ij= τII[I...]
    # yield function @ center
    #F = τII_ij - C - Pr[I...] * sinϕ
#=
    if is_pl && !iszero(τII_ij) && F > 0
        # stress correction @ center
        λ[I...] =
            (1.0 - relλ) * λ[I...] +
            relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
        dQdτij = @. 0.5 * (τij + dτij) / τII_ij
        # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
        εij_pl = λ[I...] .* dQdτij
        dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
        τij = dτij .+ τij
        setindex!.(τ, τij, I...)
        #setindex!.(ε_pl, εij_pl, I...)
        #τII[I...] = GeoParams.second_invariant(τij)
        η_vep[I...] = 0.5 * τII_ij / εII_ve
    else
 =#       # stress correction @ center
        setindex!.(τ, dτij .+ τij, I...)
        #η_vep[I...] = ηij
        #τII[I...] = τII_ij
 #   end

    #Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
end

return nothing
end

@parallel_indices (I...)  function update_stresses_center_vertex_psADTemp!(
    ε::NTuple{3},           # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{3},       # whole Voigt tensor @ centers
    EII,                      # accumulated plastic strain rate @ centers
    τ::NTuple{3} ,         # whole Voigt tensor @ centers
    τshear_v::NTuple{1},  # shear tensor components @ vertices
    τ_o::NTuple{3},
    τshear_ov::NTuple{1}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep, 
    relλ, 
    dt, 
    θ_dτ,
    rheology, 
    phase_center,
    phase_vertex,
    phase_xy,
    phase_yz,
    phase_xz,
)

τxyv = τshear_v[1]
τxyv_old = τshear_ov[1]
ni = size(Pr)
Ic = clamped_indices(ni, I...)

# interpolate to ith vertex
Pv_ij = av_clamped(Pr, Ic...)
εxxv_ij = av_clamped(ε[1], Ic...)
εyyv_ij = av_clamped(ε[2], Ic...)
τxxv_ij = av_clamped(τ[1], Ic...)
τyyv_ij = av_clamped(τ[2], Ic...)
τxxv_old_ij = av_clamped(τ_o[1], Ic...)
τyyv_old_ij = av_clamped(τ_o[2], Ic...)
EIIv_ij = av_clamped(EII, Ic...)

## vertex
phase = @inbounds phase_vertex[I...]
is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
Kv = fn_ratio(get_bulk_modulus, rheology, phase)
volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
#if isinf(Kv)
#    Kv = 0.0
#end
#volumev = Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
ηv_ij = av_clamped(η, Ic...)
dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

# stress increments @ vertex
dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
dτxyv = compute_stress_increment(
    τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
)
#τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
# trial stress is known from forward solve
τIIv_ij= av_clamped(τII, Ic...)

λtempv =  λv[I...]
# yield function @ vertex
Fv = τIIv_ij - Cv - Pv_ij * sinϕv
if is_pl && !iszero(τIIv_ij) && Fv > 0
    # stress correction @ vertex
    #λv[I...] =
    #    (1.0 - relλ) * λv[I...] +
    #    relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
    #λv[I...] =
    #    (1.0 - relλ) * λv[I...] +
    #    relλ * (Fv / (ηv_ij * dτ_rv + η_regv + volumev))
    λtempv =
        (1.0 - relλ) * λtempv +
        relλ * (Fv / (ηv_ij * dτ_rv + η_regv + volumev))
    dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
    τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λtempv * dQdτxy * dτ_rv
else
    # stress correction @ vertex
    τxyv[I...] += dτxyv
end

## center
if all(I .≤ ni)
    # Material properties
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
    K = fn_ratio(get_bulk_modulus, rheology, phase)
    volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
    #if isinf(K)
    #    K = 0.0
    #end
    #volume =  K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
    ηij = η[I...]
    dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

    # cache strain rates for center calculations
    τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

    # visco-elastic strain rates @ center
    εij_ve = @. εij + 0.5 * τij_o * _Gdt
    εII_ve = GeoParams.second_invariant(εij_ve)
    # stress increments @ center
    # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
    dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
    #τII_ij = GeoParams.second_invariant(dτij .+ τij)
    # trial stress is known from forward solve
    τII_ij= τII[I...]

    λtemp = λ[I...]
    # yield function @ center
    F = τII_ij - C - Pr[I...] * sinϕ
    if is_pl && !iszero(τII_ij) && F > 0
        # stress correction @ center
        #λ[I...] =
        #    (1.0 - relλ) * λ[I...] +
        #    relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
        #λ[I...] =
        #    (1.0 - relλ) * λ[I...] +
        #    relλ * (F / (η[I...] * dτ_r + η_reg + volume))
        λtemp =
            (1.0 - relλ) * λtemp +
            relλ * (F / (η[I...] * dτ_r + η_reg + volume))
        dQdτij = @. 0.5 * (τij + dτij) / τII_ij
        # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
        εij_pl = λtemp .* dQdτij
        dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
        τij = dτij .+ τij
        setindex!.(τ, τij, I...)
        #setindex!.(ε_pl, εij_pl, I...)
        #τII[I...] = GeoParams.second_invariant(τij)
        #η_vep[I...] = 0.5 * τII_ij / εII_ve
    else
        # stress correction @ center
        setindex!.(τ, dτij .+ τij, I...)
        #η_vep[I...] = ηij
        #τII[I...] = τII_ij
    end

    #Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
end

return nothing
end
@parallel_indices (I...) function assemble_parameter_matrices!(
    EII,
    Gv,
    Gc,
    ϕv,
    ϕc,
    Cv,
    Cc,
    rheology, 
    phase_center,
    phase_vertex,

)

    ni = size(Gc)

    Ic       = clamped_indices(ni, I...)
    EIIv_ij  = av_clamped(EII, Ic...)
    phase    = @inbounds phase_vertex[I...]
    Gv[I...] = fn_ratio(get_shear_modulus, rheology, phase)

    is_pl, Cvi, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    ϕv[I...] =  sinϕv #asind(sinϕv)
    Cv[I...] = Cvi

    if all(I .≤ ni)
        phase = @inbounds phase_center[I...]
        Gc[I...] = fn_ratio(get_shear_modulus, rheology, phase)

        is_pl, Cci, sinϕc, cosϕc, sinψc, η_regc = plastic_params_phase(rheology, EII[I...], phase)
        ϕc[I...] = sinϕc # asind(sinϕc)
        Cc[I...] = Cci
    end

return nothing

end

# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_psSensTest!(
    ε::NTuple{3},         # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{3},      # whole Voigt tensor @ centers
    EII,                  # accumulated plastic strain rate @ centers
    τ::NTuple{3},         # whole Voigt tensor @ centers
    τshear_v::NTuple{1},  # shear tensor components @ vertices
    τ_o::NTuple{3},
    τshear_ov::NTuple{1}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep,
    relλ,
    dt,
    θ_dτ,
    rheology,
    phase_center,
    phase_vertex,
    Gv,
    Gc,
    ϕv,
    ϕc,
    Cv,
    Cc,
)
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij       = av_clamped(Pr, Ic...)
    εxxv_ij     = av_clamped(ε[1], Ic...)
    εyyv_ij     = av_clamped(ε[2], Ic...)
    τxxv_ij     = av_clamped(τ[1], Ic...)
    τyyv_ij     = av_clamped(τ[2], Ic...)
    τxxv_old_ij = av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = av_clamped(τ_o[2], Ic...)
    EIIv_ij     = av_clamped(EII, Ic...)

    ## vertex
    phase = av_clamped(phase_vertex,Ic...)
    Gv_ij       = av_clamped(Gv,Ic...)
    ϕv_ij       = av_clamped(ϕv,Ic...)
    Cv_ij       = av_clamped(Cv,Ic...)
    is_pl, Cv, sinϕvNot, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt = inv(Gv_ij * dt)
    sinϕv = ϕv_ij
    #cosϕv = cosd(ϕv_ij)
    Cv    = Cv_ij

    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    if isinf(Kv)
        Kv = 0.0
    end
    
    volumev = Kv * dt * sinϕv * sinψv # plastic volumetric #change K * dt * sinϕ * sinψ

    ηv_ij = av_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
    dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
    dτxyv = compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
    )
    τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)

    # yield function @ center
    Fv = τIIv_ij - Cv - Pv_ij * sinϕv
    if is_pl && !iszero(τIIv_ij) && Fv > 0
        # stress correction @ vertex
        λv[I...] =
            (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
        τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λv[I...] * dQdτxy * dτ_rv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
    end

    ## center
    if all(I .≤ ni)
        # Material properties
        phase = phase_center[I...]
        Gc_ij       = Gc[I...]
        ϕc_ij       = ϕc[I...]
        Cc_ij       = Cc[I...]
        _Gcdt = inv(Gc_ij * dt)
        is_pl, C, sinϕNot, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        sinϕ = ϕc_ij
        #cosϕ = cosd(ϕv_ij)
        C    = Cc_ij

        K = fn_ratio(get_bulk_modulus, rheology, phase)
        if isinf(K)
            K = 0.0
        end
        volume = K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = 1.0 / (θ_dτ + ηij * _Gcdt + 1.0)

    # cache strain rates for center calculations
    τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

    # visco-elastic strain rates @ center
    εij_ve = @. εij + 0.5 * τij_o * _Gcdt
    εII_ve = GeoParams.second_invariant(εij_ve)
    # stress increments @ center
    # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
    dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gcdt, dτ_r)
    τII_ij = GeoParams.second_invariant(dτij .+ τij)
    # yield function @ center
    F = τII_ij - C - Pr[I...] * sinϕ

    if is_pl && !iszero(τII_ij) && F > 0
        # stress correction @ center
        λ[I...] =
            (1.0 - relλ) * λ[I...] +
            relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
        dQdτij = @. 0.5 * (τij + dτij) / τII_ij
        # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
        εij_pl = λ[I...] .* dQdτij
        dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
        τij = dτij .+ τij
        setindex!.(τ, τij, I...)
        #setindex!.(ε_pl, εij_pl, I...)
        #τII[I...] = GeoParams.second_invariant(τij)
        #η_vep[I...] = 0.5 * τII_ij / εII_ve
    else
        # stress correction @ center
        setindex!.(τ, dτij .+ τij, I...)
        #η_vep[I...] = ηij
        #τII[I...] = τII_ij
    end

    #Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
end

    return nothing
end


@parallel_indices (I...)  function update_stresses_center_vertex_psADSens!(
    ε::NTuple{3},           # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{3},       # whole Voigt tensor @ centers
    EII,                      # accumulated plastic strain rate @ centers
    τ::NTuple{3} ,         # whole Voigt tensor @ centers
    τshear_v::NTuple{1},  # shear tensor components @ vertices
    τ_o::NTuple{3},
    τshear_ov::NTuple{1}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep, 
    relλ, 
    dt, 
    θ_dτ,
    rheology, 
    phase_center,
    phase_vertex,
    phase_xy,
    phase_yz,
    phase_xz,
    Gv,
    Gc,
    ϕv,
    ϕc,
    Cv,
    Cc,
)
#=
τxyv = τshear_v[1]
τxyv_old = τshear_ov[1]
ni = size(Pr)
Ic = clamped_indices(ni, I...)

# interpolate to ith vertex
Pv_ij = av_clamped(Pr, Ic...)
εxxv_ij = av_clamped(ε[1], Ic...)
εyyv_ij = av_clamped(ε[2], Ic...)
τxxv_ij = av_clamped(τ[1], Ic...)
τyyv_ij = av_clamped(τ[2], Ic...)
τxxv_old_ij = av_clamped(τ_o[1], Ic...)
τyyv_old_ij = av_clamped(τ_o[2], Ic...)
EIIv_ij = av_clamped(EII, Ic...)

## vertex
phase = @inbounds phase_vertex[I...]
Gv_ij       = av_clamped(Gv,Ic...)
ϕv_ij       = av_clamped(ϕv,Ic...)
Cv_ij       = av_clamped(Cv,Ic...)
is_pl, CvNot, sinϕvNot, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
_Gvdt = inv(Gv_ij * dt)
sinϕv = ϕv_ij
Cv    = Cv_ij

Kv = fn_ratio(get_bulk_modulus, rheology, phase)
#volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
volumev=0.0
ηv_ij = av_clamped(η, Ic...)
dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

# stress increments @ vertex
dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
dτxyv = compute_stress_increment(
    τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
)
#τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + #dτxyv)^2)

#=
# yield function @ center
Fv = τIIv_ij - Cv - Pv_ij * sinϕv
if is_pl && !iszero(τIIv_ij) && Fv > 0
    # stress correction @ vertex
    λv[I...] =
        (1.0 - relλ) * λv[I...] +
        relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
    dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
    τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λv[I...] * dQdτxy * dτ_rv
else
=#    # stress correction @ vertex
    τxyv[I...] += dτxyv
#end

## center
if all(I .≤ ni)
    # Material properties
    phase = @inbounds phase_center[I...]
    Gc_ij       = Gc[I...]
    ϕc_ij       = ϕc[I...]
    Cc_ij       = Cc[I...]
    is_pl, CNot, sinϕNot, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
    _Gdt = inv(Gc_ij * dt)
    sinϕ = ϕc_ij
    C    = Cc_ij

    K = fn_ratio(get_bulk_modulus, rheology, phase)
    #volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
    volume=0.0
    ηij = η[I...]
    dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

    # cache strain rates for center calculations
    τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

    # visco-elastic strain rates @ center
    εij_ve = @. εij + 0.5 * τij_o * _Gdt
    εII_ve = GeoParams.second_invariant(εij_ve)
    # stress increments @ center
    # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
    dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
    #τII_ij = GeoParams.second_invariant(dτij .+ τij)
    # yield function @ center
    #F = τII_ij - C - Pr[I...] * sinϕ
#=
    if is_pl && !iszero(τII_ij) && F > 0
        # stress correction @ center
        λ[I...] =
            (1.0 - relλ) * λ[I...] +
            relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
        dQdτij = @. 0.5 * (τij + dτij) / τII_ij
        # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
        εij_pl = λ[I...] .* dQdτij
        dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
        τij = dτij .+ τij
        setindex!.(τ, τij, I...)
        #setindex!.(ε_pl, εij_pl, I...)
        #τII[I...] = GeoParams.second_invariant(τij)
        η_vep[I...] = 0.5 * τII_ij / εII_ve
    else
    =#    # stress correction @ center
        setindex!.(τ, dτij .+ τij, I...)
        #η_vep[I...] = ηij
        #τII[I...] = τII_ij
    #end

    #Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
end
=#
return nothing
end
=#
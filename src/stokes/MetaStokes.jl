abstract type AbstractStokesModel end
abstract type AbstractViscosity end
abstract type Viscous <: AbstractStokesModel end
abstract type AbstractElasticModel <: AbstractStokesModel end
abstract type ViscoElastic <: AbstractElasticModel end
abstract type ViscoElastoPlastic <: AbstractElasticModel end

function make_velocity_struct!(ndim::Integer; name::Symbol=:Velocity)
    dims = (:Vx, :Vy, :Vz)
    fields = [:($(dims[i])::T) for i in 1:ndim]
    @eval begin
        struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2,T}) where {T}
                return new{$PTArray}(@zeros(ni[1]...), @zeros(ni[2]...))
            end

            function $(name)(ni::NTuple{3,T}) where {T}
                return new{$PTArray}(@zeros(ni[1]...), @zeros(ni[2]...), @zeros(ni[3]...))
            end
        end
    end
end

function make_viscosity_struct!()
    @eval begin
        struct Viscosity{T}
            η::T # with no plasticity
            η_vep::T # with plasticity
            ητ::T # PT viscosity

            function Viscosity(ni::NTuple{N,Int}) where {N}
                η = @allocate(ni...)
                η_vep = @allocate(ni...)
                ητ = @allocate(ni...)
                return new{typeof(η)}(η, η_vep, ητ)
            end
        end
    end
end

function make_symmetrictensor_struct!(nDim::Integer; name::Symbol=:SymmetricTensor)
    dims = (:x, :y, :z)
    fields = [:($(Symbol((dims[i]), (dims[j])))::T) for i in 1:nDim, j in 1:nDim if j ≥ i]

    fields_c = if nDim == 2
        [:($(:xy_c)::T)]
    elseif nDim == 3
        [:($(:yz_c)::T), :($(:xz_c)::T), :($(:xy_c)::T)]
    end

    @eval begin
        struct $(name){T}
            $(fields...)
            $(fields_c...)
            II::T

            function $(name)(ni::NTuple{2,T}) where {T}
                return new{$PTArray}(
                    @zeros(ni...), # xx
                    @zeros(ni[1] + 1, ni[2] + 1), # xy
                    @zeros(ni...), # yy
                    @zeros(ni...), # xy @ cell center
                    @zeros(ni...) # II (second invariant)
                )
            end

            function $(name)(ni::NTuple{3,T}) where {T}
                return new{$PTArray}(
                    @zeros(ni...), # xx
                    @zeros(ni[1] + 1, ni[2] + 1, ni[3]), # xy
                    @zeros(ni...), # yy
                    @zeros(ni[1] + 1, ni[2], ni[3] + 1), # xz
                    @zeros(ni[1], ni[2] + 1, ni[3] + 1), # yz
                    @zeros(ni...), # zz
                    @zeros(ni...), # yz @ cell center
                    @zeros(ni...), # xz @ cell center
                    @zeros(ni...), # xy @ cell center
                    @zeros(ni...), # II (second invariant)
                )
            end
        end
    end
end

function make_residual_struct!(ndim; name::Symbol=:Residual)
    dims = (:Rx, :Ry, :Rz)
    fields = [:($(dims[i])::T) for i in 1:ndim]
    @eval begin
        struct $(name){T}
            $(fields...)
            RP::T

            function $(name)(ni::NTuple{3,T}) where {T}
                Rx = @zeros(ni[1]...)
                Ry = @zeros(ni[2]...)
                RP = @zeros(ni[3]...)
                return new{typeof(Rx)}(Rx, Ry, RP)
            end

            function $(name)(ni::NTuple{4,T}) where {T}
                Rx = @zeros(ni[1]...)
                Ry = @zeros(ni[2]...)
                Rz = @zeros(ni[3]...)
                RP = @zeros(ni[4]...)
                return new{typeof(Rx)}(Rx, Ry, Rz, RP)
            end
        end
    end
end

function make_stokes_struct!()
    @eval begin
        struct StokesArrays{M<:AbstractStokesModel,A,B,C,T,nDim}
            P::T
            P0::T
            V::A
            ∇V::T
            τ::B
            ε::B
            τ_o::Union{B,Nothing}
            R::C

            # 2D CONSTRUCTORS

            function StokesArrays(ni::NTuple{2,T}, model::Type{Viscous}) where {T}
                P = @zeros(ni...)
                P0 = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(((ni[1] + 1, ni[2] + 2), (ni[1], ni[2] + 2)))
                τ = SymmetricTensor(ni)
                ε = SymmetricTensor(ni)
                R = Residual(((ni[1] - 1, ni[2]), (ni[1], ni[2] - 1)), ni)

                return new{model,typeof(V),typeof(τ),typeof(R),typeof(P),2}(
                    P, P0, V, ∇V, τ, ε, nothing, R
                )
            end

            function StokesArrays(
                ni::NTuple{2,T}, model::Type{<:AbstractElasticModel}
            ) where {T}
                P = @zeros(ni...)
                P0 = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(((ni[1] + 1, ni[2] + 2), (ni[1] + 2, ni[2] + 1)))
                τ = SymmetricTensor(ni)
                ε = SymmetricTensor(ni)
                R = Residual(((ni[1] - 1, ni[2]), (ni[1], ni[2] - 1), ni))

                return new{model,typeof(V),typeof(τ),typeof(R),typeof(P),2}(
                    P, P0, V, ∇V, τ, ε, deepcopy(τ), R
                )
            end

            # 3D CONSTRUCTORS

            function StokesArrays(ni::NTuple{3,T}, model::Type{Viscous}) where {T}
                P = @zeros(ni...)
                P0 = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity((
                    (ni[1] + 1, ni[2], ni[3]),
                    (ni[1], ni[2] + 1, ni[3]),
                    (ni[1], ni[2], ni[3] + 1),
                ))
                τ = SymmetricTensor(ni)
                ε = SymmetricTensor(ni)
                R = Residual((
                    (ni[1] - 1, ni[2] - 2, ni[3] - 2),
                    (ni[1] - 2, ni[2] - 1, ni[3] - 2),
                    (ni[1] - 2, ni[2] - 2, ni[3] - 1),
                    ni,
                ))

                return new{model,typeof(V),typeof(τ),typeof(R),typeof(P),3}(
                    P, P0, V, ∇V, τ, ε, nothing, R
                )
            end

            function StokesArrays(
                ni::NTuple{3,T}, model::Type{<:AbstractElasticModel}
            ) where {T}
                P = @zeros(ni...)
                P0 = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity((
                    (ni[1] + 1, ni[2] + 2, ni[3] + 2),
                    (ni[1] + 2, ni[2] + 1, ni[3] + 2),
                    (ni[1] + 2, ni[2] + 2, ni[3] + 1),
                ))
                τ = SymmetricTensor(ni)
                ε = SymmetricTensor(ni)
                R = Residual((
                    (ni[1] - 1, ni[2], ni[3]),
                    (ni[1], ni[2] - 1, ni[3]),
                    (ni[1], ni[2], ni[3] - 1),
                    ni,
                ))

                return new{model,typeof(V),typeof(τ),typeof(R),typeof(P),3}(
                    P, P0, V, ∇V, τ, ε, deepcopy(τ), R
                )
            end
        end
    end
end

function make_PTstokes_struct!()
    @eval begin
        struct PTStokesCoeffs{T}
            CFL::T
            ϵ::T # PT tolerance
            Re::T # Reynolds Number
            r::T # 
            Vpdτ::T
            θ_dτ::T
            ηdτ::T

            function PTStokesCoeffs(
                li::NTuple{N,T},
                di;
                ϵ=1e-8,
                Re=3π,
                CFL=(N == 2 ? 0.9 / √2.1 : 0.9 / √3.1),
                r=0.7,
            ) where {N,T}
                lτ = min(li...)
                Vpdτ = min(di...) * CFL
                θ_dτ = lτ * (r + 2.0) / (Re * Vpdτ)
                ηdτ = Vpdτ * lτ / Re

                return new{typeof(r)}(CFL, ϵ, Re, r, Vpdτ, θ_dτ, ηdτ)
            end
        end
    end
end
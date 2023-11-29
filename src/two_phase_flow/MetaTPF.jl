abstract type PoreEvolution <: AbstractStokesModel end

function make_pressure_struct!(; name::Symbol=:TPF_Pressure)
    dims = (:Pt, :Pe)
    fields = [:($(dims[i])::T) for i in 1:2]
    @eval begin
        struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2,T}) where {T}
                return new{$PTArray}(@zeros(ni), @zeros(ni))
            end
        end
    end
end

function make_P_residual_struct!(; name::Symbol=:P_Residual)
    dims = (:RPt, :RPe)
    fields = [:($(dims[i])::T) for i in 1:2]
    @eval begin
        struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2,T}) where {T}
                return new{$PTArray}(@zeros(ni), @zeros(ni))
            end
        end
    end
end

function make_TPF_struct!()
    @eval begin
        struct TPFArrays{M<:AbstractStokesModel,A,B,C,D,E,T,nDim}
            P::A
            dP::A
            V::B
            dV::B
            qD::B
            dqD::B
            τ::C
            τ_o::Union{C,Nothing}
            R::D
            RqD::D
            RP::E
            φ::T
            η_φ::T
            k_ηf::T
            k_ηfτ::T
            η_φτ::T
            ρtg::T

            # 2D CONSTRUCTORS

            function TPFArrays(ni::NTuple{2,T}, model::Type{Viscous}) where {T}
                P = TPF_Pressure(ni)
                dP = TPF_Pressure(ni)
                V = Velocity(((ni[1] + 1, ni[2]), (ni[1], ni[2] + 1)))
                dV = Velocity(((ni[1] - 1, ni[2] - 2), (ni[1] - 2, ni[2] - 1)))
                qD = Velocity(((ni[1] + 1, ni[2]), (ni[1], ni[2] + 1)))
                # qD y to (ni[1], ni[2] + 1)))
                dqD = Velocity(((ni[1] - 1, ni[2] - 2), (ni[1] - 2, ni[2] - 1)))
                τ = SymmetricTensor(ni)
                R = Residual(((ni[1] - 1, ni[2] - 2), (ni[1] - 2, ni[2] - 1)))
                RqD = Residual(((ni[1] - 1, ni[2] - 2), (ni[1] - 2, ni[2] - 1)))
                RP = P_Residual(ni)
                φ = @zeros(ni...)
                return new{
                    model,typeof(P),typeof(V),typeof(τ),typeof(R),typeof(RP),typeof(φ),2
                }(
                    P, dP, V, dV, qD, dqD , τ, nothing, R, RqD, RP, φ, deepcopy(φ), deepcopy(φ), deepcopy(φ), deepcopy(φ), deepcopy(φ)
                )
            end
        end
    end
end

function make_PTTPF_struct!()
    @eval begin
        struct PTTPFCoeffs{M<:AbstractStokesModel,T,nDim}
            CFL::T
            ϵ::T # PT tolerance
            Re_M::T # mechanical Reynolds number
            Re_F::AbstractArray{T,nDim} # Reynolds number for fluid
            r::T # ration between elastic and bulk moduli
            Vpdτ::T # P-wave velocity
            Rhodτ_M::AbstractArray{T,nDim} # mechanical numerical density
            Rhodτ_F::AbstractArray{T,nDim} # numerical density for fluid
            Gdτ::AbstractArray{T,nDim} # elastic modulus
            Betadτ::AbstractArray{T,nDim} 
            function PTTPFCoeffs(
                ni::NTuple{nDim,A},
                di::NTuple{nDim,T},
                model::Type{Viscous};
                ϵ=1e-8,
                Re_M=3.0 * sqrt(10.0) / 2.0 * π,
                CFL=0.9 / √2,
                r=0.5
            ) where {nDim,A,T}
                Vpdτ = min(di...) * CFL
                Gdτ = @zeros(ni...)
                Re_F = @zeros(ni...)
                Rhodτ_M = @zeros(ni...)
                Rhodτ_F = @zeros(ni...)
                Betadτ = @zeros(ni...)
                
                return new{model, typeof(ϵ),2}(
                    CFL,
                    ϵ,
                    Re_M,
                    Re_F,
                    r,
                    Vpdτ,
                    Rhodτ_M,
                    Rhodτ_F,
                    Gdτ,
                    Betadτ
                )
            end
        end
    end
end

function make_TPF_parameter_struct!()
    @eval begin
        struct PTTPFParams{M<:AbstractStokesModel,A,T,nDim}
            n::A
            ρm::AbstractArray{T,nDim}
            lc::T
            kr::T
            φr::T
            ηr::T
            g::T
            dt::T
            function PTTPFParams(
                ni::NTuple{nDim,A},
                ρ::T,
                model::Type{Viscous};
                n=3,
                lc=1.0,
                φr=0.1,
                ηr=1.0,
                g=1.0
            ) where{nDim,A,T}
                kr=lc^2/ηr
                ρm=@zeros(ni...)
                dt=φr/(ρ*g*lc)*1e-3
            
                return new{model, typeof(n), typeof(kr),2}(n, ρm, lc, kr, φr, ηr, g, dt)
            end
        end
    end
end
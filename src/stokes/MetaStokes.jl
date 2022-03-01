abstract type AbstractStokesModel end
abstract type Viscous <: AbstractStokesModel end
abstract type ViscoElastic <: AbstractStokesModel end

function make_velocity_struct!(ndim::Integer; name::Symbol = :Velocity)
    dims = (:Vx, :Vy, :Vz)
    fields = [:( $(dims[i])::T ) for i in 1:ndim]
    @eval begin
        struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2, T}) where T
                new{$PTArray}(
                    @zeros(ni[1]...),
                    @zeros(ni[2]...)
                )
            end

            function $(name)(ni::NTuple{3, T}) where T
                new{$PTArray}(
                    @zeros(ni[1]...),
                    @zeros(ni[2]...),
                    @zeros(ni[3]...)
                )
            end

        end
    end
end

function make_symmetrictensor_struct!(nDim::Integer; name::Symbol = :SymmetricTensor)
    dims = (:x, :y, :z)
    fields = [:( $(Symbol((dims[i]), (dims[j])))::T ) for i in 1:nDim, j in 1:nDim if j≥i]

    @eval begin
        struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2, T}) where T
                new{$PTArray}(
                    @zeros(ni...), # xx
                    @zeros(ni[1]-1, ni[2]-1), # xy
                    @zeros(ni...) # yy
                )
            end
            
            function $(name)(ni::NTuple{3, T}) where T
                new{$PTArray}(
                    @zeros(ni[1]  , ni[2]-2, ni[3]-2), # xx
                    @zeros(ni[1]-1, ni[2]-1, ni[3]-2), # xy
                    @zeros(ni[1]-2, ni[2]  , ni[3]-2), # yy
                    @zeros(ni[1]-1, ni[2]-2, ni[3]-1), # xz
                    @zeros(ni[1]-2, ni[2]-1, ni[3]-1), # yz
                    @zeros(ni[1]-2, ni[2]-2, ni[3]  ), # zz
                )
            end

        end
    end
end

function make_residual_struct!(ndim; name::Symbol = :Residual)
    dims = (:Rx, :Ry, :Rz)
    fields = [:( $(dims[i])::T ) for i in 1:ndim]
    @eval begin
        struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2, T}) where T
                new{$PTArray}(
                    @zeros(ni[1]...),
                    @zeros(ni[2]...)
                )
            end

            function $(name)(ni::NTuple{3, T}) where T
                new{$PTArray}(
                    @zeros(ni[1]...),
                    @zeros(ni[2]...),
                    @zeros(ni[3]...)
                )
            end

        end
    end
end

function make_stokes_struct!()
    @eval begin
        struct StokesArrays{M <: AbstractStokesModel, A,B,C,T, nDim}
            P::T
            V::A
            dV::A
            ∇V::T
            τ::B
            τ_o::Union{B, Nothing}
            R::C
            
            # 2D CONSTRUCTORS

            function StokesArrays(ni::NTuple{2, T}, model::Type{Viscous}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    ((ni[1]+1, ni[2]), (ni[1], ni[2]+1))
                )
                τ = SymmetricTensor(ni)
                dV = Velocity(
                    ((ni[1]-1, ni[2]-2), (ni[1]-2, ni[2]-1))
                )
                R = Residual(
                    ((ni[1]-1, ni[2]-2), (ni[1]-2, ni[2]-1))
                )

                new{model, typeof(V), typeof(τ), typeof(R), typeof(P), 2}(P, V, dV, ∇V, τ, nothing, R)
            end

            function StokesArrays(ni::NTuple{2, T}, model::Type{ViscoElastic}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    ((ni[1]+1, ni[2]), (ni[1], ni[2]+1))
                )
                τ = SymmetricTensor(ni)
                dV = Velocity(
                    ((ni[1]-1, ni[2]-2), (ni[1]-2, ni[2]-1))
                )
                R = Residual(
                    ((ni[1]-1, ni[2]-2), (ni[1]-2, ni[2]-1))
                )

                new{model, typeof(V), typeof(τ), typeof(R), typeof(P), 2}(P, V, dV, ∇V, τ, deepcopy(τ), R)
            end

            # 3D CONSTRUCTORS

            function StokesArrays(ni::NTuple{3, T}, model::Type{Viscous}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    ((ni[1]+1, ni[2], ni[3]), (ni[1], ni[2]+1, ni[3]), (ni[1], ni[2], ni[3]+1))
                )
                τ = SymmetricTensor(ni)
                dV = Velocity(
                    ((ni[1]-1, ni[2]-2, ni[3]-2), (ni[1]-2, ni[2]-1, ni[3]-2), (ni[1]-1, ni[2]-1, ni[3]-1))
                )
                R = Residual(
                    ((ni[1]-1, ni[2]-2, ni[3]-2), (ni[1]-1, ni[2]-1, ni[3]-1), (ni[1]-2, ni[2]-1, ni[3]-2))
                )

                new{model, typeof(V), typeof(τ), typeof(R), typeof(P), 3}(P, V, dV, ∇V, τ, nothing, R)
            end

            function StokesArrays(ni::NTuple{3, T}, model::Type{ViscoElastic}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    ((ni[1]+1, ni[2], ni[3]), (ni[1], ni[2]+1, ni[3]), (ni[1], ni[2], ni[3]+1))
                )
                τ = SymmetricTensor(ni)
                dV = Velocity(
                    ((ni[1]-1, ni[2]-2, ni[3]-2), (ni[1]-2, ni[2]-1, ni[3]-2), (ni[1]-1, ni[2]-1, ni[3]-1))
                )
                R = Residual(
                    ((ni[1]-1, ni[2]-2, ni[3]-2), (ni[1]-1, ni[2]-1, ni[3]-1), (ni[1]-2, ni[2]-1, ni[3]-2))
                )

                new{model, typeof(V), typeof(τ), typeof(R), typeof(P), 3}(P, V, dV, ∇V, τ, deepcopy(τ), R)
            end

        end

    end
end
    
function make_PTstokes_struct!()
    @eval begin
        struct PTStokesCoeffs{T, nDim}
            CFL::T
            ϵ::T # PT tolerance
            Re::T # Reynolds Number
            r::T # 
            Vpdτ::T
            dτ_Rho::AbstractArray{T, nDim} 
            Gdτ::AbstractArray{T, nDim}
        
            function PTStokesCoeffs(ni::NTuple{nDim, T}, di; ϵ = 1e-8, Re = 5π, CFL = 0.9/√2, r=1e0) where {nDim, T}
            
                Vpdτ = min(di...)*CFL
                Gdτ = @zeros(ni...)
                dτ_Rho = @zeros(ni...)
        
                new{eltype(Gdτ), nDim}(CFL, ϵ, Re, r, Vpdτ, dτ_Rho, Gdτ)
            end
        
        end
    end
end
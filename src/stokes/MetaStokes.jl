abstract type AbstractStokesModel end
abstract type Viscous <: AbstractStokesModel end
abstract type ViscoElastic <: AbstractStokesModel end

function make_velocity_struct!(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct Velocity{T} \n"*
        join("V$(dims[i])::T\n" for i in 1:nDim)*
        "end"
    )
    eval(str)
end

function make_tensor_struct!(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct SymmetricTensor{T} \n"*
        join("$(dims[i])$(dims[j])::T\n" for i in 1:nDim, j in 1:nDim if j≥i)*
        "end"
    )
    eval(str)
end

function make_residual_struct!(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct Residual{T} \n"*
        join("R$(dims[i])::T\n" for i in 1:nDim)*
        "end"
    )
    eval(str)
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
            
            function StokesArrays(ni::NTuple{2, T}, model::Type{Viscous}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    @zeros(ni[1]+1, ni[2]),
                    @zeros(ni[1], ni[2]+1)
                )
                τ = SymmetricTensor(
                    @zeros(ni...), # xx
                    @zeros(ni[1]-1, ni[2]-1), # xy
                    @zeros(ni...) # yy
                )
                dV = Velocity(
                    @zeros(ni[1]-1, ni[2]-2),
                    @zeros(ni[1]-2, ni[2]-1)
                )
                R = Residual(
                    @zeros(ni[1]-1, ni[2]-2),
                    @zeros(ni[1]-2, ni[2]-1)
                )

                new{model, typeof(V), typeof(τ), typeof(R), typeof(P), 2}(P, V, dV, ∇V, τ, nothing, R)
            end

            function StokesArrays(ni::NTuple{2, T}, model::Type{ViscoElastic}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    @zeros(ni[1]+1, ni[2]),
                    @zeros(ni[1], ni[2]+1)
                )
                τ = SymmetricTensor(
                    @zeros(ni...), # xx
                    @zeros(ni[1]-1, ni[2]-1), # xy
                    @zeros(ni...) # yy
                )
                dV = Velocity(
                    @zeros(ni[1]-1, ni[2]-2),
                    @zeros(ni[1]-2, ni[2]-1)
                )
                R = Residual(
                    @zeros(ni[1]-1, ni[2]-2),
                    @zeros(ni[1]-2, ni[2]-1)
                )

                new{model, typeof(V), typeof(τ), typeof(R), typeof(P), 2}(P, V, dV, ∇V, τ, deepcopy(τ), R)
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
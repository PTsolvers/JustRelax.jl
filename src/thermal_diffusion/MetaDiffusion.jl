function make_thermal_arrays!(ndim)
    flux_sym1 = (:qTx, :qTy, :qTz)
    flux_sym2 = (:qTx2, :qTy2, :qTz2)
    flux1 = [:($(flux_sym1[i])::_T) for i in 1:ndim]
    flux2 = [:($(flux_sym2[i])::_T) for i in 1:ndim]

    @eval begin
        struct ThermalArrays{_T}
            T::_T
            T0::_T
            Told::_T
            $(flux1...)
            $(flux2...)
            ResT::_T

            function ThermalArrays(ni::NTuple{1,Integer})
                nx = ni[1]
                T, T0, Told = @zeros(ni...), @zeros(ni...), @zeros(ni...)
                qTx = @zeros(nx - 1)
                qTx2 = @zeros(nx - 1)
                ResT = @zeros(nx - 2)
                return new{typeof(T)}(T, T0, Told, qTx, qTx2, ResT)
            end

            function ThermalArrays(ni::NTuple{2,Integer})
                nx, ny = ni
                T, T0, Told = @zeros(ni...), @zeros(ni...), @zeros(ni...)
                qTx = @zeros(nx - 1, ny - 2)
                qTy = @zeros(nx - 2, ny - 1)
                qTx2 = @zeros(nx - 1, ny - 2)
                qTy2 = @zeros(nx - 2, ny - 1)
                ResT = @zeros((ni .- 2)...)
                return new{typeof(T)}(T, T0, Told, qTx, qTy, qTx2, qTy2, ResT)
            end

            function ThermalArrays(ni::NTuple{3,Integer})
                nx, ny, nz = ni
                T, T0, Told = @zeros(ni...), @zeros(ni...), @zeros(ni...)
                qTx = @zeros(nx - 1, ny - 2, nz - 2)
                qTy = @zeros(nx - 2, ny - 1, nz - 2)
                qTz = @zeros(nx - 2, ny - 2, nz - 1)
                qTx2 = @zeros(nx - 1, ny - 2, nz - 2)
                qTy2 = @zeros(nx - 2, ny - 1, nz - 2)
                qTz2 = @zeros(nx - 2, ny - 2, nz - 1)
                ResT = @zeros((ni .- 2)...)
                return new{typeof(T)}(T, T0, Told, qTx, qTy, qTz, qTx2, qTy2, qTz2, ResT)
            end
        end
    end
end

function make_PTthermal_struct!()
    @eval begin
        struct PTThermalCoeffs{T,M,nDim}
            CFL::T
            ϵ::T
            max_lxyz::T
            max_lxyz2::T
            Vpdτ::T
            θr_dτ::M
            dτ_ρ::M

            function PTThermalCoeffs(
                K, ρCp, dt, di::NTuple{nDim,T}, li::NTuple{nDim,Any}; ϵ=1e-8, CFL=0.9 / √3
            ) where {nDim,T}
                Vpdτ = min(di...) * CFL
                max_lxyz = max(li...)
                max_lxyz2 = max_lxyz^2
                Re = @. π + √(π * π + ρCp * max_lxyz2 / K / dt) # Numerical Reynolds number
                θr_dτ = @. max_lxyz / Vpdτ / Re
                dτ_ρ = @. Vpdτ * max_lxyz / K / Re

                return new{eltype(Vpdτ),typeof(dτ_ρ),nDim}(
                    CFL, ϵ, max_lxyz, max_lxyz2, Vpdτ, θr_dτ, dτ_ρ
                )
            end
        end
    end
end

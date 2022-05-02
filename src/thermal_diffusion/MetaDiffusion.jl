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

            function ThermalArrays(ni::NTuple{2,Integer})
                nx, ny = ni
                T, T0, Told = @zeros(ni...), @zeros(ni...), @zeros(ni...)
                qTx = @zeros(nx - 1, ny - 2)
                qTy = @zeros(nx - 2, ny - 1)
                qTx2 = @zeros(nx - 1, ny - 2)
                qTy2 = @zeros(nx - 2, ny - 1)
                ResT = @zeros((ni .- 2)...)
                return new{typeof(T)}(T, T0, Told, qTx, qTy, qTz, qTx2, qTy2, qTz2, ResT)
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
        struct PTThermalCoeffs{T,nDim}
            CFL::T
            ϵ::T
            Resc::T
            max_lxyz::T
            max_lxyz2::T
            Vpdτ::T

            function PTThermalCoeffs(
                di::NTuple{nDim,T},
                li::NTuple{nDim,Any};
                ϵ=1e-8,
                Resc=1 / 1.2,
                CFL=0.9 / √3,
            ) where {nDim, T}
                Vpdτ = min(di...) * CFL
                max_lxyz = max(li...)
                max_lxyz2 = max_lxyz^2

                return new{eltype(Vpdτ),nDim}(
                    promote(CFL, ϵ, Resc, max_lxyz, max_lxyz2, Vpdτ)...
                )
            end
        end
    end
end

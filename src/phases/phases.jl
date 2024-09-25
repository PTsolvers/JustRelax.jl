"""
    fn_ratio(fn::F, rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio) where {N, F}

Average the function `fn` over the material phases in `rheology` using the phase ratios `ratio`.
"""
@generated function fn_ratio(
    fn::F, rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio
) where {N,F}
    quote
        Base.@_inline_meta
        x = 0.0
        Base.@nexprs $N i -> x += iszero(ratio[i]) ? 0.0 : fn(rheology[i]) * ratio[i]
        return x
    end
end

@generated function fn_ratio(
    fn::F, rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, args::NamedTuple
) where {N,F}
    quote
        Base.@_inline_meta
        x = 0.0
        Base.@nexprs $N i -> x += begin
            r = ratio[i]
            isone(r) && return fn(rheology[i], args) * r
            iszero(r) ? 0.0 : fn(rheology[i], args) * r
        end
        return x
    end
end
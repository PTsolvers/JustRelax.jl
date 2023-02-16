function unroll(f::F, args::NTuple{N,T}) where {F,N,T}
    ntuple(Val(N)) do i
        f(args[i])
    end
end

macro unroll(f, args)
    return esc(:(unroll($f, $args)))
end

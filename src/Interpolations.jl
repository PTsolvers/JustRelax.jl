# From cell vertices to cell center

@parallel_indices (i, j) function vertex2center!(C, V)

    @inbounds C[i, j] = 0.25 * (
        V[i  , j  ] +
        V[i+1, j  ] +
        V[i  , j+1] +
        V[i+1, j+1]
    )
    
    return nothing
end

@parallel_indices (i, j, k) function vertex2center!(C, V)

    @inbounds C[i,j,k] = 0.125 * (
        V[i  , j  , k  ] +
        V[i  , j  , k+1] +
        V[i  , j+1, k  ] +
        V[i  , j+1, k+1] +
        V[i+1, j  , k  ] +
        V[i+1, j  , k+1] +
        V[i+1, j+1, k  ] +
        V[i+1, j+1, k+1] 
    )

    return nothing
end
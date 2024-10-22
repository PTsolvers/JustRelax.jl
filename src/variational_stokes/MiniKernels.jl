# masked versions
for fn in (:center, :next, :left, :right, :back, :front)
    @eval begin
        Base.@propagate_inbounds @inline ($fn)(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = ($fn)(A, inds...) * ($fn)(ϕ, inds...)        
    end
end


# finite differences
Base.@propagate_inbounds @inline _d_xa(A::T, ϕ::T, _dx, I::Vararg{Integer, N}) where {N, T} = (-center(A, ϕ, I...) + right(A, ϕ, I...)) * _dx
Base.@propagate_inbounds @inline _d_ya(A::T, ϕ::T, _dy, I::Vararg{Integer, N}) where {N, T} = (-center(A, ϕ, I...) + front(A, ϕ, I...)) * _dy
Base.@propagate_inbounds @inline _d_za(A::T, ϕ::T, _dz, I::Vararg{Integer, N}) where {N, T} = (-center(A, ϕ, I...) + front(A, ϕ, I...)) * _dz
Base.@propagate_inbounds @inline _d_xi(A::T, ϕ::T, _dx, I::Vararg{Integer, N}) where {N, T} = (-front(A, ϕ, I...) + next(A, ϕ, I...)) * _dx
Base.@propagate_inbounds @inline _d_yi(A::T, ϕ::T, _dy, I::Vararg{Integer, N}) where {N, T} = (-right(A, ϕ, I...) + next(A, ϕ, I...)) * _dy
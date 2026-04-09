import ImplicitGlobalGrid:
    global_grid,
    GGNumber,
    GGArray,
    @coordx,
    @nx,
    @nx_g,
    @olx,
    @periodx,
    @coordy,
    @ny,
    @ny_g,
    @oly,
    @periody,
    @coordz,
    @nz,
    @nz_g,
    @olz,
    @periodz

# Functions modified from ImplicitGlobalGrid.jl

x_g(idx::Integer, dxi::GGNumber, nxi::GGNumber) = _x_g(idx, dxi, nxi)
x_g(idx::Integer, dxi::GGNumber, A::GGArray) = _x_g(idx, dxi, size(A, 1))

function _x_g(idx::Integer, dxi::GGNumber, nxi::GGNumber)
    x0i = 0.5 * (@nx() - nxi) * dxi
    xi = (@coordx() * (@nx() - @olx()) + idx - 1) * dxi + x0i
    if @periodx()
        xi = xi - dxi # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (xi > (@nx_g() - 1) * dxi)
            xi = xi - @nx_g() * dxi
        end # It must not be (nx_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (xi < 0)
            xi = xi + @nx_g() * dxi
        end
    end
    return xi
end

y_g(idx::Integer, dxi::GGNumber, nxi::GGNumber) = _y_g(idx, dxi, nxi)
y_g(idx::Integer, dxi::GGNumber, A::GGArray) = _z_g(idx, dxi, size(A, 2))

function _y_g(idx::Integer, dxi::GGNumber, nxi::GGNumber)
    x0i = 0.5 * (@ny() - nxi) * dxi
    xi = (@coordy() * (@ny() - @oly()) + idx - 1) * dxi + x0i
    if @periody()
        xi = xi - dxi # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (xi > (@ny_g() - 1) * dxi)
            xi = xi - @ny_g() * dxi
        end # It must not be (ny_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (xi < 0)
            xi = xi + @ny_g() * dxi
        end
    end
    return xi
end

z_g(idx::Integer, dxi::GGNumber, nxi::GGNumber) = _z_g(idx, dxi, nxi)
z_g(idx::Integer, dxi::GGNumber, A::GGArray) = _z_g(idx, dxi, size(A, 3))

function _z_g(idx::Integer, dxi::GGNumber, nxi::GGNumber)
    x0i = 0.5 * (@nz() - nxi) * dxi
    xi = (@coordz() * (@nz() - @olz()) + idx - 1) * dxi + x0i
    if @periodz()
        xi = xi - dxi # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (xi > (@nz_g() - 1) * dxi)
            xi = xi - @nz_g() * dxi
        end # It must not be (nz_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (xi < 0)
            xi = xi + @nz_g() * dxi
        end
    end
    return xi
end

###############################
# MACROS TO INDEX GRID ARRAYS #
###############################

macro dxi(args...)
    return :(get_dxi($(esc.(args)...)))
end

Base.@propagate_inbounds @inline get_dxi(dxi::NTuple{2, Union{Number, AbstractVector}}, I::Integer, J::Integer) = get_dx(dxi, I), get_dy(dxi, J)
Base.@propagate_inbounds @inline get_dxi(dxi::NTuple{3, Union{Number, AbstractVector}}, I::Integer, J::Integer, K::Integer) = get_dx(dxi, I), get_dy(dxi, J), get_dz(dxi, K)

macro dx(args...)
    return :(get_dx($(esc.(args)...)))
end

Base.@propagate_inbounds @inline get_dx(dx::NTuple{N, Union{Number, AbstractVector}}, I::Integer) where {N} = getindex_dxi(dx[1], I)

macro dy(args...)
    return :(get_dy($(esc.(args)...)))
end

Base.@propagate_inbounds @inline get_dy(dy::NTuple{N, Union{Number, AbstractVector}}, I::Integer) where {N} = getindex_dxi(dy[2], I)

macro dz(args...)
    return :(get_dz($(esc.(args)...)))
end

Base.@propagate_inbounds @inline get_dz(dz::NTuple{3, Union{Number, AbstractVector}}, I::Integer) = getindex_dxi(dz[3], I)

Base.@propagate_inbounds @inline getindex_dxi(dxi::AbstractVector, I::Integer) = dxi[I]
Base.@propagate_inbounds @inline getindex_dxi(dxi::Number, ::Integer) = dxi

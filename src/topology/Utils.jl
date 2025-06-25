import ImplicitGlobalGrid:
    global_grid,
    GGNumber,
    GGArray,
    update_halo!,
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

const JR_T = Union{
    StokesArrays,
    SymmetricTensor,
    ThermalArrays,
    Velocity,
    Displacement,
    Vorticity,
    Residual,
    Viscosity,
}

# convenience function to get the field halos updated
function update_halo!(x::T) where {T <: JR_T}
    nfields = fieldcount(T)
    exprs = Expr[]
    for i in 1:nfields
        push!(exprs, quote
            field = getfield(x, $i)
            if field !== nothing
                ImplicitGlobalGrid.update_halo!(field)
            end
        end)
    end
    return quote
        $(exprs...)
        return nothing
    end
end

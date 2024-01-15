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

x_g(idx::Integer, dxi::GGNumber, A::GGArray) = x_g(idx, dxi, size(A, 1))

@inline function x_g(idx::Integer, dxi::GGNumber, nxi::GGNumber)
    x0i = 0.5 * (@nx() - nxi) * dxi
    xi = (@coordx() * (@nx() - @olx()) + idx - 1) * dxi + x0i
    if @periodx()
        xi -= dxi # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (xi > (@nx_g() - 1) * dx)
            xi -= @nx_g() * dxi
        end # It must not be (nx_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (xi < 0)
            xi += @nx_g() * dxi
        end
    end
    return xi
end

y_g(idx::Integer, dxi::GGNumber, A::GGArray) = y_g(idx, dxi, size(A, 2))

@inline function y_g(idx::Integer, dxi::GGNumber, nxi::GGNumber)
    x0i = 0.5 * (@ny() - nxi) * dxi
    xi = (@coordy() * (@ny() - @oly()) + idx - 1) * dxi + x0i
    if @periody()
        xi -= dxi # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (xi > (@ny_g() - 1) * dx)
            xi -= @ny_g() * dxi
        end # It must not be (ny_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (xi < 0)
            xi += @ny_g() * dxi
        end
    end
    return xi
end

z_g(idx::Integer, dxi::GGNumber, A::GGArray) = z_g(idx, dxi, size(A, 3))

@inline function z_g(idx::Integer, dxi::GGNumber, nxi::GGNumber)
    x0i = 0.5 * (@nz() - nxi) * dxi
    xi = (@coordz() * (@nz() - @olz()) + idx - 1) * dxi + x0i
    if @periodz()
        xi -= dxi # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (xi > (@nz_g() - 1) * dx)
            xi -= @nz_g() * dxi
        end # It must not be (nz_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (xi < 0)
            xi += @nz_g() * dxi
        end
    end
    return xi
end
using ParallelStencil
using JustRelax
using Printf, LinearAlgebra, CairoMakie
using ParallelStencil.FiniteDifferences3D

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# struct Viscosity2{T, M, F}
#     val::T
#     β::M
#     f::F

#     function Viscosity2(ni::NTuple{3, T}, β::M) where {T, M}
#         val = @zeros(ni...)
#         f = _viscosity(x::T, y::T, z::T, β::T) where T<:Real = exp(1 - β*( (1-x) + y*(1-y) + y*(1-z) ))
        
#         new{typeof(val), M, Function}(val, β, f)
#     end
# end

# function viscosity!(η::Viscosity2, xi, β)

#     x = [x for x in xi[1], _ in xi[2], _ in xi[3]]
#     y = [y for _ in xi[1], y in xi[2], _ in xi[3]]
#     z = [z for _ in xi[1], _ in xi[2], z in xi[3]]
#     @parallel _viscosity!(η, x, y, z, β) 

#     return η
# end

# @parallel function _viscosity!(η::Viscosity2, x, y, z) 
#     # @all(η) = exp(1 - β*( @all(x)*(1-@all(x)) + @all(y)*(1-@all(y)) + @all(z)*(1-@all(z)) ))
#     @all(η) = η.f( @all(x), @all(y), @all(z), η.β)
#     return
# end

macro allocate(nx, ny, nz) :(PTArray(undef, $nx, $ny, $nz)) end
macro allocate(ni...) :(PTArray(undef, $(ni...))) end

_viscosity(x::T, y::T, z::T, β::T) where T<:Real = exp(1 - β*( (1-x) + y*(1-y) + y*(1-z) ))

@parallel function _viscosity!(η, x, y, z, β) 
    # @all(η) = exp(1 - β*( @all(x)*(1-@all(x)) + @all(y)*(1-@all(y)) + @all(z)*(1-@all(z)) ))
    @all(η) = _viscosity(@all(x), @all(y), @all(z), β)
    return
end

function viscosity(xi, ni, β)

    η = @allocate ni...
    x = [x for x in xi[1], _ in xi[2], _ in xi[3]]
    y = [y for _ in xi[1], y in xi[2], _ in xi[3]]
    z = [z for _ in xi[1], _ in xi[2], z in xi[3]]
    @parallel _viscosity!(η, x, y, z, β) 

    return η
end

function body_forces(xi::NTuple{3, T}, η, β) where T
    x, y, z = xi
    nx, ny, nz = length.(xi)
    fx = [body_forces_x(x[ix], y[iy], z[iz], η[ix, iy, iz], β) for ix in 1:nx, iy in 1:ny, iz in 1:nz]
    fy = [body_forces_y(x[ix], y[iy], z[iz], η[ix, iy, iz], β) for ix in 1:nx, iy in 1:ny, iz in 1:nz]
    fz = [body_forces_z(x[ix], y[iy], z[iz], η[ix, iy, iz], β) for ix in 1:nx, iy in 1:ny, iz in 1:nz]
    return fx, fy, fz
end

static(x, y, z, η, β) = (1, -η, (1-2*x)*β*η, (1-2*y)*β*η, (1-2*z)*β*η)

function body_forces_x(x, y, z, η, β)
    fx = (
        y*z + 3*x^2*y^3*z,
        2+6x*y,
        2+4x+2y+6x^2*y,
        x+y+2x*y^2+x^3,
        -3z-10x*y*z
    )

    st = static(x, y, z, η, β)

    return dot(st, fx)
       
end

function body_forces_y(x, y, z, η, β)
    fy = (
        x*z + 3*x^3*y^2*z,
        2 + 2x^2 + 2y^2,
        x + y + 2x*y^2 + x^3,
        2 + 2x + 4y + 4x^2*y,
        -3z - 5x^2*z
    )

    st = static(x, y, z, η, β)

    return dot(st, fy)
       
end

function body_forces_z(x, y, z, η, β)
    fz = (
        x*y + x^3*y^3,
        -10y*z,
        -3z - 10x*y*z,
        -3z - 5x^2*z,
        -4 - 6x - 6y - 10x^2*y
    )

    st = static(x, y, z, η, β)

    return dot(st, fz)
       
end

function Burstedde()
    nx=ny=nz=32
    lx=ly=lz=1e0
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li/ni # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    xci = Tuple([di[i]/2:di[i]:(li[i]-di[i]/2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, Viscous)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di, Re = 6π, CFL = 0.9/√3)

    ## Setup-specific parameters and fields
    β = 1.0
    η = viscosity(xci, ni, β) # add reference 

    ## Boundary conditions
    
    freeslip = (
        freeslip_x = true,
        freeslip_y = true,
        freeslip_z = true
    )

    # Physical time loop
    t = 0.0
    ρg = body_forces(xci, η, β) # => ρ*(gx, gy, gz)

    local iters
    while t < ttot
        iters = solve!(stokes, pt_stokes, di, li, max_li, freeslip, ρ, η; iterMax = 10e3)
        t += Δt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

# vtk_grid("fields", xci...) do vtk
#     vtk["viscosity"] = η
# end

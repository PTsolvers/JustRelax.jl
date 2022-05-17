using GeophysicalModelGenerator
using StaticArrays

abstract type AbstractGrid end
struct Grid{T} <: AbstractGrid
    X::T
    Y::T
    Z::T
    function Grid(a::T, b::T, c::T) where {T}
        x, y, z = XYZGrid(a, b, c)
        return new{typeof(x)}(x, y, z)
    end
end

## NOTE: 
# functions below have been modified 
# and adapted from GeophysicalModelGenerator.jl

# Internal function that rotates the coordinates
function Rot3D(x, y, z, R)
    v = @SVector [x, y, z]
    CoordRot = R * v
    x = CoordRot[1]
    y = CoordRot[2]
    z = CoordRot[3]
    return x, y, z
end

function Rot3D!(X, Y, Z, StrikeAngle, DipAngle)

    # rotation matrixes
    Ry = @SMatrix [
        cosd(-DipAngle) 0 sind(-DipAngle)
        0 1 0
        -sind(-DipAngle) 0 cosd(-DipAngle)
    ]
    Rz = @SMatrix [
        cosd(StrikeAngle) -sind(StrikeAngle) 0
        sind(StrikeAngle) cosd(StrikeAngle) 0
        0 0 1
    ]
    R = Ry * Rz

    Threads.@threads for i in eachindex(X)
        @inbounds X[i], Y[i], Z[i] = Rot3D(X[i], Y[i], Z[i], R)
    end

    return nothing
end

function Compute_Phase(Phase, Temp, Z, s::LithosphericPhases)
    Layers, Phases, Tlab = s.Layers, s.Phases, s.Tlab

    Phase .= Phases[end]
    Ztop = 0
    for i in 1:length(Layers)
        Zbot = Ztop - Layers[i]

        Threads.@threads for j in eachindex(Z)
            @inbounds if (Z[j] ≥ Zbot) && (Z[j] ≥ Ztop)
                Phase[j] = Phases[i]
            end
        end

        Ztop = Zbot
    end

    # set phase to mantle if requested
    if Tlab != nothing
        Threads.@threads for j in eachindex(Temp)
            @inbounds if Temp[j] ≥ Tlab
                Phase[j] = Phases[end]
            end
        end
    end

    return Phase
end

function myAddBox!(
    Phase,
    Temp,
    grid::Grid;         # required input
    xlim=Tuple{2},
    ylim=nothing,
    zlim=Tuple{2},     # limits of the box
    Origin=nothing,
    StrikeAngle=0,
    DipAngle=0,      # origin & dip/strike
    phase=ConstantPhase(1),                       # Sets the phase number(s) in the box
    T=nothing,
)                                     # Sets the thermal structure (various fucntions are available)

    # Limits of block                
    if ylim == nothing
        ylim = extrema(grid.Y)
    end

    if Origin == nothing
        Origin = (xlim[1], ylim[1], zlim[2])  # upper-left corner
    end

    # Perform rotation of 3D coordinates:
    Xrot = grid.X .- Origin[1]
    Yrot = grid.Y .- Origin[2]
    Zrot = grid.Z .- Origin[3]

    Rot3D!(Xrot, Yrot, Zrot, StrikeAngle, DipAngle)

    # Set phase number & thermal structure in the full domain
    ztop = zlim[2] - Origin[3]
    zbot = zlim[1] - Origin[3]
    ind = findall(
        (Xrot .>= (xlim[1] - Origin[1])) .&&
        (Xrot .<= (xlim[2] - Origin[1])) .&&
        (Yrot .>= (ylim[1] - Origin[2])) .&&
        (Yrot .<= (ylim[2] - Origin[2])) .&&
        (Zrot .>= zbot) .&&
        (Zrot .<= ztop),
    )

    # Compute thermal structure accordingly. See routines below for different options
    if T != nothing
        Temp[ind] = Compute_ThermalStructure(Temp[ind], Xrot[ind], Yrot[ind], Zrot[ind], T)
    end

    # Set the phase. Different routines are available for that - see below.
    Phase[ind] = Compute_Phase(Phase[ind], Temp[ind], Zrot[ind], phase)

    return nothing
end

function generate_phases(
    x,
    y,
    z;
    Trench_x_location=500,     # trench location
    Length_Subduct_Slab=200,     # length of subducted slab
    Length_Horiz_Slab=500,     # length of overriding plate of slab
    Width_Slab=250,     # Width of slab (in case we run a 3D model)         
    SubductionAngle=34,     # Subduction angle
    ThicknessCrust=10,
    ThicknessML=75,     # Thickness of mantle lithosphere
    T_mantle=1350,   # in Celcius
    T_surface=0,
)
    grid = Grid(x, y, z)

    ThicknessSlab = ThicknessCrust + ThicknessML

    Phases = zeros(Int64, size(grid.X)) # Rock numbers
    Temp = ones(size(grid.X)) * T_mantle # Temperature in C
    T = LinearTemp(0, T_mantle)
    T1 = SpreadingRateTemp(; Tsurface=0, Tmantle=T_mantle, maxAge=100, MORside="right")
    T2 = HalfspaceCoolingTemp(0, T_mantle, 100, 0)

    # Create horizontal part of slab with crust & mantle lithosphere
    myAddBox!(
        Phases,
        Temp,
        grid;
        xlim=(Trench_x_location, Trench_x_location + Length_Horiz_Slab),
        ylim=(0, Width_Slab),
        zlim=(-ThicknessSlab, 0.0),
        phase=LithosphericPhases(; Layers=[ThicknessSlab ThicknessML], Phases=[1 2 0]),
        T=T,
    )

    # Add inclined part of slab                            
    myAddBox!(
        Phases,
        Temp,
        grid;
        xlim=(Trench_x_location - Length_Subduct_Slab, Trench_x_location),
        ylim=(0, Width_Slab),
        zlim=(-ThicknessSlab, 0.0),
        DipAngle=-SubductionAngle,
        Origin=(Trench_x_location, 0, 0),
        phase=LithosphericPhases(; Layers=[ThicknessSlab ThicknessML], Phases=[1 2 0]),
        T=T,
    )

    Threads.@threads for i in eachindex(Phases)
        @inbounds if Phases[i] == 0
            Phases[i] = 1
        end
    end

    return Phases, Temp
end
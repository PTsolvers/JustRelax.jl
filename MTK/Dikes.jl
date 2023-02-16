struct Tracer2{I,F,T,V}
    Phase::I
    T0::F
    Phi_melt::F
    coord::T
    T::V
    time_vec::V

    function Tracer2(
        dim;
        T0::F=900.0,      # temperature
        Phi_melt::F=0.0,  # Melt fraction on Tracers
    ) where {F}
        coord = ntuple(Val(dim)) do i
            F[]
        end
        I = Int64
        return new{Vector{I},F,typeof(coord),Vector{F}}(I[], T0, Phi_melt, coord, F[], F[])
    end
end
"""

This contains a number of routines that are related to inserting new dikes to the simulation,
defining a velocity field that "opens" the host rock accordingly and to inserting the dike temperature to the
temperature field

"""

"""
    Structure that holds the geometrical parameters of the dike, which are slightly different
    depending on whether we consider a 2D or a 3D case

    General form:
        Dike(Width=.., Thickness=.., Center=[], Angle=[], DikeType="..", T=.., ΔP=.., E=.., ν=.., Q=..)

        with:

            [Width]:      width of dike  (optional, will be computed automatically if ΔP and Q are specified)

            [Thickness]:  (maximum) thickness of dike (optional, will be computed automatically if ΔP and Q are specified)
    
            Center:     center of the dike
                            2D - [x; z]
                            3D - [x; y; z]
            
            Angle:      Dip (and strike) angle of dike
                            2D - [Dip]
                            3D - [Strike; Dip]
            
            DikeType:           DikeType of dike
                            "SquareDike"    -   square dike area   
                            "SquareDike_TopAccretion"           -   square dike area, which grows by underaccreting   
                            "CylindricalDike_TopAccretion"      -   cylindrical dike area, which grows by underaccreting   
                            "CylindricalDike_TopAccretion_FullModelAdvection"      -   cylindrical dike area, which grows by underaccreting; also material to the side of the dike is moved downwards   
                            "ElasticDike"   -   penny-shaped elastic dike in elastic halfspace
                            "EllipticalIntrusion" - elliptical dike intrusion area with radius Width/2 and height Height/2 
            
            T:          Temperature of the dike [Celcius]   
            
            ν:          Poison ratio of host rocks
            
            E:          Youngs modulus of host rocks [Pa]
            
            [ΔP]:       Overpressure of dike w.r.t. host rock [Pa], (optional in case we want to compute width/length directly)

            [Q]:        Volume of magma within dike [m^3], 
            
           
    All parameters can be specified through keywords as shown above. 
    If keywords are not given, default parameters are employed.
    
 The 
    
"""
struct Dike{V,S,_T,N,P} # stores info about dike
    # Note: since we utilize the "Parameters.jl" package, we can add more keywords here w/out breaking the rest of the code 
    #
    # We can also define only a few parameters here (like Q and ΔP) and compute Width/Thickness from that
    # Or we can define thickness 
    Angle::V
    DikeType::S
    T::_T
    E::_T
    ν::_T
    ΔP::_T
    Q::_T
    W::_T
    H::_T
    Center::N
    Phase::P

    function Dike(;
        Angle::V=0.0,                           # Strike/Dip angle of dike
        DikeType::S=:SquareDike,                  # DikeType of dike
        T::_T=950.0,                              # Temperature of dike
        E::_T=1.5e10,                             # Youngs modulus (only required for elastic dikes)
        ν::_T=0.3,                                # Poison ratio of host rocks
        ΔP::_T=1e6,                               # Overpressure of elastic dike
        Q::_T=1e3,                                # Volume of elastic dike
        W::_T=(3 * E * Q / (16 * (1 - ν^2) * ΔP))^(1.0 / 3.0),  # Width of dike/sill   
        H::_T=8 * (1 - ν^2) * ΔP * W / (π * E),               # (maximum) Thickness of dike/sill            
        Center::N=(20e3, -10e3),                  # Center
        Phase::P=2,                               # Phase of newly injected magma
    ) where {V,S,_T,N,P}
        return new{V,S,_T,N,P}(Angle, Symbol(DikeType), T, E, ν, ΔP, Q, W, H, Center, Phase)
    end
end

struct DikePoly{T}    # polygon that describes the geometry of the dike (only in 2D)
    x::T # x-coordinates
    z::T # z-coordinates
end

"""
    This injects a dike in the computational domain in an instantaneous manner,
    while "pushing" the host rocks to the sides. 

    The orientation and the type of the dike are described by the structure     

    General form:
        T, Velocity, VolumeInjected = InjectDike(Tracers, T, Grid, FullGrid, dike, nTr_dike; AdvectionMethod="RK2", InterpolationMethod="Quadratic", dike_poly=[])

    with:
        T:          Temperature grid (will be modified)

        Tracers:    StructArray that contains the passive tracers (will be modified)

        Grid:       regular grid on which the temperature is defined
                    2D - (X,Z)
                    3D - (X,Y,Z)

        FullGrid:   2D or 3D matrixes with the full grid coordinates
                    2D - (X,Z)
                    3D - (X,Y,Z)

        nTr_dike:   Number of new tracers to be injected into the new dike area

    optional input parameters with keywords (add them with: AdvectionMethod="RK4", etc.):

        AdvectionMethod:    Advection algorithm 
                    "Euler"     -    1th order accurate Euler timestepping
                    "RK2"       -    2nd order Runga Kutta advection method [default]
                    "RK4"       -    4th order Runga Kutta advection method
                
        InterpolationMethod: Interpolation Algorithm to interpolate data on advected points 
                    
                    Note:  higher order is more accurate for smooth fields, but if there are very sharp gradients, 
                        it may result in a 'Gibbs' effect that has over and undershoots.   

                    "Linear"    -    Linear interpolation
                    "Quadratic" -    Quadratic spline
                    "Cubic"     -    Cubic spline

        dike_poly: polygon that describes the circumferrence of a diking area (in 2D)
                    Will be advected if specified

"""
function InjectDike(
    Tracers,
    parts_semilagrange::SemiLagrangianParticles,
    T::Array,
    Grid::NTuple{dim,_T},
    dike::Dike,
    nTr_dike::Int64;
    α=0.5,
    InterpolationMethod="Linear",
    dike_poly=[],
) where {dim,_T}

    # Some notes on the algorithm:
    #   For computational reasons, we do not open the dike at once, but in sufficiently small pseudo timesteps
    #   Sufficiently small implies that the motion per "pseudotimestep" cannot be more than 0.5*{dx|dy|dz}

    (; H) = dike

    if dim == 2
        X = [x for x in Grid[1], x in Grid[2]]
        Z = [z for x in Grid[1], z in Grid[2]]
        GridFull = (X, Z)
    elseif dim == 3
        X = [x for x in Grid[1], y in Grid[2], z in Grid[3]]
        Y = [y for x in Grid[1], y in Grid[2], z in Grid[3]]
        Z = [z for x in Grid[1], y in Grid[2], z in Grid[3]]
        GridFull = (X, Y, Z)
    end

    Spacing = SVector{dim}(Grid[i][2] - Grid[i][1] for i in 1:dim)
    d = minimum(Spacing) * 0.5 # maximum distance the dike can open per pseudotimestep 
    nsteps = max(ceil(H / d), 2) # the number of steps (>=10)

    # Compute velocity required to create space for dike
    dt = 1.0 / nsteps
    Δ = H
    Velocity = HostRockVelocityFromDike(GridFull, Δ, 1.0, dike) # compute velocity field/displacement used for a full dt. 

    # Move hostrock & already existing tracers to the side to create space for new dike
    # Tnew = zeros(size(T))
    for _ in 1:nsteps
        semilagrangian_advection_RK2!(T, parts_semilagrange, Velocity, Grid, dt)
        if !isempty(Tracers.coord[1])
            advection_RK2_vertex!(Tracers.coord, Velocity, Grid, dt; α=α)
        end
    end

    # Insert dike in T profile and add new tracers
    Tnew, Tracers = AddDike(T, Tracers, Grid, dike, nTr_dike) # Add dike to T-field & insert tracers within dike

    # Compute volume of newly injected magma
    _, InjectedVolume = volume_dike(dike)

    # Advect dike polygon
    if ~isempty(dike_poly)
        advect_dike_polygon!(dike_poly, Grid, Velocity)
    end

    return Tracers, Tnew, InjectedVolume, dike_poly, Velocity
end

#--------------------------------------------------------------------------
"""
    Host rock velocity obtained during opening of dike

    General form:
        Velocity = HostRockVelocityFromDike( Grid, Points, Δ, dt, dike);

        with:

            Grid: coordinates of regular grid @ which we compute velocity
                    2D - [X; Z]
                    3D - [x; y; z]

            dike: structure that holds info about dike
            dt:   time in which the full dike is opened
            
    Note: the velocity is computed in such a manner that a maximum opening 
        increment of Δ = Vmax*dt is obtained after this timestep


"""

function HostRockVelocityFromDike(Points::NTuple{2,T}, Δ, dt, dike::Dike) where {T}
    # Prescibe the velocity field to inject the dike with given orientation
    # 
    (; H, W) = dike

    # if dim == 2
    #X,Z          =  Points[1], Points[2];

    # Rotate and shift coordinate system into 'dike' reference frame
    (; Angle, DikeType) = dike
    α = Angle[1]
    RotMat = SMatrix{2,2}([cosd(α) -sind(α); sind(α) cosd(α)])    # 2D rotation matrix
    # Xrot,Zrot   =   zeros(size(Points[1])), zeros(size(Points[2]));
    Points[1] .-= dike.Center[1]
    Points[2] .-= dike.Center[2]
    RotatePoints_2D!(Points[1], Points[2], Points[1], Points[2], RotMat)
    Vx_rot, Vz_rot = zeros(size(Points[1])), zeros(size(Points[1]))
    Vx, Vz = zeros(size(Points[1])), zeros(size(Points[1]))

    Vint::Float64 = Δ / dt # open the dike by a maximum amount of Δ in one dt (no 1/2 as that is taken care off inside the routine below)

    if DikeType === :SquareDike
        Vint *= 0.5                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
        Vz_rot[(Points[2] .≤ 0) .& (abs.(Points[1]) .≤ W * 0.5)] .= -Vint
        Vz_rot[(Points[2] .> 0) .& (abs.(Points[1]) .< W * 0.5)] .= Vint
        Vx_rot[abs.(Points[1]) .< W] .= 0.0      # set radial velocity to zero at left boundary

    elseif DikeType === :SquareDike_TopAccretion
        Vz_rot[(Points[2] .≤ 0) .& (abs.(Points[1]) .≤ W * 0.5)] .= -Vint
        #Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;
        Vx_rot[abs.(Points[1]) .≤ W] .= 0.0      # set radial velocity 

    elseif DikeType === :CylindricalDike_TopAccretion
        Vz_rot[(Points[2] .≤ 0) .& (Points[1] .≤ (W * 0.5))] .= -Vint
        #Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;
        Vx_rot[abs.(Points[1]) .≤ W] .= 0.0      # set radial velocity 

    elseif DikeType === :CylindricalDike_TopAccretion_FullModelAdvection
        Vz_rot[(Points[2] .≤ 0)] .= -Vint
        #Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;
        Vx_rot[abs.(Points[1]) .≤ W] .= 0.0      # set radial velocity to zero at left boundary

    elseif DikeType === :ElasticDike
        Threads.@threads for i in eachindex(Vz_rot)
            # use elastic dike solution to compute displacement4
            Displacement, Bmax = DisplacementAroundPennyShapedDike(
                dike, SVector(Points[1][i], Points[2][i]), 2
            )
            Displacement = Displacement ./ Bmax     # normalize such that 1 is the maximum
            Vz_rot[i] = Vint * Displacement[2]
            Vx_rot[i] = Vint * Displacement[1]
        end

    elseif DikeType === :EllipticalIntrusion
        AR = H / W      # aspect ratio of ellipse
        H = Δ           # we don't open the dike @ once but piece by piece
        W = H / AR
        a_inject = W * 0.5                  # half axis
        Vol_inject = 4 / 3 * pi * a_inject^3 * AR    # b axis = a_inject*AR
        for I in eachindex(Points[1])
            x = Points[1][I]
            z = Points[2][I]
            a = sqrt(z^2 / AR^2 + x^2)
            a3 = a^3
            da = ((Vol_inject + 4 / 3 * pi * a3 * AR) / (4 / 3 * pi * AR))^(1 / 3) - a # incremental displcement
            # note the the coordinates are already rotated & centered around the sill intrusion region
            Vx_rot[I] = x * (da ./ a)
            Vz_rot[I] = z * (da ./ a)
            if x == 0 && z == 0.0
                Vx_rot[I] = 0.0
                Vz_rot[I] = 0.0
            end
        end
    else
        error("Unknown Dike DikeType: $DikeType")
    end

    # "unrotate" vector fields and points using the transpose of RotMat
    RotatePoints_2D!(Vx, Vz, Vx_rot, Vz_rot, RotMat')
    RotatePoints_2D!(Points[1], Points[2], Points[1], Points[2], RotMat')
    Points[1] .+= dike.Center[1]
    Points[2] .+= dike.Center[2]

    return (Vx, Vz)
end

# else
#     @unpack Angle, DikeType = dike
#     α, β = Angle[1], Angle[end]
#     RotMat_y = SMatrix{3,3}([cosd(α) 0.0 -sind(α); 0.0 1.0 0.0; sind(α) 0.0 cosd(α)])                      # perpendicular to y axis
#     RotMat_z = SMatrix{3,3}([cosd(β) -sind(β) 0.0; sind(β) cosd(β) 0.0; 0.0 0.0 1.0])                      # perpendicular to z axis
#     RotMat = RotMat_y * RotMat_z

#     # Xrot,Yrot,Zrot  =   zeros(size(Points[1])),  zeros(size(Points[2])), zeros(size(Points[3]));
#     Points[1] .= Points[1] .- dike.Center[1]
#     Points[2] .= Points[2] .- dike.Center[2]
#     Points[3] .= Points[3] .- dike.Center[3]
#     RotatePoints_3D!(
#         Points[1], Points[2], Points[3], Points[1], Points[2], Points[3], RotMat
#     )                 # rotate coordinates 

#     Vx_rot, Vy_rot, Vz_rot = zeros(size(Points[1])),
#     zeros(size(Points[2])),
#     zeros(size(Points[3]))
#     Vx, Vy, Vz = zeros(size(Points[1])), zeros(size(Points[2])), zeros(size(Points[3]))

#     if DikeType == "SquareDike"
#         @unpack H, W = dike                          # Dimensions of square dike
#         Vint = Δ / dt * 0.5                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)

#         Vz_rot[(Points[3] .< 0) .& (abs.(Points[1]) .< W * 0.5) .& (abs.(Points[2]) .< W * 0.5)] .=
#             -Vint
#         Vz_rot[(Points[3] .> 0) .& (abs.(Points[1]) .< W * 0.5) .& (abs.(Points[2]) .< W * 0.5)] .=
#             Vint

#         Vx_rot[abs.(Points[1]) .< W] .= 0.0      # set radial velocity to zero at left boundary
#         Vy_rot[abs.(Points[2]) .< W] .= 0.0      # set radial velocity to zero at left boundary

#     elseif (DikeType == "SquareDike_TopAccretion")
#         @unpack H, W = dike                          # Dimensions of square dike
#         Vint = Δ / dt / 1.0                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)

#         Vz_rot[(Points[3] .< 0) .& (abs.(Points[1]) .< W * 0.5) .& (abs.(Points[2]) .< W * 0.5)] .=
#             -Vint
#         # Vz_rot[(Points[3].>0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .=  Vint;

#         Vx_rot[abs.(Points[1]) .< W] .= 0.0      # set radial velocity to zero at left boundary
#         Vy_rot[abs.(Points[2]) .< W] .= 0.0      # set radial velocity to zero at left boundary
#     elseif (DikeType == "CylindricalDike_TopAccretion") ||
#         (DikeType == "CylindricalDike_TopAccretion_FullModelAdvection")
#         @unpack H, W = dike                          # Dimensions of square dike
#         Vint = Δ / dt / 1.0                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)

#         Vz_rot[(Points[3] .≤ 0.0) .& ((Points[1] .^ 2 + Points[2] .^ 2) .≤ (W * 0.5) .^ 2)] .=
#             -Vint
#         # Vz_rot[(Points[3].>0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .=  Vint;

#         Vx_rot[abs.(Points[1]) .≤ W] .= 0.0      # set radial velocity to zero at left boundary
#         Vy_rot[abs.(Points[2]) .≤ W] .= 0.0      # set radial velocity to zero at left boundary

#     elseif DikeType == "ElasticDike"
#         @unpack H, W = dike                          # Dimensions of dike
#         Vint = Δ / dt                            # open the dike by a maximum amount of Δ in one dt (no 1/2 as that is taken care off inside the routine below)

#         Threads.@threads for i in eachindex(Vx_rot)

#             # use elastic dike solution to compute displacement
#             Displacement, Bmax = DisplacementAroundPennyShapedDike(
#                 dike, SVector(Points[1][i], Points[2][i], Points[3][i]), dim
#             )

#             Displacement .= Displacement / Bmax     # normalize such that 1 is the maximum

#             Vz_rot[i] = Vint .* Displacement[3]
#             Vy_rot[i] = Vint .* Displacement[2]
#             Vx_rot[i] = Vint .* Displacement[1]
#         end

#     else
#         error("Unknown Dike DikeType: $DikeType")
#     end

#     # "unrotate" vector fields
#     RotatePoints_3D!(Vx, Vy, Vz, Vx_rot, Vy_rot, Vz_rot, RotMat')           # rotate velocities back
#     RotatePoints_3D!(
#         Points[1], Points[2], Points[3], Points[1], Points[2], Points[3], RotMat'
#     )           # rotate coordinates back

#     Points[1] .= Points[1] .+ dike.Center[1]
#     Points[2] .= Points[2] .+ dike.Center[2]
#     Points[3] .= Points[3] .+ dike.Center[3]

#     return (Vx, Vy, Vz)
# end
# end

function RotatePoints_2D!(Xrot, Zrot, X, Z, RotMat)
    @simd for i in eachindex(X) # linear indexing
        pt_rot = RotMat * SVector(X[i], Z[i])
        Xrot[i] = pt_rot[1]
        Zrot[i] = pt_rot[2]
    end
end

function RotatePoints_3D!(Xrot, Yrot, Zrot, X, Y, Z, RotMat)
    @simd for i in eachindex(X) # linear indexing
        pt_rot = RotMat * SVector(X[i], Y[i], Z[i])

        Xrot[i] = pt_rot[1]
        Yrot[i] = pt_rot[2]
        Zrot[i] = pt_rot[3]
    end
end

"""
    dike_poly = CreateDikePolygon(dike::Dike, numpoints=101)

    Creates a new dike polygon with given orientation and width. 
    This polygon is used for plotting, and described by the struct dike

"""
function CreateDikePolygon(dike::Dike, nump=101)
    @unpack DikeType = dike

    if DikeType == "CylindricalDike_TopAccretion"
        dx = dike.W * 0.5 / nump * 0.5
        xx = Vector(dike.Center[1]:dx:(dike.W * 0.5))
        x = [xx; xx[end:-1:1]]
        z = [ones(size(xx)) * dike.H; -ones(size(xx)) * dike.H * 0.5] .+ dike.Center[2]
        poly = [x, z]

    elseif DikeType == "EllipticalIntrusion"
        dp = 2 * pi / nump
        p = 0.0:0.01:(2 * pi)
        a_ellipse = dike.W * 0.5
        b_ellipse = dike.H * 0.5
        x = cos.(p) * a_ellipse
        z = -sin.(p) * b_ellipse .+ dike.Center[2]
        poly = [x, z]

    else
        println(
            "WARNING: Polygon not yet implemented for dike type $DikeType; leaving it empty"
        )
        poly = []
    end

    return poly
end

"""
    dike_poly = CreatDikePolygon(dike::Dike, numpoints=101)

    Creates a new dike polygon with given orientation and width. 
    This polygon is used for plotting, and described by the struct dike

"""
function CreatDikePolygon(dike::Dike, nump=101)
    @unpack DikeType = dike

    if DikeType === :CylindricalDike_TopAccretion
        dx = dike.W * 0.5 / nump * 0.5
        xx = Vector(dike.Center[1]:dx:(dike.W * 0.5))
        x = [xx; xx[end:-1:1]]
        z = [ones(size(xx)) * dike.H; -ones(size(xx)) * dike.H * 0.5] .+ Dikes.Center[2]
        poly = [x, z]

    elseif DikeType === :EllipticalIntrusion
        dp = 2 * pi / nump
        p = 0.0:0.01:(2 * pi)
        a_ellipse = dike.W * 0.5
        b_ellipse = dike.H * 0.5
        x = cos.(p) * a_ellipse
        z = -sin.(p) * b_ellipse .+ dike.Center[2]
        poly = [x, z]

    else
        println(
            "WARNING: Polygon not yet implemented for dike type $DikeType; leaving it empty"
        )
        poly = []
    end

    return poly
end

"""
    advect_dike_polygon!(poly, Grid, Velocity, dt=1.0)

Advects a dike polygon
"""
function advect_dike_polygon!(poly, Grid, Velocity, dt=1.0)
    poly_vel = AdvectPoints((poly[1], poly[2]), Grid, Velocity, dt)

    Threads.@threads for i in eachindex(poly)
        @inbounds poly[i] .*= poly_vel[i] * dt
    end

    return nothing
end

"""
    in = isinside_dike(pt, dike::Dike)

    Computes if a point [pt] is inside a dike area or not depending on the type of dike

"""
function isinside_dike(pt::SVector{2,T}, dike::Dike) where {T}
    # important: this is a "unit" dike, which has the center at [0,0,0] and width given by dike.Size
    isin = false
    (; DikeType, W, H) = dike
    if DikeType === :SquareDike
        if (abs(pt[1]) < W * 0.5) && (abs(pt[2]) < H * 0.5)
            isin = true
        end

    elseif (DikeType === :SquareDike_TopAccretion) ||
        (DikeType === :CylindricalDike_TopAccretion) ||
        (DikeType === :CylindricalDike_TopAccretion_FullModelAdvection)
        if (abs(pt[1]) ≤ W * 0.5) && (abs(pt[2]) ≤ H * 0.5)
            isin = true
        end

    elseif (DikeType === :ElasticDike) || (DikeType === :EllipticalIntrusion)
        eq_ellipse = (pt[1]^2) / ((W * 0.5)^2) + (pt[2]^2) / ((H * 0.5)^2) # ellipse
        if eq_ellipse ≤ 1.0
            isin = true
        end

    else
        error("Unknown dike type $DikeType")
    end

    return isin
end

function isinside_dike(pt::SVector{3,T}, dike::Dike) where {T}
    # important: this is a "unit" dike, which has the center at [0,0,0] and width given by dike.Size
    isin = false
    (; DikeType, W, H) = dike
    if DikeType === :SquareDike
        if (abs(pt[1]) < W * 0.5) && (abs(pt[3]) < H * 0.5) && (abs(pt[2]) < W * 0.5)
            isin = true
        end

    elseif (DikeType === :SquareDike_TopAccretion) ||
        (DikeType === :CylindricalDike_TopAccretion) ||
        (DikeType === :CylindricalDike_TopAccretion_FullModelAdvection)
        if (abs(pt[1]) < W * 0.5) && (abs(pt[3]) < H * 0.5) && (abs(pt[2]) < W * 0.5)
            isin = true
        end

    elseif (DikeType === :ElasticDike) || (DikeType === :EllipticalIntrusion)
        eq_ellipse = (pt[1]^2 + pt[2]^2) / ((W * 0.5)^2) + (pt[3]^2) / ((H * 0.5)^2) # ellipsoid
        if eq_ellipse ≤ 1.0
            isin = true
        end

    else
        error("Unknown dike type $DikeType")
    end

    return isin
end

"""
    A,V = volume_dike(dike::Dike)

    Returns the area A and volume V of the injected dike
    
    In 2D, the volume is compute by assuming a penny-shaped dike, with length=width
    In 3D, the cross-sectional area in x-z direction is returned 

"""
function volume_dike(dike::Dike)
    # important: this is a "unit" dike, which has the center at [0,0,0] and width given by dike.Size
    (; W, H, DikeType) = dike

    if DikeType === :SquareDike
        area = W * H                  #  (in 2D, in m^2)
        volume = W * W * H                #  (equivalent 3D volume, in m^3)
    elseif DikeType === :SquareDike_TopAccretion
        area = W * H                  #  (in 2D, in m^2)
        volume = W * W * H                #  (equivalent 3D volume, in m^3)
    elseif (DikeType === :CylindricalDike_TopAccretion) ||
        (DikeType === :CylindricalDike_TopAccretion_FullModelAdvection)
        area = W * H                  #  (in 2D, in m^2)
        volume = pi * (W * 0.5)^2 * H       #  (equivalent 3D volume, in m^3)

    elseif (DikeType === :ElasticDike) || (DikeType === :EllipticalIntrusion)
        area = π * W * 0.5 * H * 0.5                    #   (in 2D, in m^2)  - note that W,H are the diameters
        volume = 4 / 3 * pi * (W * 0.5) * (W * 0.5) * (H * 0.5)      #   (equivalent 3D volume, in m^3)
    else
        error("Unknown dike type $DikeType")
    end

    return area, volume
end

#--------------------------------------------------------------------------
"""
    T, Tracers, dike_poly = AddDike(T,Tracers, Grid, dike,nTr_dike)
    
Adds a dike, described by the dike polygon dike_poly, to the temperature field T (defined at points Grid).
Also adds nTr_dike new tracers randomly distributed within the dike, to the 
tracers array Tracers.

"""
function AddDike(Tfield, Tr, Grid::NTuple{2,_T}, dike, nTr_dike) where {_T}
    # dim = length(Grid)
    (; Angle, Center, W, H, T, Phase) = dike
    SCenter = SVector(Center...)
    PhaseDike = Phase
    α = Angle[1]

    RotMat = @SMatrix [
        cosd(α) -sind(α)
        sind(α) cosd(α)
    ]

    # Add dike to temperature field
    x, z = Grid[1], Grid[2]
    @inbounds for j in eachindex(z), i in eachindex(x)
        pt = SVector(x[i], z[j]) .- SCenter
        pt_rot = RotMat * pt # rotate
        isinside_dike(pt_rot, dike) && (Tfield[i, j] = T)
    end

    # Add new tracers to the dike area
    for _ in 1:nTr_dike

        # 1) Randomly initialize tracers to the approximate dike area
        pt = SVector{2}(rand(2)) .- 0.5
        Size = W, H

        pt = pt .* Size
        pt_rot = (RotMat') * pt .+ SCenter # rotate backwards (hence the transpose!) and shift

        # 2) Add them to the tracers structure
        if isinside_dike(pt, dike) # we are inside the dike
            pt_new = (pt_rot[1], pt_rot[2])

            # if !isassigned(Tr, 1)
            #     number = 1
            # else
            #     number = Tr.num[end] + 1
            # end
            push!(Tr.coord[1], pt_rot[1])
            push!(Tr.coord[2], pt_rot[2])
            push!(Tr.T, T)
            push!(Tr.Phase, PhaseDike)

            # new_tracer = Tracer(; num=number, coord=pt_new, T=T, Phase=PhaseDike) # Create new tracer

            # if !isassigned(Tr, 1)
            #     StructArrays.foreachfield(v -> deleteat!(v, 1), Tr) # Delete first (undefined) row of tracer StructArray. Assumes that Tr is defined as Tr = StructArray{Tracer}(undef, 1)

            #     Tr = StructArray([new_tracer]) # Create tracer array
            # else
            #     push!(Tr, new_tracer) # Add new point to existing array
            # end
        end
    end

    return Tfield, Tr
end

"""
    This computes the displacement around a fluid-filled penny-shaped sill that is 
    inserted inside in an infinite elastic halfspace.

    Displacement, Bmax, p = DisplacementAroundPennyShapedDike(dike, CartesianPoint)

    with:

            dike:           Dike structure, containing info about the dike

            CartesianPoint: Coordinate of the point @ which we want to compute displacments
                        2D - [dx;dz]
                        3D - [dx;dy;dz]
            
            Displacement:   Displacements of that point  
                        2D - [Ux;Uz]
                        3D - [Ux;Uy;Uz]

            Bmax:           Max. opening of the dike 
            p:              Overpressure of dike
    
    Reference: 
        Sun, R.J., 1969. Theoretical size of hydraulically induced horizontal fractures and 
        corresponding surface uplift in an idealized medium. J. Geophys. Res. 74, 5995–6011. 
        https://doi.org/10.1029/JB074i025p05995

        Notes:    
            - We employ equations 7a and 7b from that paper, which assume that the dike is in a 
                horizontal (sill-like) position; rotations have to be performed outside this routine
            
            - The center of the dike should be at [0,0,0]
            
            - This does not account for the presence of a free surface. 
            
            - The values are in absolute displacements; this may have to be normalized

"""
function DisplacementAroundPennyShapedDike(dike::Dike, CartesianPoint::SVector, dim)

    # extract required info from dike structure
    (; ν, E, W, H) = dike

    # Compute r & z; note that the Sun solution is defined for z>=0 (vertical)
    if dim == 2
        r = sqrt(CartesianPoint[1]^2)
        z = abs(CartesianPoint[2])
    elseif dim == 3
        r = sqrt(CartesianPoint[1]^2 + CartesianPoint[2]^2)
        z = abs(CartesianPoint[3])
    end

    if r == 0
        r = 1e-3
    end

    B = H # maximum thickness of dike
    a = W * 0.5 # radius

    # note, we can either specify B and a, and compute pressure p and injected volume Q
    # Alternatively, it is also possible to:
    #       - specify p and a, and compute B and Q 
    #       - specify volume Q & p and compute radius and B 
    #
    # What is best to do is to be decided later (and doesn't change the code below) 
    Q = B * (2π * a^2) / 3.0               # volume of dike (follows from eq. 9 and 10a)
    p = 3E * Q / (16.0 * (1.0 - ν^2) * a^3)    # overpressure of dike (from eq. 10a) = 3E*pi*B/(8*(1-ν^2)*a)

    # Compute displacement, using complex functions
    R1 = sqrt(r^2 + (z - im * a)^2)
    R2 = sqrt(r^2 + (z + im * a)^2)

    # equation 7a:
    U =
        im * p * (1 + ν) * (1 - 2ν) / (2pi * E) * (
            r * log((R2 + z + im * a) / (R1 + z - im * a)) -
            r *
            0.5 *
            (
                (im * a - 3z - R2) / (R2 + z + im * a) +
                (R1 + 3z + im * a) / (R1 + z - im * a)
            ) -
            (2z^2 * r) / (1 - 2ν) *
            (1 / (R2 * (R2 + z + im * a)) - 1 / (R1 * (R1 + z - im * a))) +
            (2 * z * r) / (1 - 2ν) * (1 / R2 - 1 / R1)
        )
    # equation 7b:
    Ω =
        2 * im * p * (1 - ν^2) / (pi * E) * (
            z * log((R2 + z + im * a) / (R1 + z - im * a)) - (R2 - R1) -
            1 / (2 * (1 - ν)) * (
                z * log((R2 + z + im * a) / (R1 + z - im * a)) -
                im * a * z * (1 / R2 + 1 / R1)
            )
        )

    # Displacements are the real parts of U and W. 
    #  Note that this is the total required elastic displacement (in m) to open the dike.
    #  If we only want to open the dike partially, we will need to normalize these values accordingly (done externally)  
    Uz = real(Ω)  # vertical displacement should be corrected for z<0
    Ur = real(U)
    if (CartesianPoint[end] < 0)
        Uz = -Uz
    end
    if (CartesianPoint[1] < 0)
        Ur = -Ur
    end

    if dim == 2
        Displacement = (Ur, Uz)

        return Displacement, B, p

    elseif dim == 3
        # Ur denotes the radial displacement; in 3D we have to decompose this in x and y components
        x = abs(CartesianPoint[1])
        y = abs(CartesianPoint[2])
        Ux = x / r * Ur
        Uy = y / r * Ur

        Displacement = (Ux, Uy, Uz)

        return Displacement, B, p
    end

    # return Displacement, B, p
end

"""

Creates a polygon that describes the initial diking area
"""
function CreateDikePolygon(Dike, nump=100) end

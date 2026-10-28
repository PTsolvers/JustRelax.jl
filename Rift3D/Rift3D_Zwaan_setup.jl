# Initial geometry, temperature and plastic strain of the reference models of
# Zwaan et al. (2026, JGR Solid Earth, doi:10.1029/2025JB033255), Section 2.1 and
# Appendix A2. Lengths are in km and temperatures in °C unless stated otherwise;
# `x = 0` is the center of the 450 km wide domain and the vertical coordinate, the last
# axis (`y` in 2D, `z` in 3D), is zero at the initial surface. The structure varies
# along `x` only, so a 3D model is the 2D section extruded along `y`.

using Random: MersenneTwister

const ZWAAN_GEOMETRY = (
    lx = 450.0,          # domain width
    depth = 200.0,       # depth of the bottom boundary
    h_uc = 20.0,         # upper-crust thickness at the model sides
    Δh_uc = 5.0,         # extra upper-crust thickness at the center (25 km in total)
    σ_uc = 25.0,         # standard deviation of the Gaussian crustal thickening
    h_lc = 15.0,         # lower-crust thickness
    lab = 115.0,         # depth of the lithosphere-asthenosphere boundary
    T_surface = 0.0,
    T_lab = 1284.0,
    T_bottom = 1361.0,
    noise_depth = 50.0,  # initial plastic strain noise is confined to this depth
    noise_max = 0.5,     # maximum initial plastic strain
)

# depth of the upper/lower crust boundary at `x`; the thickening is a Gaussian centered
# on the domain so that it spans 150 km ≤ x ≤ 300 km of the paper's 0-450 km domain
upper_crust_base(x, g = ZWAAN_GEOMETRY) = g.h_uc + g.Δh_uc * exp(-x^2 / (2 * g.σ_uc^2))

"""
    zwaan_phase(x, depth)

Phase index at horizontal position `x` and `depth` below the initial surface [km]:
1 upper crust, 2 lower crust, 3 lithospheric mantle, 4 sub-lithospheric mantle, 5 air.
"""
function zwaan_phase(x, depth, g = ZWAAN_GEOMETRY)
    depth < 0 && return 5.0
    huc = upper_crust_base(x, g)
    depth ≤ huc && return 1.0
    depth ≤ huc + g.h_lc && return 2.0
    depth ≤ g.lab && return 3.0
    return 4.0
end

"""
    zwaan_geotherm(x, depth)

Steady-state 1D conductive geotherm in the lithosphere (layered conductivity and heat
production of Table A1, `T_surface` at the surface and `T_lab` at the LAB), and a linear
adiabat from `T_lab` to `T_bottom` below it. Temperature in °C, `depth` in km.
"""
function zwaan_geotherm(x, depth, g = ZWAAN_GEOMETRY)
    depth ≤ 0 && return g.T_surface
    if depth ≥ g.lab
        return g.T_lab + (g.T_bottom - g.T_lab) * (depth - g.lab) / (g.depth - g.lab)
    end
    huc = upper_crust_base(x, g)
    # layer bases [m], conductivities [W/m/K], heat production [W/m³]
    bases = (huc, huc + g.h_lc, g.lab) .* 1.0e3
    k = (ZWAAN_k.upper_crust, ZWAAN_k.lower_crust, ZWAAN_k.lith_mantle)
    H = (ZWAAN_H.upper_crust, ZWAAN_H.lower_crust, ZWAAN_H.lith_mantle)

    # T at the LAB is affine in the surface heat flow q0: T_lab = T(q0 = 0) + q0 * Σ h/k
    T0, Σhk, top = g.T_surface, 0.0, 0.0
    q = 0.0
    for i in 1:3
        h = bases[i] - top
        T0 += q * h / k[i] - H[i] * h^2 / (2 * k[i])
        q -= H[i] * h
        Σhk += h / k[i]
        top = bases[i]
    end
    q0 = (g.T_lab - T0) / Σhk

    # integrate down to `depth` with the surface heat flow that hits T_lab at the LAB
    z = depth * 1.0e3
    T, q, top = g.T_surface, q0, 0.0
    for i in 1:3
        h = min(bases[i], z) - top
        T += q * h / k[i] - H[i] * h^2 / (2 * k[i])
        z ≤ bases[i] && return T
        q -= H[i] * h
        top = bases[i]
    end
    return T
end

function init_phases!(phases, particles, CharDim)
    ni = size(particles.index) .- 2
    to_km = ustrip(dimensionalize(1.0, km, CharDim))
    return @parallel (@idx ni) _init_phases!(phases, particles.coords, particles.index, to_km)
end

@parallel_indices (I...) function _init_phases!(phases, pcoords, index, to_km)
    # particle cell arrays carry one ghost cell per side: grid cell I is particle cell I .+ 1
    Ip = I .+ 1
    for ip in cellaxes(phases)
        @index(index[ip, Ip...]) == 0 && continue
        x = @index(pcoords[1][ip, Ip...]) * to_km
        depth = -@index(pcoords[end][ip, Ip...]) * to_km
        @index phases[ip, Ip...] = zwaan_phase(x, depth)
    end
    return nothing
end

"""
    init_plastic_strain!(pEII, particles, CharDim; seed = 1234)

Random initial plastic strain, uniform in `[0, noise_max]`, on the particles in the
uppermost `noise_depth` of the model; zero elsewhere.
"""
function init_plastic_strain!(pEII, particles, CharDim; seed = 1234, g = ZWAAN_GEOMETRY)
    to_km = ustrip(dimensionalize(1.0, km, CharDim))
    depth = -Array(particles.coords[end].data) .* to_km
    noise = g.noise_max .* rand(MersenneTwister(seed), size(depth)...)
    copyto!(pEII.data, ifelse.(0 .≤ depth .≤ g.noise_depth, noise, 0.0))
    return nothing
end

"""
    init_temperature!(T, xci, CharDim)

Fill the cell-centered temperature `T` (with one ghost cell per side) with the geotherm
of [`zwaan_geotherm`](@ref), nondimensionalized with `CharDim`.
"""
function init_temperature!(T, xci, CharDim)
    to_km = ustrip(dimensionalize(1.0, km, CharDim))
    xc = Array(xci[1]) .* to_km
    zc = Array(xci[end]) .* to_km
    T_host = zeros(size(T))
    for I in CartesianIndices(length.(xci))
        T_C = zwaan_geotherm(xc[I[1]], -zc[I[end]])
        T_host[(Tuple(I) .+ 1)...] = nondimensionalize((T_C + 273.0) * K, CharDim)
    end
    copyto!(T, T_host)
    return nothing
end

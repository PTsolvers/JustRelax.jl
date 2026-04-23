# mantle_profile.jl
#
# Build a 1D Earth-like mantle temperature profile:
# - conductive top boundary layer
# - adiabatic interior
# - conductive bottom boundary layer
#
# Output:
#   z_m                  depth from surface [m]
#   T_K                  temperature [K]
#
# Convention:
#   z = 0 at surface
#   z = H at bottom of model domain

using Printf

function thermal_profile(
        z;
        H           = 2.89e6,          # mantle thickness [m] ~ 2890 km
        T_surface   = 273.0,           # surface temperature [K]
        T_potential = 1600.0,          # mantle potential temperature [K]
        T_cmb       = 4000.0,          # temperature at base of mantle / near CMB [K]
        delta_top   = 120e3,           # top thermal boundary layer thickness [m]
        delta_bot   = 200e3,           # bottom thermal boundary layer thickness [m]
        # Adiabatic gradient: dT/dz = alpha * g * T / cp
        alpha       = 3.5e-5,          # thermal expansivity [1/K]
        g           = 9.81,            # gravity [m/s^2]
        cp          = 1250.0,          # heat capacity [J/(kg K)]
    )
    # If you want a simpler fixed adiabat instead of computing from alpha*g*T/cp,
    # you can override with something like:
    # const dTdz_ad = 0.3 / 1e3         # 0.3 K/km in K/m


    # Use a constant adiabatic gradient based on potential temperature.
    # This gives a realistic order-of-magnitude value (~0.3-0.4 K/km).
    dTdz_ad = alpha * g * T_potential / cp   # [K/m]

    # Match the interior adiabat to the base of the top boundary layer
    T_top_bl_base = T_potential + dTdz_ad * delta_top

    # Temperature at top of bottom boundary layer, following the adiabat downward
    z_bot_bl_top = H - delta_bot
    T_bot_bl_top = T_top_bl_base + dTdz_ad * (z_bot_bl_top - delta_top)

    # -----------------------------
    # Piecewise temperature profile
    # -----------------------------
    function temperature_profile(z)
        if z <= delta_top
            # Top conductive boundary layer: linear from surface T to adiabat
            return T_surface + (T_top_bl_base - T_surface) * (z / delta_top)

        elseif z < H - delta_bot
            # Adiabatic interior
            return T_top_bl_base + dTdz_ad * (z - delta_top)

        else
            # Bottom conductive boundary layer: linear from adiabat to CMB temperature
            return T_bot_bl_top + (T_cmb - T_bot_bl_top) * ((z - (H - delta_bot)) / delta_bot)
        end
    end

    T = temperature_profile(z) + rand()*0.01
    return T
end

# lines(T, z)
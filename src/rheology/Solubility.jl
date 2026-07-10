import GeoParams: compute_dissolved!
export compute_dissolved!

"""
    compute_dissolved!(mH2O, mCO2, phase_ratios::JustPIC.PhaseRatios, rheology, args)

Fill the dissolved H2O and CO2 mass-fraction arrays from the GeoParams
solubility closures (`Liu2005_Solubility`, `Mafic_Solubility`). Mirrors
[`compute_melt_fraction!`](@ref) but writes two arrays, because
`compute_dissolved` returns the `(m_h2o, m_co2)` pair. `args` supplies `P`,
`T`, and the CO2 mole fraction of the gas `X_co2` (scalars or index-matched
arrays).
"""
function compute_dissolved!(mH2O, mCO2, phase_ratios::JustPIC.PhaseRatios, rheology, args)
    ni = size(mH2O)
    @parallel (@idx ni) compute_dissolved_kernel!(
        mH2O, mCO2, phase_ratios.center, rheology, args
    )
    return nothing
end

@parallel_indices (I...) function compute_dissolved_kernel!(
        mH2O, mCO2, phase_ratios, rheology, args
    )
    args_ijk = getindex_NamedTuple(args, I...)
    mh, mc = compute_dissolved_ratio(@cell(phase_ratios[I...]), rheology, args_ijk)
    @inbounds mH2O[I...] = mh
    @inbounds mCO2[I...] = mc
    return nothing
end

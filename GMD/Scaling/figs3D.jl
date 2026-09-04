# using Typst
using JLD2
using DataFrames
using GLMakie

fldr       = "WeakScaling3D"
fls_weak   = filter(x -> contains(x,"jld2"), readdir(fldr, join=true))
nGPUs      = parse.(Int, getindex.(split.(fls_weak, "_"), 5))
perms      = sortperm(nGPUs)
fls_weak   = fls_weak[perms]
nGPUs      = nGPUs[perms]

nGPUs      = [1, 8, 32, 27, 64, 216, 125, 512]
perms      = sortperm(nGPUs)

fls_weak   = fls_weak[perms]
nGPUs      = nGPUs[perms]

raw_data  = jldopen.(fls_weak)
# GPUs = np = [1, 8, 64, 512]
# N         = [255, 127, 63, 31]

data        = [d["iters"] for d in raw_data]
τ_kernels   = [d.timers.wtime0 for d in data] 
τ_shear     = [d.timers.wtime_shear_stress for d in data] 
τ_vel       = [d.timers.wtime_velocity for d in data] 
τ           = @. (τ_kernels + τ_shear + τ_vel)  / 101

# efficiency
η  = τ[1] ./ τ

popat!(nGPUs, 3)
popat!(η, 3)

nGPUs[5] = 128
nGPUs[6] = 256

df = DataFrame(GPUs=nGPUs, τ = τ, η = η)

f = Figure(size=(800, 800))
ax = Axis(f[1, 1], aspect = 1, xlabel="GPUs", ylabel="efficiency")
scatterlines!(ax, nGPUs, η, color=:green, linewidth=3)
ax.xticks = nGPUs
f

######################

fldr       = "StrongScaling3D"
day        = "2025-05-12"
fls_strong = filter(x -> contains(x, day) && contains(x, "jld2"), readdir(fldr, join=true))
nGPUs      = parse.(Int, getindex.(split.(fls_strong, "_"), 5))
perms      = sortperm(nGPUs)
fls_strong = fls_strong[perms]
nGPUs      = nGPUs[perms]

raw_data  = jldopen.(fls_strong)
# GPUs = np = [1, 8, 64, 512]
# N         = [255, 127, 63, 31]

data        = [d["iters"] for d in raw_data]
τ_kernels   = [d.timers.wtime0 for d in data] 
τ_shear     = [d.timers.wtime_shear_stress for d in data] 
τ_vel       = [d.timers.wtime_velocity for d in data] 
τ           = @. (τ_kernels + τ_shear + τ_vel)  / 101

# speed up
σ = τ[1] ./ τ
# efficiency
η = σ ./ nGPUs

df = DataFrame(GPUs=nGPUs, τ = τ, σ = σ, η = η)
# tb = tablex(df)


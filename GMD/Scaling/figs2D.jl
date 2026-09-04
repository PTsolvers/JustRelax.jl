# using Typst
using JLD2
using DataFrames
using GLMakie

fldr       = "WeakScaling2D"
fls_weak   = filter(x -> contains(x,"jld2"), readdir(fldr, join=true))
nGPUs      = parse.(Int, getindex.(split.(fls_weak, "_"), 4))
perms      = sortperm(nGPUs)
fls_weak   = fls_weak[perms]
nGPUs      = nGPUs[perms]

perms      = [1, 2, 4, 8, 11]
nGPUs      = nGPUs[perms]
fls_weak   = fls_weak[perms]

raw_data  = jldopen.(fls_weak)
# GPUs = np = [1, 8, 64, 512]
# N         = [255, 127, 63, 31]

data        = [d["iters"] for d in raw_data]
τ_kernels   = [d.wtime_solver for d in data] 
τ_shear     = [d.wtime_shear_stress for d in data] 
τ_vel       = [d.wtime_velocity for d in data] 
τ           = @. (τ_kernels + τ_shear + τ_vel)  / 101

# efficiency
η  = τ[1] ./ τ

df = DataFrame(GPUs=nGPUs, τ = τ, η = η)

f = Figure(size=(600, 600))
ax = Axis(f[1, 1], aspect = 1, title="Weak Scaling", xlabel="GPUs", ylabel="efficiency")
scatterlines!(ax, nGPUs, η, color=:green)

f

######################

fldr       = "StrongScaling2D"
day        = "2025-05-12"
fls_strong = filter(x -> contains(x, day) && contains(x, "jld2"), readdir(fldr, join=true))
nGPUs      = parse.(Int, getindex.(split.(fls_strong, "_"), 4))
perms      = sortperm(nGPUs)
fls_strong = fls_strong[perms]
nGPUs      = nGPUs[perms]

raw_data  = jldopen.(fls_strong)
# GPUs = np = [1, 8, 64, 512]
# N         = [255, 127, 63, 31]

data        = [d["iters"] for d in raw_data]
iters       = [d.iter for d in data] 
τ_kernels   = [d.wtime_solver for d in data] 
τ_shear     = [d.wtime_shear_stress for d in data] 
τ_vel       = [d.wtime_velocity for d in data] 
τ           = @. (τ_kernels + τ_shear + τ_vel)  / iters

# speed up
σ = τ[1] ./ τ
# efficiency
η = σ ./ nGPUs

df = DataFrame(GPUs=nGPUs, τ = τ, σ = σ, η = η)
# tb = tablex(df)


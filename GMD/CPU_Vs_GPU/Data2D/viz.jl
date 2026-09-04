using JLD2, GLMakie

load = false

if load 
    pth  = "ShearBands2D_DYREL"
    nt1  = jldopen(joinpath(pth, "SB2D_CPU_1threads_256x256.jld2"));
    nt8  = jldopen(joinpath(pth, "SB2D_CPU_8threads_256x256.jld2"));
    nt16 = jldopen(joinpath(pth, "SB2D_CPU_16threads_256x256.jld2"));
    nt32 = jldopen(joinpath(pth, "SB2D_CPU_32threads_256x256.jld2"));

    data = (
        jldopen(joinpath(pth, "SB2D_CPU_1threads_256x256.jld2")),
        jldopen(joinpath(pth, "SB2D_CPU_8threads_256x256.jld2")),
        jldopen(joinpath(pth, "SB2D_CPU_16threads_256x256.jld2")),
        jldopen(joinpath(pth, "SB2D_CPU_32threads_256x256.jld2")),
    )
    data_cuda = jldopen(joinpath(pth, "SB2D_GPU_256x256.jld2"))
   
end
nt     = [d["nthreads"] for d in data]
iters  = [d["pt_iterations"] for d in data]
t      = [d["stokes_walltimes"] for d in data]
tsteps = length(iters[1])

fig = Figure(size = (900, 600), fontsize=20)
ax = Axis(fig[1,1], xlabel = "timestep", ylabel = "time [s]")
for i in eachindex(t)
    scatterlines!(ax, 1:tsteps, t[i], label="$(nt[i]) threads")
end
axislegend(ax)

ax = Axis(fig[2, 1],  xlabel = "timestep", ylabel = "speed up")
for i in eachindex(t)
    scatterlines!(ax, 1:tsteps, t[1]./t[i], label="$(nt[i]) threads")
end
scatterlines!(ax, 1:tsteps, t[1] ./ data_cuda["stokes_walltimes"], label="GH200")
fig
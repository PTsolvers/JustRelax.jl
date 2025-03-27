using CairoMakie

func(x)  = 5*x + 2*x^2
dfunc(x) = 5 + 4*x

dp = [1e-2 / 10^i for i in 0:12]

function FD_1(func,dp,x)
    return dfdp = (func(x+dp)-func(x))/dp
end

#=
function FD_2(func,dp,x)
    return dfdp = (func(x+dp)-func(x-dp))/(2*dp)
end
=#

function FD_3(func,dp,x)
    return dfdp =  imag(func(x + im*dp))/dp
end


dfdp  = zeros(length(dp))
dfdp3 = zeros(length(dp))
ana   =  zeros(length(dp))
for i in 1:length(dp)
    dfdp[i]  = FD_1(func,dp[i],4)
    dfdp3[i] = FD_3(func,dp[i],4)
    ana[i]   = dfunc(4)
end

logh = abs.(log10.(dp))
err1 = (log10.(abs.(dfdp.-ana)))
err2 = (log10.(abs.(dfdp3.-ana)))
err1 = ((abs.(dfdp.-ana)))
err2 = ((abs.(dfdp3.-ana)))

fig = Figure(size = (600, 600), title = "Compare first order Finite Difference Methods",fontsize=16);
ax1 = Axis(fig[1,1])
#ax2 = Axis(fig[2,1])
lines!(ax1,logh,err1)
lines!(ax1,logh,reverse(err2))
fig

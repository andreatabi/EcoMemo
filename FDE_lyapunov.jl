using FdeSolver
using Plots
using StatsBase
using Distributions
using DataFrames

function oup(T, n, nu, lambda, sigma, x0)
    dw = rand(Normal(0, sqrt(T/n)), n)  
    dt = T / n
    x = zeros(n+1)  
    time = zeros(n+1)  
    x[1] = x0       
    time[1] = 0
    for i in 2:(n+1)
        x[i] = x[i-1] + lambda * (nu - x[i-1]) * dt + sigma * dw[i-1]
    end
    return x
end

function gLV_OUP(t, n, parms)
    q = parms[:q] 
    A = parms[:A] 
    cc = ceil(t / parms[:h]) + 1
   
    r = values(parms[:r][Int(cc),:])
    if any(n .< 0)
        n[n .< 0] .= 0
    end    
    dndt = q .+ n .* (r .- (A * n)) 
return dndt
end


##  gLV model ---------------------------------------------------------

## inputs
S = 5
tSpan = [0, 100]         # [intial time, final time]
y0 = fill(1, S)     # initial values
beta = 1
β = fill(beta, S)            # order of derivatives (memory)


A = [1.0000000  0.85868842 0.394451933  0.54998279 0.27420033;
    -1.3016048  1.00000000 0.091555780  0.14680490 0.04883888;
    -0.7599866 -0.12595646 1.000000000  0.05943078 0.00000000;
     0.9531117 -0.17249240 0.049319976  1.00000000 0.01741451;
     0.5983599  0.08701049 0.006325647 -0.03335709 1.00000000]
r = fill(1, S)   
h = 0.001
sigma2 = 1
q = fill(0.5, S)   
T = 1000
n = Int(T/h)+1
lambda = 0.1
r0 = r
r_oup = DataFrame([oup(T, n, r[i], lambda, sigma2, r0[i]) for i in 1:S], Symbol.("r" .* string.(1:S)))
    
par = Dict( :A => A,  :r => r_oup, :q => q, :h => h)
   
t, Yapp = FDEsolver(gLV_OUP, tSpan, y0, β, par, h=0.001)

df = DataFrame(hcat(t, Yapp), :auto)

r_bar = r0
y_star = A \ r_bar


## Lyapunov analysis
function lyapunov_gLV(y, y_star)
    y_safe = map(x -> max(x, 1e-8), y)
    y_star_safe = map(x -> max(x, 1e-8), y_star)
    return sum(y_safe .- y_star_safe .- y_star_safe .* log.(y_safe ./ y_star_safe))
end


V = [lyapunov_gLV(Yapp[i, :], y_star) for i in 1:size(Yapp, 1)]


# plotting
plot(t, Yapp, linewidth = 2, title = "LV model with memory", xaxis = "Time (t)", yaxis = "N(t)" )
plot(t, V, xlabel="Time", ylabel="Lyapunov V(t)", title="Lyapunov Function of Stochastic gLV")



# Calcualte ensemble average

num_paths = 30
V_ensemble = zeros(length(t), num_paths)

for path in 1:num_paths
    # regenerate new stochastic r(t)
    r_oup = DataFrame([oup(T, n, r[i], lambda, sigma2, r0[i]) for i in 1:S], Symbol.("r" .* string.(1:S)))
    par = Dict(:A => A, :r => r_oup, :q => q, :h => h)

    # solve the system
    t, Yapp = FDEsolver(gLV_OUP, tSpan, y0, β, par, h=h)

    # compute Lyapunov function
    V = [lyapunov_gLV(Yapp[i, :], y_star) for i in 1:size(Yapp, 1)]
    
    V_ensemble[:, path] .= V
end

# Compute ensemble mean and std
V_mean = mean(V_ensemble, dims=2)
V_std = std(V_ensemble, dims=2)


plot(t, V_mean[:], ribbon=V_std[:], lw=2, title="Mean Lyapunov Function",
     xlabel="Time", ylabel="E[V(t)] ± std", label="Mean ± std")
# If V_mean(t) decreases, then it is stable
# If it levels off, you may have stationarity or boundedness
# If V_mean(t) grows, the system is not stable
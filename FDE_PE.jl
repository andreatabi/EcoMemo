using FdeSolver
using Plots
using StatsBase
using Distributions
using DataFrames
using LinearAlgebra
using Statistics
using StatsBase
using Combinatorics
using DynamicalSystems
using DelayEmbeddings
using Printf
using StaticArrays

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

function gLV(t, n, parms)  
    r = parms[:r] 
    q = parms[:q] 
    A = parms[:A]
     dndt = q .+ n .* (r .- (A * n)) 
    return dndt
end


##  gLV model ---------------------------------------------------------

## inputs
S = 5
tSpan = [0, 100]         # [intial time, final time]
y0 = fill(1, S)     # initial values
beta = 0.7
β = fill(beta, S)            # order of derivatives (memory)


A = [1.0000000  0.85868842 0.394451933  0.54998279 0.27420033;
    -1.3016048  1.00000000 0.091555780  0.14680490 0.04883888;
    -0.7599866 -0.12595646 1.000000000  0.05943078 0.00000000;
     0.9531117 -0.17249240 0.049319976  1.00000000 0.01741451;
     0.5983599  0.08701049 0.006325647 -0.03335709 1.00000000]
r = fill(1, S)   
h = 0.001
sigma2 = 0.5
q = fill(0.1, S)   
T = 1000
n = Int(T/h)+1
lambda = 0.1
r0 = r
r_oup = DataFrame([oup(T, n, r[i], lambda, sigma2, r0[i]) for i in 1:S], Symbol.("r" .* string.(1:S)))
    
par = Dict( :A => A,  :r => r_oup, :q => q, :h => h)
#par = Dict( :A => A,  :r => r, :q => q)
   
t, Yapp = FDEsolver(gLV_OUP, tSpan, y0, β, par, h=0.001)
#t, Yapp = FDEsolver(gLV, tSpan, y0, β, par, h=0.001)

df = DataFrame(hcat(t, Yapp), :auto)


plot(t, Yapp, linewidth = 2, title = "LV model with memory", xaxis = "Time (t)", yaxis = "N(t)" )


# Compute Permutation Entropy (PE) 

function permutation_entropy(x::Vector{Float64}, m::Int, τ::Int)
    N = length(x)
    patterns = Dict{Vector{Int}, Int}()
    total = 0

    for i in 1:(N - (m - 1)*τ)
        pattern = x[i:τ:i + (m - 1)*τ]
        order = sortperm(pattern)  # ordinal pattern
        patterns[order] = get(patterns, order, 0) + 1
        total += 1
    end

    probs = [v / total for v in values(patterns)]
    H = -sum(p * log(p) for p in probs)
    H_max = log(factorial(m))

    return H / H_max  # normalized entropy [0,1]
end

E = 5    # embedding dimension
τ = 1    # delay

PE_values = [permutation_entropy(collect(Yapp[:, i]), E, τ) for i in 1:size(Yapp, 2)]
PE_mean = mean(PE_values)
println("Mean PE across species: ", round(PE_mean, digits=4))

PE_std = std(PE_values)
println("Std PE across species: ", round(PE_std, digits=4))

#Compute Weighted Permutation Entropy (WPE) 

function weighted_permutation_entropy(x::Vector{Float64}, m::Int, τ::Int)
    N = length(x)
    weights = Dict{Vector{Int}, Float64}()
    total_weight = 0.0

    for i in 1:(N - (m - 1)*τ)
        pattern = x[i:τ:i + (m - 1)*τ]
        order = sortperm(pattern)               # ordinal pattern
        w = Statistics.var(pattern)             # local variance as weight
        weights[order] = get(weights, order, 0.0) + w
        total_weight += w
    end

    # Normalize weights to get probability distribution
    probs = [w / total_weight for w in values(weights)]

    H_w = -sum(p * log(p) for p in probs)
    H_max = log(factorial(m))

    return H_w / H_max  # normalized WPE in [0, 1]
end

E = 5   # embedding dimensions between 3-5 depends on the length of the time series
τ = 1

WPE_values = [weighted_permutation_entropy(collect(Yapp[:, i]), E, τ) for i in 1:size(Yapp, 2)]

# Summarize into a scalar metric
WPE_mean = mean(WPE_values)
WPE_std = std(WPE_values)

println("Mean WPE across species: ", round(WPE_mean, digits=4))
println("Std  WPE across species: ", round(WPE_std, digits=4))


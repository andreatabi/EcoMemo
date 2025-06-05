using Random, Distributions
using FdeSolver
using Plots
using StatsBase
using DataFrames, Statistics
using JLD2, FileIO
using CSV

#Random.seed!(19850827)  
include("toolbox.jl")


T = 100
S = 10            
k = floor(log2(S))-1     
stren = sort(repeat([0.01, 0.05, 0.1, 0.5, 1], inner=1000))  
sims = length(stren) 
out = DataFrame(sim=Int[], time=Float64[], sp_pool=Int[], beta=Float64[], sigma2=Float64[], stren=Float64[], con=Float64[],
                sdA=Float64[], meanA=Float64[], states=Float64[], 
                ep_exp = Float64[], ep_exp_r2 = Float64[], ep_length = Float64[],ep_power = Float64[], ep_power_r2 = Float64[], 
                mi_exp = Float64[], mi_exp_r2 = Float64[], mi_length = Float64[],mi_power = Float64[], mi_power_r2 = Float64[], 
                be_lin = Float64[], be_lin_r2 = Float64[], be_power = Float64[], be_power_r2 = Float64[], be_length = Float64[],
                FI = Float64[]
             #   R =  Float64[], R_ts =  Float64[], VCR_ts =  Float64[]
                )

for u in 1:sims

    println("iteration :", u)
    body_size = sort(10 .^ rand(Normal(1, 0.3), S))
    frac = rand()
   # int = interaction_matrix(body_size, frac, mu[u] )
    int = random_interaction_matrix(S, frac, stren[u])
    
 #   A = int["A"]  
    A = int
   # print(A)
   # pred_all = int["pred"]  

    #fr = length(pred_all) / S
    q = body_size .^ (-0.25 .+ rand(Normal(0, 0.1), S))        
   # K = 1000 .* body_size .^ (-0.75 .+ rand(Normal(0, 0.1), S))    
    r = body_size .^ (-0.25 .+ rand(Normal(0, 0.1), S))      
    d = fill(0.05, S)  
    y0 = fill(10, S)

  #  beta = body_size .^ (-0.25)
  #  beta = (beta .- minimum(beta) .+ 0.01) / (maximum(beta) - minimum(beta) + 0.01)    
  #  beta = round.(beta, digits = 2)
  #  β = beta  

    #beta = round(rand() * (1 - 0.006) + 0.006, digits = 2) 
    beta = rand(0.1:0.1:1) 
    β = fill(beta, S)     
    
    tSpan = [0, T]
    sigma =  1    
    h = 0.01

    # Parameters
    n = Int(T/h)+1
    r = body_size .^ (-0.25 .+ rand(Normal(0, 0.1), S))      
    lambda = 0.1
    r0 = r
    #sigma2 = round(rand() * 2, digits = 2)
    sigma2 = rand(0:0.5:2)
    r_oup = DataFrame([oup(T, n, r[i], lambda, sigma2, r0[i]) for i in 1:S], Symbol.("r" .* string.(1:S)))
    #plot(r_oup, linewidth = 2, title = "OUP", 
    #                xaxis = "Time (t)", yaxis = "N(t)" )

    par = Dict( :A => A, :d => d, :q => q, :r => r_oup, :h =>h)
            
    t, Yapp = FDEsolver(gLV_OUP, tSpan, y0, β, par, h=h)

    df = DataFrame(hcat(t, Yapp), :auto)
    #print(df[:, 2:end])

    df1 = (df .> 0) .|> Int
    ct = [join(row, "_") for row in eachrow(df1)]
    ms = unique(ct)
   # res_sample = HOMC(ct, memo)
    #res_sample_rev = HOMC(reverse(ct), memo)
    
    # entropy production
    ep = entprod2(ct,50)
    EP = compute_entropy_prod_exponent(ep, 1:50)
    ep_exp = EP[:slope_exp]
    ep_r2_exp = EP[:r_sq_exp]
    ep_power = EP[:slope_power]
    ep_r2_power = EP[:r_sq_power]
    ep_length = EP[:length]
    
  #  plt = plot( log.(1:100), log.(ep),seriestype = :scatter,line = (:solid, :blue),markershape = :circle,
   #     xlabel = "Lag", ylabel = "Entropy Production", title = "Entropy Production vs Lag",
    #    legend = false, grid = true)
    #display(plt)
    # mutual information
  
    I = compute_mutual_information(ct, 50)
    MI = compute_mutual_info_exponent(I, 1:50)
    mi_exp = MI[:slope_exp]
    mi_r2_exp = MI[:r_sq_exp]
    mi_power = MI[:slope_power]
    mi_r2_power = MI[:r_sq_power]
    mi_length = MI[:length]
    
    # block entropy
    Hn = block_entropy(ct, 50)
    BE = block_entropy_exponent(Hn, 50)
    be_lin = BE[:linear_slope]
    be_r2_lin = BE[:linear_r2]
    be_power = BE[:power_slope]
    be_r2_power = BE[:power_r2]
    be_length = BE[:length]

    # reactivity
   # eq = neq(A, r, d)
   # J = LVjac(A, r, eq, d)
   # react = Reactivity(J)
   # df2 = df[:, 2:end]
   # reac_ts = [Reactivity(LVjac(A, Vector(r_oup[i,:]),Vector(df2[i, :]), d)) for i in 1:nrow(df)]
   # med_react = median(reac_ts)
   
    # Compute VCR
   # vcr_ts = [VCR(LVjac(A, Vector(r_oup[i,:]),Vector(df2[i, :]), d)) for i in 1:nrow(df)]
   # med_vcr = median(vcr_ts)

   FI = calculate_weighted_frustration(A, verbose=false)

    A[diagind(A)] .= NaN
    AA = filter(!isnan, A )

    push!(out, (u, T, S, mean(beta), sigma2, stren[u], frac, std(vec(AA)), mean(vec(AA)), length(ms), 
                 ep_exp, ep_r2_exp, ep_length,ep_power, ep_r2_power,  
                 mi_exp, mi_r2_exp, mi_length,mi_power, mi_r2_power,  
                 be_lin, be_r2_lin, be_power, be_r2_power, be_length,
                 FI["weighted_frustration_index"]
               #  react, med_react, med_vcr
                 ))
                
end

#print(out)

CSV.write("/Users/Andrea/Dropbox/HOMC/results/FDE_results_bitstring.csv", out)
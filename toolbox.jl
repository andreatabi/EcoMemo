using Random, Optim
using LinearAlgebra
using DataFrames
using JuMP
using GLPK
using HiGHS
using Distributions
using Observables
using StatsBase
using Statistics
using GLM
using InformationMeasures
using Printf
using Plots
using MultivariateStats
using GaussianMixtures
using StatsBase: argmax
using UMAP
using Combinatorics
#using ClusteringGMM


using Random, Statistics, LinearAlgebra, Optim

function random_interaction_matrix(S::Int, con::Float64, stren::Float64)
    n = S * (S - 1)

    # Generate off-diagonal interaction values
    Inte = randn(n) .* stren

    # Apply connectance by zeroing out some interactions
    num_nonzero = floor(Int, n * con)
    mask = vcat(ones(Int, num_nonzero), zeros(Int, n - num_nonzero))
    shuffle!(mask)
    Inte[mask .== 0] .= 0.0

    # Fill the off-diagonal entries of matrix M
    M = fill(NaN, S, S)
    inds = findall(I -> I[1] != I[2], CartesianIndices((S, S)))
    for (idx, val) in enumerate(Inte)
        M[inds[idx]] = val
    end

    # Set diagonal to 1.0
    for i in 1:S
        M[i, i] = 1.0
    end

    # Objective function to match target SD of off-diagonal entries
    function objective(a0)
        A = a0 .* M
        for i in 1:S
            A[i, i] = NaN  # exclude diagonal from SD calculation
        end
        current_sd = std(skipmissing(vec(A)))
        return (current_sd - stren)^2
    end

    result = optimize(objective, 0.0, 100.0, Brent(); abs_tol=1e-14)
    a0 = Optim.minimizer(result)

    # Scale matrix and restore diagonal
    A = a0 .* M
    for i in 1:S
        A[i, i] = 1.0
    end

    return A
end


function interaction_matrix(M, frac, mu, e=1)
    S = length(M)
    probs_M = M ./ sum(M)
    pred_sp = findall(in(sample(M,  Weights(probs_M), floor(Int, S * frac))), M)
    
    m_j = (1/4 + 2/3) # consumer scaling Rall et al 2012
    m_i = 2/3         # resource scaling Rall et al 2012
    
    function f(a0)
        IntePP0 = (repeat(M, 1, S) .^ m_i) .* (repeat(M', S, 1) .^ m_j)
        IntePP0[diagind(IntePP0)] .= NaN
        IntePP = rescale_matrix(IntePP0)
        IntePP[tril!(trues(S, S), -1)] .*= -1
        IntePP[diagind(IntePP)] .= 1
        
        InteC0 = (repeat(M', S, 1) ./ repeat(M, 1, S)) .^ (3/4)
        InteC0[diagind(InteC0)] .= NaN
        InteC = rescale_matrix(InteC0)
        InteC[diagind(InteC)] .= 1
        
        A0 = copy(InteC)
        for i in pred_sp
            A0[i, 1:(i-1)] .= e .* IntePP[i, 1:(i-1)] .+ InteC[i, 1:(i-1)]
            A0[1:(i-1), i] .= e .* IntePP[1:(i-1), i] .+ InteC[1:(i-1), i]
        end
        A = a0 * A0
        A[diagind(A)] .= 1
        return ((sum(A) - S) / ((S-1) * S) - mu)^2
    end
    
    res = optimize(f, 0, 100, abs_tol=1e-14)
    a0 = Optim.minimizer(res)
    
    IntePP0 = (repeat(M, 1, S) .^ m_i) .* (repeat(M', S, 1) .^ m_j)
    IntePP0[diagind(IntePP0)] .= NaN
    IntePP = rescale_matrix(IntePP0) 
    IntePP[tril!(trues(S, S), -1)] .*= -1
    IntePP[diagind(IntePP)] .= 1
    
    InteC0 = (repeat(M', S, 1) ./ repeat(M, 1, S)) .^ (3/4)
    InteC0[diagind(InteC0)] .= NaN
    InteC = rescale_matrix(InteC0)
    InteC[diagind(InteC)] .= 1
    
    A0 = copy(InteC)
    for i in pred_sp
        A0[i, 1:(i-1)] .= e * IntePP[i, 1:(i-1)] .+ InteC[i, 1:(i-1)]
        A0[1:(i-1), i] .= e * IntePP[1:(i-1), i] .+ InteC[1:(i-1), i]
    end
    A = a0 * A0
    A[diagind(A)] .= 1
 
    return Dict("A" => A, "pred" => pred_sp)
end


function rescale_matrix(mat)
    d = diagind(mat)
    min_val = minimum(mat[:][setdiff(1:length(mat), d)])
    max_val = maximum(mat[:][setdiff(1:length(mat), d)])
    return (mat .- min_val) ./ (max_val - min_val)
end


function oup(T, n, nu, lambda, sigma, x0)
    dw = rand(Normal(0, sqrt(T/n)), n)  
    dt = T / n
    x = zeros(n+1)  
    time = zeros(n+1)  
    x[1] = x0       
    time[1] = 0
    for i in 2:(n+1)
        x[i] = x[i-1] + lambda * (nu - x[i-1]) * dt + sigma * dw[i-1]
        #time[i] = time[i-1] + dt
    end
    #df = DataFrame(time = time, r = x )
    return x
end


function gLV(t, n, parms)
    
    r = parms[:r] 
    d = parms[:d] 
    q = parms[:q] 
    A = parms[:A]
     dndt = q .+ n .* (r .- (A * n)) .- d .* n 
    return dndt
end



function homc_q(x, r; indices=false, as_embed=true)
    n = length(x) - (r - 1)
    if n <= 0
        error("Insufficient observations for the requested embedding")
    end

    out = [collect(1:n) .+ (i - 1) for i in 1:r] |> hcat

    if as_embed
        out = out[:, 1:r]
    end

    if !indices
        out = reshape(x[out], n, r)
    end

    return out
end

function homc_embed(ct, s, indices=false, as_embed=true)
    n = length(ct) - (s - 1)
    print(n)
    if n <= 0
        error("Insufficient observations for the requested embedding")
    end
    
    X = collect(1:length(ct))  
    out = reshape(repeat(X[1:n], s), n, s)
    
    for i in 2:s
        out[:, i] .+= (i - 1)
    end
    
    if as_embed
        out = out[:, 1:s]
    end

    if !indices
        out = reshape(ct[out], n, s)
    end

    return out
end


function HOMC(ct, memo)
    Q_matrix = []
    pp = homc_embed(ct, memo + 1)
    #pp = filter(row -> !any(isnan, row), eachrow(pp)) |> collect
    
    if isempty(pp)
        return fill(NaN, memo)
    else
        states = unique(skipmissing(ct))
        for i in 1:memo
            com = unique(pp[:, [1, i + 1]], dims=1)
            
            Qs0 = zeros(length(states), length(states))
            for v in 1:size(com, 1)
                pairs = findall(pp[:, 1] .== com[v, 1] .&& pp[:, i+1] .== com[v, 2])
                Qs0[findfirst(states .== com[v, 2]), findfirst(states .== com[v, 1])] = length(pairs)
            end
            Qs = mapslices(col -> sum(col) != 0 ? col ./ sum(col) : col, Qs0, dims=1)
            
            push!(Q_matrix, Qs)
        end

        ct_counter = countmap(ct)
        total_count = sum(values(ct_counter))
        X_hat = [get(ct_counter, s, 0) / total_count for s in states]
       
        Q_X_hats = [Q_matrix[i] * X_hat for i in 1:memo]
        QXh = hcat(Q_X_hats...)
        X_hat_column = reshape(X_hat, :, 1)
        
       # eq_matrix1 = hcat(-QXh, -ones(size(QXh, 1)))
       # eq_matrix2 = hcat( QXh, -ones(size(QXh, 1)))
        eq_matrix1 = hcat(-QXh)
        eq_matrix2 = hcat( QXh)
       #print(eq_matrix1)

        # Optimization using JuMP
        #model = Model(GLPK.Optimizer)
        model = Model(HiGHS.Optimizer)
        @variable(model, x[1:memo])
        @variable(model, z )
        @objective(model, Min, z)  
        @constraint(model, sum(x[1:memo]) == 1)
        @constraint(model, eq_matrix1 * x  + X_hat_column .<=  z)
        @constraint(model, eq_matrix2 * x  - X_hat_column .<=  z)
        @constraint(model, x[1:memo] >=0)
        @constraint(model, z >=0)

        optimize!(model)
        #print(model) 
        return Dict("x" => value.(x), "z" => objective_value(model))
    end
end


function gLV_OUP(t, n, parms)
    d = parms[:d] 
    q = parms[:q] 
    A = parms[:A] 
    cc = ceil(t / parms[:h]) + 1
   
    r = values(parms[:r][Int(cc),:])
    if any(n .< 0)
        n[n .< 0] .= 0
    end    
    dndt = q .+ n .* (r .- (A * n)) .- d .* n 
    #dndt =  n .* (r .- (A * n)) .- d .* n 
return dndt
end


function mutinfo(ts, memo)
    Q_matrix = []
    pp = homc_embed(ts, memo + 1)

    states = unique(ts)
    num_states = length(states)
    #state_index = Dict(s => i for (i, s) in enumerate(states))
    
    for i in 1:memo
        com = unique(pp[:, [1, i + 1]], dims=1)
        Qs0 = zeros(num_states, num_states)
        for v in 1:size(com, 1)
            pairs = findall((com[v,1] .== pp[:,1]) .& (com[v,2] .== pp[:,i+1]))
            #Qs0[state_index[com[v][1]], state_index[com[v][2]]] = length(pairs)
            Qs0[findfirst(states .== com[v,2]), findfirst(states .== com[v,1])] = length(pairs)
        end
        
        if any(Qs0 .== 0)
            Qs0 .+= 1
        end

        Qs = Qs0 ./ sum(Qs0, dims=2)
        push!(Q_matrix, Qs)
    end

    # Mutual information
    I = []
    for j in 1:memo
        Tr = Q_matrix[j]
        counts = countmap(ts)  
        total_count = sum(values(counts))  
        P0 = Dict(k => v / total_count for (k, v) in counts)
        P = [P0[c] for c in unique(ts)]  
        E = -sum(P .* log.(P))
        CE =  -sum( P[i] * sum(Tr[i, :] .* log.(Tr[i,: ])) for i in 1:num_states )
        push!(I, E - CE)
    end

    return I
end

function mutualinfo(x::Vector, y::Vector)
    @assert length(x) == length(y)

    # Create joint and marginal frequency tables
    joint = countmap(zip(x, y))
    px = countmap(x)
    py = countmap(y)
    N = length(x)
    
    mi = 0
    for ((xi, yi), pxy_count) in joint
        pxy = pxy_count / N
        px_i = px[xi] / N
        py_i = py[yi] / N
        mi += pxy * log(pxy / (px_i * py_i)) 
    end

    return mi
end

function compute_mutual_information(symbols::Vector{String}, max_lag::Int)
    mi_vals = zeros(Float64, max_lag)    
    for τ in 1:max_lag
        x = symbols[1:end - τ]
        y = symbols[1 + τ:end]
        mi_vals[τ] = mutualinfo(x, y)
    end
    return mi_vals
end



function PRG(df::DataFrame; norm=false, bin=false)
    mat = Matrix(df)

    if norm
        mat = mat .- mean(mat, dims=1)
    end

    
    if bin
        mat = ifelse.(mat .> 0, 1, 0)
    end
   
    new_mat = Matrix{Float64}(undef, size(mat, 1), 0)  

    for i in 1:fld(size(mat, 2), 2)  
        co = corspearman(mat)  
        co[diagind(co)] .= 0 
        max_corr = maximum(co)  
        ind = Tuple(findall(x -> x == max_corr, co)[1])
        sum_col = vec(sum(mat[:, [ind...]], dims=2))  

        if i == 1
            new_mat = sum_col
        else
            new_mat = hcat(new_mat, sum_col)
        end

        mat = mat[:, Not([ind...])]  
    end

    final_mat = hcat(new_mat, mat)
    return DataFrame(final_mat, :auto)
end

function LVjac(A::Matrix{Float64}, r::Vector{Float64}, n::Vector{Float64}, d::Vector{Float64})
    Jac = zeros(size(A))
    num_species = size(A, 1)

    for j in 1:num_species
        cols = setdiff(1:num_species, j)
        Jac[j, j] = r[j] - d[j] - sum(A[j, :] .* n)
        Jac[j, cols] = -A[j, cols] .* n[j]
    end

    return Jac
end

function Reactivity(J::Matrix{Float64})
    H = (J + transpose(J)) / 2
    λ = eigvals(H)
    return maximum(real(λ))
end

function neq(A, r, d)
    eq = A \ (r .- d)
    return eq
end

function VCR(J::Matrix{Float64})
    Tr = tr(J)
    return Tr
end


function mutinfo_TL(ts::Vector, memo::Int)
    Q_matrix = Vector{Matrix{Float64}}()
    pp = homc_embed(ts, memo + 1)
    states = unique(ts)
    state_index = Dict(s => i for (i, s) in enumerate(states))
    n_states = length(states)

    for i in 1:memo
        Qs0 = zeros(Float64, n_states, n_states)
        for row in 1:size(pp, 1)
            from = pp[row, 1]
            to = pp[row, i + 1]
            Qs0[state_index[to], state_index[from]] += 1
        end

        # Laplace smoothing
        if any(Qs0 .== 0)
            Qs0 .+= 1
        end

        Qs = Qs0 ./ sum(Qs0, dims=2)
        push!(Q_matrix, Qs)
    end

    counts = countmap(ts)
    total = sum(values(counts))
    P = [counts[s] / total for s in states]
    E = -sum(p * log(p) for p in P)

    I = []
    for j in 1:memo
        Tr = Q_matrix[j]
        CE = -sum(P[i] * sum(Tr[i, :] .* log.(Tr[i, :])) for i in 1:n_states)
        push!(I, E - CE)
    end
    
    return I
end

function block_entropy(sym_series::Vector{String}, max_block::Int=12)
    Hn = zeros(Float64, max_block)
    N = length(sym_series)

    for n in 1:max_block
        block_strings = [join(sym_series[i:i+n-1], "") for i in 1:(N - n + 1)]
        freqs = countmap(block_strings)
        total = sum(values(freqs))
        probs = [v / total for v in values(freqs)]
        Hn[n] = -sum(p * log2(p) for p in probs)
    end

    return Hn
end


function block_entropy_exponent(Hn::Vector{Float64}, max_block::Int)
  #  dH = diff(Hn)  
  #  threshold = std(dH)
    n_vals = collect(1:max_block)

  #  valid_max_n = findall(dH .> threshold) 
  #  cut_off = maximum(valid_max_n) + 1

  valid_max_n =  max_block
  cut_off = max_block

    if cut_off < 2
        @warn "Too few usable points for slope estimation."
        return Dict(
            :linear_slope => NaN,
            :linear_r2 => NaN,
            :power_slope => NaN,
            :power_r2 => NaN,
            :H1 => Hn[1],
            :length => cut_off
        )
    else

        # Linear fit
        df_linear = DataFrame(n = n_vals[1:cut_off], H = Hn[1:cut_off])
        fit_linear = lm(@formula(H ~ n), df_linear)
        linear_slope = coef(fit_linear)[2]
        linear_r2 = r2(fit_linear)

        # Log-log fit
        log_n = log.(n_vals[1:cut_off])
        log_H = log.(Hn[1:cut_off])
        df_log = DataFrame(log_n = log_n, log_H = log_H)
        fit_log = lm(@formula(log_H ~ log_n), df_log)
        power_slope = coef(fit_log)[2]
        power_r2 = r2(fit_log)

        return Dict(
            :linear_slope => linear_slope,
            :linear_r2 => linear_r2,
            :power_slope => power_slope,
            :power_r2 => power_r2,
            :H1 => Hn[1],
            :length => cut_off
        )
    end
end


function compute_mutual_info_exponent(mi::Vector{Float64}, lags::UnitRange{Int64})
    valid = findall(x -> x > 0, mi)
   # min_len = min(length(valid), 10)

    min_len = length(valid)

    if all(mi .== 0.0) || length(valid) < 2
        return Dict(
            :r_sq_power => NaN,
            :slope_power => NaN,
            :r_sq_exp => NaN,
            :slope_exp => NaN,
            :length => length(valid)
        )
    else
        tau = lags[valid]
        log_tau = log.(tau)
        log_mi = log.(mi[valid])

        best_r2 = -Inf
        best_range = (1, min_len)
        best_fit = nothing
        n_power = min_len 
     #   best_aic_power = -Inf

        best_r2_exp = -Inf
        best_range_exp = (1, min_len)
        best_fit_exp = nothing
        n_exp = min_len
       # best_aic_exp = -Inf

        n = length(log_tau)

        for i in min_len:n
            x = log_tau[1:i]
            y = log_mi[1:i]
            df = DataFrame(x = x, y = y)
            fit = lm(@formula(y ~ x), df)
            r2_power = r2(fit)
           # aic_power = AIC(fit)
            if r2_power > best_r2
                best_r2 = r2_power
                best_range = (1, i)
                best_fit = fit
                n_power = i
              #  best_aic_power = aic_power
            end
        end

        for j in min_len:n
            x_exp = tau[1:j]
            y_exp = log_mi[1:j]
            df_exp = DataFrame(x = x_exp, y = y_exp)
            fit_exp = lm(@formula(y ~ x), df_exp)
            r2_exp = r2(fit_exp)
            if r2_exp > best_r2_exp
                best_r2_exp = r2_exp
                best_range_exp = (1, j)
                best_fit_exp = fit_exp
                n_exp = j
            end
        end

        # Plotting
      #  plot(log_tau, log_mi, label = "Data", marker = :circle, title = "Mutual Information Decay (log)", xlabel = "log(Lag)", ylabel = "log(MI)")
      #  scatter!(log_tau[best_range[1]:best_range[2]], log_mi[best_range[1]:best_range[2]], label = "Best Fit Range", color = :red)

      #  plot(tau, log_mi, label = "Data", marker = :circle, title = "Mutual Information Decay", xlabel = "Lag", ylabel = "log(MI)")
      #  scatter!(tau[best_range_exp[1]:best_range_exp[2]], log_mi[best_range_exp[1]:best_range_exp[2]], label = "Best Exp Fit", color = :blue)

        return Dict(
            :r_sq_power => best_r2,
            :slope_power => coef(best_fit)[2],
            :r_sq_exp => best_r2_exp,
            :slope_exp => coef(best_fit_exp)[2],
            :length => n_power
        )
    end
end


function embed_sequence(x::Vector, order::Int)
    n = length(x) - order + 1
    if n <= 0
        error("Insufficient observations for the requested embedding")
    end
    return [x[i:(i + order - 1)] for i in 1:n]
end

function entprod2(sym_series::Vector{String}, max_lag::Int; plot_result::Bool=false)
    states = sort(unique(skipmissing(sym_series)))
    num_states = length(states)
    epsilon = 1e-12

    # Empirical distribution
    valid_series = collect(skipmissing(sym_series))
    state_counts = countmap(valid_series)
    P = [get(state_counts, s, 0) / length(valid_series) for s in states]

    # Transition matrix function
    function transition_matrices(series::Vector, max_lag::Int)
        matrices = Vector{Matrix{Float64}}(undef, max_lag)
        emb = embed_sequence(series, max_lag + 1)
        emb = filter(e -> !any(ismissing, e), emb)

        for lag in 1:max_lag
            mat = zeros(num_states, num_states)
            for row in emb
                from = row[1]
                to = row[lag + 1]
                i = findfirst(==(to), states)
                j = findfirst(==(from), states)
                if !isnothing(i) && !isnothing(j)
                    mat[i, j] += 1
                end
            end
            # Normalize columns
            col_sums = sum(mat, dims=1)
            for j in 1:num_states
                if col_sums[1, j] > 0
                    mat[:, j] ./= col_sums[1, j]
                end
            end
            matrices[lag] = mat
        end
        return matrices
    end

    forward_mats = transition_matrices(sym_series, max_lag)
    backward_mats = transition_matrices(reverse(sym_series), max_lag)

    entropy_prod = zeros(max_lag)
    for lag in 1:max_lag
        Tij = forward_mats[lag] .* repeat(P, 1, num_states)
        Tji = backward_mats[lag] .* repeat(P, 1, num_states)
        Tij .+= epsilon
        Tji .+= epsilon
        entropy_prod[lag] = sum((Tij .- Tji) .* log.(Tij ./ Tji))
    end

    if plot_result
        plot(1:max_lag, entropy_prod, seriestype=:scatter, title="Entropy Production vs Lag",
             xlabel="Lag", ylabel="Entropy Production", legend=false)
    end

    return entropy_prod
end

function entprod(sym_series::Vector{String}, max_lag::Int; plot_result::Bool=false) 
    states = unique(skipmissing(sym_series))
    state_idx = Dict(s => i for (i, s) in enumerate(states))
    N = length(states)
    total_length = count(!ismissing, sym_series)
    prop = countmap(skipmissing(sym_series))
    P_vec = [get(prop, s, 0) / total_length for s in states]

    # Embedding
    pp = homc_embed(sym_series, max_lag + 1)
    if size(pp, 1) == 0
        return fill(NaN, max_lag)
    end

    # Forward transitions
    forward_tr = Vector{Matrix{Float64}}(undef, max_lag)
    for lag in 1:max_lag
        mat = zeros(N, N)
        for row in eachrow(pp)
            i = state_idx[row[1]]
            j = state_idx[row[lag+1]]
            mat[j, i] += 1
        end
        forward_tr[lag] = reduce(hcat, [sum(col) > 0 ? col ./ sum(col) : zeros(N) for col in eachcol(mat)])
    end

    # Reverse
    rev_series = reverse(sym_series)
    pp_rev = homc_embed(rev_series, max_lag + 1)
    if size(pp_rev, 1) == 0
        return fill(NaN, max_lag)
    end

    # Backward transitions
    backward_tr = Vector{Matrix{Float64}}(undef, max_lag)
    for lag in 1:max_lag
        mat = zeros(N, N)
        for row in eachrow(pp_rev)
            i = state_idx[row[1]]
            j = state_idx[row[lag+1]]
            mat[j, i] += 1
        end
        backward_tr[lag] = reduce(hcat, [sum(col) > 0 ? col ./ sum(col) : zeros(N) for col in eachcol(mat)])
    end

    # Entropy Production
    entropy_production = zeros(Float64, max_lag)
    ϵ = 1e-12

    for lag in 1:max_lag
        Tij = forward_tr[lag] .* repeat(P_vec', N, 1)
        Tji = backward_tr[lag] .* repeat(P_vec', N, 1)
        Tij .+= ϵ
        Tji .+= ϵ
        entropy_production[lag] = sum((Tij .- Tji) .* log.(Tij ./ Tji))
    end

    if plot_result
        plot(1:max_lag, entropy_production, seriestype = :scatter, 
             title = "Entropy Production vs Lag", 
             xlabel = "Lag", ylabel = "Entropy Production")
    end

    return entropy_production
end


function compute_entropy_prod_exponent(ep::Vector{Float64}, lags::UnitRange{Int64})
    valid = findall(x -> x > 0, ep)
   # min_len = min(length(valid), 10)

    min_len= length(valid)

    if all(ep .== 0.0) || length(valid) < 2
        return Dict(
            :r_sq_power => NaN,
            :slope_power => NaN,
            :r_sq_exp => NaN,
            :slope_exp => NaN,
            :length => length(valid)
        )
    else
        tau = lags[valid]
        log_tau = log.(tau)
        log_ep = log.(ep[valid])

        best_r2 = -Inf
        best_range = (1, min_len)
        best_fit = nothing
        n_power = min_len

        best_r2_exp = -Inf
        best_range_exp = (1, min_len)
        best_fit_exp = nothing
        n_exp = min_len

        n = length(log_tau)

        for i in min_len:n
            x = log_tau[1:i]
            y = log_ep[1:i]
            df = DataFrame(x = x, y = y)
            fit = lm(@formula(y ~ x), df)
            r2_power = r2(fit)
            if r2_power > best_r2
                best_r2 = r2_power
                best_range = (1, i)
                best_fit = fit
                n_power = i
            end
        end

        for j in min_len:n
            x_exp = tau[1:j]
            y_exp = log_ep[1:j]
            df_exp = DataFrame(x = x_exp, y = y_exp)
            fit_exp = lm(@formula(y ~ x), df_exp)
            r2_exp = r2(fit_exp)
            if r2_exp > best_r2_exp
                best_r2_exp = r2_exp
                best_range_exp = (1, j)
                best_fit_exp = fit_exp
                n_exp = j
            end
        end

        # Plotting
      #  plot(log_tau, log_mi, label = "Data", marker = :circle, title = "Mutual Information Decay (log)", xlabel = "log(Lag)", ylabel = "log(MI)")
      #  scatter!(log_tau[best_range[1]:best_range[2]], log_mi[best_range[1]:best_range[2]], label = "Best Fit Range", color = :red)

      #  plot(tau, log_mi, label = "Data", marker = :circle, title = "Mutual Information Decay", xlabel = "Lag", ylabel = "log(MI)")
      #  scatter!(tau[best_range_exp[1]:best_range_exp[2]], log_mi[best_range_exp[1]:best_range_exp[2]], label = "Best Exp Fit", color = :blue)

        return Dict(
            :r_sq_power => best_r2,
            :slope_power => coef(best_fit)[2],
            :r_sq_exp => best_r2_exp,
            :slope_exp => coef(best_fit_exp)[2],
            :length => n_power
        )
    end
end


function CSC(df::Matrix{Float64}, max_cluster::Int)
    # 1. CLR transform with pseudocount
    function clr_transform(data::Matrix{Float64}; pseudocount=1e-6)
        # Ensure all values are positive before the log transformation
        data = copy(data)
        if any(data .<= 0)
            println("Warning: Data contains zero or negative values before pseudocount adjustment.")
            println("Min value before adjustment: ", minimum(data))
        end

        data .+= pseudocount  # Add pseudocount
        if any(data .<= 0)
            println("Warning: Data still contains zero or negative values after pseudocount adjustment.")
            println("Min value after adjustment: ", minimum(data))
        end
        
        # Take the log and center the data
        log_data = log.(data)
        gm = mean(log_data, dims=2)
        
        # Check for any NaN values after log transformation
        if any(isnan, log_data)
            println("Warning: NaN values encountered after log transformation.")
            println("Row indices with NaNs: ", findall(isnan, log_data))
        end
        
        return log_data .- gm
    end

    df_clr = clr_transform(copy(df))
    df_clr_matrix = convert(Matrix{Float64}, df_clr)
    #print(df_clr_matrix)
    # 2. PCA (retain 95% variance)
    pca_model = MultivariateStats.fit(MultivariateStats.PCA, df_clr_matrix; method=:svd, mean=nothing, maxoutdim=size(df_clr_matrix, 2) )
    # 2. Compute explained variance ratio
    #explained = cumsum(pca_model.prinvars) / sum(pca_model.prinvars)
    explained = vec(cumsum(pca_model.prinvars)) / sum(pca_model.prinvars)

    n_components = findfirst(x -> x >= 0.95, explained)
    if n_components === nothing
        n_components = size(df_clr_matrix, 2)  # fallback: keep all components
        println("Warning: 95% variance not reached, using all ", n_components, " components.")
    end
    #pca_data = MultivariateStats.predict(pca_model, df_clr_matrix) #[:, 1:n_components]
    pca_data = projection(pca_model)
    #print(pca_data)

    # 3. GMM Clustering with BIC model selection
    best_bic = Inf
    best_model = nothing
    best_k = 0
    bics = Float64[]
    N = size(pca_data, 1)

    for k in 2:min(max_cluster, N)
        gmm = GMM(k, pca_data)
        
        loglikelihood = sum(llpg(gmm, pca_data))
        #loglikelihood = sum(logpdf(gmm, pca_data)) 
        d = size(pca_data, 2)
        num_params = k * (2 * d + 1)  # 2d for means and covariances, +1 for the mixture weights
        
        # Calculate BIC using the formula
        bic_val = log(N) * num_params - 2 * loglikelihood

        push!(bics, bic_val)
        if bic_val < best_bic
            best_bic = bic_val
            best_model = gmm
            best_k = k  
        end
    end

    # Check that best_model is valid before using it
    if best_model !== nothing
        resp, _ = gmmposterior(best_model, pca_data)
        state_labels = map(argmax, eachrow(resp))
        return state_labels
    else
        println("No valid GMM model found.")
        return []
    end

    
end


function CSC_umap(df::Matrix{Float64}, max_cluster::Int)
    
    # 1. UMAP dimensionality reduction
    df2 = permutedims(df)
    umap_data = umap(df2)
    print(size(df2))
    print(size(umap_data))

    # 3. GMM Clustering with BIC model selection
    best_bic = Inf
    best_model = nothing
    best_k = 0
    bics = Float64[]
    N = size(umap_data, 1)

    for k in 2:min(max_cluster, N)
        gmm = GMM(k, umap_data)
        
        loglikelihood = sum(llpg(gmm, umap_data))
        #loglikelihood = sum(logpdf(gmm, pca_data)) 
        d = size(umap_data, 2)
        num_params = k * (2 * d + 1)  # 2d for means and covariances, +1 for the mixture weights
        
        # Calculate BIC using the formula
        bic_val = log(N) * num_params - 2 * loglikelihood

        push!(bics, bic_val)
        if bic_val < best_bic
            best_bic = bic_val
            best_model = gmm
            best_k = k  
        end
    end

    # Check that best_model is valid before using it
    if best_model !== nothing
        resp, _ = gmmposterior(best_model, umap_data)
        state_labels = map(argmax, eachrow(resp))
        return state_labels
    else
        println("No valid GMM model found.")
        return []
    end

    
end

function first_index_of_consecutive_trues(bool_vector, min_length)
    count = 0
    for (i, val) in enumerate(bool_vector)
        if val
            count += 1
            if count == min_length
                return i - min_length + 1  # return starting index
            end
        else
            count = 0
        end
    end
    return nothing  # if no such sequence exists
end



function calculate_frustration_continuous(mat::AbstractMatrix{<:Real})
    n = size(mat, 1)
    triads = collect(combinations(1:n, 3))
    
    total = length(triads)
    frustrated = 0

    for triad in triads
        i, j, k = triad
        # Check all 3 directed cycles: i→j→k→i, j→k→i→j, k→i→j→k
        perms = [(i, j, k), (j, k, i), (k, i, j)]

        for (a, b, c) in perms
            M_ab = mat[a, b]
            M_bc = mat[b, c]
            M_ca = mat[c, a]

            if isnan(M_ab) || isnan(M_bc) || isnan(M_ca)
                continue
            end

            product = M_ab * M_bc * M_ca

            if product < 0
                frustrated += 1
                break  # Only count one frustration per triad
            end
        end
    end

    frustration_index = frustrated / total
    return Dict(
        "total_triads" => total,
        "frustrated_triads" => frustrated,
        "frustration_index" => frustration_index
    )
end


function calculate_weighted_frustration(mat::AbstractMatrix{<:Real}; verbose::Bool=true)
    n = size(mat, 1)
    triads = collect(combinations(1:n, 3))

    frustrated_sum = 0.0
    total_sum = 0.0
    frustrated_triads = String[]

    for triad in triads
        i, j, k = triad
        # All cyclic permutations of the triad
        perms = [(i, j, k), (j, k, i), (k, i, j)]

        for (a, b, c) in perms
            M_ab = mat[a, b]
            M_bc = mat[b, c]
            M_ca = mat[c, a]

            if isnan(M_ab) || isnan(M_bc) || isnan(M_ca)
                continue
            end

            product = M_ab * M_bc * M_ca
            total_sum += abs(product)

            if product < 0
                frustrated_sum += abs(product)
                if verbose
                    push!(frustrated_triads,
                          "Triad: $a→$b→$c→$a | Frustration: $(round(product, digits=3))")
                end
            end
        end
    end

    if verbose && !isempty(frustrated_triads)
        println("Frustrated triads:")
        println(join(frustrated_triads, "\n"))
        println()
    end

    index = isnan(total_sum) || total_sum == 0 ? NaN : frustrated_sum / total_sum

    return Dict(
        "total_weight" => total_sum,
        "frustrated_weight" => frustrated_sum,
        "weighted_frustration_index" => index
    )
end


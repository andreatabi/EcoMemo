using Random, Optim
using LinearAlgebra
using DataFrames
using JuMP
using GLPK
using HiGHS
using Distributions
using Observables
using StatsBase

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
    sigma = parms[:sigma] 
    r = body_size .^ (-0.25 .+ rand(Normal(0, sigma), length(n) ))   
    d = parms[:d] 
    q = parms[:q] 
    A = parms[:A]
    if any(n .< 0)
        n[n .< 0] .= 0
    end    
    # dndt = n .* (r .- (A * n)) 
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

function EntProd(ct, memo)
    m = unique(ct)
    pp = homc_embed(ct, memo+1)   
    #dropmissing!(pp)

    if length(pp) == 0
        return NaN
    else
        Tr = []
        for j in 1:memo
            com = unique(pp[:, [1, j + 1]], dims=1)
            Tr0 = zeros(length(m), length(m))
            for v in 1:size(com, 1)
                pairs = findall((com[v,1] .== pp[:,1]) .& (com[v,2] .== pp[:,j+1]))
                Tr0[findfirst(m .== com[v,2]), findfirst(m .== com[v,1])] = length(pairs)
            end

            Tr1 = mapslices(c -> sum(c) != 0 ? c ./ sum(c) : c, Tr0, dims=2)
            push!(Tr, Tr1)
        end

        cst_rev = reverse(ct)
        m_rev = unique(cst_rev)
        pp_rev = homc_embed(cst_rev, memo+1)
        #dropmissing!(pp_rev)

        Tr_rev = []
        for j in 1:memo
            com = unique(pp[:, [1, j + 1]], dims=1)
            Tr0 = zeros(length(m_rev), length(m_rev))

            for v in 1:size(com, 1)
                pairs = findall((com[v,1] .== pp_rev[:,1]) .& (com[v,2] .== pp_rev[:,j+1]))
                Tr0[findfirst(m_rev .== com[v,2]), findfirst(m_rev .== com[v,1])] = length(pairs)
            end

            Tr1 = mapslices(c -> sum(c) != 0 ? c ./ sum(c) : c, Tr0, dims=2)
            push!(Tr_rev, Tr1)
        end

        #H = Dict()
        H = []
        for u in 1:memo
            counts = countmap(ct)  
            total_count = sum(values(counts))  
            P0 = Dict(k => v / total_count for (k, v) in counts)
            P = [P0[c] for c in unique(ct)]  #
            Pij = Tr[u] .* reshape(repeat(reverse(P), length(unique(ct))), length(unique(ct)), :)
            #Pij = Pij[Pij .> 0]
            Pij = vec(Pij)

            Pji = Tr_rev[u] .* reshape(repeat(reverse(P), length(unique(ct))), length(unique(ct)), :)
            #Pji = Pji[Pji .> 0]
            Pji = vec(Pji)
            
            EP = vec( (Pij - Pji) .* log.(Pij ./ Pji) )
            EP = filter(isfinite, EP)
            #print(EP)

            push!(H, sum(EP) )
            #H[u] = sum((Pij - Pji) .* log.(Pij ./ Pji))
        end
    end

    return H
end

function gLV_OUP(t, n, parms)
    d = parms[:d] 
    q = parms[:q] 
    A = parms[:A] 
    cc = ceil(t / parms[:h]) + 1
   # print(cc)
    r = values(parms[:r][Int(cc),:])
    if any(n .< 0)
        n[n .< 0] .= 0
    end    
    #dndt = q .+ n .* (r .- (A * n)) .- d .* n 
    dndt =  n .* (r .- (A * n)) .- d .* n 
return dndt
end

# Example
#S = [1, 1, 2, 2, 1, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2]
#memo = 2
#res = HOMC(S, memo)
#EntProd(S,1)
#print(res["z"])

#h = 0.1
#r = r_oup
#parms = Dict( :A => A,
 #           :r => r_oup, :h => h)
#cc = 2
#r = parms[:r][cc, :]
#print(parms[:A][1,:])
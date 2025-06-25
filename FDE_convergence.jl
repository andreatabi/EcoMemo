using Plots
using Interpolations
using DataFrames

#include("/Users/Andrea/Dropbox/HOMC/julia/toolbox.jl")

function gLV(t, n, parms)    
    r = parms[:r] 
    A = parms[:A]
     dndt = n .* (r .- (A * n)) 
    return dndt
end


S = 5
y0 = fill(1.0, S)
β = fill(1.0, S)
A = [1.0000000  0.85868842 0.394451933  0.54998279 0.27420033;
    -1.3016048  1.00000000 0.091555780  0.14680490 0.04883888;
    -0.7599866 -0.12595646 1.000000000  0.05943078 0.00000000;
     0.9531117 -0.17249240 0.049319976  1.00000000 0.01741451;
     0.5983599  0.08701049 0.006325647 -0.03335709 1.00000000]
r = fill(1.0, S)
par = Dict(:A => A, :r => r)

h_fixed = 0.0001

# Increasing final times
Tvals = [10, 50, 100, 500]

# To store final states for each T
final_states = zeros(length(Tvals), S)
changed_flags = falses(length(Tvals), S)
ϵ = 1e-6

# Store results
equilibrium_flags = falses(length(Tvals), S)
final_vals = zeros(length(Tvals), S)

# Run simulations for increasing T
for (i, T) in enumerate(Tvals)
    tSpan = [0.0, T]
    t, Y = FDEsolver(gLV, tSpan, y0, β, par, h=h)
    final_vals[i, :] .= Y[end, :]

    # Compare last two time steps for each variable
    for j in 1:S
        Δ = abs(Y[end, j] - Y[end-1, j])
        if Δ < ϵ
            equilibrium_flags[i, j] = true
        end
    end
end

# Print equilibrium report
println("Equilibrium check (Δ < $ϵ):")
for i in 1:length(Tvals)
    println("At T = $(Tvals[i]):")
    for j in 1:S
        status = equilibrium_flags[i, j] ? "✓ equilibrium" : "✗ not yet"
        println("  State $j: $status (Δ = $(round(abs(final_vals[i,j] - (i > 1 ? final_vals[i-1,j] : NaN)), digits=4)))")
    end
end

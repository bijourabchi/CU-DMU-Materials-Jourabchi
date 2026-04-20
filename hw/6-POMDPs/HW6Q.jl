using POMDPs
using LinearAlgebra: dot
using QMDP
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, has_consistent_distributions, Uniform
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
# using SARSOP: SARSOPSolver
using NativeSARSOP: SARSOPSolver
using Statistics: mean, std
using ProgressMeter

##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    bp_vec = zeros(length(states(up.m)))
    
    for sp in states(up.m)
        bp_vec[stateindex(up.m,sp)] = Z(up.m,a,sp,o) * sum(s->T(up.m,s,a,sp) * beliefvec(b)[stateindex(up.m,s)], states(up.m))
    end
    
    # Handle case where sum is 0 (no observation match)
    total = sum(bp_vec)
    if total > 0
        bp_vec = bp_vec/total
    else
        # If no valid observation, return uniform belief
        bp_vec = ones(length(states(up.m))) / length(states(up.m))
    end
    
    return DiscreteBelief(up.m, bp_vec)
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.
m = TigerPOMDP()
up = HW6Updater(m)
b0 = POMDPs.initialize_belief(up,Uniform(states(m)))

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)

    # Fill in code to choose action based on alpha vectors
    return p.alpha_actions[argmax(a -> dot(p.alphas[a],beliefvec(b)), 1:length(p.alpha_actions))]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order


#------
# QMDP
#------

function qmdp_solve(m, discount=discount(m))

    # Fill in Value Iteration to compute the Q-values
    A = actions(m)
    R = reward_vectors(m)
    T = transition_matrices(m)

    S = states(m)
    
    tol = 1e-6
    n_states = length(S)
    n_actions = length(A) 


        # Value fxn initialization
    V = ones(n_states)
    V_prev = zeros(n_states)

    # State Value fxn
    Q = zeros(n_states,n_actions)

    while maximum(abs.(V-V_prev))  > tol

        V = copy(V_prev)

        for (i,a) in enumerate(A)
            # R -> [S, 1], T-> [S x S], V -> [S x 1]
            Q[:, i] = R[a] + discount * (T[a] * V)
        end

        V_prev = vec(maximum(Q, dims=2)) # Col wise max
 
    end

    acts = actiontype(m)[]
    
    alphas = Vector{Float64}[]

    for a in actions(m)

        # Fill in pseudo alpha vector calculation
        # Note that the ordering of the entries in the pseudo alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
        push!(acts,a)
        alpha = [Q[stateindex(m,s),actionindex(m,a)] for s in S]
        push!(alphas, alpha)
        
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

MC_qmdp = [simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000]
MC_sarsop = [simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000]
qmdp_mean = mean(MC_qmdp)
sarsop_mean = mean(MC_sarsop) 

@show qmdp_mean
@show std(MC_qmdp)/5000
@show sarsop_mean
@show std(MC_sarsop)/5000

solver = QMDPSolver(max_iterations=200,
                    belres=1e-6,
                    verbose=false
                   ) 

# run the solver
policy = solve(solver, m)

# @show qmdp_p.alphas
# @show policy.alphas

#-----------
# Plot Alpha Vectors
#-----------

using Plots

# For TigerPOMDP, belief is 2D but we can parameterize by b(TIGER_LEFT) ∈ [0,1]
# where b(TIGER_RIGHT) = 1 - b(TIGER_LEFT)

# Generate belief points to evaluate
n_belief_points = 100
b_left = range(0, 1, length=n_belief_points)

# Evaluate QMDP alpha vectors
qmdp_plot = plot(
    xlabel = "b(TIGER_LEFT)",
    ylabel = "Value",
    title = "QMDP Alpha Vectors",
    legend = :outertopright,
    lw = 2
)

for (i, alpha) in enumerate(qmdp_p.alphas)
    values = [alpha[1] * bl + alpha[2] * (1 - bl) for bl in b_left]
    action_label = "$(qmdp_p.alpha_actions[i])"
    plot!(qmdp_plot, b_left, values, label = action_label, marker = :circle, ms = 3)
end

# Evaluate SARSOP alpha vectors
sarsop_plot = plot(
    xlabel = "b(TIGER_LEFT)",
    ylabel = "Value",
    title = "SARSOP Alpha Vectors",
    legend = :outertopright,
    lw = 2
)

for (i, alpha) in enumerate(sarsop_p.alphas)
    values = [alpha[1] * bl + alpha[2] * (1 - bl) for bl in b_left]
    action_label = "$(sarsop_p.action_map[i])"
    plot!(sarsop_plot, b_left, values, label = action_label, marker = :circle, ms = 3)
end

# Combine into side-by-side layout
combined_plot = plot(qmdp_plot, sarsop_plot, layout = (1, 2), size = (1400, 500), margin = 10Plots.mm)
savefig(combined_plot, "alpha_vectors_comparison.png")

###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(
    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],
    transition = function (s, a)
    if s == :healthy
    return SparseCat([:healthy, :in_situ], [0.98, 0.02])
    elseif s == :in_situ
    if a == :treat
    return SparseCat([:healthy, :in_situ], [0.6, 0.4])
    else
    return SparseCat([:in_situ, :invasive], [0.9, 0.1])
    end
    elseif s == :invasive
    if a == :treat
    return SparseCat([:healthy, :death, :invasive],
    [0.2, 0.2, 0.6])
    else
    return SparseCat([:invasive, :death], [0.4, 0.6])
    end
    else
    return Deterministic(:death)
    end
    end,
    observation = function (a, sp)
    if a == :test
    if sp == :healthy
    return SparseCat([true, false], [0.05, 0.95])
    elseif sp == :in_situ
    return SparseCat([true, false], [0.8, 0.2])
    elseif sp == :invasive
    return Deterministic(true)
    end
    elseif a == :treat
    if sp in (:in_situ, :invasive)
    return Deterministic(true)
    end
    end
    return Deterministic(false)
    end,
    reward = function (s, a)
    if s == :death

    return 0.0
    elseif a == :wait
    return 1.0
    elseif a == :test
    return 0.8
    elseif a == :treat
    return 0.1
    end
    end,
    discount = 0.99,
    initialstate = Deterministic(:healthy),
    isterminal = s->s==:death,
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)

                               # Fill in your heuristic policy here
                               
                                qmdp_action = POMDPs.action(qmdp_p, b)
                                qmdp_value = maximum(a -> dot(qmdp_p.alphas[a], beliefvec(b)), 1:length(qmdp_p.alphas))
                                
                                values = [dot(qmdp_p.alphas[a], beliefvec(b)) for a in 1:length(qmdp_p.alphas)]
                                sort!(values, rev=true)
                                uncertainty = abs(values[1] - values[2])
    
                                # Test more when uncertainty is high
                                threshold = 0.5  # Tunable parameter
                                if uncertainty < threshold
                                    return :test  # Gather information when uncertain
                                else
                                    return qmdp_action  # Follow QMDP when confident
                                end
                           end
                          ) 


@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79
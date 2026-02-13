using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime

############
# Question 5
############

function rollout(mdp, policy_function, s0, max_steps=100)
    r_total = 0.0
    t = 0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s) # replace this with a policy
        s, r = @gen(:sp,:r)(mdp, s, a)
        r_total += discount(m)^t*r
        t += 1
    end
    return r_total # replace this with the reward
end



function smart_policy(m,s)
    x,y = s

    AS = [
        (:down, (x, y-1)),
        (:up, (x, y+1)),
        (:right, (x+1, y)),
        (:left, (x-1, y))
    ]


    # Find closest goal from CURRENT state
    goals = [[20,20], [20,40], [40,20], [40,40]]
    closest_goal_cell = goals[argmin([sqrt((x - g[1])^2 + (y - g[2])^2) for g in goals])]
    #@show s
    #@show closest_goal_cell
    
    # Find which next state is closest to that goal
    distances_to_goal = [sqrt((next_s[1] - closest_goal_cell[1])^2 + (next_s[2] - closest_goal_cell[2])^2) for (a,next_s) in AS]

    #@show distances_to_goal
    #@show AS[argmin(distances_to_goal)][1]
    return AS[argmin(distances_to_goal)][1]

end

function simulate_MCTS(π, s, d = 100)
    
    # Unpack
    m = π.m

    if d <= 0 # Base case

        return rollout(m, smart_policy, s) # U(s)
    end

    A = actions(m)
    γ = discount(m)
    
    # If we haven't checked the actions at state s yet, assign all dict values to 0
    if !haskey(n, (s, first(A))) 
        for a in A
            π.n[(s,a)] = 0
            π.q[(s,a)] = 0.0

        end
        # Value funciton estimate
        return rollout(m, smart_policy, s)
    end

    # Explore decision tree
    a = explore(π,s)
    sp, r = @gen(:sp,:r)(m, s, a)
    Q = r + γ * simulate_MCTS(π, sp, d-1)

    π.n[(s,a)] += 1
    π.q[(s,a)] += (Q-π.q[(s,a)])/π.n[(s,a)]
    tval = get(π.t, (s, a, sp), 0)
    π.t[(s,a,sp)] = tval + 1

    return Q
end

function explore(π, s)
    m = π.m
    n = π.n
    q = π.q
    c = π.c
    A = actions(m)

    bonus = (Nsa, Ns) -> Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)
    Ns = sum(n[(s,a)] for a in A)
    return argmax(a -> q[(s,a)] + c*bonus(n[(s,a)], Ns), A)

end

m = DenseGridWorld()

S = statetype(m)
A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()
c = 2

mutable struct MCTS
    m
    n
    q
    t
    c
end

π = MCTS(m,n,q,t,c)

# This is an example state - it is a StaticArrays.SVector{2, Int}
s = SA[19,19]

@assert s isa statetype(m)

for _ in 1:7
    simulate_MCTS(π, s)
end

inchrome(visualize_tree(π.q, π.n, π.t, s))

using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime

############
# Question 6
############


### rollout sim from state s0
function rollout(π, policy_function, s0, max_steps=100)
    r_total = 0.0
    t = 0
    s = s0
    while !isterminal(π.m, s) && t < max_steps
        a = policy_function(π, s) # replace this with a policy
        s, r = @gen(:sp,:r)(π.m, s, a)
        r_total += discount(π.m)^t*r
        t += 1
    end
    return r_total # replace this with the reward
end


### Policy for DenseGridWorld
function smart_policy(m,s)
    x,y = s

    AS = [
        (:down, (x, y-1)),
        (:up, (x, y+1)),
        (:right, (x+1, y)),
        (:left, (x-1, y))
    ]


    # Find closest goal from CURRENT state
    goals = [[20,20], [20,40], [20,60], [20,80], [40,20], [40,40], [40,60], [40,80],[60,20], [60,40], [60,60], [60,80],[80,20], [80,40], [80,60], [80,80]]
    closest_goal_cell = goals[argmin([(x - g[1])^2 + (y - g[2])^2 for g in goals])]
    #@show s
    #@show closest_goal_cell
    
    # Find which next state is closest to that goal
    distances_to_goal = [(next_s[1] - closest_goal_cell[1])^2 + (next_s[2] - closest_goal_cell[2])^2 for (a,next_s) in AS]

    #@show distances_to_goal
    #@show AS[argmin(distances_to_goal)][1]
    return AS[argmin(distances_to_goal)][1]

end

### main MCTS simulation
function simulate_MCTS(π, s, d = 15)
    
    # Unpack
    m = π.m

    if d <= 0 # Base case
        return rollout(π, smart_policy, s) # U(s)
    end

    A = actions(m)
    γ = discount(m)
    
    # If we haven't checked the actions at state s yet, assign all dict values to 0
    if !haskey(π.n, (s, first(A))) 
        for a in A
            π.n[(s,a)] = 0
            π.q[(s,a)] = 0.0

        end
        # Value funciton estimate
        return rollout(π, smart_policy, s)
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

### Explore decision tree for mcts
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

### Get an action based on MCTS tree
function select_action(m, s)
    S = statetype(m)
    A = actiontype(m)

    # These would be appropriate containers for your Q, N, and t dictionaries:
    n = Dict{Tuple{S, A}, Int}()
    q = Dict{Tuple{S, A}, Float64}()
    t = Dict{Tuple{S, A, S}, Int}()
    c = sqrt(2)

    π = MCTS(m,n,q,t,c)

    for k in  1:1000
        simulate_MCTS(π,s) # 1000 iterations to choose each action, d = 100 by default
    end
    return argmax(a -> π.q[(s,a)], actions(π.m))
end

m = DenseGridWorld()

mutable struct MCTS
    m
    n
    q
    t
    c
end

s0 = rand(initialstate(m))
a = select_action(m, s0)

@show s0
@show a

HW3.evaluate(select_action,"bijan.jourabchi@colorado.edu")
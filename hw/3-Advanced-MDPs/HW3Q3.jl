using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime

m = HW3.DenseGridWorld(seed = 3)

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

### Part a)
function heuristic_policy(m, s)
    # put a smarter heuristic policy here
    return rand(actions(m))
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

# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
# Part a)
n = 260

results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:n]

@show mean(results)
@show std(results)/sqrt(n)

# part b)
n = 260
results = [rollout(m, smart_policy, rand(initialstate(m))) for _ in 1:n]

@show mean(results)
@show std(results)/sqrt(n)

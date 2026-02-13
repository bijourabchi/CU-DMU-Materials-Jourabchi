using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW3 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

Please make sure to update DMUStudent to gain access to the HW3 module.

=#

############
# Question 3
############

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
n = 250

results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:n]

# @show mean(results)
# @show std(results)/sqrt(n)

# part b)
n = 250
results = [rollout(m, smart_policy, rand(initialstate(m))) for _ in 1:n]

# @show mean(results)
# @show std(results)/sqrt(n)

############
# Question 4
############

m = DenseGridWorld()

S = statetype(m)
A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

# This is an example state - it is a StaticArrays.SVector{2, Int}
s = SA[19,19]
@show typeof(s)
@assert s isa statetype(m)

# here is an example of how to visualize a dummy tree (q, n, and t should actually be filled in your mcts code, but for this we fill it manually)
q[(SA[1,1], :right)] = 0.0
q[(SA[2,1], :right)] = 0.0
n[(SA[1,1], :right)] = 1
n[(SA[2,1], :right)] = 0
t[(SA[1,1], :right, SA[2,1])] = 1

inchrome(visualize_tree(q, n, t, SA[1,1])) # use inbrowser(visualize_tree(q, n, t, SA[1,1]), "firefox") etc. if you want to use a different browser

############
# Question 5
############

# A starting point for the MCTS select_action function (a policy) which can be used for Questions 4 and 5
function select_action(m, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()


    for _ in 1:1000
    # while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
        break # replace this with mcts iterations to fill n and q
    end

    # select a good action based on q and/or n

    return rand(actions(m)) # this dummy function returns a random action, but you should return your selected action
end

@btime select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

# use the code below to evaluate the MCTS policy
@show results = [rollout(m, select_action, rand(initialstate(m)), max_steps=100) for _ in 1:100]

############
# Question 6
############

HW3.evaluate(select_action, "your.gradescope.email@colorado.edu")

# If you want to see roughly what's in the evaluate function (with the timing code removed), check sanitized_evaluate.jl

########
# Extras
########

# With a typical consumer operating system like Windows, OSX, or Linux, it is nearly impossible to ensure that your function *always* returns within 50ms. Do not worry if you get a few warnings about time exceeded.

# You may wish to call select_action once or twice before submitting it to evaluate to make sure that all parts of the function are precompiled.

# Instead of submitting a select_action function, you can alternatively submit a POMDPs.Solver object that will get 50ms of time to run solve(solver, m) to produce a POMDPs.Policy object that will be used for planning for each grid world.

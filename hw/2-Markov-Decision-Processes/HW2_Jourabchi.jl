import Cairo
import Fontconfig

using DMUStudent.HW2
using POMDPs: states, actions, discount
using POMDPTools: ordered_states, render

############
# Question 3
############


function value_iteration(m)

    # Problem Setup
    A = actions(m)
    T = transition_matrices(m, sparse = true)
    R = reward_vectors(m)
    S = states(m)
    gamma = discount(m)
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
            Q[:, i] = R[a] + gamma * (T[a] * V)
        end

        V_prev = vec(maximum(Q, dims=2)) # Col wise max
 
    end

    return V
end

@show actions(grid_world) 
V = value_iteration(grid_world)
render(grid_world, color = V)

#for V in V_hist
#    render(grid_world, color = V)
#end

############
# Question 4
############

m = UnresponsiveACASMDP(20)


V = value_iteration(m)
@show HW2.evaluate(V, "bijan.jourabchi@colorado.edu")
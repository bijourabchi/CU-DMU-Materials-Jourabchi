using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states, render

############
# Question 3
############

@show actions(grid_world) 

V = zeros(length(states(m)))

render(grid_world, color = V)
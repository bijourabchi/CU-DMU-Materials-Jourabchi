using POMDPs
using QMDP
using DMUStudent.HW6
using POMDPTools: DiscreteBelief, DiscreteUpdater
using POMDPTools.Policies: FunctionPolicy, AlphaVectorPolicy
using BasicPOMCP

m = LaserTagPOMDP()

qsolver = QMDPSolver(
    max_iterations=200,
    belres=1e-6,
    verbose=false
)

qmdp_policy = solve(qsolver, m)

function qmdp_state_action(m, qmdp_policy, s)
    si = stateindex(m, s)
    best_i = argmax(i -> qmdp_policy.alphas[i][si], eachindex(qmdp_policy.action_map))
    return qmdp_policy.action_map[best_i]
end

fo_qmdp_rollout = FunctionPolicy(s -> qmdp_state_action(m, qmdp_policy, s))

up = DiscreteUpdater(m)

function pomcp_solve(m, rollout_policy)
    solver = POMCPSolver(
        tree_queries=10000,
        max_depth=15,
        c=10.0,
        default_action=first(actions(m)),
        estimate_value=FORollout(rollout_policy)
    )
    return solve(solver, m)
end

pomcp_p = pomcp_solve(m, fo_qmdp_rollout)


#@show HW6.evaluate((pomcp_p,up), "bijan.jourabchi@colorado.edu")
@show HW6.evaluate((pomcp_p, up), n_episodes=100)
# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((qmdp_p, up), "REPLACE_WITH_YOUR_EMAIL@colorado.edu")

#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
#=using POMDPGifs
import Cairo, Fontconfig # needed to display properly

makegif(m, pomcp_p, up, max_steps=30, filename="lasertag.gif")

# You can render a single frame like this
using POMDPTools: stepthrough, render
using Compose: draw, PNG

history = []
for step in stepthrough(m, pomcp_p, up, max_steps=10)
    push!(history, step)
end
displayable_object = render(m, last(history))
# display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
draw(PNG("lasertag.png"), displayable_object) =#

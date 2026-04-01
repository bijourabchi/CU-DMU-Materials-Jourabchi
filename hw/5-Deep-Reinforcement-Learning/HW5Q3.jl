using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
using CommonRLInterface
using Flux
using Random
using ProgressMeter
using CommonRLInterface.Wrappers: QuickWrapper
using ElectronDisplay
ElectronDisplay.CONFIG.single_window = true
import POMDPs


env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2])

function loss(Q, Q_target, s, a_ind, r, sp, done)
    if ! done
        return (r + 0.99*maximum(Q_target(sp)) - Q(s)[a_ind])^2
    else return (r-Q(s)[a_ind])^2 end
end

function policy(env,s,eps,Q)
    if rand() < eps
        return rand(1:length(actions(env)))
    else
        
        return argmax(i -> Q(s)[i], 1:length(actions(env)))
    end
end

function simulate(env,Q)
    reset!(env)
    r_tot = 0
    done = false
    count = 0
    eps = 0
    MAX_STEPS = 1000

    while !done && count < MAX_STEPS
        s = observe(env)
        a_ind = policy(env,s,eps, Q)
        r = act!(env, actions(env)[a_ind])
        sp = observe(env)
        done = terminated(env)
        r_tot += (0.99^count) * r
        s = sp
        count += 1
    end
    return r_tot / count
end

function dqn(env)
    Q = Chain(Dense(2,128), relu,
            Dense(128, length(actions(env)))) # Tune as needed
    opt = Flux.setup(ADAM(0.0005), Q)

    Q_target = deepcopy(Q)
    N_EPSISODES = 2000
    MAX_STEPS = 10000
    WARMUP_STEPS = 2000 # Idea is to let it explore for a while to gain experience_tuples before training to ensure NN has data
    REPLAY_CAPACITY = 200_000
    TRAIN_SAMPLES_PER_EPISODE = 2_000

    buffer = []
    learning_curve = Float64[]
    

    @showprogress for i = 1:N_EPSISODES
        
        reset!(env)
        count = 0
        eps = max(0.01, 1-i/N_EPSISODES)
        done = false

        while !done && count < MAX_STEPS
            # Create our experience tuple
            s = observe(env)
            a_ind = policy(env,s,eps,Q)
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)

            experience_tuple = (s, a_ind, r, sp, done)
            push!(buffer, experience_tuple)
            if length(buffer) > REPLAY_CAPACITY
                popfirst!(buffer) # drop oldest transition
            end
             
            count += 1
        end
        
        ## Save deepcopy only if Q policy is better than prev Q
        r_now = simulate(env,Q)
        
        push!(learning_curve, r_now)
        r_trial = simulate(env,Q_target)

        if r_now > r_trial # This will ensure Q_target is always a better Q than the previous one
            
            Q_target = deepcopy(Q)

        end
        
        Q_target = deepcopy(Q)

        if length(buffer) >= WARMUP_STEPS
            n_updates = min(TRAIN_SAMPLES_PER_EPISODE, length(buffer))
            for data in rand(buffer, n_updates)
                # this runs a forward and backward pass to calculate the loss and gradient
                loss_value, grads = Flux.withgradient(loss, Q, Q_target, data...)

                # this will take a gradient step
                Flux.update!(opt, Q, grads[1])
            end
        end

        ## Stuff for learning curve
    end
    return Q, learning_curve
end

Q, learning_curve = dqn(env)

scr = HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=10_000).score
println("\n score = $scr")
if scr > 40
    HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "bijan.jourabchi@colorado.edu") # you will need to remove the n_episodes=100 keyword argument and add your email as a positional argument to create a json file; evaluate needs to run 10_000 episodes to produce a json
end
#----------
# Rendering
#----------

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
ElectronDisplay.display(render(env))

# The following code allows you to render the value function
using Plots
lc_plot = plot(
    1:length(learning_curve),
    learning_curve,
    xlabel="Episode",
    ylabel="Average Return",
    title="DQN Learning Curve",
    label="Episode Return"
)
savefig(lc_plot, "DQN_LC.png")
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")

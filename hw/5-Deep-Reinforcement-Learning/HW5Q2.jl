using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator, stepthrough, RandomPolicy
using Statistics: mean
using POMDPs: simulate
import POMDPs

cancer = QuickPOMDP(
        states = ["Healthy","ISC", "IC", "Death"],
        actions = ["wait","test","treat"],
        observations = ["positive","negative"],
        initialstate = Deterministic("Healthy"),
        discount = 0.99,

        transition = function(s,a)
            if s == "Healthy"
                return SparseCat(["Healthy", "ISC"], [1-0.02, 0.02])
            elseif a == "treat" && s == "ISC"
                return SparseCat(["Healthy", "ISC"],[0.6,0.4])
            elseif a != "treat" && s == "ISC"
                return SparseCat(["IC","ISC"],[0.1,0.9])
            elseif a == "treat" && s == "IC"
                return SparseCat(["IC","Healthy","Death"],[0.6,0.2,0.2])
            elseif a != "treat" && s == "IC"
                return SparseCat(["IC","Death"],[0.4,0.6])
            elseif s == "Death"
                return Deterministic("Death")
            end
        end,

        observation = function(s,a,sp)
            if a == "test"
                if sp == "Healthy"
                    return SparseCat(["positive","negative"],[0.05,0.95])
                elseif sp == "ISC"
                    return SparseCat(["positive","negative"],[0.8,0.2])
                elseif sp == "IC"
                    return SparseCat(["positive","negative"],[1.0,0])
                end
            end
            if a == "treat"
                if sp == "ISC" || sp == "IC"
                    return SparseCat(["positive","negative"],[1.0,0])
                end
            end

            return SparseCat(["positive","negative"],[0,1.0])
        end,

        reward = function(s,a)
            if s == "Death"
                return 0
            end
            if a == "wait" return 1.0 end
            if a == "test" return 0.8 end
            if a == "treat" return 0.1 end
        end
)



const policy = FunctionPolicy(o->"wait")
sim = RolloutSimulator(max_steps=100)
@show mean(POMDPs.simulate(sim, cancer, policy) for _ in 1:10_000)


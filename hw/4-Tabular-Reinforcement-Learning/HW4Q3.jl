using Flux
using Flux.Losses: kldivergence
using Plots
using Statistics: mean
using Plots.PlotMeasures: mm
using ProgressMeter
using Random

f = x -> (1 - x) * sin(20 * log(x + 0.2))
#f = x -> sin(x)

function make_training_data(f)
    x = collect(range(0,1,length=1000))
    y = f.(x)
    return x, y
end

function lossfxn(m, x, y)
    return kldivergence(m(x), y)
end

function train(x_data, y_data; learning_rate=1e-3, n_epochs=1_000, save_every=50, minibatch_size=1)
    x_data = ndims(x_data) == 1 ? reshape(Float32.(x_data), 1, :) : Float32.(x_data)
    y_data = ndims(y_data) == 1 ? reshape(Float32.(y_data), 1, :) : Float32.(y_data)
    @assert size(x_data, 2) == size(y_data, 2) "x_data and y_data must have the same number of samples"
    @assert minibatch_size >= 1 "minibatch_size must be >= 1"

    model = Chain(
        Dense(1=>32, tanh),
        Dense(32=>64, tanh),
        Dense(64=>32, tanh),
        Dense(32=>1)
    )
    opt_state = Flux.setup(Adam(learning_rate), model)

    losses = Float32[]
    models = [deepcopy(model)]


    n_samples = size(y_data, 2)
    n_minibatches = cld(n_samples, minibatch_size)

    @showprogress for epoch in 1:n_epochs
        batch_loss = zero(Float32)
        perm = randperm(n_samples)

        for i in 1:n_minibatches
            start_idx = (i - 1) * minibatch_size + 1
            end_idx = min(i * minibatch_size, n_samples)
            idxs = perm[start_idx:end_idx]
            x_minibatch = x_data[:, idxs]
            y_minibatch = y_data[:, idxs]

            function minibatch_objective(model)
                return lossfxn(model, x_minibatch, y_minibatch)
            end

            minibatch_loss, grads = Flux.withgradient(minibatch_objective, model)

            Flux.update!(opt_state, model, grads[1])

            batch_loss += minibatch_loss / n_minibatches
        end

        push!(losses, batch_loss)

        if epoch % save_every == 0
            push!(models, deepcopy(model))
        end
    end

    return models, losses
end

x, y = make_training_data(f)

training_plot = plot(
    vec(x), vec(y),
    label="Training Data",
    xlabel="x",
    ylabel="f(x)",
    title="Training Set",
    linewidth=1.8,
    marker=:circle,
    markersize=2.5,
    color=:navy,
    legend=:topright,
    gridalpha=0.25
)
savefig(training_plot, "training_set.png")

models, losses = train(x, y; learning_rate=1e-3, n_epochs=3_000, minibatch_size=5)

xs = collect(range(0.0f0, 1.0f0, length=100))
truth_vals = f.(xs)
pred_vals = [models[end](Float32[x])[1] for x in xs]

p1 = plot(
    xs, truth_vals,
    label="Truth",
    xlabel="x",
    ylabel="f(x)",
    title="Truth vs NN Approximation",
    linewidth=3,
    color=:steelblue,
    legend=:topleft,
    gridalpha=0.25,
    left_margin=8mm,
    right_margin=4mm,
    top_margin=6mm,
    bottom_margin=8mm
)
scatter!(
    p1, xs, pred_vals,
    label="NN Model",
    color=:tomato,
    markersize=3,
    markerstrokewidth=0,
    alpha=0.85
)

p2 = plot(
    1:length(losses), losses,
    label="Training Loss",
    xlabel="Epoch",
    ylabel="Loss",
    title="Loss vs Number of Epochs",
    linewidth=2.5,
    marker=:circle,
    markersize=3,
    color=:darkgreen,
    legend=:topright,
    gridalpha=0.25,
    left_margin=8mm,
    right_margin=4mm,
    top_margin=6mm,
    bottom_margin=8mm
)

P = plot(
    p1, p2,
    layout=(1, 2),
    size=(1300, 480),
    dpi=150
)

savefig("NN_Fxn_Approx.png") 

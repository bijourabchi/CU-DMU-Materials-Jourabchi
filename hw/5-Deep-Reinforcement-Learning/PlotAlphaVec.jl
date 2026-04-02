using Plots

# One Step α vector
F = 0
T = 1

x = [F,T]
y1 = [6,5]

y2 = [7, 2]

one_step = plot(
    x,y1,
    yticks=0:1:8,
    xlabel = "b(T)",
    ylabel = "Expected Reward",
    lw=3,
    color=:blue,
    marker=:circle,
    ms=8,
    label = "Action = D",
    xlims = (-0.1,1.5),
    ylims = (0,8),
    legend = :topright
)

plot!(
    x,y2,
    lw=3,
    color=:red,
    marker=:circle,
    ms=8,
    label = "Action = I",

)

plot!([F,F], [0,8], color = :black, lw=2,label = "b(T) = 0")
plot!([T,T], [0,8], color = :black, lw=2,label = "b(T) = 1")

savefig(one_step, "one_step.png")

using Plots

# One Step α vector
F = 0
T = 1

x = [F,T]
y1 = [6,5]

y2 = [7, 2]
y3 = [8, 4]

two_step = plot(
    x,y1,
    yticks=0:1:14,
    xlabel = "b(T)",
    ylabel = "Expected Reward",
    lw=3,
    color=:blue,
    marker=:circle,
    ms=8,
    label = "Action = D",
    xlims = (-0.1,1.5),
    ylims = (0,14),
    legend = :topright
)

plot!(
    x,y2,
    lw=3,
    color=:red,
    marker=:circle,
    ms=8,
    label = "Action = I",

)

plot!(
    x,y3,
    lw=3,
    color=:purple,
    marker=:circle,
    ms=8,
    label = "Two Step Plan",

)

plot!([F,F], [0,14], color = :black, lw=2,label = "b(T) = 0")
plot!([T,T], [0,14], color = :black, lw=2,label = "b(T) = 1")
savefig(two_step, "two_step.png")
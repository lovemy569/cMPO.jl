using cMPO
using Printf
using JLD2
using Plots

# 定义数据集
datasets = [
    (delta=0.0, filepath="data/delta_0.0_bond_20_beta_20.0.jld2"),
    (delta=1.0, filepath="data/delta_1.0_bond_20_beta_20.0.jld2"),
    (delta=2.0, filepath="data/delta_2.0_bond_20_beta_20.0.jld2")
]

# 定义颜色和形状
colors = [:red, :blue, :green]
shapes = [:circle, :rect, :diamond]

function plot_fig_2(datasets, cv, corr, chi, spectral)
    p1 = plot(xlabel="Temperature", ylabel="specific heat", xlims=(0.0, 2.0), ylims=(0, 0.41), foreground_color_legend = nothing, background_color_legend = nothing)
    p2 = plot(xlabel="τ", ylabel="χ(τ)", xlims=(-1.0, 21), ylims=(0, 0.26), legend=:top, foreground_color_legend = nothing, background_color_legend = nothing)
    p3 = plot(xscale=:log10, xlabel="Temperature", ylabel="Tχloc", ylims=(0, 0.26), foreground_color_legend = nothing, background_color_legend = nothing)
    p4 = plot(xlabel="ω", ylabel="S(ω)", xlims=(0.0, 3.0), ylims=(0, 2.1), foreground_color_legend = nothing, background_color_legend = nothing)

    for (i, dataset) in enumerate(datasets)
        delta, filepath = dataset.delta, dataset.filepath
        @load filepath psi Lpsi

        model = XXZModel(delta, 1.0)
        T = model.T
        s = PauliSpin()

        # specific heat
        temperatures = range(0.1, stop=2.0, length=100)
        cv_values = []

        for TT in temperatures
            beta = 1.0 / TT
            cv_value = cv(psi, Lpsi, T, beta)
            push!(cv_values, cv_value)
        end

        scatter!(p1, temperatures, cv_values, markersize = 3, markercolor=:white, markershape=shapes[i],
                 markerstrokecolor=colors[i], markerstrokewidth=1.5, markerfillcolor=:transparent,
                 label="Δ = $delta")
        plot!(p1, temperatures, cv_values, linecolor=colors[i], linestyle=:dash, linewidth=1.5, label=false)

        # correlation function at β = 20
        Tau = range(0.0, stop=20.0, length=100)
        corr_values = []

        for tau in Tau
            corr_value = corr(psi, Lpsi, T, s.Z/2, s.Z/2, 20.0, tau)
            push!(corr_values, corr_value)
        end

        scatter!(p2, Tau, corr_values, markersize = 3, markercolor=:white, markershape=shapes[i],
                 markerstrokecolor=colors[i], markerstrokewidth=1.5, markerfillcolor=:transparent,
                 label="Δ = $delta")
        plot!(p2, Tau, corr_values, linecolor=colors[i], linestyle=:dash, linewidth=1.5, label=false)

        # local susceptibility at ω = 0.0
        temperatures = 10 .^ range(-2, stop=1, length=50)
        chi_values = []

        for TT in temperatures
            beta = 1.0 / TT
            chi_value = chi(psi, Lpsi, T, s.Z/2, s.Z/2, beta, 0.0) / beta
            push!(chi_values, chi_value)
        end

        scatter!(p3, temperatures, chi_values, markersize = 3, markercolor=:white, markershape=shapes[i],
                 markerstrokecolor=colors[i], markerstrokewidth=1.5, markerfillcolor=:transparent,
                 label="Δ = $delta")
        plot!(p3, temperatures, chi_values, xscale=:log10, linecolor=colors[i], linestyle=:dash, linewidth=1.5, label=false)

        # spectral function with eta = 0.05
        omegas = range(0.0, stop=3.0, length=200)
        spectral_values = []

        for omega in omegas
            spectral_value = spectral(psi, Lpsi, T, s.Z/2, s.Z/2, 20.0, omega, 0.05)
            push!(spectral_values, spectral_value)
        end

        scatter!(p4, omegas, spectral_values, markersize = 3, markercolor=:white, markershape=shapes[i],
                 markerstrokecolor=colors[i], markerstrokewidth=1.5, markerfillcolor=:transparent,
                 label="Δ = $delta")
        plot!(p4, omegas, spectral_values, linecolor=colors[i], linestyle=:dash, linewidth=1.0, label=false)
    end

    # 将四个图形组合到一个 subplot 中
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), plot_title="bond dimension 20", titlefont=font("times new roman", 16))
    savefig(final_plot, joinpath("data", "bond20_reproduce_fig_2.png"))
    display(final_plot)
end

plot_fig_2(datasets, cv, corr, chi, spectral)

##################################################################################################

function plot_fig_S3(psi, Lpsi, T, s, beta, eta1, eta2, omega_range)
    @load "data/delta_0.0_bond_20_beta_20.0.jld2" psi Lpsi
    spectral_values5 = []
    spectral_values1 = []

    for omega in omega_range
        spectral_value5 = spectral(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega, eta1)
        push!(spectral_values5, spectral_value5)
    end

    for omega in omega_range
        spectral_value1 = spectral(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega, eta2)
        push!(spectral_values1, spectral_value1)
    end

    scatter(omega_range, spectral_values1, xscale=:log10, xlabel="ω", ylabel="S(ω)",
            markercolor=:white, markershape=:rect, markerstrokecolor=:blue, markerstrokewidth=1.5,
            markerfillcolor=:transparent, label="η = $eta2", ylims=(-0.5, 2.5))
    plot!(omega_range, spectral_values1, linecolor=:blue, linestyle=:dash, linewidth=1.0, label=false)

    scatter!(omega_range, spectral_values5, xscale=:log10, xlabel="ω", ylabel="S(ω)",
            markercolor=:white, markershape=:circle, markerstrokecolor=:red, markerstrokewidth=1.5,
            markerfillcolor=:transparent, label="η = $eta1", ylims=(-0.5, 2.5))
    plot!(omega_range, spectral_values5, linecolor=:red, linestyle=:dash, linewidth=1.0, label=false)

    savefig(current(), joinpath("data", "bond20_reproduce_fig_s3.png"))
    display(current())
end

# spectral function with η = 0.05 or 0.1
omegas = range(0.01, stop=10, length=500)
plot_fig_S3(psi, Lpsi, T, s, 20.0, 0.05, 0.1, omegas)

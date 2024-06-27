using cMPO
using Printf
using JLD2
using Plots

# 导入参数
bondD = 10
beta = 20.0
delta = 0.0
s = PauliSpin()
model = XXZModel(delta, 1.0)
T = model.T

# 生成文件名
filename = @sprintf "delta_%.1f_bond_%d_beta_%.1f.jld2" delta bondD beta

# 寻找存储路径
directory = joinpath("data")
filepath = joinpath(directory, filename)

# 保存变量到指定路径
@load filepath psi Lpsi

##################################################################################################

function plot_fig_2(psi, Lpsi, T, s, cv, corr, chi, spectral)
    # specific heat
    temperatures = range(0.1, stop=2.0, length=100)
    cv_values = []

    for TT in temperatures
        beta = 1.0 / TT
        cv_value = cv(psi, Lpsi, T, beta)
        push!(cv_values, cv_value)
    end

    p1 = scatter(temperatures, cv_values, xlabel="Temperature", ylabel="specific heat",
                 markercolor=:white, markershape=:circle, markerstrokecolor=:red, markerstrokewidth=1.5,
                 markerfillcolor=:transparent, label="Δ = 0", xlims=(0.0, 2.0), ylims=(0, Inf))
    plot!(p1, temperatures, cv_values, linecolor=:black, linestyle=:dash, linewidth=1.5, label=false)

    # correlation function at β = 20
    Tau = range(0.0, stop=20.0, length=100)
    corr_values = []

    for tau in Tau
        corr_value = corr(psi, Lpsi, T, s.Z/2, s.Z/2, 20.0, tau)
        push!(corr_values, corr_value)
    end

    p2 = scatter(Tau, corr_values, xlabel="τ", ylabel="χ(τ)",
                 markercolor=:white, markershape=:circle, markerstrokecolor=:red, markerstrokewidth=1.5,
                 markerfillcolor=:transparent, label="Δ = 0", xlims=(-1.0, 21), ylims=(0, 0.26))
    plot!(p2, Tau, corr_values, linecolor=:black, linestyle=:dash, linewidth=1.5, label=false)

    # local susceptibility at ω = 0.0
    temperatures = 10 .^ range(-2, stop=1, length=100)
    chi_values = []

    for TT in temperatures
        beta = 1.0 / TT
        chi_value = chi(psi, Lpsi, T, s.Z/2, s.Z/2, beta, 0.0) / beta
        push!(chi_values, chi_value)
    end

    p3 = scatter(temperatures, chi_values, xscale=:log10, xlabel="Temperature", ylabel="Tχloc",
                 markercolor=:white, markershape=:circle, markerstrokecolor=:red, markerstrokewidth=1.5,
                 markerfillcolor=:transparent, label="Δ = 0", ylims=(0, 0.26))
    plot!(p3, temperatures, chi_values, xscale=:log10, linecolor=:black, linestyle=:dash, linewidth=1.5, label=false)

    # spectral function with eta = 0.05
    omegas = range(0.0, stop=3.0, length=200)
    spectral_values = []

    for omega in omegas
        spectral_value = spectral(psi, Lpsi, T, s.Z/2, s.Z/2, 20.0, omega, 0.05)
        push!(spectral_values, spectral_value)
    end

    p4 = scatter(omegas, spectral_values, xlabel="ω", ylabel="S(ω)",
                 markercolor=:white, markershape=:circle, markerstrokecolor=:red, markerstrokewidth=1.5,
                 markerfillcolor=:transparent, label="Δ = 0", xlims=(0.0, 3.0), ylims=(0, 2.5))
    plot!(p4, omegas, spectral_values, linecolor=:black, linestyle=:dash, linewidth=1.0, label=false)

    # 将四个图形组合到一个 subplot 中
    plot(p1, p2, p3, p4, layout=(2, 2))
end

plot_fig_2(psi, Lpsi, T, s, cv, corr, chi, spectral)

##################################################################################################

function plot_fig_S3(psi, Lpsi, T, s, beta, eta1, eta2, omega_range)
    
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

    scatter(omega_range, spectral_values5, xscale=:log10, xlabel="ω", ylabel="S(ω)",
            markercolor=:white, markershape=:circle, markerstrokecolor=:red, markerstrokewidth=1.5,
            markerfillcolor=:transparent, label="η = $eta1", ylims=(0, 2.5))
    plot!(omega_range, spectral_values5, linecolor=:red, linestyle=:dash, linewidth=1.0, label=false)

    scatter!(omega_range, spectral_values1, xscale=:log10, xlabel="ω", ylabel="S(ω)",
            markercolor=:white, markershape=:rect, markerstrokecolor=:blue, markerstrokewidth=1.5,
            markerfillcolor=:transparent, label="η = $eta2", ylims=(0, 2.5))
    plot!(omega_range, spectral_values1, linecolor=:blue, linestyle=:dash, linewidth=1.0, label=false)

    display(current())
end

# spectral function with η = 0.05 or 0.1
omegas = range(0.01, stop=10, length=200)
plot_fig_S3(psi, Lpsi, T, s, 20.0, 0.05, 0.1, omegas)






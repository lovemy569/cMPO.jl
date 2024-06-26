using LinearAlgebra
using Printf
using Plots
using JLD2
using cMPO

bondD = 15
beta = 20.0
model1 = XXZModel(1.0, 1.0)

T = model1.T
W = model1.W
d = model1.d
ph_leg = model1.ph_leg

psi = CMPS(T.Q, T.L)
Lpsi = CMPS(T.Q, T.R)

init_step = Int(floor(log(bondD) / log(ph_leg)))
for ix in 1:init_step - 1
    psi = act(T, psi)
    Lpsi = act(transpose(T), psi)
end

power_counter = 0
step = 0
Fmin = 9.9e9
omega = 1.0
eta = 0.05
s = PauliSpin()

while power_counter < 3
    if size(psi.Q, 1) <= bondD
        Tpsi = act(T, psi)
    else
        Tpsi = psi
    end
    
    # 微分优化
    psi = variational_compr(Tpsi, beta, bondD)
    Lpsi = multiply(W, psi)

    # 可观测量
    F_value = F(psi, Lpsi, T, beta)
    Cv_value = Cv(psi, Lpsi, T, beta)
    chi_value = chi(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega) / beta
    chi2_value = chi2(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega, eta)
    spectral_value = spectral(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega, eta)

    # 打印输出
    @printf("%d %.12f %.12f %.12f %.12f %.12f\n", step, F_value, Cv_value, chi_value, chi2_value, spectral_value)

    step += 1
    
    # 连续三次达到优化阈值输出
    if F_value < Fmin - 1e-11
        power_counter = 0
        Fmin = F_value
    else
        power_counter += 1
    end
end

# filename = @sprintf "1bondD_%d_beta_%.1f_omega_%.1f_eta_%.1f.jld2" bondD beta omega eta
# @save filename psi Lpsi



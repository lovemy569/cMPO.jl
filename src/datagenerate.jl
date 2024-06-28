using LinearAlgebra
using Printf
using JLD2
using cMPO

bondD = 15
beta = 20.0
delta = 0.0
model = XXZModel(delta, 1.0)

T = model.T
W = model.W
d = model.d
ph_leg = model.ph_leg

psi = CMPS(T.Q, T.L)
Lpsi = CMPS(T.Q, T.R)

init_step = Int(floor(log(bondD) / log(ph_leg)))
for ix in 1:init_step - 1
    psi = act(T, psi)
    Lpsi = act(transpose(T), psi)
end

power_counter = 0
step = 0
fmin = 9.9e9
omega = 1.0
eta = 0.05
tau = 10.0
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
    f_value = f(psi, Lpsi, T, beta)
    cv_value = cv(psi, Lpsi, T, beta)
    corr_value = corr(psi, Lpsi, T, s.Z/2, s.Z/2, beta, tau)
    chi_value = chi(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega) / beta
    chi2_value = chi2(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega, eta)
    spectral_value = spectral(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega, eta)

    # 打印输出
    @printf("%d %.9f %.9f %.9f %.9f %.9f %.9f\n", step, f_value, cv_value, corr_value, chi_value, chi2_value, spectral_value)

    step += 1
    
    # 连续三次达到优化阈值输出
    if f_value < fmin - 1e-8
        power_counter = 0
        fmin = f_value
    else
        power_counter += 1
    end
end

# 生成文件名
filename = @sprintf "delta_%.1f_bond_%d_beta_%.1f.jld2" delta bondD beta

# 指定存储路径
directory = joinpath("data")
filepath = joinpath(directory, filename)

# 创建目录（如果尚不存在）
mkpath(directory)

# 保存变量到指定路径
@save filepath psi Lpsi





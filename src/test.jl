using LinearAlgebra
using Zygote
using Printf
using Plots
# using OMEinsum

include("cMPO.jl")
using .cMPO

bondD = 10
beta = 20.0
model1 = cMPO.cmpoaction.XXZModel(0.0, 1.0)

T = model1.T
W = model1.W
d = model1.d
ph_leg = model1.ph_leg

psi = cMPO.cmpoaction.cmps(T.Q, T.L)
Lpsi = cMPO.cmpoaction.cmps(T.Q, T.R)


# Do = size(T.Q, 1)
# Ds = size(psi.Q, 1)
# d = size(psi.R, 1)
# Io = I(Do)
# Is = I(Ds)
# Is = I(Ds)
# Is_reshape = reshape(Is, 1 , Ds, Ds)
# # repeat(Is_reshape, d, 1, 1)[1, :, :]
# # repeat(Is_reshape, d, 1, 1)[2, :, :]
# # repeat(Is_reshape, d, 1, 1)[3, :, :]

# z = ein"mab,mcd->mcadb"(T.L, repeat(Is_reshape, d, 1, 1))
# reshape(z, d, Do*Ds, Do*Ds)[1, :, :]
# reshape(z, d, Do*Ds, Do*Ds)[2, :, :]
# reshape(z, d, Do*Ds, Do*Ds)[3, :, :]

# # Checkpoint
# psi.Q
# psi.R[1, :, :]
# psi.R[2, :, :]
# psi.R[3, :, :]
# Lpsi.Q
# Lpsi.R[1, :, :]
# Lpsi.R[2, :, :]
# Lpsi.R[3, :, :]

init_step = Int(floor(log(bondD) / log(ph_leg)))
for ix in 1:init_step - 1
    psi = cMPO.cmpoaction.act(T, psi)
    Lpsi = cMPO.cmpoaction.act(cMPO.cmpoaction.transpose(T), psi)
end

# psi.Q
# psi.R[1, :, :]
# psi.R[2, :, :]
# psi.R[3, :, :]
# Lpsi.Q
# Lpsi.R[1, :, :]
# Lpsi.R[2, :, :]
# Lpsi.R[3, :, :]


# psi = cMPO.cmpoaction.diagQ(psi)
# Lpsi = cMPO.cmpoaction.diagQ(Lpsi)
# eigen(psi.Q)

# _, U = cMPO.cmpoaction.eigensolver(psi.Q)

# # Checkpoint
# psi.Q
# psi.R[1, :, :]
# psi.R[2, :, :]
# psi.R[3, :, :]
# Lpsi.Q
# Lpsi.R[1, :, :]
# Lpsi.R[2, :, :]
# Lpsi.R[3, :, :]

power_counter = 0
step = 0
Fmin = 9.9e9
omega = 1.0
s = cMPO.cmpoaction.PauliSpin()
# chkp_loc = "checkpoint.jld2"

while power_counter < 3
    if size(psi.Q, 1) <= bondD
        Tpsi = cMPO.cmpoaction.act(T, psi)
    else
        Tpsi = psi
    end
    
    psi = cMPO.cmpoaction.variational_compr(Tpsi, beta, bondD)
    Lpsi = cMPO.cmpoaction.multiply(W, psi)

    F_value = cMPO.cmpoaction.F(psi, Lpsi, T, beta)
    Cv_value = cMPO.cmpoaction.Cv(psi, Lpsi, T, beta)
    chi_value = cMPO.cmpoaction.chi(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega) / beta
    chi2_value = cMPO.cmpoaction.chi2(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega)
    spectral_value = cMPO.cmpoaction.spectral(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega)

    # 打印输出
    @printf("%d %.12f %.12f %.12f %.12f %.12f\n", step, F_value, Cv_value, chi_value, chi2_value, spectral_value)

    step += 1
    
    if F_value < Fmin - 1e-11
        power_counter = 0
        # power_counter += 1
        Fmin = F_value
    else
        power_counter += 1
    end
end

# 优化结束后，计算并绘制图像
omegas = 10.0 .^ range(-2, stop=1, length=100)
spectral_values = []

for omega in omegas
    spectral_value = cMPO.cmpoaction.spectral(psi, Lpsi, T, s.Z/2, s.Z/2, beta, omega)
    push!(spectral_values, spectral_value)
end

# 绘制图像
plot(omegas, spectral_values, xscale=:log10, xlabel="omega", ylabel="spectral value", title="Spectral function vs omega")




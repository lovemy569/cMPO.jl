using JLD2
using cMPO
using Plots

bondD = 10
beta = 20.0
delta = 0.0

# 生成文件名
filename = @sprintf "delta_%.1f_bond_%d_beta_%.1f.jld2" delta bondD beta

# 寻找存储路径
directory = joinpath("data")
filepath = joinpath(directory, filename)

# 保存变量到指定路径
@load filepath psi Lpsi

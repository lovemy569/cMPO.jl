module cMPO
__precompile__()

using LinearAlgebra
using Zygote
using Optim
using OMEinsum
using Printf
using ChainRulesCore
using ChainRules
using Parameters
using SparseArrays
import Base: transpose

include("cMPOAction.jl")

export CMPO, CMPS, XXZModel, IsingModel, PauliSpin

export act, transpose, multiply, variational_compr,F,Cv,chi,chi2,spectral, Corr

end

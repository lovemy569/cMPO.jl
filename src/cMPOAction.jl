### ### ### ### ### ### ### ### ### ###
###         Basic structure         ###
### ### ### ### ### ### ### ### ### ###

""" 
The object for cMPO
d: the physical dimension of the cMPO
D: the bond dimension of the cMPO
The structure of cMPO 
    --                   --
    | I + ε Q    √(ε) L   |
    |                     |
    | √(ε) R     P        |
    --                   --
"""
struct CMPO{T<:Number}
    Q::Matrix{T}    # 2 legs: D x D
    L::Array{T, 3}  # 3 legs: d x D x D
    R::Array{T, 3}  # 3 legs: d x D x D
    P::Array{T, 4}  # 4 legs: d x d x D x D

    function CMPO(Q::AbstractMatrix{T}, L::Array{T, 3}, R::Array{T, 3}, P::Array{T, 4}) where T
        @assert size(Q, 1) == size(Q, 2)
        @assert size(L, 2) == size(Q, 1) && size(L, 3) == size(Q, 1) 
        @assert size(R, 2) == size(Q, 1) && size(R, 3) == size(Q, 1) 
        @assert size(P, 3) == size(Q, 1) && size(P, 4) == size(Q, 1)
        new{T}(Q, L, R, P)
    end
end

""" 
The object for cMPS
d: the physical dimension of the cMPS
D: the bond dimension of the cMPS
The structure of cMPS
    --         --
    | I + ε Q   |
    |           |
    | √(ε) R    |
    --         --
"""
struct CMPS{T<:Number}
    Q::Matrix{T}    # 2 legs: D x D
    R::Array{T, 3}  # 3 legs: d x D x D

    function CMPS(Q::AbstractMatrix{T}, R::Array{T, 3}) where T
        @assert size(Q, 1) == size(Q, 2) 
        @assert size(R, 2) == size(Q, 1) && size(R, 3) == size(Q, 1) 
        new{T}(Q, R)
    end
end

""" 
The 1/2-PauliSpin operator
"""
struct PauliSpin
    Id::Matrix{Float64}
    X::Matrix{Float64}
    iY::Matrix{ComplexF64}
    Z::Matrix{Float64}
    Sp::Matrix{Float64}
    Sm::Matrix{Float64}

    function PauliSpin()
        new(
            I(2),                      # Identity matrix
            [0.0 1.0; 1.0 0.0],        # Pauli X matrix
            [0.0 0.0-im; 0.0+im 0.0],  # Pauli iY matrix
            [1.0 0.0; 0.0 -1.0],       # Pauli Z matrix
            [0.0 1.0; 0.0 0.0],        # Raising operator
            [0.0 0.0; 1.0 0.0]         # Lowering operator
        )
    end
end


### ### ### ### ### ### ### ### ### ###
###            Basic Model          ###
### ### ### ### ### ### ### ### ### ###

""" 
The Hamiltonian of transverse field Ising chain reads
H = -Γ∑ᵢXᵢ - J∑⟨i,j⟩ZᵢZⱼ
The corresponding cMPO tensor reads
    --                     --
    | 1 + εΓX   sqrt(εJ)Z   |
    |                       |
    | sqrt(εJ)Z             |
    --                     --
Physical dimension d = 2
Virtual bond dimension D = 2   
"""
struct IsingModel{T<:Number}
    T::CMPO{T}
    W::Matrix{T}
    ph_leg::Int
    d::Int

    function IsingModel(Gamma::T, J::T) where T
        s = PauliSpin()
        Q = Gamma * s.X
        L = sqrt(J) * reshape(s.Z, 1, 2, 2)
        R = sqrt(J) * reshape(s.Z, 1, 2, 2)
        P = zeros(T, 1, 1, 2, 2)
        new{T}(CMPO(Q, L, R, P), Diagonal(ones(T, 1)), 2, 1)
    end
end

""" 
The Hamiltonian of XXZ Heisenberg chain reads
H = Σ⟨i,j⟩ (Sᵢˣ Sⱼˣ + Sᵢʸ Sⱼʸ + Δ Sᵢᶻ Sⱼᶻ) 
The corresponding cMPO tensor reads
    --                                  --
    |     1       √(ε)S⁺  √(ε)S⁻  √(ε)Sᶻ |
    | -√(ε)S⁻/2                          |
    | -√(ε)S⁺/2                          |
    | -Δ√(ε)Sᶻ                           |
    --                                  --
Physical dimension d = 2
Virtual bond dimension D = 4   
"""
struct XXZModel{T<:Number}
    T::CMPO{T}
    W::Matrix{T}
    ph_leg::Int
    d::Int

    function XXZModel(Jz::T, Jxy::T) where T
        s = PauliSpin()
        Q = zeros(T, 2, 2)
        L = vcat(sqrt(abs(Jxy) / 2) * reshape(s.Sp, 1, 2, 2),
                 sqrt(abs(Jxy) / 2) * reshape(s.Sm, 1, 2, 2),
                 sqrt(abs(Jz)) / 2 * reshape(s.Z, 1, 2, 2))
        R = vcat(sqrt(abs(Jxy) / 2) * sign(Jxy) * reshape(s.Sm, 1, 2, 2),
                 sqrt(abs(Jxy) / 2) * sign(Jxy) * reshape(s.Sp, 1, 2, 2),
                 sqrt(abs(Jz)) / 2 * -sign(Jz) * reshape(s.Z, 1, 2, 2))
        P = zeros(T, 3, 3, 2, 2)

        row_indices = [1, 2, 3]  
        col_indices = [2, 1, 3]  
        values = [sign(Jxy), sign(Jxy), -sign(Jz)]
        W = sparse(row_indices, col_indices, values, 3, 3)
        new{T}(CMPO(Q, L, R, P), W, 2, 3)
    end
end


### ### ### ### ### ### ### ### ### ###
###          Basic operation        ###
### ### ### ### ### ### ### ### ### ###

function transpose(mpo::CMPO{T}) where T
    Q_t = mpo.Q
    L_t = mpo.R
    R_t = mpo.L
    P_t = ein"abmn->abnm"(mpo.P)
    return CMPO(Q_t, L_t, R_t, P_t)
end

""" 
Act the cMPO to the right of cMPS
    --                   -- --         --
    | I + ε Q    √(ε) L   | | I + ε Q   |
    |                     | |           |
    | √(ε) R     P        | | √(ε) R    |
    --                   -- --         --
"""
function act(mpo::CMPO{T}, mps::CMPS{T}) where T
    Do = size(mpo.Q, 1)
    Ds = size(mps.Q, 1)
    d = size(mps.R, 1)
    Io = I(Do)
    Is = I(Ds)
    
    Q_rslt = ein"ab,cd->cadb"(mpo.Q, Is) .+ 
             ein"ab,cd->cadb"(Io, mps.Q) .+ 
             ein"mab,mcd->cadb"(mpo.R, mps.R)      
    Q_rslt = reshape(Q_rslt, Do*Ds, Do*Ds)

    Is_reshape = reshape(Is, 1 , Ds, Ds)
    R_rslt = ein"mab,mcd->mcadb"(mpo.L, repeat(Is_reshape, d, 1, 1)) .+ 
             ein"mnab,ncd->mcadb"(mpo.P, mps.R)
    R_rslt = reshape(R_rslt, d, Do*Ds, Do*Ds)

    return CMPS(Q_rslt, R_rslt)
end

""" 
Manually symmetrize M before the eigen decomposition
"""
function eigensolver(M::Matrix{Float64})
    M_sym = eigen(0.5 * (M + M'))
    return M_sym
end

""" 
Perform a unitary transformation in the imaginary-time direction
U is the eigenvector that projects QR to the truncated state
"""
function project(mps::CMPS, U::Matrix)
    Q = U' * mps.Q * U
    R_new = ein"(ip,lpq),qj -> lij"(U', mps.R, U)
    return CMPS(Q, R_new)
end

""" 
Transform the cMPS to the gauge where Q is a diagonalized matrix 
"""
function diagQ(mps::CMPS)
    _, U = eigensolver(mps.Q)
    return project(mps, U)
end

"""
Initialize the isometry 
Keep the chi largest eigenvalues in the Q matrix of the cMPS
"""
function energy_cut(mps::CMPS, bond::Int)
    _, v = eigensolver(mps.Q)
    P = v[:, end-bond+1:end]
    return P
end

""" 
Act the LEFT of cMPS to the right of cMPS
    --                   -- --         --
    |                     | | I + ε Q   |
    | I + ε Q    √(ε) R   | |           |
    |                     | | √(ε) R    |
    --                   -- --         --
"""
function density_matrix(mps1::CMPS, mps2::CMPS)
    D1 = size(mps1.Q, 1)
    D2 = size(mps2.Q, 1)
    I1 = Matrix{Float64}(I, D1, D1)
    I2 = Matrix{Float64}(I, D2, D2)

    M = ein"ab,cd->cadb"(mps1.Q, I2) .+ 
        ein"ab,cd->cadb"(I1, mps2.Q) .+ 
        ein"mab,mcd->cadb"(mps1.R, mps2.R)  
    M = reshape(M, D1*D2, D1*D2)
    
    return M
end

"""
Calculate the logarithm and free energy function
"""
function logsumexp(x)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function log_tr_expm(beta, mat)
    w,_ = eigensolver(mat)
    y = logsumexp(beta .* w)
    return y
end

"""
For Zygote forward propagation
"""
function log_tr_expm_forward(beta, mat)
    y = log_tr_expm(beta, mat)
    return y
end

"""
For Zygote backward propagation
∂y/∂β = Σᵢ λᵢ ∂/∂β (e^(βλᵢ - y))
∂y/∂λᵢ = Σᵢ β ∂/∂β (e^(βλᵢ - y))
"""
function ChainRules.rrule(::typeof(log_tr_expm_forward), beta, mat)
    vals, vecs = eigensolver(mat)
    y = logsumexp(beta .* vals)
    function log_tr_expm_pullback(deltay)
        partialy_partialbeta = sum(vals .* exp.(beta .* vals .- y))
        betaback = deltay * partialy_partialbeta

        lambda = exp.(beta .* vals .- y)
        partialy_partialmat = transpose(vecs * Diagonal(lambda) * vecs')
        matback = deltay * beta * partialy_partialmat
        return ChainRules.NoTangent(), betaback, matback
    end
    return y, log_tr_expm_pullback
end

"""
Calculate the logarithm and free energy function
"""
function ln_ovlp(mps1::CMPS, mps2::CMPS, beta::Float64)
    M = density_matrix(mps1, mps2)
    y = log_tr_expm_forward(beta, M)
    return y
end

"""
More general free energy function
F = ⟨ψ | H | 𝜆⟩ / √(⟨ψ | ψ⟩)
"""
function Fidelity(psi::CMPS, mps::CMPS, beta::Float64)
    up = ln_ovlp(psi, mps, beta)
    dn = ln_ovlp(psi, psi, beta)
    return up - 0.5 * dn
end

"""
Interpolation Optimization
"""
function interpolate_cut(cut1::Matrix, cut2::Matrix, theta::Float64)
    mix = sin(theta) * cut1 + cos(theta) * cut2
    U, _, V = svd(mix)
    return U * V'
end

function adaptive_mera_update(mps::CMPS, beta::Float64, bond::Int, atol::Float64=1e-5, btol::Float64=1e-9, maxiter::Int=50, interpolate::Bool=true)
    step = 1
    n₀ = exp(ln_ovlp(mps, mps, beta))

    # Defining the loss function
    loss(P) = ln_ovlp(project(mps, P), mps, beta) - 0.5 * ln_ovlp(project(mps, P), project(mps, P), beta)

    # Truncating virtual bond dimensions
    Pc = energy_cut(mps, bond)

    # Initialize the loss function value
    Lp = 9.9e9
    Lc = loss(Pc)

    # Calculate the difference between the current fidelity and 1.0
    ΔF = abs(exp(Lc) / n₀ - 1.0)
    # Calculate the difference in logfidelity between the current step and the previous step
    ΔlnF = abs(Lc - Lp)
    
    println("Adaptive MERA Update\n")
    println("step        θ/π               ΔlnF                1.0 - F      ")     
    println("----  ----------------  -----------------   -------------------")
    println(@sprintf("%03i   %.10e   %.10e   %.10e", step, 1.0, ΔlnF, ΔF))

    while step < maxiter
        step += 1   
        grad = Zygote.gradient(loss, Pc)[1]
        F = svd(grad)
        Pn = F.U * F.Vt
        θ = π
        proceed = interpolate
        while proceed
            θ /= 2
            if θ < π / (1.9^12)
                Pn = Pc
                proceed = false
            else
                Pi = interpolate_cut(Pn, Pc, θ)
                Li = loss(Pi)
                if Li > Lc
                    Pn = Pi
                    Lc = Li
                    proceed = false
                end
            end     
        end

        ΔF = abs(exp(Lc) / n₀ - 1.0)
        ΔlnF = abs(Lc - Lp)
        println(@sprintf("%03i   %.10e   %.10e   %.10e", step, θ/π, ΔlnF, ΔF))

        Pc = Pn
        Lp = Lc

        if ΔF < atol || ΔlnF < btol
            break
        end
    end

    return project(mps, Pc)
end

function variational_compr(mps::CMPS, beta::Float64, bond::Int, tol::Float64=1e-9)
    psi = adaptive_mera_update(mps, beta, bond)
    psi = diagQ(psi)
    Q = diagm(diag(psi.Q))
    R = copy(psi.R)

    function loss_function(QR)
        Q = reshape(QR[1:bond*bond], bond, bond)
        R = reshape(QR[bond*bond+1:end], size(R))
        psi = CMPS(Q, R)
        return -Fidelity(psi, mps, beta)
    end

    # Vectorized stitching
    QR_initial = vcat(vec(Q), vec(R))
    result = optimize(loss_function, QR_initial, LBFGS(), Optim.Options(f_tol = 2.220446049250313e-9,g_tol=tol, iterations=50))

    QR_optimized = result.minimizer
    Q_optimized = reshape(QR_optimized[1:bond*bond], bond, bond)
    R_optimized = reshape(QR_optimized[bond*bond+1:end], size(R))

    Q_optimized .-= maximum(Q_optimized)
    psi = CMPS(Q_optimized, R_optimized)
    println(result)

    return psi
end

"""
Convert right vector to left vector
"""
function multiply(W::Matrix, mps::CMPS)
    R1 = ein"mn, nab->mab"(W, mps.R)
    return CMPS(mps.Q, R1)
end


### ### ### ### ### ### ### ### ### ###
###       Physical observables      ###
### ### ### ### ### ### ### ### ### ###

""" 
Calculate the free energy by
f = -(1/beta) * ln [<Lpsi|T|psi> / <Lpsi|psi>]
"""
function f(psi::CMPS, Lpsi::CMPS, T::CMPO, beta::Float64)
    Tpsi = act(T, psi)
    up = ln_ovlp(Lpsi, Tpsi, beta)
    dn = ln_ovlp(Lpsi, psi, beta)
    return (- up + dn) / beta
end

""" 
Calculate the specific heat 
c = β² [ (⟨K_{|-+-|}²⟩_{K_{|-+-|}} - ⟨K_{|-+-|}⟩_{K_{|-+-|}}²) - ...
    (⟨K_{|--|}²⟩_{K_{|--|}} - ⟨K_{|--|}⟩_{K_{|--|}}²) ]
"""
function cv(psi::CMPS, Lpsi::CMPS, T::CMPO, beta::Float64)
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)
    w, _ = eigensolver(M)
    w_nm = w .- logsumexp(beta .* w) / beta
    up = dot(exp.(beta .* w_nm), w.^2) - (dot(exp.(beta .* w_nm), w))^2

    M = density_matrix(Lpsi, psi)
    w, _ = eigensolver(M)
    w_nm = w .- logsumexp(beta .* w) / beta
    dn = dot(exp.(beta .* w_nm), w.^2) - (dot(exp.(beta .* w_nm), w))^2

    return (beta^2) * (up - dn)
end

""" 
Calculate unequal-imaginary-time correlator
⟨Aᵢ(τ) Bᵢ⟩ = Tr(e^{-(β-τ)K_{|-+-|}} A e^{-τ K_{|-+-|}} B) / Tr(e^{-β K_{|-+-|}})
"""
function corr(psi::CMPS, Lpsi::CMPS, T::CMPO, O1::AbstractMatrix, O2::AbstractMatrix, beta::Float64, tau::Float64)
    totalD = size(O1, 1) * size(psi.Q, 1) * size(psi.Q, 1)
    matI = Matrix(I, size(psi.Q, 1), size(psi.Q, 1))
    matO1 = reshape(ein"(ab,cd),ef->acebdf"(matI, O1, matI), totalD, totalD)
    matO2 = reshape(ein"(ab,cd),ef->acebdf"(matI, O2, matI), totalD, totalD)
    
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    eig = eigensolver(M)
    w = eig.values .- maximum(eig.values)
    v = eig.vectors

    matO1 = v' * matO1 * v
    matO2 = v' * matO2 * v

    expw_a = Diagonal(exp.((beta - tau) .* w))
    expw_b = Diagonal(exp.(tau .* w))
    expw = Diagonal(exp.(beta .* w))

    numerator = tr(expw_a * matO1 * expw_b * matO2)
    denominator = tr(expw)

    return numerator / denominator
end

""" 
Calculate the Matsubara frequency susceptibility
χ(iω) = (1/Z) Σ_{m,n} (Ṡ_{nm}ᶻ Ṡ_{mn}ᶻ (e^{-βΛₘ} - e^{-βΛₙ})) / (iω - Λₘ + Λₙ)
where Z = Σₙ e^{-βΛₙ} and Ṡᶻ = U† (I_{|-} ⊗ Sᶻ ⊗ I_{-|}) U
"""
function chi(psi::CMPS, Lpsi::CMPS, T::CMPO, O1::AbstractMatrix, O2::AbstractMatrix, beta::Float64, iomega::Float64)
    totalD = size(O1, 1) * size(psi.Q, 1) * size(psi.Q, 1)
    matI = Matrix(I, size(psi.Q, 1), size(psi.Q, 1))
    matO1 = reshape(ein"(ab,cd),ef->acebdf"(matI, O1, matI), totalD, totalD)
    matO2 = reshape(ein"(ab,cd),ef->acebdf"(matI, O2, matI), totalD, totalD)
    
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    eig = eigensolver(M)
    w = eig.values .- maximum(eig.values)
    v = eig.vectors

    matO1 = v' * matO1 * v
    matO2 = v' * matO2 * v
    expw = Diagonal(exp.(beta .* w))

    Envec = repeat(w, 1, length(w))
    Emvec = repeat(w', length(w), 1) .+ 1e-8
    F = (exp.(beta .* Envec) .- exp.(beta .* Emvec)) ./ (iomega .+ Envec .- Emvec)

    result = tr(matO1 * (matO2 .* F)) / tr(expw)
    return result
end

""" 
Calculate the dynamic susceptibility
χ''(ω) = - (π/Z) Σ_{m,n} Ṡ_{nm}ᶻ Ṡ_{mn}ᶻ (e^{-βΛₘ} - e^{-βΛₙ}) δ(ω - Λₘ + Λₙ),
where δ(x) = lim_{η→0} (1/π) (η / (x² + η²)).
"""
function chi2(psi::CMPS, Lpsi::CMPS, T::CMPO, O1::AbstractMatrix, O2::AbstractMatrix, beta::Float64, omega::Float64, eta::Float64)
    totalD = size(O1, 1) * size(psi.Q, 1) * size(psi.Q, 1)
    matI = Matrix(I, size(psi.Q, 1), size(psi.Q, 1))
    matO1 = reshape(ein"(ab,cd),ef->acebdf"(matI, O1, matI), totalD, totalD)
    matO2 = reshape(ein"(ab,cd),ef->acebdf"(matI, O2, matI), totalD, totalD)
    
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    eig = eigensolver(M)
    w = eig.values .- maximum(eig.values)
    v = eig.vectors

    matO1 = v' * matO1 * v
    matO2 = v' * matO2 * v
    expw = Diagonal(exp.(beta .* w))

    delta(x, eta) = 1 / π * eta / (x^2 + eta^2)

    Envec = repeat(w, 1, length(w))
    Emvec = repeat(w', length(w), 1)
    F = (exp.(beta .* Envec) .- exp.(beta .* Emvec)) .* delta.(omega .+ Envec .- Emvec, eta)

    result = -π * tr(matO1 * (matO2 .* F)) / tr(expw)
    return result
end

""" 
Calculate spectral function
S(ω) = 2χ''(ω) / (1 - e^{-βω})
"""
function spectral(psi::CMPS, Lpsi::CMPS, T::CMPO, O1::AbstractMatrix, O2::AbstractMatrix, beta::Float64, omega::Float64, eta::Float64)
    return 2 * chi2(psi, Lpsi, T, O1, O2, beta, omega, eta) / (1 - exp(-beta * omega))
end




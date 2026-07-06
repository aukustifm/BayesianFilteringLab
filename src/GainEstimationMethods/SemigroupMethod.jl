"""
    SemigroupMethod(epsilon, delta, max_iter)

Semigroup method from Algorithm 1 in [1].

[1] Taghvaei, A., & Mehta, P. G. (2016). Gain function approximation in the feedback particle filter. In 2016 IEEE 55th Conference on Decision and Control (CDC) (pp. 5446–5452). IEEE. https://doi.org/10.1109/CDC.2016.7799105

    SemigroupMethod(epsilon, delta, max_iter, lambda)

Semigroup method with regularization parameter `lambda`.
"""
mutable struct SemigroupMethod <: GainEstimationMethod
    epsilon::Float32
    delta::Float32
    max_iter::Int
    lambda::Float32
    SemigroupMethod(eps, delta, max_iter, lambda=0.) = new(eps, delta, max_iter, lambda)
end

SemigroupMethod(epsilon::Float64, delta::Float64) = SemigroupMethod(epsilon, delta, 200)

function solve!(eq::PoissonEquation, method::SemigroupMethod; gain=true)
    N = size(eq.positions,2)
    M̃  = m(eq)
    
    #=
    Σ⁻ = inv(diagm([sqrt(median(pairwise(Euclidean(), eq.positions[dₓ, :]))^2/log((N-1)/(sqrt(2)*1e-6)^2)) for dₓ in 1:size(eq.positions,1)].^2))
    method.epsilon = median(pairwise(Mahalanobis(Σ⁻), eq.positions, dims=2))/log(N) #Euclidean()
    =#
    ###=
    #method.epsilon = 0.5*[1.06(min(std(eq.positions), iqr(eq.positions)/(1.34))*N^(-1/(size(eq.positions,1)+4)))][1]
    method.epsilon = median(pairwise(Euclidean(), eq.positions, dims=2))/log(N) #Euclidean()
    ##=#
    #println(method.epsilon)
    #method.epsilon = 0.1

    broadcast!(*, M̃, M̃, method.epsilon)

    # compute T operator
    T = zeros(eltype(eq.positions), N, N)
    Dₓ = state_dim(eq)
    #L₀ = eq.positions.-mean(eq.positions,dims=2)
    for i in 1:N
        T[i,i] = one(eltype(eq.positions))
        for j in i+1:N
            sq = zero(eltype(eq.positions))
            #sq₂ = zero(eltype(eq.positions))
            for l in 1:Dₓ
                sq += (eq.positions[l,i] - eq.positions[l,j])^2#*Σ⁻[l,l]#*Σ⁻[l,l]#^2#corr_dist(eq.positions[:,i], eq.positions[:, j])^2#
                #sq += (L₀[l,i]-L₀[l,j])^2*Σ⁻[l,l]#^2#corr_dist(eq.positions[:,i], eq.positions[:, j])^2#
            end
            T[i,j] = exp(-sq/(4*method.epsilon))#exp(-sq₂)#-sq₂/(4*method.epsilon^2))
            T[j,i] = T[i,j]
        end
    end
    
    #D = sum(T,dims=2)/sum(T)

    qₓ = sum(T,dims=1); qᵧ = sum(T,dims=2); 
    broadcast!(/, T, T, sqrt.(qₓ .* qᵧ))
    broadcast!(/, T, T, sum(T,dims=2))

    # add noise to regularize T
    if method.lambda > 0.
        for i in 1:N
            T[i,i] -=  method.lambda * rand(Distributions.Uniform(0.9,1))
        end
    end

    # solve fixed-point equation for potential
    newpotential = copy(eq.potential)::Array{eltype(eq.positions),2}
    fluctuation = 1.
    n = 1
    while fluctuation > method.delta
        LinearAlgebra.mul!(newpotential, eq.potential, T')
        broadcast!(+, newpotential, newpotential, M̃)
        broadcast!(-, newpotential, newpotential, Statistics.mean(newpotential, dims=2))
        fluctuation = maximum(abs.(newpotential - eq.potential))
        eq.potential .= newpotential
        n += 1
        if n == method.max_iter
            #print("!")
            break
        end
    end

    eq.potential .= broadcast(+, eq.potential, -method.epsilon*eq.L)
    eq.potential .-= mean(eq.potential)

    # compute gain from potential
    gainhelper_semigroup!(eq, T) 
    broadcast!(/, eq.gain, eq.gain, 2*method.epsilon)
end

function gainhelper_semigroup!(eq::PoissonEquation, T::AbstractMatrix)
    pos     = eq.positions
    pot     = eq.potential
    gain    = eq.gain
    
    Dₓ      = state_dim(eq)
    Dᵧ      = obs_dim(eq)
    N       = no_of_particles(eq)
    
    @inbounds for l in 1:Dₓ, dᵧ in 1:Dᵧ, i in 1:N
        Tpotpos = zero(eltype(gain))
        Tpot    = zero(eltype(gain))
        Tpos    = zero(eltype(gain))
        for j in 1:N
            Tpot        += T[i,j] * pot[dᵧ,j]
            Tpos        += T[i,j] * pos[l,j]
            Tpotpos     += T[i,j] * pot[dᵧ,j] * pos[l,j]
        end
        gain[l,i,dᵧ] = (Tpotpos - Tpot * Tpos)
    end
end
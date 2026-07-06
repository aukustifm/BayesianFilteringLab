"""
    WeightedParticleEnsemble{T}

An ensemble of `N` particles, each of dimension `n`.
"""
mutable struct WeightedParticleEnsemble{T} <: WeightedParticleRepresentation{Vector{T}}
    positions::Matrix{T}
    weights::StatsBase.ProbabilityWeights
    resweights::StatsBase.ProbabilityWeights
    ancestors::Vector{Int}
end

particle_dim(ens::WeightedParticleEnsemble)    = size(ens.positions, 1)
no_of_particles(ens::WeightedParticleEnsemble) = size(ens.positions, 2)
get_pos(ens::WeightedParticleEnsemble, i) = view(ens.positions, :, i)
get_weight(ens::WeightedParticleEnsemble, i) = ens.weights[i]
get_vweight(ens::WeightedParticleEnsemble, i) = ens.resweights[i]
sum_of_weights(ens::WeightedParticleEnsemble) = sum(ens.weights)
sum_of_vweights(ens::WeightedParticleEnsemble) = sum(ens.resweights)

Base.show(io::IO, ens::WeightedParticleEnsemble) = print(io, "Weighted particle ensemble
    # of particles: ", no_of_particles(ens),"
    particle type:  ", particle_dim(ens),"-dimensional ", eltype(ens))

function WeightedParticleEnsemble(positions::Matrix)
    N = size(positions)[2]
    return WeightedParticleEnsemble(positions, StatsBase.ProbabilityWeights(fill(1/N, N)), StatsBase.ProbabilityWeights(fill(1/N, N)), Vector(1:N))
end

function WeightedParticleEnsemble(model::HiddenStateModel, N::Int)
    return WeightedParticleEnsemble(hcat([initialize(model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)), StatsBase.ProbabilityWeights(fill(1/N, N)), Vector(1:N))
end

mean(ens::WeightedParticleEnsemble) = Statistics.mean(ens.positions, ens.weights, dims=2)
cov(ens::WeightedParticleEnsemble)  = Statistics.cov(ens.positions, ens.weights, 2, corrected=false)
var(ens::WeightedParticleEnsemble)  = Statistics.var(ens.positions, ens.weights, 2, corrected=false)

myskewness = function(ens::WeightedParticleEnsemble)
    x1 = ens.positions[1,:]
    x2 = ens.positions[2, :]
    ww = ens.weights
    m1 = mean(ens)
    m2 = cov(ens)
    m3 = [sum(ww.*(x1.-m1[1]).^3)/(m2[1, 1])^1.5;    
          sum(ww.*(x2.-m1[2]).^3)/(m2[2, 2])^1.5]
    return m3
end

mykurtosis = function(ens::WeightedParticleEnsemble)
    x1 = ens.positions[1,:]
    x2 = ens.positions[2, :]
    ww = ens.weights
    m1 = mean(ens)
    m2 = cov(ens)
    m4 = [sum(ww.*(x1.-m1[1]).^4)/(m2[1, 1])^2    ;    
          sum(ww.*(x2.-m1[2]).^4)/(m2[2, 2])^2]
    return m4
end

mardiaskewness  = function(ens::WeightedParticleEnsemble)

    n = length(ens.weights)
    
    μᵢ = mean(ens)
    σᵢ = inv(cov(ens))
    
    x1₀ = ens.positions .- μᵢ
    x2₀ = ens.positions .- μᵢ

    ww = ens.weights
    m3 = 0.0
    Base.Threads.@threads for i in 1:n 
        for j in 1:n       
            m3 += ww[i] * ww[j] * (x1₀[:, i]'*σᵢ*x2₀[:,j])^3 
        end
    end
    return m3
end

mardiakurtosis = function(ens::WeightedParticleEnsemble)
    n = length(ens.weights)

    μᵢ = mean(ens)
    σᵢ = inv(cov(ens))
    
    x1₀ = ens.positions .- μᵢ
    
    ww = ens.weights
    m4 = 0.0
    Base.Threads.@threads for i in 1:n 
        m4 += ww[i] * (x1₀[:, i]' * σᵢ * x1₀[:,i])^2
    end
    return m4
end

function inverse_cdf(W, su)
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        su: (N,) ndarray
            N sorted uniform variates (i.e. M ordered points in [0,1]).
        Returns
        -------
        A: (M,) ndarray
            a vector of N indices in range 0, ..., N-1
    """
    j = 1
    s = W[1]
    N = length(su)
    A = Array{Int}(undef, N)
    for n in 1:N
        while (su[n] > s)
            j += 1
            s += W[j]
        end
        A[n] = j
    end
    return A
end

function stratified(W, N)
    su = (rand(N) + Vector(0:N-1)) / N
    return inverse_cdf(W, su)
end

function systematic(W, N)
    su = (Vector(0:N-1) .+ rand(1)) / N
    return inverse_cdf(W, su)
end

function multinomial_res(W, N)
    return StatsBase.sample(1:N, W, N)
end

function uniform_spacings(N)
    """ Generate ordered uniform variates in O(N) time. (Standard algorithms require O(NlogN))
    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates
    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)
    Note
    ----
    This is equivalent to::
        from numpy import random
        u = sort(random.rand(N))
    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).
    """
    z = cumsum(-log.(rand(N + 1)))
    return z[1:end-1] / z[end]
end

function multinomial(W, N)
    return inverse_cdf(W, uniform_spacings(N))
end

function residual(W, N)
    A = Array{Int}(undef, N)
    NW = N .* W
    intpart = Integer.(floor.(NW))
    sip = sum(intpart)
    res = NW .- intpart
    sres = N - sip
    aux = [[i] for i in 1:N]
    A[1:sip] = collect(Iterators.flatten(repeat.(aux, intpart)))

    # each particle n is repeated intpart[n] times
    if sres > 0
        A[sip+1:end] = multinomial(res / sres, sres)
    end
    return A
end

function resample!(ens::WeightedParticleEnsemble, res; auxiliary = false)

    N   = no_of_particles(ens)
 
    if auxiliary
        weights = ens.resweights
        weights = ProbabilityWeights(weights./sum(weights))
    else 
        weights = ens.weights
    end
   
    if res == "Multinomial"
        idx = multinomial_res(weights, N)
    elseif res == "Stratified"
        idx = stratified(weights, N)
    elseif res == "Systematic"
        idx = systematic(weights, N)
    elseif res == "Residual"
        idx = residual(weights, N)
    else
        println("Please choose a correct resampling method")
    end

    ens.ancestors = idx
    ens.positions .= view(ens.positions, :, idx)
    
    if auxiliary 
        resweights = view(ens.resweights, idx)   
        ens.resweights = ProbabilityWeights(resweights./sum(resweights))
    else
        ens.weights .= 1/N     
    end
    
end
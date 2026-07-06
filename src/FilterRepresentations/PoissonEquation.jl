@doc raw"""
    PoissonEquation(l, ensemble) ::GainEquation

Returns a Poisson equation struct representing the equation ``\nabla\cdot(pM^{-1}\nabla \phi) = - \tilde m``, where ``p`` is a probability density and ``\tilde m = -(L-\int L p dx)``. 
The container contains the following fields
* `:m': ``m`` itself
* `:positions': an i.i.d. sample from ``p``, represented as a matrix
* `:L': the evaluation of ``L`` at the sample points
* `:mean_L': the sample average of `L'
* `:potential': the evaluation of ``\phi`` at the sample points
* `:gain': the evaluation of ``K=M^{-1}\nabla \phi`` at the sample points

    solve!(eq::PoissonEquation, method::GainEstimationMethod)

Fills the field `:gain' with appropriate values.
The fields `:L', `:mean_L', and `:potential' are stored to be re-used.

    update!(eq::PoissonEquation, ensemble)

Fills the fields `:positions', `:L', and `:mean_L' according to the new samples from `ensemble'.

    update!(eq::PoissonEquation)

Updates fields `:L', and `:mean_L' to be consistent with `:positions'.
"""
mutable struct PoissonEquation{T, TF, MM} <: GainEquation
    l::TF
    positions::Matrix{T}
    L::Matrix{T}
    mean_L::Matrix{T}
    potential::SharedMatrix{T}
    gain::SharedArray{T,3}    
    mass::MM
    PoissonEquation(l, positions, L, mean_L, potential, gain, mass) = 
        if  size(positions, 2) == size(L, 2) == size(potential, 2) == size(gain, 2) && 
            size(L, 1) == size(mean_L, 1) == size(potential, 1) == size(gain, 3) &&
            size(positions, 1) == size(gain, 1) &&
            size(mean_L,2) == 1
            new{eltype(positions), typeof(l), typeof(mass)}(l, positions, L, mean_L, potential, gain, mass)
        else
            throw(DimensionMismatch("Inconsistent dimensions of positions, L, mean_L, potential, or gain."))
        end
end
    
function PoissonEquation(l::AbstractMatrix, pos::AbstractMatrix, mass)
    L = l
    mean_L = mean(L, dims=2)
    return PoissonEquation(l, pos, L, mean_L, zeros(eltype(pos), size(L, 1), size(pos, 2)), zeros(eltype(pos), size(pos, 1), size(pos, 2), size(L, 1)), mass)
end

function PoissonEquation(l::Function, pos::AbstractMatrix, mass)
    L = mapslices(l, pos, dims=1)
    mean_L = Statistics.mean(L, dims=2)
    return PoissonEquation(l, pos, L, mean_L, 
        SharedArray{eltype(pos)}(size(L, 1), size(pos, 2)), 
        SharedArray{eltype(pos)}(size(pos, 1), size(pos, 2), size(L, 1)), 
        mass)
end

function PoissonEquation(l::Function, ens::UnweightedParticleEnsemble)
    pos = copy(ens.positions)
    return PoissonEquation(l, pos)
end
    
state_dim(eq::PoissonEquation)       = size(eq.positions, 1)
no_of_particles(eq::PoissonEquation) = size(eq.positions, 2)
obs_dim(eq::PoissonEquation) = size(eq.L, 1)
    
Base.show(io::IO, eq::PoissonEquation) = print(io, "Poisson equation for the gain
    # of particles:        ", size(eq.positions, 2),"
    hidden dimension:      ", state_dim(eq))
    
function m(eq::PoissonEquation)
    return -(eq.L .- eq.mean_L)
end
    
function update!(eq::PoissonEquation)
    #eq.L         .= mapslices(eq.L, eq.positions, dims=1)
    eq.mean_L    .= Statistics.mean(eq.L, dims=2)
    return eq
end
    
function update!(eq::PoissonEquation, pos::AbstractMatrix)
    eq.positions .= pos
    update!(eq)
end    
    
    
function update!(eq::PoissonEquation, ens::UnweightedParticleEnsemble)
    update!(eq, ens.positions)
end 
    
function GainEquation(state_model::DiffusionStateModel, obs_model::DiffusionObservationModel, ens::UnweightedParticleEnsemble)
    return PoissonEquation(obs_model.observation_function, ens)
end
    
function GainEquation(state_model::DiffusionStateModel, obs_model::DiffusionObservationModel, N::Int)
    ens = UnweightedParticleEnsemble(state_model, N)
    return GainEquation(state_model, obs_model, ens)
end
    
function GainEquation(filt_prob::AbstractFilteringProblem, ens::UnweightedParticleEnsemble)
    return GainEquation(state_model(filt_prob), obs_model(filt_prob), ens)
end
    
function GainEquation(filt_prob::AbstractFilteringProblem, N::Int)
    return GainEquation(state_model(filt_prob), obs_model(filt_prob), N)
end
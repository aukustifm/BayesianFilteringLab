"""
    WeightedParticleRepresentation{S} <: ParticleRepresentation{S} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by weighted particles (samples).
"""
abstract type WeightedParticleRepresentation{S} <: ParticleRepresentation{S} end
"""
    get_weight(ensemble::WeightedParticleRepresentation, i)

Return the weight of the particle with index `i` from `ensemble`.
"""
function get_weight(ensemble::WeightedParticleRepresentation, i) end
"""
    list_of_weights(ensemble::WeightedParticleRepresentation)

Return a list of particle weights from `ensemble`.
"""
list_of_weights(ens::WeightedParticleRepresentation) = [get_weight(ens, i) for i in 1:no_of_particles(ens)]
"""
    sum_of_weights(ensemble::WeightedParticleRepresentation)

Return the sum of the importance weights in `ensemble`.
"""
function sum_of_weights(ens::WeightedParticleRepresentation) 
    sum = 0.
    for i in 1:no_of_particles(ens)
        sum += get_weight(ens, i)
    end
    return sum
end
"""
    resample!(ensemble::WeightedParticleRepresentation)

Resample the positions by drawing a weighted sample with replacement.
"""
function resample!(ens::WeightedParticleRepresentation) end
function eff_no_of_particles(ensemble::WeightedParticleRepresentation; auxiliary = false)
    sum = 0.
    
    
    if auxiliary
        @inbounds for i in 1:no_of_particles(ensemble)
            sum += get_vweight(ensemble, i).^2
        end
        return sum_of_vweights(ensemble)^2/sum
    else 
        @inbounds for i in 1:no_of_particles(ensemble)
            sum += get_weight(ensemble, i).^2
        end
        return sum_of_weights(ensemble)^2/sum
    end
end
dim(ensemble::WeightedParticleRepresentation) = (particle_dim(ensemble) + 1) * no_of_particles(ensemble)
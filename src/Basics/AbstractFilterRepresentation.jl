"""
    AbstractFilterRepresentation{S}

Abstract type for representation of the conditional distribution over the hidden state of type `S`.
"""
abstract type AbstractFilterRepresentation{S} end
"""
    represented_type(rep::AbstractFilterRepresentation)

Return the type which is represented by `rep`.
"""
represented_type(rep::AbstractFilterRepresentation{S}) where S = S
"""
    dim(rep::AbstractFilterRepresentation{S})

Return the dimensionality of the filter representation `rep`.
"""
function dim(rep::AbstractFilterRepresentation{S}) where S end




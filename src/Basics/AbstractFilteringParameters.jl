"""
    AbstractFilteringProblem{S1, S2, T1<:TimeType, T2<:TimeType}

Abstract type for a filtering problem for observations of type `S2` in `T2` and hidden states of type `S1` in `T1`.
"""
abstract type AbstractFilteringParameters{T} end


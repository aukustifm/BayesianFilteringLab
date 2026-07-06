"""
    FilteringParameters

Specify a filtering algorithm to solve the filtering problem for a hidden state model `state_model` and observation model `obs_model`.

"""
struct FilteringParameters{T} <: AbstractFilteringParameters{T}
    values::T
end
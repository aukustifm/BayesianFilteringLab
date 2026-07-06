"""
    AbstractFilteringProblem{S1, S2, T1<:TimeType, T2<:TimeType}

Abstract type for a filtering problem for observations of type `S2` in `T2` and hidden states of type `S1` in `T1`.
"""
abstract type AbstractFilteringProblem{T1<:TimeType, T2<:TimeType, D1<:Int, D2<:Int, D3<:Int, D4<:Int} end

"""
    state_model(problem::AbstractFilteringProblem)

Return the hidden state model underlying `problem`.
"""
function state_model(problem::AbstractFilteringProblem) end
"""
    obs_model(problem::AbstractFilteringProblem)

Return the observation model underlying `problem`.
"""
function obs_model(problem::AbstractFilteringProblem) end
"""
    state_dim(problem::AbstractFilteringProblem)

Return the dimensionality of the hidden state in `problem`.
"""
state_dim(problem::AbstractFilteringProblem) = state_dim(state_model(problem))
"""
    obs_dim(problem::AbstractFilteringProblem)

Return the dimensionality of the observed state in `problem`.
"""
obs_dim(problem::AbstractFilteringProblem) = obs_dim(obs_model(problem))
hidden_time_type(problem::AbstractFilteringProblem{T1, T2}) where {T1, T2} = T1
obs_time_type(problem::AbstractFilteringProblem{T1, T2}) where {T1, T2}    = T2
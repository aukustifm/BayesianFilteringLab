mutable struct FilterState{S1, S2, TF<:AbstractFilterState}
    mean::S1
    cov::S2
end
"""
    simulate!(filtering_algorithm, filtering_problem, no_of_timesteps, dt)
Runs a simulation of the hidden state, observation, and filtering algorithm for a duration of `no_of_timesteps`.
"""
function FilterState(filt_prob, filt_algo)
    hidden_state = deepcopy(initialize(state_model(filt_prob)))
    obs          = obs_model(filt_prob)(hidden_state, zero(eltype(hidden_state)))
    filter_state = deepcopy(initialize(filt_algo))
    return SimulationState(hidden_state, obs, filter_state)
end
hidden_state(st::SimulationState) = st.hidden_state
obs(st::SimulationState)          = st.obs
filter_state(st::SimulationState) = st.filter_state
cond_mean(x::SimulationState)     = mean(filter_state(x))
cond_cov(x::SimulationState)      = cov(filter_state(x))
cond_var(x::SimulationState)      = var(filter_state(x))
"""
    propagate!(sfs, filtering_problem, filtering_algorithm; dt) --> sfs
Propagates the system and filter states for one time-step according to the specified filtering problem and algorithm.
"""
function filter!(
    sfs::FilterState, 
    filt_algo::AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}, 
    dt) where {S1, S2}
    
    filter_state = sfs.filter_state
    
    propagate!(filter_state, filt_algo, dt)
    update!(filter_state, filt_algo, obs, dt)
end
"""
    FilteringProblem(state_model, obs_model)

Specify a filtering problem for a hidden state model `state_model` and observation model `obs_model`.

Constraints: `state_model` and `obs_model` must be chosen such that
* `state_model` is of type `HiddenStateModel{S1, T1}`,
* `obs_model` is of type `ObservationModel{S1, S2, T2}`,
where `T1<:TimeType`, `T2 <:TimeType`, and `S1`, `S2` are arbitrary.
This ensures that the filtering problem can be evaluated, i.e. the hidden states are of appropriate type to generate observations.
"""
struct FilteringProblem{T1, T2, D1, D2, D3, D4, M1, M2} <: AbstractFilteringProblem{T1, T2, D1, D2, D3, D4}
    m::D1
    n::D2
    mp::D3
    np::D4
    state_model::M1
    obs_model::M2

    function FilteringProblem(mod1::HiddenStateModel{T1}, mod2::ObservationModel{T2}, m::D1, n::D2, mp::D3, np::D4) where {T1<:TimeType, T2<:TimeType, D1, D2, D3, D4} 
        return new{T1, T2, typeof(m), typeof(n), typeof(mp), typeof(np), typeof(mod1), typeof(mod2),}(m, n, mp, np, mod1, mod2)
    end
end
# Mandatory methods
initial_condition(problem::FilteringProblem) = problem.state_model.init
state_model(problem::FilteringProblem)       = problem.state_model
obs_model(problem::FilteringProblem)         = problem.obs_model
# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", problem::FilteringProblem{T1, T2, D1, D2, M1, M2}) where {T1, T2, D1, D2, M1, M2}
    print(io, T1, " - ", T2," filtering problem
    hidden state model:                         ", problem.state_model,"
    hidden state dimension:                     ", problem.m,"
    observation model:                          ", problem.obs_model,"
    observation dimension:                      ", problem.n)
end
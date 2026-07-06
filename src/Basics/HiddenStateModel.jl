"""
    HiddenStateModel{T<:TimeType} <: AbstractModel

Abstract type for any model of the hidden state, discrete or continuous 
"""
abstract type HiddenStateModel{T} <: AbstractModel{T} end
function initialize(state_model::HiddenStateModel)
    init = initial_condition(state_model)
    if init isa Distributions.Sampleable
        return rand(init)
    else
        return init
    end
end  
    
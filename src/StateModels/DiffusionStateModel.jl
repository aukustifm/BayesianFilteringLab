@doc raw"""
    DiffusionStateModel(f::Function, g::Function, init)

Returns a diffusion process hidden state model ``dX_t = f(X_t)dt + g(X_t)dW_t``, where ``f`` is the `drift_function`, ``g`` is the `observation_function`, ``X_t`` is the ``n``-dimensional hidden state at time ``t``, and ``W_t`` is an ``m``-dimensional Brownian motion process.

Argument `init` stands for the initial condition of the process, which is either
* A vector of length `n` for a fixed (deterministic) initial condition
* A `Distributions.Sampleable` type for a random initial condition
"""
struct DiffusionStateModel{F1, TI} <: HiddenStateModel{ContinuousTime}
    dynamic_function::F1
    init::TI
    function DiffusionStateModel(dynamics::Function, init::TI) where TI<:Union{Distributions.Sampleable, Any}
        return new{typeof(dynamics), typeof(init)}(dynamics, init)
    end
end
                
#####################
### BASIC METHODS ###
#####################
                   
"""
drift_function(model)

Returns the drift function ``f`` of the diffusion model ``dX_t = f(X_t)dt + g(X_t)dW_t`` at time t.
"""                
drift_function(model::DiffusionStateModel, t) = drift(model, t)
drift_diff_function(model::DiffusionStateModel, t) = drift_diff(model, t)
function drift(model::DiffusionStateModel, t)

    return model.dynamic_function(t)[1] # returns f(x, p | t)
end        
function drift_diff(model::DiffusionStateModel, t)
    return model.dynamic_function(t) # returns f(x, p | t) and g(x, p | t)
end
                     
"""
    diffusion_function(model)

Returns the diffusion function ``g`` of the diffusion model ``dX_t = f(X_t)dt + g(X_t)dW_t``.
"""                            
diffusion_function(model::DiffusionStateModel,  t) = diffusion(model, t)
function diffusion(model::DiffusionStateModel, t)
    return model.dynamic_function(t)[2] # returns g(x, p | t)
end
                    
initial_condition(model::DiffusionStateModel) = model.init
                
function Base.show(io::IO, ::MIME"text/plain", model::DiffusionStateModel)
    print(io, "Diffusion process model for the hidden state
    initial condition:                      ", model.init isa Distributions.Sampleable ? "random" : "fixed")
end  
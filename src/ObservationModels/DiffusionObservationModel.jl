struct DiffusionObservationModel{S1, S2, TF} <: ObservationModel{ContinuousTime}
    n::Int
    m::Int
    observation_function::TF
    function DiffusionObservationModel(dim_x, dim_y, h::Function)
        return new{typeof(dim_x),typeof(dim_y),typeof(h)}(dim_x,dim_y,h)
    end
end

#####################
### BASIC METHODS ###
#####################
              
state_dim(model::DiffusionObservationModel) = model.n
obs_dim(model::DiffusionObservationModel) = model.m
noise_dim(model::DiffusionObservationModel) = model.m

function Base.show(io::IO, ::MIME"text/plain", model::DiffusionObservationModel)
    print(io, "Diffusion process model for the observation
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    type of observation:                    ", obs_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model))
end

function Base.show(io::IO, model::DiffusionObservationModel)
    print(io, obs_dim(model),"-dimensional diffusion with ", noise_dim(model), "-dimensional Brownian motion")
end

function Base.show(io::IO, ::MIME"text/plain", model::Type{DiffusionObservationModel{S,T}}) where {S,T}
    print(io, "DiffusionObservationModel{", S, ",", T, "}")
end

function Base.show(io::IO, model::Type{DiffusionObservationModel{S,T}}) where {S,T}
    print(io, "DiffusionObservationModel")
end

######################
### TIME EVOLUTION ###
######################

function (model::DiffusionObservationModel{S1, S2, TF})(x::AbstractVector{S1}, dt) where {S1, S2, TF}
    dV = randn(S1, noise_dim(model))
    h  = observation_function(model)
    return h(x) * dt + dV * sqrt(dt)
end

function (model::DiffusionObservationModel)(x::AbstractMatrix{T}, dt) where T
    dV = randn(T, noise_dim(model), size(x, 2))
    h  = observation_function(model)
    return mapslices(h, x, dims=1) * dt + dV * sqrt(dt)
end

function (model::DiffusionObservationModel{S1, S2, TF})(x::AbstractVector{Vector{S1}}, dt) where {S1, S2, TF}
    return [model(y, dt) for y in x]
end
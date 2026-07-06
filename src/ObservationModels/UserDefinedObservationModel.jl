struct UserDefinedDiscreteObservationModel{TF} <: ObservationModel{DiscreteTime}
    observation_function::TF
    function UserDefinedDiscreteObservationModel(h::Function)
        return new{typeof(h)}(h)
    end
end

#####################
### BASIC METHODS ###
#####################
         
observation_function(obs_model::UserDefinedDiscreteObservationModel, t) = emission(obs_model, t)
function emission(model::UserDefinedDiscreteObservationModel, t)
    return model.observation_function
end
function Base.show(io::IO, ::MIME"text/plain", model::UserDefinedDiscreteObservationModel)
    print(io, "User defined model for the observation")
end


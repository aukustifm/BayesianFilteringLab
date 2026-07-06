##########################################
### Generate solutions to ODEs/SDEs ######
##########################################

"""
    SimulationMeta

Meta data for a collection of simulations.
"""
mutable struct SimulationMeta{DM, OM, Int64, Int64, IN, SP, IM, IS, T1, Int64} <: AbstractSimulation
    dynamical_model::DM
    obs_model::OM
    state_dims::Int64
    obs_dims::Int64
    inputs::IN
    simulation_parameters::SP
    initialμ::IM
    initialΣ::IS
    T::T1
    nruns::Int64
    function SimulationMeta(f, h, nx, ny, inputs, simpar, initialμ, initialΣ,  T, nruns)
        new{typeof(f), typeof(h), typeof(nx), typeof(ny), typeof(inputs), typeof(simpar), typeof(initialμ), typeof(initialΣ), typeof(T), typeof(nruns)}(f, h, nx, ny, inputs, simpar, initialμ, initialΣ, T, nruns)
    end 
end

"""
    run!(SimulationMeta)

Runs collection of simulations.
"""
function run(sim::AbstractSimulation; records = ())
        
    f = sim.dynamical_model
    h = sim.obs_model
    Nx = sim.state_dims 
    Ny = sim.obs_dims
    inputs = sim.inputs
    δᵤ, δₓ, δᵧ = sim.simulation_parameters 
    initialμ = sim.initialμ 
    initialΣ = sim.initialΣ
    T = sim.T
    nruns = sim.nruns     
    
    runs_memoryₓ = Array{Float64}(undef, nruns, Integer(T/δₓ)+1, Nx)        
    runs_memoryᵧ = Array{Float64}(undef, nruns, Integer(T/δₓ)+1, Ny)        

    runs_memoryₓ[:, 1, :] = initialμ

    Base.Threads.@threads for i in 1:nruns
        println("Iteration ", i, " of ", nruns)
        initd = MvNormal(initialμ[i, :], initialΣ);
        output = run!(SystemSimulation(f, h, Nx, Ny, inputs, δᵤ, δₓ, δᵧ, initd, T));
        X, Y = output
        runs_memoryₓ[i, :, :] = X[:x]
        runs_memoryᵧ[i, :, :] = Y[:y]
    end;

    datetimeₓ = X[:t]
    datetimeᵧ = Y[:t]

    X = Dict(:t =>  datetimeₓ , :x_dt => runs_memoryₓ)
    Y = Dict(:t => datetimeᵧ, :y_dt => runs_memoryᵧ)

    return [X, Y]
end


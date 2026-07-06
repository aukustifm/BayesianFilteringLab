abstract type AbstractSimulation end
abstract type Simulation end
abstract type System end
abstract type Integration end
abstract type Algorithm end

struct DynamicalSystem{SM <: Function,
                OM <: Function,
                DIMₓ <: Integer,
                DIMᵧ <: Integer} <: System 
   
    state_model::SM
    observation_model::OM
    dimₓ::DIMₓ
    dimᵧ::DIMᵧ
end

struct DiffEqIntegration{IN <: Function,
                DTᵤ <: Real,
                DTₓ <: Real,
                DTᵧ <: Real,
                TI} <: Integration 

    inputs::IN
    dtᵤ::DTᵤ
    dtₓ::DTₓ
    dtᵧ::DTᵧ 
    init_d::TI
end

mutable struct IntAlgorithm{A1} <: Algorithm 
    name::A1
end

struct SystemSimulation{DS<:DynamicalSystem, DEI<: DiffEqIntegration , TIME <: Real, ALG <: IntAlgorithm} <: Simulation 

    system::DS
    integration::DEI
    T::TIME
    alg::ALG

end

using DifferentialEquations

SystemSimulation(state_model, 
                    observation_model, 
                    dimₓ, dimᵧ, 
                    inputs, 
                    dtᵤ, dtₓ, dtᵧ,
                    init_d,
                    T; 
                    alg = nothing) = SystemSimulation(DynamicalSystem(state_model, observation_model, dimₓ, dimᵧ), 
                                                            DiffEqIntegration(inputs, dtᵤ, dtₓ, dtᵧ, init_d), 
                                                            T,
                                                            IntAlgorithm(alg)
                                                        )

SystemSimulation(system, 
    int, 
    T, 
    ; 
    alg = IntAlgorithm(nothing)) = SystemSimulation(system, int, T, alg)               



function Base.show(io::IO, sim::SystemSimulation)
    print(io, "Simulation of system dynamics and observations
    Stopping time T:                        ", sim.T,"
    dimension of states:                    ", sim.system.dimₓ,"
    dimension of observations:              ", sim.system.dimᵧ,"
    size of input time step:                ", sim.integration.dtᵤ,"
    size of state time step:                ", sim.integration.dtₓ,"
    size of observation time step:          ", sim.integration.dtᵧ,"
    Initial position from:                  ", sim.integration.init_d)
end

"""
    run!(simulation)

Runs the simulation.
"""
function run!(sim::Simulation; records = ())
    
    dimₓ = sim.system.dimₓ
    dimᵧ = sim.system.dimᵧ

    if length(sim.integration.dtᵤ)==1 
        dtᵤ = round(sim.integration.dtᵤ, digits=3)
        dtₓ = round(sim.integration.dtₓ, digits=3)
        dtᵧ = round(sim.integration.dtᵧ ,digits=3)
    end 

    if dtₓ == dtᵧ  
        has_continuous_obs = true
        obs_model2 = DiffusionObservationModel(sim.system.dimₓ, sim.system.dimᵧ, sim.system.observation_model)
        yₖ = zeros(dimᵧ) #[0.2, 1.0]
    else 
        has_continuous_obs = false 
        obs_model2 = UserDefinedDiscreteObservationModel(sim.system.observation_model)
    end

    Mx = Integer(round(dtᵤ/dtₓ, digits=3))
    My = round(dtᵤ/dtᵧ, digits=3)

    T = sim.T

    Nu = Integer(round(T/dtᵤ))
    Ny = Integer(round(T/dtᵧ))+1
    Nx = Integer(round(T/dtₓ))+1

    ty_vec = range(0., length = Ny, step = dtᵧ)
    tu_vec = range(0., stop = ty_vec[end], step = dtᵤ)
    tx_vec = range(0., stop = ty_vec[end], step = dtₓ)

    algname = sim.alg.name

    t0 = 0.

    u_func = sim.integration.inputs
            
    x0 = rand(sim.integration.init_d) .* ones(dimₓ)
    u0 = copy(x0)

    XX = Array{Float64}(undef, Nx, dimₓ)
    YY = Array{Float64}(undef, Ny, dimᵧ) #For now, observations are as frequent as inputs u(t)

    XX[1, :] = x0
    YY[1, :] = zeros(dimᵧ)
    
    # Observing Initial Point
    inp = u_func(t0)
    p = (inp[:η][:u], inp[:η][:d], inp[:p][:pₓ], inp[:w][:wₓ])
    prob = sim.system.state_model(t0)(x0, (0., dtᵤ), p)
    function default_algorithm(prob)
        if prob isa ODEProblem
            return Tsit5()
        elseif prob isa SDEProblem
            return SOSRI()
        else
            error("No default algorithm defined")
        end
    end
    if isnothing(algname)
        algname = default_algorithm(prob)
    end

    print("Running simulation with 
    Algorithm:                        ", algname)

    @showprogress for k in 1:1:Nu
        
        t0 = tu_vec[k]
    
        dyn_func = sim.system.state_model(t0)
        inp = u_func(t0)

        u = inp[:η][:u]
        d = inp[:η][:d]

        pₓ = inp[:p][:pₓ]
        pᵧ = inp[:p][:pᵧ]
        
        wₓ = inp[:w][:wₓ]
        wᵧ = inp[:w][:wᵧ]

        # Run Dynamical Model for range         
        tvec = t0:dtₓ:(t0+dtᵤ)
        p = (u, d, pₓ, wₓ)
        sol = solve(dyn_func(u0, (tvec[1], tvec[end]), p), algname);
   
        xvec = sol(tvec[2:end]).u'
        xvec = permutedims(reshape(hcat(xvec...),(length(xvec[1]), length(xvec))))

        x₀ = sol(tvec[1])
     
        XX[(k-1) * Mx + 2: k*Mx+1, :] = xvec; ## XX is populated with initial value in the beginning

        tyₖ =  round(tvec[end], digits=3)
        if (has_continuous_obs)
            # Run Continuous Observation Model
            inp = u_func(tyₖ)
            u = inp[:η][:u]
            d = inp[:η][:d]
            pᵧ = inp[:p][:pᵧ]
            wᵧ = inp[:w][:wᵧ]
            p = (u, d, pᵧ, wᵧ)
            hₕ, kₕ  = obs_model2.observation_function(tyₖ) 
            
            ###=
            # Fine computation
            drift_cum = zeros(dimᵧ)
            for i in 1:length(sol.t)-1
                drift_cum += hₕ(sol.u[i], p, sol.t[i]) * (sol.t[i+1]-sol.t[i]) 
            end
            if obs_model2.m==1
                dyₖ =  drift_cum +   kₕ(x₀, p, tyₖ) .* rand(MvNormal(zeros(obs_model2.m), Matrix(I, dimᵧ, dimᵧ) .* (dtᵧ)))
            else
                A = kₕ(x₀, p, tyₖ)
                dyₖ =  drift_cum +  sqrt(dtᵧ)*randn(obs_model2.m, 1).*@view(A[diagind(A)]) 
            end
            yₖ = YY[Integer(round(k*My)),:] +  dyₖ
            ##=#

            #=
            # Sparse computation
            if obs_model2.m==1
                dyₖ =  hₕ(sol.u[end], p, sol.t[end]) * dtᵧ +  kₕ(x₀, p, tyₖ) .* rand(MvNormal(zeros(obs_model2.m), Matrix(I, dimᵧ, dimᵧ) .* (dtᵧ)))# drift_cum +   kₕ(x₀, p, tyₖ) .* rand(MvNormal(zeros(obs_model2.m), Matrix(I, dimᵧ, dimᵧ) .* (dtᵧ)))
            else
                A = kₕ(x₀, p, tyₖ)
                dyₖ =  hₕ(sol.u[end], p, sol.t[end]) * dtᵧ +  sqrt(dtᵧ)*randn(obs_model2.m, 1).*@view(A[diagind(A)]) 
            end
            =#
            #zₖ = dyₖ#/dtₓ  
            
            YY[Integer(round(k*My))+1,:] .= yₖ;
        else 
            # Run Discrete Observation Model
            if (tyₖ in ty_vec)
                inp = u_func(tyₖ)
                u = inp[:η][:u]
                d = inp[:η][:d]
                pᵧ = inp[:p][:pᵧ]
                wᵧ = inp[:w][:wᵧ]
                p = (u, d, pᵧ, wᵧ)
                hₕ, kₕ = obs_model2.observation_function(tyₖ)
                yₖ = hₕ(xvec[end,:], p, tyₖ)+kₕ(xvec[end,:], p, tyₖ)*randn(dimᵧ)
                YY[Integer(round(k*My)),:] .= yₖ;
            end    
        end
       u0 = xvec[end, :]

    end
    
    X = Dict(:t => tx_vec, :x => XX)
    Y = Dict(:t => ty_vec, :y => YY)

    return [X, Y]
end

"""
    simulate!(filtering_algorithm, filtering_problem, no_of_timesteps, dt)

Runs a simulation of the hidden state, observation, and filtering algorithm for a duration of `no_of_timesteps`.
"""
#=
function simulate!(int_algo::OrdinaryDiffEqAlgorithm, f::Function, h::Function, T, dtᵤ, dtₓ, dtᵧ)
    simulation = ContinuousTimeSimulation(f, h, int_algo, T, dtᵤ, dtₓ, dtᵧ)
    run!(simulation)
end
=#



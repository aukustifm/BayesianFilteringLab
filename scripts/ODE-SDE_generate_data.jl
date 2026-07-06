import Pkg
Pkg.activate(".")

# LIBRARIES ______________________________
using  Random, LinearAlgebra, DifferentialEquations
#, using Plots, DifferentialEquations, Distributions,
using BayesianFilteringLab

Random.seed!(1234)

# EXPERIMENT _____________________________________________________

## DYNAMICS

## WRITE A FUNCTION THAT,
## given u(t), p(t)=p, w(t) or (wdB)
## returns a Differential Equation system, with its independent variables 
## being the Initial value of the states x₀ and the time span for which it needs to be solved.  

function f(t; kwargs...)

    # Drift 
    f1 = function(x, p, t)
      b, s = x
      u, d, pₓ, _ = p
      # Flow input
      F = u
      # Disturbances input (flow concentration)
      bᵢₙ, sᵢₙ = d
      # Volume, yield coefficient, maximum growth rate, half-saturation
      V, k, μ_max, Kₛ, Iₛ = pₓ
      # Dilution rate
      D = F/V
      [    mu_f(s, μ_max, Kₛ, Iₛ) * b + D * (bᵢₙ - b);   # db/dt
      -k * mu_f(s, μ_max, Kₛ, Iₛ) * b + D * (sᵢₙ - s);   # ds/dt
      ]
    end
  
    # Diffusion
    f2 = function(x, p, t)
      b, s = x
      _, _, _, wₓ = p
      # Modelling unknown phenomena via brownian motion
      σ₁, σ₂ = wₓ
      b>=0 ? diff1 = σ₁*b : diff1 = 0
      s>=0 ? diff2 = σ₂*s : diff2 = 0
      [diff1; diff2]
    end
  
    return function(x0, tspan, p; kwargs...) SDEProblem(f1, f2, x0, tspan, p; kwargs...) end
  
end
  
## AUXILIARY
function mu_f(s, μ_max, Kₛ, Iₛ)
    s==0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end

#########################################################################
## OBSERVATION

## WRITE A FUNCTION THAT,
## given u(t), p(t), w(t) or (wdB)
## returns an observation y_k with its independent variable 
## being the vector of states for the specified time t_k  

function h(t; kwargs...)
    
    function hₕ(x, p, t; kwargs...)
        u, d, pᵧ, wᵧ = p
        V, μ_max, Kₛ, Iₛ, γ  = pᵧ
        if (x[1]>0 && x[2]>0)
            return [γ*mu_f(x[2], μ_max, Kₛ, Iₛ)*x[1], x[2], x[1]] # [biogas flow rate Q(t), substrate concentration s(t), biomass concentration b(t)]
        else
          return zeros(3)
        end
    end

    function kₕ(x, p, t; kwargs...)
        u, d, pᵧ, wᵧ = p
        Dr = 3
        # Observation noise
        K = Matrix(I, Dr, Dr) .* [wᵧ[i] for i in 1:Dr] 
        return K
    end

    return hₕ, kₕ    
end

####################################
########## MONOD ###################
####################################

# Biogas, Substrate and Biomass obs

# Means (initial points)
μlist = [1.5 10.0;
        3.0 3.0;
        0.75  0.75;
        1.0 6.0;
        2.0 0.0;
        6.00 2.0;
        1.0 1.0]
# Variance
Σ = 1e-8*Matrix(I, 2,2);

# Parameters
pₓ = (V = 100., k = 10.,  μ_max = 0.3,  k_s = 10.0, K = 0.0)
pᵧ = (V = 100., μ_max = 0.3, k_s = 10.0, K = 0.0, γ = 27.5)

####################################
########## MONOD ODE ###############
####################################

#Noise intensities
wₓ = (σ₁ = 0.0, σ₂ = 0.0)
wᵧ = (σ₁ = 0.0075, σ₂ = 0.0075, σ₃ = 0.0075)

# More parameters
function flow(t) 1. end

# Inputs
function inputs(t)
  d = t -> (bᵢₙ = 0., sᵢₙ  = 100.)            
  η = Dict(:u => flow(t), :d => d(t))
  p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
  w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
end

T = 500 # (hours)
# Time step
δᵤ = 0.01
δₓ = 0.01
δᵧ = 0.01

sim = SimulationMeta(f, h, 2, 3, inputs, (δᵤ, δₓ, δᵧ), μlist, Σ, T, size(μlist, 1))
output = BayesianFilteringLab.run(sim);

using JLD2
jldsave("./simulations/data/1A-continuous/500h_001h_ODE_7runs.jld2", x_dict=output[1], y_dict=output[2])
#x_dict, y_dict = load("./simulations/data/1A-continuous/500h_001h_ODE_7runs.jld2")

####################################
########## MONOD SDE ###############
####################################

wₓ = (σ₁ = 0.05,   σ₂ = 0.05)
function inputs(t)
  d = t -> (bᵢₙ = 0., sᵢₙ = 100.)            
  η = Dict(:u => flow(t), :d => d(t))
  p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
  w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
end
sim = SimulationMeta(f, h, 2, 3, inputs, (δᵤ, δₓ, δᵧ), μlist, Σ, T, size(μlist, 1))
output = BayesianFilteringLab.run(sim);

using JLD2
jldsave("./simulations/data/1A-continuous/500h_001h_SDE_7runs.jld2", x_dict=output[1], y_dict=output[2])
#x_dict, y_dict = load("./simulations/data/1A-continuous/500h_001h_SDE_7runs.jld2")

####################################
########## HALDANE #################
####################################

μlist = [3.0 3.0;
        2.25 2.75;
        0.75 0.5;
        3.5  1.5;
        0.75  0.0;
        0.5 0.5;
        0.1 0.0;
        0.0 0.5;
        1.75 4.0;
        1.65 2.5]
Σ = 1e-8*Matrix(I, 2,2);

pₓ = ( V = 100., κ = 1., μmax = 5.,  Kₛ = 0.5, Iₛ = 5.0)
pᵧ = (V = 100., μmax = 5., Kₛ = 0.5, Iₛ = 5.0, γ = 2.75)

####################################
########## HALDANE ODE #############
####################################

wₓ = (σ₁ = 0.0, σ₂ = 0.0);
wᵧ  = (σ₁ = 0.01, σ₂ = 0.01, σ₃ = 0.01)

function flow(t) 70. end
function inputs(t)
  d = t -> (bᵢₙ = 0., sᵢₙ = 2.)            
  η = Dict(:u => flow(t), :d => d(t))
  p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
  w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
end

T = 25
# Time step (in hours)
δᵤ = 0.005
δₓ = 0.005
δᵧ = 0.005

sim = SimulationMeta(f, h, 2, 3, inputs, (δᵤ, δₓ, δᵧ), μlist, Σ, T, size(μlist, 1))
output = BayesianFilteringLab.run(sim)
jldsave(rootpath .* "/simulations/data/2A-continuous/25h_0005h_ODE_10runs.jld2", x_dict=output[1], y_dict=output[2])
#x_dict, y_dict = load(rootpath .* "/simulations/data/2A-continuous/25h_0005h_ODE_10runs.jld2")

####################################
########## HALDANE SDE #############
####################################

wₓ = (σ₁ = 0.075,   σ₂ = 0.075)
function inputs(t)
  d = t -> (bᵢₙ = 0., sᵢₙ = 2.)            
  η = Dict(:u => flow(t), :d => d(t))
  p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
  w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
end
sim = SimulationMeta(f, h, 2, 3, inputs, (δᵤ, δₓ, δᵧ), μlist, Σ, T, size(μlist, 1))
output = BayesianFilteringLab.run(sim);
jldsave(rootpath .* "/simulations/data/2A-continuous/25h_0005h_SDE_10runs.jld2", x_dict=output[1], y_dict=output[2])
#x_dict, y_dict = load(rootpath .* "/simulations/data/2A-continuous/25h_0005h_SDE_10runs.jld2")

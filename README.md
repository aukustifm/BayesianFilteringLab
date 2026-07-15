# Bayesian Filtering Lab

This repository provides implementations of **traditional filtering methods** and **sequential Monte Carlo methods (particle filters)** for stochastic dynamical systems, with a particular focus on systems driven by **Brownian motion**.

The framework is designed for **low- and moderate-dimensionals state estimation problems** arising in applications such as the control of bioreactors. For filters better suited to high-dimensional problems, we refer the interested reader to the [LocalParticleFlowFilters](https://github.com/aukustifm/LocalParticleFlowFilters) repository.

Although our examples focus on **biochemical systems**, the algorithms are problem-agnostic and can be applied to general nonlinear state-space models.

## Our Approach

As a baseline, we suggest using the bootstrap particle filter with a sufficiently large number of particles or grid-based methods to approximate the solution of the Zakai equation.

As a more advanced approach, we present a novel method that integrates *particle flow* and *hyperbolic cross approximations*: 

- **Particle flow**: We reposition predicted particles through a dynamic law of motion, mitigating sample impoverishment by diversifying particle positions without relying solely on resampling.

- **Hyperbolic cross approximations**: We employ sparse hyperbolic cross expansions to approximate the solution of the weighted Poisson equation arising in feedback particle filtering. This approach exploits mixed regularity in the solution, significantly reducing the computational cost compared with full tensor-product approximations while maintaining high approximation accuracy.

By combining particle flow and sparse approximations, our approach: 

- Reduces the need for resampling, mitigating sample impoverishment;
- Maintains computational efficiency through sparse representations;
- Provides reliable state estimation across a range of nonlinear stochastic systems.

# Associated Publications

This repository forms the computational backbone of the following publications.

Magalhães, J. A. F., Emzir, M. F., & Corona, F.  
*Pathwise approximations to solving the filtering problem for the stochastic chemostat*.  
Under review.

Magalhães, J. A. F., Emzir, M. F., Harjunkoski, I., & Corona, F.  
*Sparse grids for the weighted Poisson equation in particle filtering-based estimation*.  
Under review.

# Repository Structure

```text
src/
├── Basics/                         # Core abstractions and interfaces
│   ├── Abstractions.jl
│   ├── AbstractModel.jl
│   ├── AbstractFilteringProblem.jl
│   ├── AbstractFilteringAlgorithm.jl
│   ├── AbstractFilteringParameters.jl
│   ├── AbstractFilterRepresentation.jl
│   ├── AbstractKolmogorovForwardState.jl
│   ├── FilteringAlgorithm.jl
│   ├── FilteringProblem.jl
│   ├── HiddenStateModel.jl
│   ├── IntegrationAlgorithm.jl
│   ├── ObservationModel.jl
│   ├── ParticleRepresentation.jl
│   ├── UnweightedParticleRepresentation.jl
│   └── WeightedParticleRepresentation.jl
│
├── Filtering/                     # Generic filtering framework
│   ├── Filtering.jl
│   ├── FilteringState.jl
│   └── OptimalFilter.jl
│
├── FilteringAlgorithms/           # State estimation algorithms
│   ├── BPF.jl                     # Bootstrap particle filter
│   ├── BPFGrid.jl                 # Grid-based particle filter
│   ├── SREKF.jl                   # Square-root extended Kalman filter
│   └── FPF.jl                     # Feedback particle filter
│
├── FilterRepresentations/         # Particle ensemble representations
│   ├── WeightedParticleEnsemble.jl
│   └── UnweightedParticleEnsemble.jl
│
├── GainEstimationMethods/         # Gain approximation methods for FPF
│   ├── ConstantGainApproximation.jl
│   ├── GeneralPolynomials.jl
│   └── SemigroupMethod.jl
│
├── StateModels/                   # Hidden state dynamics
│   ├── DiffusionStateModel.jl
│   └── DiscreteTimeStateModel.jl
│
├── ObservationModels/             # Observation processes
│   ├── DiffusionObservationModel.jl
│   └── UserDefinedObservationModel.jl
│
├── Simulation/                    # Simulation utilities
│   ├── Simulation.jl
│   └── create_simulation.jl
│
└── StateEstimationChemostat.jl     # Main package entry point

simulations/
├── data/                          # Generated benchmark datasets
│   ├── 1A-continuous/
│   └── 2A-continuous/

scripts/                           # Reproducible numerical experiments
├── ODE-SDE_generate_data.jl
├── ODE-SDE_generate_plots.jl
├── KFE.jl
├── KSE-monod.jl
├── KSE-haldane.jl
└── hellinger_matrix.jl

Project.toml                       # Julia package dependencies
Manifest.toml                      # Julia environment lock file
init.jl                            # Package initialisation

# Requirements

The code is written in **Julia**.
```

Activate the environment before running the scripts: 

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

# Example 1: Stochastic chemostat (single species, state estimation)

We estimate a 2-dimensional chemostat system. Below is the Hovmöller diagram of a chaotic attractor (θ = 8) for the first 64 coordinates in a simulated trajectory. The initial condition is spatially autocorrelated (with correlation 0.9) and 𝑥 ∼ N(0, 0.01²). The state evolution is discretised with Δ𝑡 = 0.01. The first 100 steps are considered a burn-in period, and are discarded from this plot and subsequent filtering procedures.

```julia
# LIBRARIES ______________________________
using  LinearAlgebra, DifferentialEquations, Parameters, Distributions
using BayesianFilteringLab

"""
Construct the dynamical system: 

The function defines the system dynamics for a given control input `u(t)`,
parameter vector `p`, and process noise `w(t)` (or its stochastic differential
form `w dB`).

  Inputs            : 
- `x₀`              : initial state vector;
- `tspan`           : time interval over which the system is integrated;
- `u(t)`            : control input;
- `p`               : model parameters;
- `w(t)` (or `w dB`): process noise.

Returns: 
- A `DifferentialEquations.jl` problem (or equivalent) representing the system
  dynamics, ready to be solved with DifferentialEquations.jl.
"""
function f(t; kwargs...)

    # Drift 
      f1 = function(x, p, t)
    b, s = x
    u, d, pₓ, _ = p
      # Flow input
      F = u
      # Disturbances (flow concentrations)
      bᵢₙ, sᵢₙ = d
      # Volume, yield coefficient, maximum growth rate, half-saturation, inhibition
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
      # Modelling unknown phenomena via Brownian motion
      σ₁, σ₂ = wₓ
         b>  = 0 ? diff1 = σ₁*b : diff1 = 0
         s>  = 0 ? diff2 = σ₂*s : diff2 = 0
      [diff1; diff2]
    end
  
    return function(x0, tspan, p; kwargs...) SDEProblem(f1, f2, x0, tspan, p; kwargs...) end
  
  end

# Growth kinetics
function mu_f(s, μ_max, Kₛ, Iₛ)
    s == 0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end

# version for filtering: 
function f_filt(t; kwargs...)

  function f(x, p, t; kwargs...)
    b, s    = x
    u, b_in, s_in, V, k, mu_max, k_s, K, σ1, σ2 = p
      F     = u
      D     = F/V
    [    mu_f(s, mu_max, k_s, K) * b + D * (b_in - b);   # db/dt
    -k * mu_f(s, mu_max, k_s, K) * b + D * (s_in - s);   # ds/dt
    ]
  end

  function g(x, p, t; kwargs...)
    u, b_in, s_in, V, k, mu_max, k_s, K, σ1, σ2 = p
      w_x   = [σ1, σ2]
      D     = length(x)
      aux   = Matrix{Float64}(I, D, D);
    for i in 1: D
      aux[i, i] = w_x[i]*x[i]
    end
    return aux
  end
  return f, g
end


"""
Construct the observation model for the dynamical system.

The function should define the observation equation for a given control input
`u(t)`, parameter vector `p(t)`, and measurement noise `w(t)`.

The returned function computes the observation `yₖ` at a specified time `tₖ`
from the corresponding state vector `x(tₖ)`.

  Inputs : 
- `x(tₖ)`: state vector at time `tₖ`;
- `u(tₖ)`: control input at time `tₖ`;
- `p(tₖ)`: model parameters;
- `w(tₖ)`: measurement noise.

  Returns: 
- `yₖ`   : observation vector at time `tₖ`.
"""
# the version for simulation: 
function h(t; kwargs...)
    
    function hₕ(x, p, t; kwargs...)
        _, _,     pᵧ, _  = p
        _, μ_max, Kₛ, Iₛ, γ = pᵧ
        if (x[1]>0 && x[2]>0)
            return [γ*mu_f(x[2], μ_max, Kₛ, Iₛ)*x[1], x[2], x[1]] # [biogas flow rate Q(t), substrate concentration s(t), biomass concentration b(t)]
        else
          return zeros(3)
        end
    end

    function kₕ(x, p, t; kwargs...)
        _, _, _, wᵧ = p
          Dr = 3
        # Observation noise
        K = Matrix(I, Dr, Dr) .* [wᵧ[i] for i in 1:Dr]
        return K
    end

    return hₕ, kₕ    
end

# the version for filtering (using only biogas observations): 
function h_filt(x, r, p, t; kwargs...)
  _, _, _, _, μ_max, Kₛ, Iₛ, γ, σ = p
  return (x[2]>0 && x[1]>0) ? γ*[mu_f(x[2], μ_max, Kₛ, Iₛ)*x[1]] .+ r: zeros(1)
end  

# Initial mean and variance
μ = [1.65, 2.5]; Σ = 1e-16*Matrix(I, 2,2);

# Parameters
pₓ = (V = 100., k = 1.,  μ_max = 5.0,  k_s = 0.5, K = 5.0)
pᵧ = (V = 100., μ_max = 5.0, k_s = 0.5, K = 5.0, γ = 2.75)

function flow(t) 70. end

##############
# Simulation #
##############

wₓ = (σ₁ = 0.05, σ₂ = 0.05)
wᵧ = (σ₁ = 0.01, σ₂ = 0.01, σ₃ = 0.01)
function inputs(t)
d = t -> (bᵢₙ = 0., sᵢₙ = 100.)
η = Dict(:u => flow(t), :d => d(t))
p = Dict(:pₓ => pₓ, :pᵧ => pᵧ)
w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)
return  Dict(:η => η, :p => p, :w => w)
end

T = 16 # (hours)
# Time step
δᵤ = 0.005; δₓ = 0.005; δᵧ = 0.005
    
#Dimensions for simulating the system
Nx = 2; Ny = 3

  initd  = MvNormal(μ, Σ);
  output = run!(SystemSimulation(f, h, Nx, Ny, inputs, δᵤ, δₓ, δᵧ, initd, T));
X, Y     = output
  
x = X[:x]'
tvec  = X[:t][1:cutoff]
dY    = mapslices(diff,Y[:y][:, 1],dims=1)
z     = dY/δᵧ
tᵧvec = collect(Y[:t][2:end])

#############
# Filtering #
#############

wᵧ      = (σ = 0.01)
K       = diagm([wᵧ])
h_aux   = x -> h_filt(x, zeros(1), [zeros(3); [pₓ[:V], pₓ[:μ_max], pₓ[:k_s], pₓ[:K], pᵧ[:γ], wᵧ]], 0.0)
dỸₜvec = inv(K).*dY

# Inputs
function inputs(t)
d = t -> (bᵢₙ = 0., sᵢₙ = 2.)
η = Dict(:u => flow(t), :d => d(t))
p = Dict(:pₓ => pₓ, :pᵧ => pᵧ)
w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)
return  Dict(:η => η, :p => p, :w => w)
end

# Initial condition for filtering
μ = [1.65, 2.5] # remains the same as the initial condition for simulation
Σ = 0.0025*diagm(ones(2)) # uncertainty around the initial condition

my_state_model = DiffusionStateModel(f_filt, MvNormal(μ,Σ))
my_obs_model   = UserDefinedDiscreteObservationModel(h_filt)
my_filt_prob   = FilteringProblem(my_state_model, my_obs_model, 2, 1, 10, 10)

filt_parameters = @with_kw (
dtᵤ = δᵤ,
dtₓ = δₓ,
dtᵧ = δᵧ
)
my_filt_params_continuous = FilteringParameters(filt_parameters())

# Bootstrap particle filter Np = 128
res_method   = "Stratified"
α            = 1.0
Nₚ           = 128
my_filt_algo = FilteringAlgorithm(BPF, (Nₚ, res_method, α))
myfilter     = Filter(my_filt_prob, my_filt_algo, my_filt_params_continuous, inputs, tᵧvec, dY)
output   = run!(myfilter)
bpf_μ, _bpf_Σ, ensemble, details = output
```

# Stochastic State Estimation

This repository provides implementations of **traditional filtering methods** and **sequential Monte Carlo methods (particle filters)** for stochastic dynamical systems, with a particular focus on systems driven by **Brownian motion**.

The framework is designed for **low- and moderate-dimensional state estimation problems** arising in applications such as the control of bioreactors. For filters better suited to high-dimensional problems, we refer the interested reader to the [LocalParticleFlowFilters repository](https://github.com/aukustifm/LocalParticleFlowFilters).

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
init.jl                            # Package initialization

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

# Example 2: Stochastic chemostat (two species, state and parameter estimation)
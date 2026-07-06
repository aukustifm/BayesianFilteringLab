module BayesianFilteringLab

using LinearAlgebra
using Distributions
using Distances
using Statistics
using Random
using PDMats
using SharedArrays
using ProgressMeter
using StatsBase
using DifferentialEquations
using OrdinaryDiffEq
using StochasticDiffEq
using ReverseDiff
using ForwardDiff
using Polynomials
using SpecialPolynomials
using SparseArrays
using RecursiveArrayTools
using Base.Threads

import Statistics.mean
import Statistics.cov
import Statistics.var

####################################
########## CORE FUNCTIONS ##########
####################################

include("Basics/Abstractions.jl")
export
	TimeType,
	DiscreteTime,
	ContinuousTime

include("Basics/AbstractModel.jl")
export
	AbstractModel,
	state_type,
	obs_type,
	time_type

include("Basics/AbstractFilteringAlgorithm.jl")
export
	AbstractFilteringAlgorithm
	AbstractFilterState

include("Basics/AbstractFilteringParameters.jl")
export
	AbstractFilteringParameters
	
include("Basics/AbstractFilteringProblem.jl")
export
	AbstractFilteringProblem,
	state_model,
	obs_model,
	state_dim,
	obs_dim,
	hidden_time_type,
	obs_time_type
	
include("Basics/AbstractFilterRepresentation.jl")
export
	AbstractFilterRepresentation,
	represented_type,
	dim

include("Basics/AbstractKolmogorovForwardState.jl")
export
	AbstractKolmogorovForwardState

include("Basics/HiddenStateModel.jl")
export
	initialize,
	HiddenStateModel
	
include("Basics/ObservationModel.jl")
export
	ObservationModel,
	state_dim,
	obs_dim,
	state_type,
	obs_type,
	time_type,
	emit!

include("Basics/ParticleRepresentation.jl")
export
	ParticleRepresentation,
	get_pos,
	list_of_pos,
	no_of_particles,
	eff_no_of_particles,
	particle_dim,
	dim,
	propagate!

include("Basics/WeightedParticleRepresentation.jl")
export
	WeightedParticleRepresentation,
	get_weight,
	list_of_weights,
	sum_of_weights,
	eff_no_of_particles,
	dim,
	myskewness,
	mykurtosis,
	resample!
	
include("Basics/GainEstimation.jl")
	export
	GainEstimationMethod

include("Basics/UnweightedParticleRepresentation.jl")
export
	UnweightedParticleRepresentation
	
include("Basics/FilteringAlgorithm.jl")
export
	FilteringAlgorithm

	include("Basics/FilteringProblem.jl")
export
	FilteringProblem

include("Basics/FilteringParameters.jl")
export
	FilteringParameters


####################################
########### STATE MODELS ###########
####################################

include("StateModels/DiffusionStateModel.jl")
export
	DiffusionStateModel,
	drift,
	diffusion,
	initial_condition,
	state_dim,
	noise_dim,
	initialize,
	drift_function,
	drift_diff_function,
	diffusion_function

include("StateModels/DiscreteTimeStateModel.jl")
export
	DiscreteTimeStateModel,
	drift,
	initial_condition,
	initialize,
	drift_function
				
####################################
######## OBSERVATION MODELS ########
####################################

include("ObservationModels/UserDefinedObservationModel.jl")
export
	UserDefinedDiscreteObservationModel,
	initial_condition,
	userdefined_h,
	state_dim,
	obs_dim,
	noise_dim,
	observation_function

include("ObservationModels/DiffusionObservationModel.jl")
export
	DiffusionObservationModel,
	state_dim,
	obs_dim,
	noise_dim

####################################
###### FILTER REPRESENTATIONS ######
####################################

include("FilterRepresentations/WeightedParticleEnsemble.jl")
export
	WeightedParticleEnsemble

include("FilterRepresentations/UnweightedParticleEnsemble.jl")
export
	UnweightedParticleEnsemble

include("FilterRepresentations/PoissonEquation.jl")
export
	PoissonEquation
	m

####################################
####### FILTERING ALGORITHMS #######
####################################

# Square root  EKF
include("FilteringAlgorithms/SREKF.jl")
export
	SREKF,
	SREKState,
	propagate_state,
	update_state!

# Particle Filters

# BPF
include("FilteringAlgorithms/BPF.jl")
export
	BPF,
	BPFState,
	initialize,
	no_of_particles,
	mean,
	cov,
	var,
	propagate_state,
	update_state!

# BPF Grid (for Hellinger distances)
include("FilteringAlgorithms/BPFGrid.jl")
export
	BPFGrid,
	BPFGridState,
	initialize,
	no_of_particles,
	mean,
	cov,
	var,
	propagate_state,
	update_state!

# Feedback Particle Filter
include("FilteringAlgorithms/FPF.jl")
export
	FPF,
	FPFState,
	initialize,
	no_of_particles,
	mean,
	cov,
	var,
	propagate_state,
	update_state!


####################################
####### GAIN ESTIMATION METHODS ####
####################################
include("GainEstimationMethods/ConstantGainApproximation.jl")
export
	ConstantGainApproximation

include("GainEstimationMethods/SemigroupMethod.jl")
export
	SemigroupMethod
	
include("GainEstimationMethods/GeneralPolynomials.jl")
export	
	GeneralPolynomialBasis,
	optimise

####################################
############ FILTERING ############
####################################

include("Filtering/Filtering.jl")
export
	Filter,
	run!,
	filter!

####################################
############ SIMULATION ############
####################################

include("Simulation/Simulation.jl")
export
	SystemSimulation,
	DynamicalSystem,
	DiffEqIntegration,
	IntAlgorithm,
	run!,
	simulate!

include("Simulation/create_simulation.jl")
export
	SimulationMeta,
	run

####################################
########## OPTIMAL FILTER ##########
####################################

include("Filtering/OptimalFilter.jl")
export
	FPState,
	KF_discretise,
	KF_assemble,
	KF_propagate,
	quad_trap,
	fd_2d,
	mypad,
	mypad0,
	pf_counter
end
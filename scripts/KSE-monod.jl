import Pkg
Pkg.activate(".")

# LIBRARIES ______________________________
using LinearAlgebra, JLD2

using BayesianFilteringLab
####################################
########## MONOD ###################
####################################

rootpath = pwd()

output_ode = load(rootpath .* "/simulations/data/1A-continuous/500h_001h_ODE_7runs.jld2")
Xode_dict, Yode_dict = output_ode["x_dict"], output_ode["y_dict"]
Xode = Dict(:t => Xode_dict[:t] , :x => Xode_dict[:x_dt][end, :, :])
Yode = Dict(:t => Yode_dict[:t] , :y => Yode_dict[:y_dt][end, :, :])

output_sde = load(rootpath .* "/simulations/data/1A-continuous/500h_001h_SDE_7runs.jld2")
Xsde_dict, Ysde_dict = output_sde["x_dict"], output_sde["y_dict"]
Xsde = Dict(:t => Xsde_dict[:t] , :x => Xsde_dict[:x_dt][end, :, :])
Ysde = Dict(:t => Ysde_dict[:t] , :y => Ysde_dict[:y_dt][end, :, :])

l₁  = 256; l₂ = 256
x₁ = range(.0, stop = 5.0, length = l₁+1)
x₂ = range(.0, stop = 5.0, length = l₂+1)

Δx₁ = x₁[2]-x₁[1]
Δx₂ = x₂[2]-x₂[1]

center₁ = x₁[1:end-1] .+ Δx₁/2
center₂ = x₂[1:end-1] .+ Δx₂/2

p = collect.(Iterators.product(center₁, center₂))
points = copy(p)
points_reshaped = reduce(hcat,points)
points_matrix₁ = reshape(points_reshaped[1,:], l₁, l₂)
points_matrix₂ = reshape(points_reshaped[2,:], l₁, l₂)

# Parameters
pₓ = (V = 100., k = 10.,  μ_max = 0.3,  Kₛ = 10.0, Iₛ = 0.0)
pᵧ = (V = 100., k = 10.,  μ_max = 0.3,  Kₛ = 10.0, Iₛ = 0.0, γ = 27.5)

# More parameters
function flow(t) 1. end

# Inputs
function inputs(t)
  d = t -> (bᵢₙ = 0., sᵢₙ = 100.)            
  η = Dict(:u => flow(t), :d => d(t))
  p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
  w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
end

## Prior - Likelihood - Posterior

δt = 0.01
tvec = 0:δt:16

# 1A (Biogass measurement)

function mu_f(s, μ_max, Kₛ, Iₛ)
  s==0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end
function h(x, r, p, t; kwargs...)
  V, μ_max, Kₛ, Iₛ, γ, σ = p
  return (x[2]>0 && x[1]>0) ? γ*[mu_f(x[2], μ_max, Kₛ, Iₛ)*x[1]] .+ r : zeros(1)
end  

wᵧ = 0.01

h_aux = x -> h(x, zeros(1), [pₓ[:V], pₓ[:μ_max], pₓ[:Kₛ], pₓ[:Iₛ], pᵧ[:γ], wᵧ], 0.0)

cutoff = 1601
obs = Ysde[:y][201:200:cutoff, 1]

tᵧvec = Ysde[:t][collect(201:200:cutoff)]
Δ = tᵧvec[2]-tᵧvec[1]
K = diagm([wᵧ*sqrt(Δ/δt)])
h̃ₜ = inv(K).*h_aux.(points)
Ỹₜvec = inv(K).*obs
Nₜ = cutoff

## Run splitting up-algorithm

# Initial conditions
μ = [1.0, 1.0]
Σ =  0.0025*diagm(ones(2))
my_state_model = DiffusionStateModel(dynamics, MvNormal(μ, Σ))
kfstate = KFState(my_state_model, points)

μ₁, μ₂, D₁, D₂, D₁₂, D₂₁ = KF_discretise(kfstate, inputs, 0.)

kfsystem = function(u,p,t)
  KF_assemble(u, μ₁, μ₂, D₁, D₂, D₁₂; Δx₁=Δx₁, Δx₂=Δx₂)
end

kse_history = Array{Float64}(undef, Integer(ceil(T/δt))+1, l₁, l₂)
kse_history[1, :, :] = copy(kfstate.pdf)

prior = Array{Float64}(undef, length(tᵧvec), l₁, l₂)
likelihood = Array{Float64}(undef, length(tᵧvec), l₁, l₂)
posterior = Array{Float64}(undef, length(tᵧvec), l₁, l₂)
idx = 1

"""
Finite-difference + Splitting-up approximation, PDE integration
Arguments
    p(0, x): Initial condition for the Kolmorogov forward equation ("KFE")

Returns
  kse_history (Array{Float64}) : the history of p(t, x)
  prior (Array{Float64}) : the history of \bar{p}^{\Delta}
  likelihood (Array{Float64}) : the history of \zeta(x)
  posterior (Array{Float64}) : the history of p^{\Delta}(t, x)
""" 

@showprogress for n in 1:(Nₜ-1)
  n == 1 ?  u₀ = kfstate.pdf :  u₀ = pₜᵟ 

  # KOLMOGOROV FORWARD (Prediction step)
  # For fast computation, explicit method Tsit5(), DP8()...
  # For high accuracy yet explicit/fast: Rodas5()
  # For high accuracy: Trapezoid()
  non_normalised_sol = KF_propagate(u₀, tvec[n], δt, Tsit5(), kfsystem) 
  pₜᵟ = non_normalised_sol ./ quad_trap(non_normalised_sol, center₂, center₁)

  # INNOVATION PROCESS (Update step), use Kallianpur–Striebel formula
  tedge = round(tvec[n]+δt, digits=3)
  if (tedge in tᵧvec) 
    prior[idx,:,:] = copy(pₜᵟ)
    ## Likelihood ratio
    z̃ₜ = Ỹₜvec[findall(x -> x == tedge, tᵧvec)][]
    innovation = x -> (x .-  z̃ₜ)'*(x .-  z̃ₜ)
    ηₜ = reshape(reduce(vcat, innovation.(h̃ₜ)), l₁, l₂)
    ϕₙᵟ = η -> exp(-0.5*Δ*η)
    Lₜ =  ϕₙᵟ.(ηₜ) 
    likelihood[idx,:,:] = Lₜ
    ## Posterior
    pₜᵟ = pₜᵟ .* Lₜ
    pₜᵟ /= quad_trap(pₜᵟ, center₂, center₁)
    posterior[idx,:,:] = copy(pₜᵟ)
    idx += 1
  end
  kse_history[n+1, :, :] = pₜᵟ
  global pₜᵟ

end

## Choose which measurement update to plot (idx ∈ (1st,2nd..., 8th))
idx = 1

ff_prior = CairoMakie.Figure(size = (800, 800) , dpi=500,
  fontsize = 40)
ax1 = ff_prior[1, 1] = Axis(ff_prior, xlabel = "", ylabel = "", title = "", 
  aspect = 1, backgroundcolor = :white)
CairoMakie.heatmap!(ax1, center₁,  center₂, prior[idx, :, :]', colormap=:coolwarm)
hidedecorations!(ax1)
ff_prior

ff_likelihood = CairoMakie.Figure(size = (800, 800) , dpi=500,
fontsize = 40)
ax1 = ff_likelihood[1, 1] = Axis(ff_likelihood, xlabel = "", ylabel = "", title = "", 
aspect = 1, backgroundcolor = :white)
CairoMakie.heatmap!(ax1, center₁,  center₂, likelihood[idx, :, :]', colormap=:coolwarm)
hidedecorations!(ax1)
ff_likelihood

ff_posterior = CairoMakie.Figure(size = (800, 800) , dpi=500,
fontsize = 40)
ax1 = ff_posterior[1, 1] = Axis(ff_posterior, xlabel = "", ylabel = "", title = "", 
aspect = 1, backgroundcolor = :white)
CairoMakie.heatmap!(ax1, center₁, center₂, posterior[idx, :, :]', colormap=:coolwarm)
hidedecorations!(ax1)
ff_posterior

## For other measurement functions:

# Substrate only
function h(x, r, p, t; kwargs...)
  V, μ_max, Kₛ, Iₛ, γ, σ = p
  return (x[2]>0 && x[1]>0) ? x[2] .+ r : zeros(1)
end   
wᵧ = 0.02
h_aux = x -> h(x, zeros(1), [pₓ[:V], pₓ[:μ_max], pₓ[:Kₛ], pₓ[:Iₛ], pᵧ[:γ], wᵧ], 0.0)
K = diagm([wᵧ*sqrt(Δ/δt)])
h̃ₜ = inv(K).*h_aux.(points)
cutoff = 1600
obs = Ysde[:y][200:200:cutoff, 2]

# [...]

# OR Biomass only

function h(x, r, p, t; kwargs...)
  V, μ_max, Kₛ, Iₛ, γ, σ = p
  return (x[2]>0 && x[1]>0) ? x[1] .+ r : zeros(1)
end  
wᵧ = 0.02
h_aux = x -> h(x, zeros(1), [pₓ[:V], pₓ[:μ_max], pₓ[:Kₛ], pₓ[:Iₛ], pᵧ[:γ], wᵧ], 0.0)
K = diagm([wᵧ*sqrt(Δ/δt)])
h̃ₜ = inv(K).*h_aux.(points)
cutoff = 1600
obs = Ysde[:y][200:200:cutoff, 2]
Ỹₜvec = inv(K).*obs
# [....]
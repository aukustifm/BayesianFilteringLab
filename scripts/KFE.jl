import Pkg
Pkg.activate(".")

# LIBRARIES ______________________________
using LinearAlgebra, DifferentialEquations, Distributions, OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock
using BayesianFilteringLab

using CairoMakie
kfe_plotting = function(center₁, center₂, kfe_history, t_idx)

    p_masked = kfe_history[t_idx, :, :]  

    i = 1
    p_b = zeros(length(center₁))
    for b in center₂
      p_b[i] =  sum(p_masked[i,:].*(x₂[2]-x₂[1]))
      i +=1 
    end
    p_b ./= sum(p_b)

    i = 1
    p_s = zeros(length(center₂))
    for s in center₁
      p_s[i] =  sum(p_masked[:,i].*(x₁[2]-x₁[1]))
      i +=1 
    end
    p_s ./= sum(p_s)

    fig = CairoMakie.Figure(size = (800, 800) , dpi=500,
    fontsize = 40)

    ax1 = fig[1,1] = Axis(fig, xlabel = "", ylabel = "", title = "", #xaxis=false,
    aspect = 4, backgroundcolor = :white; spinewidth=0)
    ax2 = fig[1, 2] = Axis(fig, xlabel = "", ylabel = "", title = "", #xaxis=false,
    aspect = 1, backgroundcolor = :white;spinewidth=0)
    ax3 = fig[2, 1] = Axis(fig, xlabel = "", ylabel = "", title = "", #xaxis=false,
    aspect = 1, backgroundcolor = :white)
    ax4 = fig[2, 2] = Axis(fig, xlabel = "", ylabel = "", title = "", #xaxis=false,
    aspect = 0.25, backgroundcolor = :white;spinewidth=0)
  
    CairoMakie.heatmap!(ax3, center₁,  center₂, kfe_history[t_idx, :, :]', colormap=:coolwarm)
    CairoMakie.lines!(ax1, center₂,  p_s,  linewidth = 2.5, color = p_s, colormap = cgrad(:coolwarm))
    CairoMakie.lines!(ax4, p_b,  center₁,  linewidth = 2.5, color = p_b, colormap = cgrad(:coolwarm))

    # Make the limits match the cell edges so the heatmap fills the axis.
    limits!(
      ax3,
      first(center₁) - Δx₁/2, last(center₁) + Δx₁/2,
      first(center₂) - Δx₂/2, last(center₂) + Δx₂/2
    )

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)
    hidedecorations!(ax4)
    rowsize!(fig.layout, 2, Auto(4))
    colsize!(fig.layout, 1, Auto(4))

    # Compute edge-aligned limits so the heatmap fills the axis
    x1min = first(center₁) - Δx₁/2
    x1max = last(center₁)  + Δx₁/2
    x2min = first(center₂) - Δx₂/2
    x2max = last(center₂)  + Δx₂/2

    limits!(ax3, x1min, x1max, x2min, x2max)
    ylims!(ax4, x2min, x2max)  # match y-limits

    linkxaxes!(ax1, ax3)
    linkyaxes!(ax3, ax4)
    fig
end

function dynamics(t; kwargs...)

  # Drift 
  function f(x, p, t; kwargs...)
    b, s = x
    F, bᵢₙ, sᵢₙ, V, k, μ_max, Kₛ, Iₛ, σ₁, σ₂ = p
    # Dilution rate
    D = F/V
    [    mu_f(s, μ_max, Kₛ, Iₛ) * b + D * (bᵢₙ - b);   # db/dt
    -k * mu_f(s, μ_max, Kₛ, Iₛ) * b + D * (sᵢₙ - s);   # ds/dt
    ]
  end

  # Diffusion
  g = function(x, p, t)
    F, bᵢₙ, sᵢₙ, V, k, μ_max, Kₛ, Iₛ, σ₁, σ₂ = p
    # Modelling unknown phenomena via brownian motion
    D = length(x)
    aux = Matrix{Float64}(I, D, D);
    wₓ = [σ₁, σ₂]
    for i in 1:D
      aux[i, i] = wₓ[i]*x[i]
    end
    return aux
  end

  return f, g
end

## AUXILIARY
function mu_f(s, μ_max, Kₛ, Iₛ)
  s==0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end

####################################
########## MONOD ###################
####################################

# Parameters
pₓ = (V = 100., k = 10.,  μ_max = 0.3,  Kₛ = 10.0, Iₛ = 0.0)
pᵧ = ()

# Noise intensities
wₓ =  (σ₁ = 0.05, σ₂ = 0.05)
wᵧ = ()

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

# Initial conditions
μ = [1.0, 1.0]
Σ =  0.0025*diagm(ones(2))

## Create Grid
lᵢ = 128 #(32, 128, 256, 512)

l₁  = lᵢ; l₂ = lᵢ
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

## KOLMOGOROV FORWARD
my_state_model = DiffusionStateModel(dynamics, MvNormal(μ, Σ))
kfstate = BayesianFilteringLab.KFState(my_state_model, points)

μ₁, μ₂, D₁, D₂, D₁₂, D₂₁ = BayesianFilteringLab.KF_discretise(kfstate, inputs, 0.)

kfsystem = function(u,p,t)
  BayesianFilteringLab.KF_assemble(u, μ₁, μ₂, D₁, D₂, D₁₂; Δx₁=Δx₁, Δx₂=Δx₂)
end

u₀ = kfstate.pdf

start = time()
δt = 0.01
T = 16.0 
tvec = 0:δt:T
Nₜ = length(tvec)

kfe_history = Array{Float64}(undef, Integer(ceil(T/δt))+1, l₁, l₂)
kfe_history[1, :, :] = copy(kfstate.pdf)
  
"""
Finite-difference approximation, PDE integration
Arguments
    p(0, x): Initial condition for the Kolmorogov forward equation ("KFE")

Returns
  kfe_history (Array{Float64}) : the history of p(t, x)
"""

using ProgressMeter 
@showprogress for n in 1:(Nₜ-1)
  n == 1 ?  u₀ = kfstate.pdf :  u₀ = pₜᵟ     
  # For fast computation, explicit method Tsit5(), DP8()...
  # For high accuracy yet explicit/fast: Rodas5()
  # For high accuracy: Trapezoid()
  non_normalised_sol = KF_propagate(u₀, tvec[n], δt, Tsit5(), kfsystem) #Vern7(), Tsit5(), Vern7, DP8(), Trapezoid(), Rodas5()
  pₜᵟ = non_normalised_sol ./ quad_trap(non_normalised_sol, center₂, center₁)  
  kfe_history[n+1, :, :] = copy(pₜᵟ)
  global pₜᵟ
  #println("Minimum value of p(x,t): ", minimum(pₜᵟ))
end    

#solution at t=0h
kfe_plotting(center₁, center₂, kfe_history, 1)

#solution at t=16h
kfe_plotting(center₁, center₂, kfe_history, 1601)

####################################
########## HALDANE #################
####################################

# Parameters
pₓ = (V = 100., κ = 1., μ_max = 5.,  Kₛ = 0.5, Iₛ = 5.0)
pᵧ = ()

# Noise intensities
wₓ = (σ₁ = 0.075, σ₂ = 0.075);
wᵧ  = ()

# More parameters
function flow(t) 70. end

# Inputs
function inputs(t)
  d = t -> (bᵢₙ = 0., sᵢₙ = 2.)            
  η = Dict(:u => flow(t), :d => d(t))
  p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
  w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
end

# Initial conditions
μ = [1.65, 2.5]
Σ =  0.0025*diagm(ones(2))

## Create Grid
lᵢ = 128 #(32, 128, 256, 512)
l₁  = lᵢ; l₂ = lᵢ;

x₁ = range(.0, stop = 3.0, length = l₁+1)
x₂ = range(.0, stop = 3.0, length = l₂+1)

Δx₁ = x₁[2]-x₁[1]
Δx₂ = x₂[2]-x₂[1]

center₁ = x₁[1:end-1] .+ Δx₁/2
center₂ = x₂[1:end-1] .+ Δx₂/2

p = collect.(Iterators.product(center₁, center₂))
points = copy(p)
points_reshaped = reduce(hcat,points)
points_matrix₁ = reshape(points_reshaped[1,:], l₁, l₂)
points_matrix₂ = reshape(points_reshaped[2,:], l₁, l₂)

## KOLMOGOROV FORWARD
my_state_model = BayesianFilteringLab.DiffusionStateModel(dynamics, MvNormal(μ, Σ))
kfstate = BayesianFilteringLab.KFState(my_state_model, points)

μ₁, μ₂, D₁, D₂, D₁₂, D₂₁ = BayesianFilteringLab.KF_discretise(kfstate, inputs, 0.)

kfsystem = function(u,p,t)
  BayesianFilteringLab.KF_assemble(u, μ₁, μ₂, D₁, D₂, D₁₂; Δx₁=Δx₁, Δx₂=Δx₂)
end

start = time()
δt = 0.005
T = 16.0 
tvec = 0:δt:T
Nₜ = length(tvec)

kfe_history = Array{Float64}(undef, Integer(ceil(T/δt))+1, l₁, l₂)
kfe_history[1, :, :] = copy(kfstate.pdf)
  

#using OrdinaryDiffEqSDIRK

@showprogress for n in 1:(Nₜ-1)
  n == 1 ?  u₀ = kfstate.pdf :  u₀ = pₜᵟ     
  # For fast computation, explicit method Tsit5(), DP8()...
  # For high accuracy yet explicit/fast: Rodas5()
  # For high accuracy: Trapezoid()
  non_normalised_sol = KF_propagate(u₀, tvec[n], δt, Rodas5(), kfsystem) 
  pₜᵟ = non_normalised_sol ./ quad_trap(non_normalised_sol, center₂, center₁)  
  kfe_history[n+1, :, :] = copy(pₜᵟ)
  global pₜᵟ
  println("Minimum value of p(x,t): ", minimum(pₜᵟ))
end

#solution at t=0h
kfe_plotting(center₁, center₂, kfe_history, 1)

#solution at t=16h
kfe_plotting(center₁, center₂, kfe_history, 3201)

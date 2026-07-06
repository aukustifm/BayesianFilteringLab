import Pkg
Pkg.activate(".")

# LIBRARIES ______________________________
using  Random, LinearAlgebra, DifferentialEquations, JLD2, Distributions, ProgressMeter, Parameters
#, using Plots, DifferentialEquations, Distributions,
using BayesianFilteringLab

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
    for i in 1:D
      aux[i, i] = wₓ[i]*x[i]
    end
    return aux
  end

  return f, g
end

# 1A (Biogass measurement)

function mu_f(s, μ_max, Kₛ, Iₛ)
  s==0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end
function h(x, r, p, t; kwargs...)
  F, bᵢₙ, sᵢₙ, V, k, μ_max, Kₛ, Iₛ, γ, σ = p
  return (x[2]>0 && x[1]>0) ? γ*[mu_f(x[2], μ_max, Kₛ, Iₛ)*x[1]] .+ r : zeros(1)
end  

####################################
########## MONOD ###################
####################################

output_sde = load("./simulations/data/1A-continuous/500h_001h_SDE_7runs.jld2")
Xsde_dict, Ysde_dict = output_sde["x_dict"], output_sde["y_dict"]
Xsde = Dict(:t => Xsde_dict[:t] , :x => Xsde_dict[:x_dt][end, :, :])
Ysde = Dict(:t => Ysde_dict[:t] , :y => Ysde_dict[:y_dt][end, :, :])

Plots.plot(Xsde[:t], Xsde[:x][:, 2], label = "Biomass", xlabel = "Time (h)", ylabel = "Concentration (g/L)")
Plots.scatter!(Ysde[:t][2:end], mapslices(diff,Ysde[:y][:, 2],dims=1)/0.01, label = "Biomass obs", xlabel = "Time (h)", ylabel = "Concentration (g/L)")

# Parameters
pₓ = (V = 100., k = 10.,  μ_max = 0.3,  Kₛ = 10.0, Iₛ = 0.0)
pᵧ = (V = 100., k = 10.,  μ_max = 0.3,  Kₛ = 10.0, Iₛ = 0.0, γ = 27.5)

# Noise intensities
wₓ =  (σ₁ = 0.05, σ₂ = 0.05)
wᵧ = (σ = 0.01)

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

cutoff = 1601; δt = 0.01

dY = mapslices(diff,Ysde[:y][:, 1],dims=1)
z = dY[1:cutoff-1]/δt
tᵧvec = Ysde[:t][collect(2:1:cutoff)]
Δ = tᵧvec[2]-tᵧvec[1]

K = diagm([wᵧ*sqrt(Δ/δt)])

h_aux = x -> h(x, zeros(1), vcat(zeros(5), [pₓ[:μ_max], pₓ[:Kₛ], pₓ[:Iₛ], pᵧ[:γ], wᵧ]), 0.0)
h̃ₜ = inv(K).*h_aux.(points)
dỸₜvec = inv(K).*dY

## KUSHNER-STRATONOVICH EQUATION with Splitting up approximation
my_state_model = DiffusionStateModel(dynamics, MvNormal(μ, Σ))
kfstate = BayesianFilteringLab.KFState(my_state_model, points)
μ₁, μ₂, D₁, D₂, D₁₂, D₂₁ = KF_discretise(kfstate, inputs, 0.)
kfsystem = function(u,p,t)
  KF_assemble(u, μ₁, μ₂, D₁, D₂, D₁₂; Δx₁=Δx₁, Δx₂=Δx₂)
end
start = time()
δt = 0.01
T = 16.0 
tvec = 0:δt:T
Nₜ = length(tvec)

pde_integration = function(method)
  """
  Finite-difference + Splitting-up approximation, PDE integration
  
  Arguments
      method  (string) : Kolmorogov forward equation ("KFE") or Kushner-Stratonovich equation ("KSE")
  
  Returns
      pp (Array{Float64}) : the history of p(t, x)
  """

  pp = Array{Float64}(undef, Integer(ceil(T/δt))+1, l₁, l₂)
  pp[1, :, :] = copy(kfstate.pdf)

  @showprogress for n in 1:(Nₜ-1)
    n == 1 ?  u₀ = kfstate.pdf :  u₀ = pₜᵟ     
  
    # KOLMOGOROV FORWARD (Prediction step)
    # For fast computation, explicit method Tsit5(), DP8()...
    # For high accuracy yet explicit/fast: Rodas5()
    # For high accuracy: Trapezoid()
    non_normalised_sol = KF_propagate(u₀, tvec[n], δt, Tsit5(), kfsystem) 
    pₜᵟ = non_normalised_sol./quad_trap(non_normalised_sol, center₂, center₁)

    # INNOVATION PROCESS (Update step), use Kallianpur–Striebel formula
    if method == "KFE"
      pp[n+1, :, :] = pₜᵟ
    elseif method == "KSE"
      tedge = round(tvec[n]+δt, digits=3)
      if (tedge in tᵧvec) 
        ## Likelihood ratio
        z̃ₜ = dỸₜvec[findall(x -> x == tedge, tᵧvec)][]/δt # since dỸₜvec is the increment of the observation process
        innovation = x -> (x .-  z̃ₜ)'*(x .-  z̃ₜ)
        ηₜ = reshape(reduce(vcat, innovation.(h̃ₜ)), l₁, l₂)
        ϕₙᵟ = η -> exp(-0.5*Δ*η)
        Lₜ =  ϕₙᵟ.(ηₜ) 
        ## Posterior
        pₜᵟ = non_normalised_sol .* Lₜ
        pₜᵟ /= quad_trap(pₜᵟ, center₂, center₁)
        pp[n+1, :, :] = pₜᵟ
      end
    else
        break;
    end
    global pₜᵟ
    println("Minimum value of p(x,t): ", minimum(pₜᵟ))
  end    
  return pp
end

kfe_history = pde_integration("KFE")
#save_object(rootpath .* "/simulations/data/1A-continuous/kfe_history", kfe_history)
kfe_history = load_object(rootpath .* "/simulations/data/1A-continuous/kfe_history")
kse_history = pde_integration("KSE")
#save_object(rootpath .* "/simulations/data/1A-continuous/kse_history", kse_history)
kse_history = load_object(rootpath .* "/simulations/data/1A-continuous/kse_history")

my_state_model = DiffusionStateModel(dynamics, MvNormal(μ,Σ))
my_obs_model   = UserDefinedDiscreteObservationModel(h)
my_filt_prob   = FilteringProblem(my_state_model, my_obs_model, 2, 1, 10, 10)

# Time step (in hours)
δᵤ = 0.01 
δₓ = 0.01
δᵧ = 0.01

filt_parameters = @with_kw (
dtᵤ = δᵤ,
dtₓ = δₓ,
dtᵧ = δᵧ
)
my_filt_params_continuous = FilteringParameters(filt_parameters())

## ------------ BPF with Stratified, Continuous, 100000 particles
res_method = "Stratified"
α = 1.0
ub = 5.0 # Grid upper bound limit
Nₗ = 128  # Grid mesh size on a single dimension
Nₚ = Integer(1e4)
    
my_filt_algo = FilteringAlgorithm(BPFGrid, (Nₚ, res_method, α, ub, Nₗ))
myfilter = Filter(my_filt_prob, my_filt_algo, my_filt_params_continuous, inputs, tᵧvec, dY)
output = run!(myfilter)
#save_object(rootpath .* "/simulations/data/1A-continuous/bpf_history", output[5])
bpf_history = output[5]

## ------------ SREKF, Continuous

myfilter = Filter(my_filt_prob, FilteringAlgorithm(SREKF), my_filt_params_continuous, inputs, tᵧvec, z)
output_srekf = run!(myfilter)
mm_srekf = copy(output_srekf[1])
ss_srekf = copy(output_srekf[2])
pp_srekf = similar(ss_srekf)
for i in 1:size(ss_srekf, 1)
  pp_srekf[i, :, :] = ss_srekf[i, :, :]*ss_srekf[i, :, :]'
end
srekf_history = Array{Float64}(undef, Nₜ, l₁, l₂)
for i in 1:Nₜ
  post_ekf_pdf = x -> pdf(MvNormal(mm_srekf[i, :], pp_srekf[i, :, :]), x)
  srekf_history[i, :, :] = post_ekf_pdf.(points)
end
# save_object(rootpath .* "/simulations/data/1A-continuous/srekf_history", srekf_history)

## Hellinger matrix
# Choose idx, idx ∈ (1:1601)

idx = 1601

myN_KFE = kfe_history[idx,:,:]
myN_KSE = kse_history[idx,:,:]
myN_BPF = bpf_history[idx,:,:]
myN_EKF = srekf_history[idx,:,:]

Plots.contour(center₂, center₁, myN_KFE)
Plots.contour(center₂, center₁, myN_KSE)
Plots.contour(center₂, center₁, myN_BPF) #output[5][end,:,:])
Plots.contour(center₂, center₁, myN_EKF)

metamatrix = (myN_KFE, myN_KSE, myN_BPF, myN_EKF)

hellinger_matrix_value = function(matrix₁, matrix₂)
    1-quad_trap(sqrt.(max.(0,matrix₁) .* max.(0,matrix₂)), center₂, center₁)
end

hellinger_matrix = [hellinger_matrix_value(i, j) for j in metamatrix, i in metamatrix]

using CairoMakie, Plots
using LaTeXStrings
Plots.heatmap(hellinger_matrix[end:-1:1, :], color = :coolwarm, clim=(0,1), 
    framestyle=:box,
    size=(800,800), legend=false, axis=nothing)

zz = [L"\textrm{\textbf{KFE}}" "" "" "";
      "" L"\textrm{\textbf{KSE \cdot PDE}}" "" "";
      "" "" L"\textrm{\textbf{KSE \cdot BPF}}" "";
      "" "" "" L"\textrm{\textbf{KSE \cdot EKF}}"]
annotate!( vec(tuple.((1:4)', (1:4), tuple.(zz[end:-1:1, :], 22),:white)))


import Pkg
Pkg.activate(".")

# LIBRARIES ______________________________
using  Random, LinearAlgebra, DifferentialEquations, Parameters, JLD2, Distributions, ProgressMeter, OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock
#, using Plots, DifferentialEquations, Distributions,
using BayesianFilteringLab

Random.seed!(1234)

# AUXILIARY FUNCTIONS __________________________________________________________
makeProgressBar(N, title) = Progress(N, desc=title, dt=0.5, barlen=50, showspeed=true, color=:white);

pde_integration = function(method, lᵢ, l₁, l₂, center₁, center₂, kfstate, kfsystem, h̃ₜ, dỸₜvec, Δ; verbose=false)
  """
  Finite-difference + Splitting-up approximation, PDE integration
  
  Arguments
      method  (string) : Kolmorogov forward equation ("KFE") or Kushner-Stratonovich equation ("KSE")
  
  Returns
      pp (Array{Float64}) : the history of p(t, x)
  """

  pp = Array{Float64}(undef, Nₜ, l₁, l₂)
  pp[1, :, :] = copy(kfstate.pdf)

  timetracker = makeProgressBar(Nₜ-1, "Filtering (Grid: lᵢ=$(lᵢ))")

  @showprogress for n = 1:(Nₜ-1); if verbose; next!(timetracker); end;
    n == 1 ?  u₀ = kfstate.pdf :  u₀ = pₜᵟ     
  
    # KOLMOGOROV FORWARD (Prediction step)
    # For fast computation, explicit method Tsit5(), DP8()...
    # For high accuracy yet explicit/fast: Rodas5()
    # For high accuracy: Trapezoid()
    non_normalised_sol = KF_propagate(u₀, tvec[n], Δ, Tsit5(), kfsystem) 
    pₜᵟ = non_normalised_sol./quad_trap(non_normalised_sol, center₂, center₁)

    # INNOVATION PROCESS (Update step), use Kallianpur–Striebel formula
    if method == "KFE"
      pp[n+1, :, :] = pₜᵟ
    elseif method == "KSE"
      tedge = round(tvec[n]+Δ, digits=3)
      if (tedge in tᵧvec) 
        ## Likelihood ratio
        z̃ₜ = dỸₜvec[findall(x -> x == tedge, tᵧvec)][]/Δ # since dỸₜvec is the increment of the observation process
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
    #println("Minimum value of p(x,t): ", minimum(pₜᵟ))
  end    
  return pp, [timetracker.tinit, timetracker.tsecond, timetracker.tlast] 
end

function means_from_joint_trap(p, center₁, center₂)
  Ny, Nx = size(p)
  
  Δx = center₁[2] - center₁[1]
  Δy = center₂[2] - center₂[1]

  wx = ones(eltype(p), Nx); wx[1] = 0.5; wx[end] = 0.5
  wy = ones(eltype(p), Ny); wy[1] = 0.5; wy[end] = 0.5

  # Ex = ∬ x p(x,y) dx dy
  Ex = 0.0
  Ey = 0.0
  for j in 1:Ny, i in 1:Nx
      w = wy[j] * wx[i]
      Ex += center₁[i] * p[j,i] * w
      Ey += center₂[j] * p[j,i] * w
  end
  Ex *= Δx * Δy
  Ey *= Δx * Δy
  
  return [Ex, Ey]
end


# EXPERIMENT _____________________________________________________


# ----------------------------------------------------------------------------- 
# Dynamical system 
# -----------------------------------------------------------------------------

"""
Construct the dynamical system:

The function defines the system dynamics for a given control input `u(t)`,
parameter vector `p`, and process noise `w(t)` (or its stochastic differential
form `w dB`).

Inputs:
- `x₀`: initial state vector;
- `tspan`: time interval over which the system is integrated;
- `u(t)`: control input;
- `p`: model parameters;
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

# Growth kinetics
function mu_f(s, μ_max, Kₛ, Iₛ)
    s==0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end

# version for filtering:
function f_filt(t; kwargs...)

  function f(x, p, t; kwargs...)
    b, s = x
    u, b_in, s_in, V, k, mu_max, k_s, K, σ1, σ2 = p
    F = u
    D = F/V
    [    mu_f(s, mu_max, k_s, K) * b + D * (b_in - b);   # db/dt
    -k * mu_f(s, mu_max, k_s, K) * b + D * (s_in - s);   # ds/dt
    ]
  end

  function g(x, p, t; kwargs...)
    u, b_in, s_in, V, k, mu_max, k_s, K, σ1, σ2 = p
    w_x = [σ1, σ2]
    D = length(x)
    aux = Matrix{Float64}(I, D, D);
    for i in 1:D
      aux[i, i] = w_x[i]*x[i]
    end
    return aux
  end
  return f, g
end

# ----------------------------------------------------------------------------- 
# Observation system 
# -----------------------------------------------------------------------------

"""
Construct the observation model for the dynamical system.

The function should define the observation equation for a given control input
`u(t)`, parameter vector `p(t)`, and measurement noise `w(t)`.

The returned function computes the observation `yₖ` at a specified time `tₖ`
from the corresponding state vector `x(tₖ)`.

Inputs:
- `x(tₖ)`: state vector at time `tₖ`;
- `u(tₖ)`: control input at time `tₖ`;
- `p(tₖ)`: model parameters;
- `w(tₖ)`: measurement noise.

Returns:
- `yₖ`: observation vector at time `tₖ`.
"""
# the version for simulation:
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

# the version for filtering:
function h_filt(x, r, p, t; kwargs...)
  _, _, _, _, μ_max, Kₛ, Iₛ, γ, σ = p
  return (x[2]>0 && x[1]>0) ? γ*[mu_f(x[2], μ_max, Kₛ, Iₛ)*x[1]] .+ r : zeros(1)
end  

Ns = 20 # Number of simulations

####################################
########## HALDANE ###################
####################################

# Biogas, Substrate and Biomass obs (although we only use the biogas flow rate for filtering)

# Initial mean and variance
μ = [1.65, 2.5]
Σ = 1e-16*Matrix(I, 2,2);

# Parameters
pₓ = (V = 100., k = 1.,  μ_max = 5.0,  k_s = 0.5, K = 5.0)
pᵧ = (V = 100., μ_max = 5.0, k_s = 0.5, K = 5.0, γ = 2.75)

function flow(t) 70. end

T = 25 # (hours)
# Time step
δᵤ = 0.005
δₓ = 0.005
δᵧ = 0.005

Nx = 2
Ny = 3

initd = MvNormal(μ, Σ);

Nₜ = cutoff = 3201; δt = 0.005

Nmethods = 7 # 4 PDE-based discretisation levels + 2 boostrap PF with different ensemble sizes + 1 EKF
RMSE = [zeros(Nmethods, Nₜ) for ns in 1:Ns]
ITAE = [zeros(Nmethods, Nₜ) for ns in 1:Ns]
runtimes = zeros(Ns,Nmethods)

for ns = 1:Ns
  
  # Simulation
  wₓ = (σ₁ = 0.075,   σ₂ = 0.075)
  wᵧ = (σ₁ = 0.01, σ₂ = 0.01, σ₃ = 0.01)
  function inputs(t)
    d = t -> (bᵢₙ = 0., sᵢₙ = 2.)            
    η = Dict(:u => flow(t), :d => d(t))
    p = Dict(:pₓ => pₓ, :pᵧ => pᵧ) 
    w = Dict(:wₓ => wₓ, :wᵧ => wᵧ)                     
  return  Dict(:η => η, :p => p, :w => w)
  end

  output = run!(SystemSimulation(f, h, Nx, Ny, inputs, δᵤ, δₓ, δᵧ, initd, T));
  X, Y = output
  
  Xsde = Dict(:t => X[:t] , :x => X[:x])
  Ysde = Dict(:t => Y[:t] , :y => Y[:y])

  x = Xsde[:x][1:cutoff, :]'
  tvec = Xsde[:t][1:cutoff]
  global tvec
  
  dY = mapslices(diff,Ysde[:y][:, 1],dims=1)[1:cutoff-1]
  z = dY/δt
  tᵧvec = Ysde[:t][collect(2:1:cutoff)]
  Δ = tᵧvec[2]-tᵧvec[1]
  global tᵧvec
  
  # Filtering
  wᵧ = (σ = 0.01)
  K = diagm([wᵧ*sqrt(Δ/δt)])
  h_aux = x -> h_filt(x, zeros(1), [zeros(3); [pₓ[:V], pₓ[:μ_max], pₓ[:k_s], pₓ[:K], pᵧ[:γ], wᵧ]], 0.0)
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
  Σ =  0.0025*diagm(ones(2)) # uncertainty around the initial condition

  lᵢvec = [32, 128, 256, 512]

  for lᵢid = 2:4

    lᵢ = lᵢvec[lᵢid]

    l₁  = lᵢ; l₂ = lᵢ

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

    h̃ₜ = inv(K).*h_aux.(points)

    ## KUSHNER-STRATONOVICH EQUATION with Splitting up approximation
    my_state_model = DiffusionStateModel(f_filt, MvNormal(μ, Σ))
    kfstate = BayesianFilteringLab.KFState(my_state_model, points)
    μ₁, μ₂, D₁, D₂, D₁₂, D₂₁ = KF_discretise(kfstate, inputs, 0.)
    kfsystem = function(u,p,t)
      KF_assemble(u, μ₁, μ₂, D₁, D₂, D₁₂; Δx₁=Δx₁, Δx₂=Δx₂)
    end

    kse_history, kse_runtime = pde_integration("KSE", lᵢ, l₁, l₂, center₁, center₂, kfstate, kfsystem, h̃ₜ, dỸₜvec, Δ;verbose=true)
   
    # Computing the mean estimate 
    xe =hcat([means_from_joint_trap(kse_history[k, :, :]', center₁, center₂) for k in 1:Nₜ]...)

    # Computing RMSE and ITAE
    RMSE[ns][lᵢid, :] = (sqrt.(sum((xe-x).^2,dims=1))[:])
    ITAE[ns][lᵢid, :] = cumsum(δt*[0.0; tᵧvec].*(sqrt.(sum((xe-x).^2,dims=1))[:]))

    # Computing running times (s/Iter)
    runtimes[ns, lᵢid] = (kse_runtime[3]-kse_runtime[1])/(Nₜ-1)
  end

  my_state_model = DiffusionStateModel(f_filt, MvNormal(μ,Σ))
  my_obs_model   = UserDefinedDiscreteObservationModel(h_filt)
  my_filt_prob   = FilteringProblem(my_state_model, my_obs_model, 2, 1, 10, 10)

  filt_parameters = @with_kw (
  dtᵤ = δᵤ,
  dtₓ = δₓ,
  dtᵧ = δᵧ
  )
  my_filt_params_continuous = FilteringParameters(filt_parameters())

  # Bootstrap particle filter Np=128
  res_method = "Stratified"
  α = 1.0
  Nₚ = 128
  my_filt_algo = FilteringAlgorithm(BPF, (Nₚ, res_method, α))
  myfilter = Filter(my_filt_prob, my_filt_algo, my_filt_params_continuous, inputs, tᵧvec, dY)
  output_bpf = run!(myfilter)
  mm_bpf = copy(output_bpf[1])'

  RMSE[ns][5, :] = (sqrt.(sum((mm_bpf-x).^2,dims=1))[:])
  ITAE[ns][5, :] = cumsum(δt*[0.0; tᵧvec].*(sqrt.(sum((mm_bpf-x).^2,dims=1))[:]))

  # Computing running times (s/Iter)
  runtimes[ns, 5] = output_bpf[4]["clocktime"]/(Nₜ-1)

  # Bootstrap particle filter Np=1024
  res_method = "Stratified"
  Nₚ = 1024
  my_filt_algo = FilteringAlgorithm(BPF, (Nₚ, res_method, α))
  myfilter = Filter(my_filt_prob, my_filt_algo, my_filt_params_continuous, inputs, tᵧvec, dY)
  output_bpf = run!(myfilter)
  mm_bpf = copy(output_bpf[1])'

  RMSE[ns][6, :] = (sqrt.(sum((mm_bpf-x).^2,dims=1))[:])
  ITAE[ns][6, :] = cumsum(δt*[0.0; tᵧvec].*(sqrt.(sum((mm_bpf-x).^2,dims=1))[:]))

  # Computing running times (s/Iter)
  runtimes[ns, 6] = output_bpf[4]["clocktime"]/(Nₜ-1)

  # Extended Kalman Filter (EKF)

  myfilter = Filter(my_filt_prob, FilteringAlgorithm(SREKF), my_filt_params_continuous, inputs, tᵧvec, z)
  output_srekf = run!(myfilter)
  mm_srekf = copy(output_srekf[1])'

  RMSE[ns][7, :] = (sqrt.(sum((mm_srekf-x).^2,dims=1))[:])
  ITAE[ns][7, :] = cumsum(δt*[0.0; tᵧvec].*(sqrt.(sum((mm_srekf-x).^2,dims=1))[:]))

  # Computing running times (s/Iter)
  runtimes[ns, 7] = output_srekf[4]["clocktime"]/(Nₜ-1)


  jldsave("./res/20runs_Haldane_run$(ns).jld2"; RMSE = RMSE[ns], ITAE=ITAE[ns], runtimes=runtimes[ns,:])
end
jldsave("./res/20runs_Haldane.jld2"; RMSE = RMSE, ITAE=ITAE, runtimes=runtimes)

#=
xe = hcat([means_from_joint_trap(kfe_history[k, :, :]', center₁, center₂) for k in 1:Nₜ]...)
Plots.plot!(xe[1,:], label = "lᵢ=128 (KFE)")

Plots.plot(xe[2,:], label = "lᵢ=128")
Plots.plot!(mm_bpf[2,:], label = "BPF Np=128")
Plots.plot!(mm_srekf[2,:], label = "EnKF")
Plots.plot!(x[2,:], label = "Exact")
=#

#=
output = load("./res/20runs_Monod.jld2")
RMSE, ITAE, runtimes = output["RMSE"], output["ITAE"], output["runtimes"]
using Plots
Plots.plot([tᵧvec], ITAE[ns][1, 2:end],yscale=:log10, label = "lᵢ=32", xlabel = "Time (h)", ylabel = "RMSE")
Plots.plot!([tᵧvec], ITAE[ns][2, 2:end], label = "lᵢ=128", xlabel = "Time (h)", ylabel = "RMSE")
Plots.plot!([tᵧvec], ITAE[ns][3, 2:end], label = "lᵢ=256", xlabel = "Time (h)", ylabel = "RMSE")
Plots.plot!([tᵧvec], ITAE[ns][4, 2:end], label = "lᵢ=512", xlabel = "Time (h)", ylabel = "RMSE")
Plots.plot!([tᵧvec], ITAE[ns][5, 2:end], label = "BPF 128", xlabel = "Time (h)", ylabel = "RMSE")
Plots.plot!([tᵧvec], ITAE[ns][6, 2:end], label = "BPF 1024", xlabel = "Time (h)", ylabel = "RMSE")
Plots.plot!([tᵧvec], ITAE[ns][7, 2:end], label = "EnKF", xlabel = "Time (h)", ylabel = "RMSE")
=#

#=
## Hellinger matrix
# Choose idx, idx ∈ (1:1601)

idx = 401

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
=#
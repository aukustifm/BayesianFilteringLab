"""
    Kolmogorov Forward (Fokker Planck): creation and propagation
"""
mutable struct KFState{T1, T2, PDF, G, SM} <: AbstractKolmogorovForwardState
  mean::T1
  cov::T2
  pdf::PDF  
  grid::G 
  state_model:: SM
  function KFState(sm::HiddenStateModel{ContinuousTime}, mygrid)

      @assert length(sm.init.μ) == 2 "The current scheme is implemented only for the 2D case."
      
      prior_pdf = x -> pdf(sm.init, x)
      pp = prior_pdf.(mygrid)

      l₁, l₂ = size(pp)
      global l₁
      global l₂
      lims = (mygrid[1, 1], mygrid[end, end])
      
      x₁ = range(lims[1][1], lims[2][1], length=l₁)       
      x₂ = range(lims[1][2], lims[2][2], length=l₂)

      Δx₁ = x₁[2]-x₁[1]
      Δx₂ = x₂[2]-x₂[1]
      
      integ = quad_trap(pp, x₂, x₁)
      #@assert integ > 0.5 "The chosen grid/spacing does not provide an accurate initial condition. Try adding more bins."
      
      pp /= integ

      points_reshaped = reduce(hcat, mygrid)
      points_matrix₁ = reshape(points_reshaped[1,:], l₁, l₂)
      points_matrix₂ = reshape(points_reshaped[2,:], l₁, l₂)

      Sum = x -> quad_trap(x, x₂, x₁)
      
      μ0 = [Sum(pp.*points_matrix₁), Sum(pp.*points_matrix₂)]
      Σ_0 = diagm([Sum(pp.*(points_matrix₁ .- μ0[1]).^2), Sum(pp.* (points_matrix₂ .- μ0[2]).^2)])
      Σ_0[1, 2] = Σ_0[2, 1] = Sum(pp.*((points_matrix₁ .- μ0[1]).*(points_matrix₂ .- μ0[2]))) 

      new{typeof(μ0), typeof(Σ_0), typeof(pp), typeof(mygrid), typeof(sm)}(μ0, Σ_0, pp, mygrid, sm)
  end 
end


################################
### CONVENIENCE CONSTRUCTORS ###
################################

struct OptimalFilter{FP<:AbstractFilteringProblem} <: AbstractFilteringAlgorithm
  filt_prob::FP
end
    
function KF_discretise(fp::AbstractKolmogorovForwardState, inputs, t)
  """Compute :math: `dp/dt` of the KF equation for two dimensional problems. 

  Parameters
  ----------
  fp: all information about dynamical model (drift and diffusion functions)
  t:  parameters for SDE (e.g. BM instensities)
  t:  current time

  Returns
  -------
  μ₁, μ₂, D₁, D₂, D₁₂, D₂₁: components of the Kolmogorov forward equation
  """
    sm = fp.state_model

    inp = inputs(t)
            
    u = inp[:η][:u]
    d = inp[:η][:d]
    pₓ = inp[:p][:pₓ]    
    wₓ = inp[:w][:wₓ]

    # PROPAGATE
  
    pₓ = vcat(collect(Iterators.flatten(u)), collect(Iterators.flatten(d)), collect(Iterators.flatten(pₓ)), collect(Iterators.flatten(wₓ)))
           
    u0 = fp.pdf
    l₁, l₂ = size(u0)

    μ₁ = Array{Float64}(undef, l₁, l₂)
    μ₂ = Array{Float64}(undef, l₁, l₂)
    D₁ = Array{Float64}(undef, l₁, l₂)
    D₂ = Array{Float64}(undef, l₁, l₂)
    D₁₂ = Array{Float64}(undef, l₁, l₂)
    D₂₁ = Array{Float64}(undef, l₁, l₂)

    μfunc, Dfunc = drift_diff_function(sm, t)
    for li in 1:l₁  
        for lj in 1:l₂
          μ₁[li, lj], μ₂[li,lj] = μfunc(fp.grid[li,lj], pₓ, t) 
            auxtensor = Dfunc(fp.grid[li,lj], pₓ, t)         
            (D₁[li, lj], D₁₂[li, lj], D₂₁[li, lj], D₂[li, lj]) = auxtensor*auxtensor'
        end
    end

    return μ₁, μ₂, D₁, D₂, D₁₂, D₂₁
end  

function KF_assemble(u, μ₁, μ₂, D₁, D₂, D₁₂; Δx₁=1.0, Δx₂=1.0)

  A1 = broadcast(*, μ₁,u)
  B1 = broadcast(*, μ₂,u)

  A11 = fd_2d(mypad(mypad0(max.(0, A1))), [Δx₁, Δx₂], "d/dx1", "backward", "first") # Campillo's 
  A11[end, :] .= 0            #reflective boundary at bmax    
  A22 = fd_2d(mypad(mypad0(max.(0, -A1))), [Δx₁, Δx₂], "d/dx1", "forward", "first")
  A1 = -(A11-A22)
  
  B11 = fd_2d(mypad(mypad0(max.(0, B1))), [Δx₁, Δx₂], "d/dx2", "backward", "first") 
  B11[:, end] .= 0            # reflective boundary at smax  
  B22 = fd_2d(mypad(mypad0(max.(0, -B1))), [Δx₁, Δx₂], "d/dx2", "forward", "first")

  B1 = -(B11-B22)

  C = broadcast(*,0.5*D₁,u)
  D = broadcast(*,0.5*D₂,u)
  E = 2*broadcast(*,0.5*(D₁₂),u)

  Ca = fd_2d(mypad(mypad0(C)), [Δx₁, Δx₂], "d^2/dx1^2",  "centered", "first")
  Ca[1, :] = zeros(l₂)    # natural boundary at b₀    
  Ca[end, :] = zeros(l₂) 

  Da = fd_2d(mypad(mypad0(D)), [Δx₁, Δx₂], "d^2/dx2^2",  "centered", "first")
  Da[:,1] = zeros(l₁);    # natural boundary at s₀    
  Da[:,end] = zeros(l₁);

  A1 + B1 + Ca + Da + E
end

function KF_propagate(u0, t, δt, KF_solver, KF_system)
  tspan = (t, t+δt)
  sol = solve(ODEProblem(KF_system, u0, tspan),  abstol=1e-12, reltol=1e-12, KF_solver,save_everystep=false; dt=0.001, adaptive=false)
  u0 = copy(sol(tspan[end]))
  return u0
end

## QUAD reshape
# 2D trapezoidal rule on a uniform grid
function quad_trap(f, x, y)
    h1 = x[2] - x[1]  # Δx
    h2 = y[2] - y[1]  # Δy
    l1 = length(x)
    l2 = length(y)

    # trapezoid weights
    wx = ones(Float64, l1); wx[1] = 0.5; wx[end] = 0.5
    wy = ones(Float64, l2); wy[1] = 0.5; wy[end] = 0.5

    # sum_{j,i} wy[j] * wx[i] * f[j,i]
    s = 0.0
    @inbounds for j in 1:l2
        row = 0.0
        row += wx[1] * f[j, 1]
        @simd for i in 2:l1-1
            row += wx[i] * f[j, i]
        end
        row += wx[end] * f[j, end]
        s += wy[j] * row
    end
    return h1 * h2 * s
end

## BY indexing
fd_2d = function(pp, dxs, mode, direction, order)
    if mode == "d/dx1"
      if direction == "centered"
        return (pp[3:end, 2:end-1] - pp[1:end-2, 2:end-1]) / (2 * dxs[1])
      elseif direction == "forward"
        if order == "corrected"
          return (pp[4:end-1, 3:end-2] - pp[3:end-2, 3:end-2]) / (dxs[1])
        elseif order == "second"
          return (-3*pp[3:end-2, 3:end-2] + 4 * pp[4:end-1, 3:end-2] - pp[5:end, 3:end-2]) / (2*dxs[1])
        else 
          return (pp[3:end, 2:end-1] - pp[2:end-1, 2:end-1]) / (dxs[1])
        end
      elseif direction == "backward"
        if order == "corrected"
          return (pp[2:end-3, 3:end-2] - pp[1:end-4, 3:end-2]) / (dxs[1])
        elseif order == "second"
          return (3*pp[3:end-2, 3:end-2] - 4 * pp[2:end-3, 3:end-2] + pp[1:end-4, 3:end-2]) / (2*dxs[1])
        else
          return (pp[2:end-1, 2:end-1] - pp[1:end-2, 2:end-1]) / (dxs[1])
        end
      end
  
    elseif mode == "d/dx2"
      if direction == "centered"
            return (pp[2:end-1, 3:end] - pp[2:end-1, 1:end-2]) / (2 * dxs[2])
      elseif direction == "forward"
        if order == "corrected"
          return (pp[3:end-2, 5:end] - pp[3:end-2, 4:end-1]) / (dxs[2])
        elseif order == "second"
          return (-3*pp[3:end-2, 3:end-2] + 4 * pp[3:end-2, 4:end-1] - pp[3:end-2, 5:end]) / (2*dxs[2])
        else
          return (pp[2:end-1, 3:end] - pp[2:end-1, 2:end-1]) / (dxs[2])
        end
      elseif direction == "backward"
        if order == "corrected"
          return (pp[3:end-2, 2:end-3] - pp[3:end-2, 1:end-4]) / (dxs[2])
        elseif order == "second"
          return (3*pp[3:end-2, 3:end-2] - 4 * pp[3:end-2, 2:end-3] + pp[3:end-2, 1:end-4]) / (2*dxs[2])
        else
          return (pp[2:end-1, 2:end-1] - pp[2:end-1, 1:end-2]) / (dxs[2])
        end
      end
  
    elseif mode == "d^2/dx1^2"
      return (pp[3:end, 2:end-1] - 2 * pp[2:end-1, 2:end-1] + pp[1:end-2, 2:end-1]) / (dxs[1]^2)
    
    elseif mode == "d^2/dx2^2"
      return (pp[2:end-1, 3:end] - 2 * pp[2:end-1, 2:end-1] + pp[2:end-1, 1:end-2]) / (dxs[2]^2)
    
    elseif mode == "d^2/dx1dx2"
      return (pp[3:end, 3:end] - pp[1:end-2, 3:end] - pp[3:end, 1:end-2] + pp[1:end-2, 1:end-2]) / (4 * dxs[1]* dxs[2])
    
    else mode == "d^2/dx2dx1"
      return (pp[3:end, 3:end] - pp[1:end-2, 3:end] - pp[3:end, 1:end-2] + pp[1:end-2, 1:end-2]) / (4 * dxs[1] * dxs[2])
    end
  end


## PADDING
mypad0 = function(pp)
    z1, z2 = size(pp)
    ppaux = [zeros(z1) pp zeros(z1)]
    newpp = ppaux = ([zeros(z2+2) ppaux' zeros(z2+2)])'
    return newpp
  end
  
mypad = function(newpp)
    z1, z2 = size(newpp)
    newpp[end, :] = newpp[end-2, :]
    newpp[:, end] = newpp[:, end-2]
    return newpp
end


## Preparing Grid for Hellinger Distances 
pf_counter = function(x::AbstractVector, y::AbstractVector, xedges::AbstractRange, yedges::AbstractRange, wei)
  # Calculate bin index from x value
  nxbins = length(xedges)
  xmin, xmax = extrema(xedges)
  δ𝑖δx = nxbins / (xmax - xmin)

  nybins = length(yedges)
  ymin, ymax = extrema(yedges)
  δ𝑗δy = nybins / (ymax - ymin)

  N = spzeros(nxbins,nybins)

  # Make sure we don't have a default by filling beyond the length of N
  # in the @inbounds loop below
  if size(N,2) < nybins
      nybins = size(N,1)
      @warn "size(N,2) < nybins; any y bins beyond size(N,2) will not be filled"
  end
  if size(N,1) < nxbins
      nxbins = size(N,2)
      @warn "size(N,1) < nxbins; any x bins beyond size(N,1) will not be filled"
  end

  # Calculate the means for each bin, ignoring NaNs
  @inbounds for n ∈ eachindex(x)
      𝑖 = (x[n] - xmin) * δ𝑖δx
      𝑗 = (y[n] - ymin) * δ𝑗δy
      if (0 <= 𝑖 < nxbins) && (0 <= 𝑗 < nybins)
          i = ceil(Int, 𝑖)
          j = ceil(Int, 𝑗)
          N[i,j] += wei[n]
      end
  end

  Δx1 = xedges[2]-xedges[1]
  Δx2 = yedges[2]-yedges[1]
  return N/(Δx1* Δx2)
end

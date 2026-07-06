@doc raw"""
    FPF

"""

mutable struct FPFState{T1, T2, T, PMEM,T3} <: AbstractFilterState
    mean::T1
    cov::T2
    ensemble::UnweightedParticleEnsemble
    _particle_memory::PMEM
    mass::T3
    function FPFState(problem::AbstractFilteringProblem, algparams)

        x0         = rand(problem.m, 1); r0 = rand(problem.n, 1); p0 = rand(problem.np, 1); t0 = rand(1, 1)
        Jresults_y = (similar(x0, problem.n, problem.m), similar(r0, problem.n, problem.n), similar(p0, problem.n, problem.np), similar(t0, problem.n, 1))
        h_tape     = ReverseDiff.JacobianTape(problem.obs_model.observation_function, (x0, r0, p0, t0))
        
        # compile `f_tape` and `g_tape` into more optimized representations
        compiled_h_tape = ReverseDiff.compile(h_tape)
        global Jresults_y 
        global compiled_h_tape

        N, l,  M = algparams
        ens = UnweightedParticleEnsemble(state_model(problem), N)
        
        xx  = similar(ens.positions, N, problem.m, 1)
        xx .= ens.positions'
        m0 = reshape(mean(ens), problem.m)
        p0 = cov(ens)
        p0 = convert(Matrix, Hermitian(p0))#PDMat(convert(Matrix, Hermitian(p0)))
        new{typeof(m0), typeof(p0), typeof(ens), typeof(xx), typeof(M)}(m0, p0, ens, xx, M)
    end 
end

struct FPF{FP<:AbstractFilteringProblem, T<:Real} <: AbstractFilteringAlgorithm
    filt_prob::FP
    N::Int
    l::T
end

#####################
###### METHODS ######
#####################  

no_of_particles(st::FPFState) = no_of_particles(st.ensemble)
    
mean(st::FPFState) = mean(st.ensemble)
cov(st::FPFState)  = cov(st.ensemble)
var(st::FPFState)  = var(st.ensemble)

function Base.show(io::IO, ::MIME"text/plain", filter::FPF)
    print(io, "Feedback particle filter algorithm
    hidden      state model: ", state_model(filter.filt_prob),"
    observation model      : ", obs_model(filter.filt_prob),"
    ensemble    size       : ", no_of_particles(filter),"
    number of iterations for DM alg                  ", l)
end

function Base.show(io::IO, filter::FPF)
    print(io, "FPF with ", no_of_particles(filter)," particles ")
end

######################
### MAIN ALGORITHM ###
######################
    
function initialize(filter::FPF)
    return FPFState(filter)
end     

function propagate(filter_state::FPFState, filt_prob::FilteringProblem, inputs, t1, t2, δx)

    Dx, N = size(filter_state.ensemble.positions)
    
    propagate_state!(filter_state, state_model(filt_prob), inputs, t1, t2, δx)
    
    tvec = t1:δx:t2
        
    mm = Array{Float64}(undef, length(tvec)-1, Dx);
    pp = Array{Float64}(undef, length(tvec)-1, Dx, Dx);

    Nt = length(tvec)-1
    for i in 1: Nt
        mm[i, :] = reshape(StatsBase.mean(filter_state._particle_memory[:,:,end-Nt+i]', dims=2), Dx)
        x0       = filter_state._particle_memory[:,:,end-Nt+i] .- mm[i, :]'
        pp[i, :, :] = cov(x0)#PDMat(cov(x0))
    end
   
    return mm, pp, filter_state
end

function update!(filter_state::FPFState, filt_prob::FilteringProblem, method, inputs, t, δy, y)
    update_particles!(filter_state, obs_model(filt_prob), method, inputs, t, δy, y)
    
    ens = filter_state.ensemble
    mm  = reshape(mean(ens), filt_prob.m)
    pp  = cov(ens)
    
    pp = convert(Matrix, Hermitian(pp))#PDMat(convert(Matrix, Hermitian(pp)))
    
    filter_state.ensemble = ens;
    filter_state.mean     = mm;
    filter_state.cov      = pp;

end

function propagate_state!(filter_state::FPFState, model::DiffusionStateModel, myparams, t1, t2, δx)
       tvec = t1:δx:t2
       x    = copy(filter_state.ensemble.positions)
    Dx, N   = size(x)
    xx = [zeros(N, Dx) for _ in 1:length(tvec)];

    #xx = Array{eltype(x)}(undef, N, Dx, length(tvec))
   
    drift_f, diff_g = drift_diff(model, t1)
    
    ###=
    function diff_g_mat(x, p, t) return Dx > 1 ? diag(diff_g(x, p, t)) : diagm(diff_g(x, p, t)) end

    function ff(du, u, p, t)
        du .= drift_f(u, p, t)
    end
    function gg(du, u, p, t)
        du .= Dx>1 ? diag(diff_g(u, p, t)) : diagm(diff_g(u, p, t))
    end  

    
    if length(myparams) == 0 myparams = zeros(1) end
    Base.Threads.@threads for n in 1:N  #
        sol = solve(SDEProblem(ff, gg, x[:, n], (tvec[1], tvec[end]), myparams),  SRIW1(); dt=δx/10)# isoutofdomain = (m,p,t) -> any(x->x.<0, m)SKSROCK(;post_processing=true); dt=δx)#, EM(); dt=0.001)#SRIW1()EM(); dt=0.001)
        xvec = sol(tvec).u'
        xvec        = permutedims(reshape(hcat(xvec...),(length(xvec[1]), length(xvec))))
        xx[1][n,:] = xvec[1,:]
        xx[end][n,:] = xvec[2,:]
    end
    ###=#

    #=
    EM_δt = δx/10
    integration_tvec = (tvec[1]:EM_δt:tvec[end])
    xx_roll = [zeros(N, Dx) for _ in 1:length(integration_tvec)];
    #xx_roll = Array{eltype(x)}(undef,, Dx, N)
    Base.Threads.@threads for n in 1:N#@sync @distributed 
        xx_roll[1][n, :] .= x[:, n]
        i=2
        for ti in (tvec[1]:EM_δt:tvec[end])[1:end-1]
            xx_roll[i][n, :] =   xx_roll[i-1][n,:] + EM_δt*drift_f(xx_roll[i-1][n,:],  myparams, ti) + diff_g(xx_roll[i-1][n,:], myparams, ti)*randn(Dx, 1)*sqrt(EM_δt)
            i+=1
        end
        xx[1][n,:] = xx_roll[1][n, :] 
        xx[end][n,:] =  xx_roll[end][n,:] 
    end
    =#

    filter_state._particle_memory   = cat(filter_state._particle_memory, xx[end], dims=3)
    filter_state.ensemble.positions = convert(Matrix, xx[end]')

end

function update_particles!(filter_state::FPFState, model::UserDefinedDiscreteObservationModel,  method::GainEstimationMethod, p, t, δy, dy)
    x  = filter_state.ensemble.positions
    M  = filter_state.mass
    Dₓ, N = size(x)
    Dᵧ = length(dy)
    
    #=
    # WITH HOMOTOPY
    wᵧ  = p[end-Dᵧ+1:end]
    R  = Diagonal(wᵧ.^2)
    # Precompute Cholesky once per update (R must be SPD)
    ch = cholesky(R)      # R = L*L'
    L = ch.L              # lower-triangular

    y_hat = L \ dy

    hh₁ = observation_function(model, t)
    function hh₂(x)
        return hh₁(x, zeros(Dᵧ), p, t)
    end
    function lf(x)
        # Whiten h for each particle column: H_hat = L \ H
        return -L \ hh₂(x)
    end


    # Initialise FPF GeneralPolynomial
    
    if hasproperty(method, :λ) 
        λopt_gain = optimise(PoissonEquation(lf, x, M), method) 
        λopt_correction = zeros(Dᵧ)

        method.λ = λopt_gain
        eq = PoissonEquation(lf, x, M)
        StochasticStateEstimation.solve!(eq, method)
        K = copy(eq.gain)

        ∇h = [zeros(Dₓ, Dᵧ) for _ in 1:N];
        Base.Threads.@threads for n in 1:N
            args = (reshape(x[:,n], Dₓ, 1), zeros(Dᵧ, 1), reshape(p, length(p), 1), [t])
            ReverseDiff.jacobian!(Jresults_y, compiled_h_tape, args)
            ∇h[n] .= (L \ Jresults_y[1])'#Jresults_y[1]'
            

        end      
        h⁻         = mapslices(hh₂, x, dims=1)
        for dᵧ in 1:Dᵧ
            h⁻²  = hcat([h⁻[dᵧ, n]'h⁻[dᵧ, n] for n in 1:N]...)
            h̃⁻² = (mean(h⁻[dᵧ,:]))^2
            L₂   = Array{Float64}(undef, N)
            for n in 1:N
                L₂[n] = dot(K[:,n,dᵧ],∇h[n][:,dᵧ])
            end
            #println("Correction\n")
            λopt_correction[dᵧ] = optimise(PoissonEquation((L₂' .- (mean(h⁻²)-h̃⁻²)), x, M), method)[1]    
        end

        println("λ Gain: ",λopt_gain, " λ Correction: ", λopt_correction) 
    end
    

    #λopt_gain = [0.0001, 0.0001]
    #λopt_correction = [0.0001, 0.0001]

    ∇h = [zeros(Dₓ, Dᵧ) for _ in 1:N];
    function fode!(du,u,p,λ)   
    #function fode(u,p,λ)   
        #println("Gain\n")      
        if hasproperty(method, :λ)  method.λ = λopt_gain end
        
        eq = PoissonEquation(lf, u, M)
        StochasticStateEstimation.solve!(eq, method)
        K = copy(eq.gain)
        
        Base.Threads.@threads for n in 1:N
            args = (reshape(u[:,n], Dₓ, 1), zeros(Dᵧ, 1), reshape(p, length(p), 1), [t])
            ReverseDiff.jacobian!(Jresults_y, compiled_h_tape, args)
            ∇h[n] .= (L \ Jresults_y[1])'#Jresults_y[1]'
        end      
        h⁻         =  L \ mapslices(hh₂, u, dims=1)
        h_mean = mean(h⁻, dims=2)
        Innovation = y_hat .- h_mean*δy 
        Ω          = [zeros(Dₓ, N) for _ in 1:Dᵧ]
        μ          = [zeros(Dₓ, N) for _ in 1:Dᵧ]
        for dᵧ in 1:Dᵧ
            h⁻²  = hcat([h⁻[dᵧ, n]'h⁻[dᵧ, n] for n in 1:N]...)
            h̃⁻² = (mean(h⁻[dᵧ,:]))^2
            L₂   = Array{Float64}(undef, N)
            for n in 1:N
                L₂[n] = dot(K[:,n,dᵧ],∇h[n][:,dᵧ])
            end
            #println("Correction\n")
            eq = PoissonEquation((L₂' .- (mean(h⁻²)-h̃⁻²)), u, M)
            if hasproperty(method, :λ) method.λ[1] = λopt_correction[dᵧ]  end
        
            StochasticStateEstimation.solve!(eq,  method; gain=true)
            Ω[dᵧ] .= eq.gain[:,:,1]
            μ[dᵧ] .= K[:,:,dᵧ] .* Innovation[dᵧ,:]' + 0.5*Ω[dᵧ]*δy#(-0.5*(h⁻.+mean(h⁻)).+y)
        end
        du .= sum(μ)
        #sum(μ)
    end
    
    # Or manually, with Euler. Use this carefully at your own discretion, as it does not ensure ergodicity/positivity
    
    # Number of iterations
    num_iterations = 100

    # Total sum should be 1
    total_sum = 1.0

    # Ratio for geometric progression: Needs to be >1 to increase increment, adjust as needed
    ratio = 1.05

    # Generate the initial values for Δλvec
    Δλ_initial = total_sum * (1 - ratio) / (1 - ratio^num_iterations)

    # Generate the sequence; each entry is a geometric progression
    Δλvec = [Δλ_initial * ratio^(i-1) for i in 1:num_iterations]
    # Normalize the sequence to ensure it sums to total_sum exactly
    Δλvec = Δλvec .* (total_sum / sum(Δλvec))
    =#
    
    #=
    Δλvec = repeat([0.01],100)
    λvec = cumsum(Δλvec)
    sol = copy(x)
    for i in 1:length(Δλvec)
        drift = Δλvec[i]*fode(sol, p, λvec[i])
        #println("maximum drift: ",maximum(drift)) 
        sol += drift
    end
    x[:,:] = copy(sol)
    println(minimum(x))
    #println(norm(sol))
    =#
    
    #=
    u₀    = copy(x)
    λspan = (0.0,1.0)
    prob  = ODEProblem(fode!,u₀,λspan,p)
    sol = solve(
        prob,
        Tsit5(),           # fast explicit Runge–Kutta
        reltol = 1e-3,
        abstol = 1e-4,
        save_everystep = false,
        dense = false
    )
    x = sol.u[end]
    #println(minimum(x))
    =#
    #println("assimilated!")

    ###=
    # WITHOUT HOMOTOPY AND STRATONOVICH

    hh₁       = observation_function(model, t);
    wᵧ        = p[end-Dᵧ+1:end]
    R⁻¹       = spdiagm([wᵧ[i]^(-2) for i in 1:Dᵧ]);
    function hh₂(x) return hh₁(x, zeros(Dᵧ), p, t) end
    function lf(x) return -hh₂(x) end     

    
    if hasproperty(method, :λ) #&& t==δy
        #method.λ = optimise(PoissonEquation(lf, x, M), method) 
        method.λ = [1e-4,1e-4]
        #println("λ Gain: ",method.λ) 
    end
    

    eq = PoissonEquation(lf, x, M)
    StochasticStateEstimation.solve!(eq, method)
 
    ∇h = [zeros(Dₓ, Dᵧ) for _ in 1:N];
    for n in 1:N
        args = (reshape(x[:,n], Dₓ, 1), zeros(Dᵧ, 1), reshape(p, length(p), 1), [t])
        ReverseDiff.jacobian!(Jresults_y, compiled_h_tape, args)
        ∇h[n] .= (Jresults_y[1])'#Jresults_y[1]'
    end      
    h⁻         =  mapslices(hh₂, x, dims=1)
    h_mean = mean(h⁻, dims=2)
    dInnovation = -0.5(h⁻.+h_mean) * δy .+ dy

    #---
    # stochastic Heun method: intermediate step
    ensemble2 = deepcopy(filter_state.ensemble)    
    applygain!(ensemble2, eq, dInnovation)
    eq2 = PoissonEquation(lf, ensemble2.positions, M)
    StochasticStateEstimation.solve!(eq2, method)
    K  = (eq.gain + eq2.gain) / 2
    
    #---
   
    #Ω          = [zeros(Dₓ, N) for _ in 1:Dᵧ]
    μ          = [zeros(Dₓ, N) for _ in 1:Dᵧ]
    for dᵧ in 1:Dᵧ
        #=
        h⁻²  = hcat([h⁻[dᵧ, n]'h⁻[dᵧ, n] for n in 1:N]...)
        h̃⁻² = (mean(h⁻[dᵧ,:]))^2
        L₂   = Array{Float64}(undef, N)
        for n in 1:N
            L₂[n] = dot(K[:,n,dᵧ],∇h[n][:,dᵧ])
        end
        
        #println("Correction\n")
        eq = PoissonEquation((L₂' .- (mean(h⁻²)-h̃⁻²)), x, M)
        if hasproperty(method, :λ) method.λ[1] = λopt_correction[dᵧ]  end
    
        StochasticStateEstimation.solve!(eq,  method; gain=true)
        Ω[dᵧ] .= eq.gain[:,:,1]
        =#
        μ[dᵧ] .= K[:,:,dᵧ] .* (R⁻¹[dᵧ,dᵧ] * dInnovation[dᵧ,:]') #+ 0.5 * Ω[dᵧ] * δy
    end
    x += sum(μ)
    #println("Assimilated!")

    #for dᵧ in 1:Dᵧ
    #    μ[dᵧ] .= K[:,:,dᵧ] .* dInnovation[dᵧ,:]' 
    #end
    #x += sum(μ)
    ##=#    
    filter_state._particle_memory[:,:,end] = x'
    filter_state.ensemble.positions = copy(x)
    #println("\nMean:", mean(x,dims=2),"\n")
end


########################
### HELPER FUNCTIONS ###
########################
    
function heun!(eq::PoissonEquation, ensemble::UnweightedParticleEnsemble, error::AbstractMatrix, method::GainEstimationMethod)
    ensemble2 = deepcopy(ensemble)    
    applygain!(ensemble2, eq, error)
    eq2 = PoissonEquation(lf, ensemble2.positions, M)
    StochasticStateEstimation.solve!(eq2, method)
    eq.gain  .= (eq.gain + eq2.gain) / 2
end    

function add_gainxerror!(out::Array{T, 2}, gain, error::Array{T, 2}) where T
    size(gain, 2) == size(error, 2) == size(out, 2) ? N = size(gain, 2) : throw(DimensionMismatch("The provided gain, error, and output array are for different numbers of particles."))
    size(gain, 3) == size(error, 1)                 ? m = size(gain, 3) : throw(DimensionMismatch("The provided gain and error are for different numbers of observed variables."))
    size(gain, 1) == size(out, 1)                   ? n = size(gain, 1) : throw(DimensionMismatch("The provided gain and output array are for different numbers of hidden variables."))
    @inbounds for k in 1:N, i in 1:n, j in 1:m
            out[i, k] += gain[i, k, j] * error[j, k]
    end
    return out
end

function applygain!(ens::UnweightedParticleEnsemble, eq, error::AbstractMatrix)
    add_gainxerror!(ens.positions, eq.gain, error)
end


#=
eqc = PoissonEquation(lf, copy(u), mass_f)
Kc  = copy(StochasticStateEstimation.solve!(eqc, ConstantGainApproximation()))
eqs = PoissonEquation(lf, copy(u), mass_f)
Ks  = copy(StochasticStateEstimation.solve!(eqs, SemigroupMethod(1e-4,1e-4)))
eqp = PoissonEquation(lf, copy(u), mass_f);
StochasticStateEstimation.solve!(eqp, GeneralPolynomialBasis(Nᵦ, Nₑ, Nₓ, Nᵧ, shifted_P, shifted_∂P, shifted_∂³P; grid="hypercross"));
Kp  = eqp.gain

using Plots
idx =  1
idy =  1

Plots.scatter(x[idx, :], Kc[idx, :, idy])
Plots.scatter!(x[idx, :], Ks[idx, :, idy])
Plots.scatter!(x[idx, :], Kp[idx, :, idy])

Plots.scatter(x[idx, :], eqc.potential[idy,:])
Plots.scatter!(x[idx, :], eqs.potential[idy,:])
Plots.scatter!(x[idx, :], eqp.potential[idy,:])

sum([û[nid]*method.L[:, nid]' for nid in 1:Nᵦexp])[:]

=#

#=
#using Plots
nₓ = 5
Plots.scatter(sfs.ensemble.positions[nₓ, :], Kc[nₓ, :, 3])
Plots.scatter!(sfs.ensemble.positions[nₓ, :], Ks[nₓ, :, 3])
Plots.scatter!(sfs.ensemble.positions[nₓ, :], Kp[nₓ, :, 3])

using BenchmarkTools
@btime Kc  = copy(StochasticStateEstimation.solve!(PoissonEquation(lf, u, M), ConstantGainApproximation()));
@btime Ks  = copy(StochasticStateEstimation.solve!(PoissonEquation(lf, u, M), SemigroupMethod(1e-2,1e-2)));
@btime Kp  = copy(StochasticStateEstimation.solve!(PoissonEquation(lf, u, M), method));
=#


#=
Σ = randn(Nₓ,Nₓ)
Σ = Σ*Σ'
x = rand(MvNormal(zeros(Nₓ),Σ), Nₑ)
rhs = function(x) -x end

eqc = PoissonEquation(rhs, copy(x), mass_f)
Kc  = copy(StochasticStateEstimation.solve!(eqc, ConstantGainApproximation()))
eqs = PoissonEquation(rhs, copy(x), mass_f)
Ks  = copy(StochasticStateEstimation.solve!(eqs, SemigroupMethod(1e-4,1e-4)))
eqp = PoissonEquation(rhs, copy(x), mass_f);
StochasticStateEstimation.solve!(eqp, GeneralPolynomialBasis(Nᵦ, Nₑ, Nₓ, Nᵧ, shifted_P, shifted_∂P, shifted_∂³P; hypercross=true));
Kp  = eqp.gain


using Plots
idx =  1
idy =  1

Plots.hline(x[idx, :], [Σ[idx, idy]],c=:black,linewidth=2.0)
Plots.scatter(x[idx, :], Kc[idx, :, idy])
Plots.scatter!(x[idx, :], Ks[idx, :, idy])
Plots.scatter!(x[idx, :], Kp[idx, :, idy])


#Plots.scatter(x[idx, :], (Σ*x)[idy,:],c=:black,markersize=5.0)
Plots.scatter(x[idx, :], eqc.potential[idy,:])
Plots.scatter!(x[idx, :], eqs.potential[idy,:])
Plots.scatter!(x[idx, :], eqp.potential[idy,:])


method = GeneralPolynomialBasis(Nᵦ, Nₑ, Nₓ, Nᵧ, shifted_P, shifted_∂P, shifted_∂³P; grid="hypercross")
method.λ =  λopt_gain
up = fode(u₀,p,0.0)
method = SemigroupMethod(1e-3,1e-3)
us = fode(u₀,p,0.0)
method = ConstantGainApproximation()
uc = fode(u₀,p,0.0)

idx =  1
idy =  1

Plots.scatter(x[idx, :], uc[idy,:])
Plots.scatter!(x[idx, :], us[idy,:])
Plots.scatter!(x[idx, :], up[idy,:])

=#
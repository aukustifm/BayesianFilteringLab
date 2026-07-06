@doc raw"""
    BPF storing only current ensemble and moments

    The effective sample size (ESS) N eff is a diagnostics tool that tells us when our weights are problematic in the sense that they are close to
    being degenerate.
    Neff = N/Eq[ω^2(x^i)] ≤ N
    We cannot evaluate N eff exactly, but we can compute an estimate1

    Neff = \frac{1}/{\sum_{i=1}^N (w^i)^2}.

    ”ESS-adaptive resampling”: When N_eff falls below some threshold N_thres
    we resample the particles, otherwise we continue without resampling
"""

mutable struct BPFGridState{T1, T2, T, RS, AL, LL,  NEM} <: AbstractFilterState
    mean::T1
    cov::T2
    ensemble::WeightedParticleEnsemble
    resampling_method::RS
    alpha::AL
    logl::LL
    _neff_memory::NEM
    function BPFGridState(problem::AbstractFilteringProblem, algparams)


        x0 = rand(problem.m, 1); r0 = rand(problem.n, 1); p0 = rand(problem.mp, 1); t0 =  rand(1, 1)        
        Jresults_y = (similar(x0, problem.n, problem.m), similar(r0, problem.n, problem.n), similar(p0, problem.n, problem.mp), similar(t0, problem.n, 1))    
        h_tape = ReverseDiff.JacobianTape(problem.obs_model.observation_function, (x0, r0, p0, t0))
        
        # compile `f_tape` and `g_tape` into more optimized representations
        compiled_h_tape = ReverseDiff.compile(h_tape)
        global Jresults_y 
        global compiled_h_tape


        N, res, alp = algparams
        ens = WeightedParticleEnsemble(state_model(problem), N)
        #ens.positions = abs.(ens.positions)

        #temporary 
        #ens.positions[1, end]  = xref[1]
        #ens.ancestors[end] = N
        ll = 0.
        
        m0 = reshape(mean(ens), problem.m)
        p0 = cov(ens)
        p0 = PDMat(convert(Matrix, Hermitian(p0)))

        Neff = [N]
        new{typeof(m0), typeof(p0), typeof(ens), typeof(res), typeof(alp), typeof(ll), typeof(Neff)}(m0, p0, ens, res, alp, ll,  Neff)
    end 
end


struct BPFGrid{FP<:AbstractFilteringProblem, T<:Real} <: AbstractFilteringAlgorithm
    filt_prob::FP
    N::Int
    alpha::T
end

#####################
###### METHODS ######
#####################  

no_of_particles(st::BPFGridState) = no_of_particles(st.ensemble)
    
mean(st::BPFGridState) = mean(st.ensemble)
cov(st::BPFGridState)  = cov(st.ensemble)
var(st::BPFGridState)  = var(st.ensemble)    

    
function Base.show(io::IO, ::MIME"text/plain", filter::BPFGrid)
    print(io, "Bootstrap particle filter algorithm
    hidden state model:                              ", state_model(filter.filt_prob),"
    observation model:                               ", obs_model(filter.filt_prob),"
    ensemble size:                                   ", no_of_particles(filter),"
    threshold for effective number of particles (%): ", 100*filter.alpha)
end

function Base.show(io::IO, filter::BPFGrid)
    print(io, "BPF with ", no_of_particles(filter)," particles ")
end

######################
### MAIN ALGORITHM ###
######################
    
function initialize(filter::BPFGrid)
    return BPFState(filter)
end     


function propagate(filter_state::BPFGridState, filt_prob::FilteringProblem, inputs, t1, t2, δx)

    ens = filter_state.ensemble
    res = filter_state.resampling_method
    alpha_eff = filter_state.alpha
    Neff = floor(eff_no_of_particles(ens))
    filter_state._neff_memory = cat(filter_state._neff_memory, [Integer(Neff)], dims=1);

    Dx, N = size(filter_state.ensemble.positions)
    
    if Neff < alpha_eff * N
        resample!(ens, res)   
    else
        ens.ancestors = 1:N
    end

    filter_state.ensemble = ens

    propagate_state!(filter_state, state_model(filt_prob), inputs, t1, t2, δx)

    mm = reshape(mean(ens), filt_prob.m)
    pp = cov(ens)
   
    return mm, pp, filter_state
end

function update!(filter_state::BPFGridState, filt_prob::FilteringProblem, inputs, t, δy, y)
    update_weights!(filter_state, obs_model(filt_prob), inputs, t, δy, y)
    
    ens = filter_state.ensemble
    mm = reshape(mean(ens), filt_prob.m)
    pp = cov(ens)
    
    pp = PDMat(convert(Matrix, Hermitian(pp)))
    
    filter_state.ensemble = ens;
    filter_state.mean = mm;
    filter_state.cov = pp;

end

function propagate_state!(filter_state::BPFGridState, model::DiffusionStateModel, p, t1, t2, δx)
    tvec = t1:δx:t2 

    #const global model
    x = copy(filter_state.ensemble.positions)
    w = copy(filter_state.ensemble.weights)
    a = copy(filter_state.ensemble.ancestors)
    Dx, N = size(x)
    xx = SharedArray{Float64}(N, Dx, length(tvec))
    ww = similar(w, N, length(tvec))
    ww .= w
    aa = similar(a, N, 1)
    aa .= a
   
    drift_f, diff_g = drift_diff(model, t1)
    diff_g_mat = Dx>1 ? (x, p, t) -> diag(diff_g(x, p, t)) : (x, p, t) -> diagm(diff_g(x, p, t))

    function ff(du, u, p, t)
        du .= drift_f(u, p, t) 
    end
    function gg(du, u, p, t)
        du .= diff_g_mat(u, p, t) 
    end

    #Manually (#You may use the code in BPF.jl when using other integration algorithms from DifferentialEquations.jl)
    EM_δt = δx/10
    integration_tvec = (tvec[1]:EM_δt:tvec[end])
    xx_roll = SharedArray{Float64}(length(integration_tvec), Dx, N)
    
    Base.Threads.@threads for n in 1:N
        xx_roll[1, :, n] .= x[:, n]
        i=2
        for ti in (tvec[1]:EM_δt:tvec[end])[1:end-1]
            xx_roll[i, :, n] =  xx_roll[i-1, :, n] + EM_δt*drift_f(xx_roll[i-1, :, n], p, ti) + diff_g_mat(xx_roll[i-1, :, n], p, ti).*randn(Dx)*sqrt(EM_δt)
            i+=1
        end
        xx[n, :, :] = [xx_roll[1, :, n] xx_roll[end, :, n]]
    end
    filter_state.ensemble.positions = convert(Matrix, xx[:, :, end]')
end

function update_weights!(filter_state::BPFGridState, model::UserDefinedDiscreteObservationModel,  p, t, δy, dy)
    w = filter_state.ensemble.weights
    x = filter_state.ensemble.positions
    Dx, N = size(x)
    Dy = length(dy)
    
    hh = observation_function(model, t);
   
    wᵧ  = p[end-Dy+1:end]
    Rinv = Diagonal(1.0 ./ wᵧ.^2)
    Rinv_scaled = (1/δy) * Rinv
    hh2(x) = hh(x, zeros(Dy), p, t)
    
    heval = mapslices(hh2,x, dims=1)
    #logwaux = [heval[:, n]' * Rinv_scaled * y - 0.5 * heval[:, n]'* Rinv_scaled * heval[:, n] for n in 1:N]
    logwaux = [
        -0.5 * (dy - heval[:,n]*δy)' * Rinv_scaled * (dy - heval[:,n]*δy) for n in 1:N
    ]

    logwaux = logwaux .- maximum(logwaux)
    w = exp.(logwaux)

    ## LogLikelihood estimator  
    filter_state.logl += log(mean(w)) 
    ## Normalising
    w = w./sum(w)

    filter_state.ensemble.weights = ProbabilityWeights(w)
end

################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function BPFGrid(filt_prob::AbstractFilteringProblem, data)
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)
    return BPFGrid(st_mod, ob_mod, data)
end
function BPFState(filter::BPFGrid)
    N   = no_of_particles(filter)
    ens = WeightedParticleEnsemble(hcat([initialize(filter.state_model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)))
    return BPFGridState(ens)
end    
function BPFState(state_model::HiddenStateModel, N)
    ens = WeightedParticleEnsemble(hcat([initialize(state_model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)))
    return BPFState(ens)
end
function BPF(filt_prob::AbstractFilteringProblem, N, alpha)
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)  
    return BPF(st_mod, ob_mod, N, alpha)
end

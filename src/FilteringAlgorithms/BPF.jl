@doc raw"""
    BPF

    The effective sample size (ESS) N eff is a diagnostics tool that tells us when our weights are problematic in the sense that they are close to
    being degenerate.
    Neff = N/Eq[ω^2(x^i)] ≤ N
    We cannot evaluate N eff exactly, but we can compute an estimate1

    Neff = \frac{1}/{\sum_{i=1}^N (w^i)^2}.

    ”ESS-adaptive resampling”: When N_eff falls below some threshold N_thres
    we resample the particles, otherwise we continue without resampling
"""

mutable struct BPFState{T1, T2, T, RS, AL, LL, PMEM, WMEM, AMEM, NEM} <: AbstractFilterState
    mean::T1
    cov::T2
    ensemble::WeightedParticleEnsemble
    resampling_method::RS
    alpha::AL
    logl::LL
    _particle_memory::PMEM
    _weight_memory::WMEM
    _ancestor_memory::AMEM
    _neff_memory::NEM
    function BPFState(problem::AbstractFilteringProblem, algparams)


        x0 = rand(problem.m, 1); r0 = rand(problem.n, 1); p0 = rand(problem.np, 1); t0 =  rand(1, 1)        
        Jresults_y = (similar(x0, problem.n, problem.m), similar(r0, problem.n, problem.n), similar(p0, problem.n, problem.mp), similar(t0, problem.n, 1))    
        h_tape = ReverseDiff.JacobianTape(problem.obs_model.observation_function, (x0, r0, p0, t0))
        
        # compile `f_tape` and `g_tape` into more optimized representations
        compiled_h_tape = ReverseDiff.compile(h_tape)
        global Jresults_y 
        global compiled_h_tape

        N, res, alp = algparams
        ens = WeightedParticleEnsemble(state_model(problem), N)
        
        xx = similar(ens.positions, N, problem.m, 1)
        ww = similar(ens.weights, N, 1)
        aa = similar(ens.ancestors, N, 1)
        ll = 0.
        xx .= ens.positions'
        ww .= ens.weights
        aa .= ens.ancestors
        m0 = reshape(mean(ens), problem.m)
        p0 = cov(ens)+1e-16I(problem.m)
        p0 = PDMat(convert(Matrix, Hermitian(p0)))
        Neff = [N]
        new{typeof(m0), typeof(p0), typeof(ens), typeof(res), typeof(alp), typeof(ll), typeof(xx), typeof(ww), typeof(aa), typeof(Neff)}(m0, p0, ens, res, alp, ll, xx, ww, aa, Neff)
    end 
end

struct BPF{FP<:AbstractFilteringProblem, T<:Real} <: AbstractFilteringAlgorithm
    filt_prob::FP
    N::Int
    alpha::T
end

#####################
###### METHODS ######
#####################  

no_of_particles(st::BPFState) = no_of_particles(st.ensemble)
    
mean(st::BPFState) = mean(st.ensemble)
cov(st::BPFState)  = cov(st.ensemble)
var(st::BPFState)  = var(st.ensemble)    

function Base.show(io::IO, ::MIME"text/plain", filter::BPF)
    print(io, "Bootstrap particle filter algorithm
    hidden state model:                              ", state_model(filter.filt_prob),"
    observation model:                               ", obs_model(filter.filt_prob),"
    ensemble size:                                   ", no_of_particles(filter),"
    threshold for effective number of particles (%): ", 100*filter.alpha)
end

function Base.show(io::IO, filter::BPF)
    print(io, "BPF with ", no_of_particles(filter)," particles ")
end

######################
### MAIN ALGORITHM ###
######################
    
function initialize(filter::BPF)
    return BPFState(filter)
end     

function propagate(filter_state::BPFState, filt_prob::FilteringProblem, inputs, t1, t2, δx)

    ens = filter_state.ensemble
    res = filter_state.resampling_method
    alpha_eff = filter_state.alpha
    Neff = floor(eff_no_of_particles(ens))
    filter_state._neff_memory = cat(filter_state._neff_memory, [Integer(Neff)], dims=1);

    Dx, N = size(filter_state.ensemble.positions)
    
    if Neff < alpha_eff * N
        resample!(ens, res)
        filter_state._weight_memory[:, end] .= ens.weights
    else
        ens.ancestors = 1:N
    end

    filter_state.ensemble = ens

    propagate_state!(filter_state, state_model(filt_prob), inputs, t1, t2, δx)
    
    tvec = t1:δx:t2
        
    mm = Array{Float64}(undef, length(tvec)-1, Dx);
    pp = Array{Float64}(undef, length(tvec)-1, Dx, Dx); 

    Nt = length(tvec)-1
    for i in 1:Nt
        mm[i, :] = reshape(StatsBase.mean(filter_state._particle_memory[:,:,end-Nt+i]', filter_state.ensemble.weights, dims=2), Dx)
        x0 = filter_state._particle_memory[:,:,end-Nt+i] .- mm[i, :]'
        w = filter_state.ensemble.weights
        pp[i, :, :] = PDMat(convert(Matrix, Hermitian((x0.*w)'*x0 + 1e-16I(Dx))))       
    end
   
    return mm, pp, filter_state
end

function update!(filter_state::BPFState, filt_prob::FilteringProblem, inputs, t, δy, y)
    update_weights!(filter_state, obs_model(filt_prob), inputs, t, δy, y)
    
    ens = filter_state.ensemble
    mm = reshape(mean(ens), filt_prob.m)
    pp = cov(ens)
    
    pp = PDMat(convert(Matrix, Hermitian(pp+1e-32I(filt_prob.m))))
    
    filter_state.ensemble = ens;
    filter_state.mean = mm;
    filter_state.cov = pp;

end
    

function propagate_state!(filter_state::BPFState, model::DiffusionStateModel, myparams, t1, t2, δx)
    tvec = t1:δx:t2 
    x = copy(filter_state.ensemble.positions)
    w = copy(filter_state.ensemble.weights)
    a = copy(filter_state.ensemble.ancestors)
    Dx, N = size(x)
    xx = Array{eltype(x),3}(undef, N, Dx, length(tvec));      # Matrix of particles
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

    if length(myparams)==0 myparams=zeros(1) end
    
    Base.Threads.@threads for n in 1:N  
        sol = solve(SDEProblem(ff, gg, x[:, n], (tvec[1], tvec[end]), myparams))#, EM(); dt=0.001)#SRIW1()EM(); dt=0.001)  SKSROCK(;post_processing=true); dt=δx
        xvec = sol(tvec).u'
        xvec = permutedims(reshape(hcat(xvec...),(length(xvec[1]), length(xvec))))
        xx[n, :, :] = xvec'; ## XX is populated with initial value in the beginning
    end

    #=if length(myparams) == 0 myparams=zeros(1) end
    @sync @distributed for n in 1:N   
        sol = solve(SDEProblem(ff, gg, x[:, n], (tvec[1], tvec[end]), myparams, noise_rate_prototype = zeros(Dx,Dx)))#    isoutofdomain = (m,myparams,t) -> any(x->x.<0, m) SKSROCK(;post_processing=true); dt=0.01)#, EM(); dt=0.001)#SRIW1()EM(); dt=0.001)        
        xvec = sol(tvec).u'
        xvec = permutedims(reshape(hcat(xvec...),(length(xvec[1]), length(xvec))))
        xx[n, :, :] = xvec'; ## XX is populated with initial value in the beginning
    end
    =#
   

    #=
    # Or manually, with Euler Maruyama. Use this carefully at your own discretion, as it does not ensure ergodicity/positivity
    EM_δt = δx/10
    integration_tvec = (tvec[1]:EM_δt:tvec[end])
    xx_roll = Array{eltype(x),3}(undef,length(integration_tvec), Dx, N)
    Base.Threads.@threads for n in 1:N#
        xx_roll[1, :, n] .= x[:, n]
        i=2
        for ti in (tvec[1]:EM_δt:tvec[end])[1:end-1]
            xx_roll[i, :, n] =   xx_roll[i-1, :, n] + EM_δt*drift_f(xx_roll[i-1, :, n],  myparams, ti) + diff_g(xx_roll[i-1, :, n], myparams, ti)*randn(Dx, 1)*sqrt(EM_δt)
            i+=1
        end
        xx[n, :, :] =  (xx_roll[:, :, n]')[:, [1,end]] 
    end
    =#
    
    filter_state._particle_memory = cat(filter_state._particle_memory, xx[:, :, end], dims=3)
    filter_state._weight_memory = cat(filter_state._weight_memory, ww[:, 2:end], dims=2)
    #filter_state._ancestor_memory = cat(filter_state._ancestor_memory, aa, dims=2)
    filter_state.ensemble.positions = convert(Matrix, xx[:, :, end]')

end

function update_weights!(filter_state::BPFState, model::UserDefinedDiscreteObservationModel,  myparams, t, δy, dy)
    w = filter_state.ensemble.weights
    x = filter_state.ensemble.positions
    Dx, N = size(x)
    Dy = length(dy)
    
    hh = observation_function(model, t);

    wᵧ  = myparams[end-Dy+1:end]
    Rinv = Diagonal(1.0 ./ wᵧ.^2)
    Rinv_scaled = (1/δy) * Rinv
    hh2(x) = hh(x, zeros(Dy), myparams, t)
    
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

    filter_state._weight_memory[:, end] .= w
    filter_state.ensemble.weights = ProbabilityWeights(w)
end

################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function BPF(filt_prob::AbstractFilteringProblem, data)
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)
    return BPF(st_mod, ob_mod, data)
end
function BPFState(filter::BPF)
    N   = no_of_particles(filter)
    ens = WeightedParticleEnsemble(hcat([initialize(filter.state_model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)))
    return BPFState(ens)
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

abstract type Filtering end
struct Filter{FP<:AbstractFilteringProblem, FA<:AbstractFilteringAlgorithm, PR<: AbstractFilteringParameters, IN<:Function, TT <: Array, YY <: Array} <: Filtering 
    filt_prob::FP
    filt_algo::FA
    params::PR
    inputs::IN
    obs_t::TT
    obs_y::YY
end
function Base.show(io::IO, filt::Filter)
    print(io, "Continuous-time propagation
    filtering problem:                    ", filt.filt_prob,"
    filtering parameters:                 ", filt.params.values,"
    filtering algorithm:                  ", filt.filt_algo.name,"
    filtering algorithm parameters:       ", filt.filt_algo.params,"
    observations time:                         ", filt.obs_t[1, :]," ...
    observations values:                         ", filt.obs_y[1, :]," ...") 
end
"""
    initialize(Filtering Problem, Chosen Algorithm)
Runs the filtering algorithm.
"""
function initialize(problem::FilteringProblem, alg::FilteringAlgorithm)
    
    if alg.name == SREKF 
        SREKFState(problem) 
    elseif alg.name == BPF 
        BPFState(problem, collect(alg.params))
    elseif alg.name == FPF 
        FPFState(problem, collect(alg.params))
    elseif alg.name == BPFGrid 
        BPFGridState(problem, collect(alg.params))
    else 
        "Error0"
    end
end


"""
    run!(filtering)
Runs the filtering algorithm.
"""

function run!(filt::Filtering; records = (), verbose=true)

    ty_vec = filt.obs_t
    Y = filt.obs_y
    
    dimₓ = filt.filt_prob.m
        
    dtᵤ = filt.params.values.dtᵤ 
    dtₓ = filt.params.values.dtₓ
    dtᵧ = filt.params.values.dtᵧ 

    #ty_vec = range(dtᵧ, length = Ny, step = dtᵧ)
    tu_vec = range(0, stop = ty_vec[end], step = dtᵤ)
    tx_vec = range(0, stop = ty_vec[end], step = dtₓ)

    Nx = length(tx_vec)
    Nu = length(tu_vec)-1

    M = Int((Nx-1)/(Nu))

    # Initialization
    sfs = deepcopy(initialize(filt.filt_prob, filt.filt_algo))
    
    MM  = Array{Float64}(undef, Nx, dimₓ);
    PP = Array{Float64}(undef, Nx, dimₓ, dimₓ); 
    
    MM[1, :] = sfs.mean
    PP[1, :, :] = cov(filt.filt_prob.state_model.init)
    
    if filt.filt_algo.name == SREKF
        PP[1, :, :] = sfs.sd
    #else
    #    PP[1, :, :] = sfs.cov
    end

    
    if filt.filt_algo.name == BPFGrid
        ub, Nₗ = collect(filt.filt_algo.params)[end-1:end]
        x₁ = range(.0, stop = ub, length = Integer(Nₗ+1))
        x₂ = range(.0, stop = ub, length = Integer(Nₗ+1))
        
        Δx₁ = x₁[2]-x₁[1]
        Δx₂ = x₂[2]-x₂[1]
        
        xedges = x₁[1:end-1] .+ Δx₁/2
        yedges = x₂[1:end-1] .+ Δx₂/2

        pp = SharedArray{Float64}(Nu+1, Integer(Nₗ), Integer(Nₗ))
        pp[1, :, :] = pf_counter(sfs.ensemble.positions[1, :], sfs.ensemble.positions[2, :], xedges, yedges, sfs.ensemble.weights)
    end
    
    prog = Progress(Nu-1, desc="Filtering:", dt=0.5, barlen=50, showspeed=true, color=:white);
    
    start = time()
    for kᵢ = 1:Nu; if verbose; next!(prog); end;

        tinit = tu_vec[kᵢ]
        tfinal = tinit + M*dtₓ
                
        inp = filt.inputs(tinit)
        
        u = inp[:η][:u]
        d = inp[:η][:d]
        
        pₓ = inp[:p][:pₓ]
        pᵧ = inp[:p][:pᵧ]
        
        wₓ = inp[:w][:wₓ]
        wᵧ = inp[:w][:wᵧ]

        # PROPAGATE
        
        #p = vcat(u, d, pₓ, wₓ)
        p = Array(vcat(collect(Iterators.flatten(u)), collect(Iterators.flatten(d)), collect(Iterators.flatten(pₓ)), collect(Iterators.flatten(wₓ))))
        pred = propagate(sfs, filt.filt_prob, p, tinit, tfinal, dtₓ)     
        #filter_state = sfs; model=state_model(filt.filt_prob); t1=tinit; t2=tinit + M*dtₓ;δx=dtₓ
            
        MM[(kᵢ-1)*M+2 : kᵢ*M+1, :] = copy(pred[1])
        PP[(kᵢ-1)*M+2 : kᵢ*M+1, :, :] = copy(pred[2])
        sfs = pred[3] 

        # Saving only last value for future emission
        sfs.mean = pred[1][end, :]
        if filt.filt_algo.name == SREKF
            sfs.sd = pred[2][end, :, :];
        elseif filt.filt_algo.name == BPFGrid
            sfs.cov = PDMat(convert(Matrix, Hermitian(pred[2])))
        elseif filt.filt_algo.name == FPF
            sfs.cov = convert(Matrix, Hermitian(pred[2][end, :, :]))
        else
            sfs.cov = PDMat(convert(Matrix, Hermitian(pred[2][end, :, :])));
        end

        tx = round(tinit + M*dtₓ, digits=4)

        # UPDATE

        if (tx in ty_vec) 
            yindex = findfirst(t->t==tx, ty_vec)     
            inp = filt.inputs(tx)          
            u = inp[:η][:u]
            d = inp[:η][:d]
            pᵧ = inp[:p][:pᵧ]
            wᵧ = inp[:w][:wᵧ]

            p = vcat(collect(Iterators.flatten(u)), collect(Iterators.flatten(d)), collect(Iterators.flatten(pᵧ)), collect(Iterators.flatten(wᵧ)))
            if filt.filt_algo.name == FPF 
                #filter_state = sfs; model=obs_model(filt.filt_prob); t=tinit; δy=dtᵧ; dy=Y[1, :];
                update!(sfs, filt.filt_prob, filt.filt_algo.params[2], p, tinit, dtᵧ, Y[yindex, :])
                #println(maximum(sfs.ensemble.positions[:, end]))
            else
                #filter_state = sfs; model=obs_model(filt.filt_prob); t=tinit; δy=dtᵧ; dy=Y[yindex, :];
                update!(sfs, filt.filt_prob, p, tinit, dtᵧ, Y[yindex, :])
            end    
            MM[kᵢ*M+1, :] = sfs.mean
            if filt.filt_algo.name == SREKF
                PP[kᵢ*M+1, :, :] = copy(sfs.sd)
            else
                PP[kᵢ*M+1, :, :] = copy(sfs.cov)
            end
        end
        
        if filt.filt_algo.name == BPFGrid
             pp[kᵢ+1, :, :] = pf_counter(sfs.ensemble.positions[1, :], sfs.ensemble.positions[2, :], xedges, yedges, sfs.ensemble.weights)
        end
        
    end

    details = Dict(
        "clocktime" => time()-start
    )
    if filt.filt_algo.name == BPFGrid
            return [MM, PP, sfs, details, pp]
    else
        return [MM, PP, sfs, details]
    end
end

# AUXILIARY FUNCTIONS __________

function FilterState(filt)
    hidden_state = deepcopy(initialize(state_model(filt.filt_algo)))
    obs          = filt.filt_algo.obs_model(hidden_state, zero(eltype(hidden_state)))
    filter_state = deepcopy(initialize(filt.filt_algo))
    return FilterState(hidden_state, obs, filter_state)
end

function makeProgressBar(K, title)
    return Progress(K, desc=title, dt=0.5, barlen=50, showspeed=true, color=:white);
end

function PF_ProgressSummary(k) 
    return () -> [(:k,k)]
end
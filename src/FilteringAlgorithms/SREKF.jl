@doc raw"""
    SREKF storing only mean and standard deviation estimates
"""
mutable struct SREKFState{T1, T2} <: AbstractFilterState
    mean::T1
    sd::T2
    function SREKFState(problem::AbstractFilteringProblem)

        # Jacobian pre-compilation
        x0 = rand(problem.m, 1); p0 = rand(problem.mp, 1); t0 =  rand(1, 1);
        Jresults_x = (similar(x0, problem.m, problem.m), similar(p0, problem.m, problem.mp), similar(t0, problem.m, 1))    
        f_tape = ReverseDiff.JacobianTape(problem.state_model.dynamic_function(0.)[1], (x0, p0, t0))
        
        x0 = rand(problem.m, 1); r0 = rand(problem.n, 1); p0 = rand(problem.np, 1); t0 =  rand(1, 1)        
        Jresults_y = (similar(x0, problem.n, problem.m), similar(r0, problem.n, problem.n), similar(p0, problem.n, problem.np), similar(t0, problem.n, 1))    
        h_tape = ReverseDiff.JacobianTape(problem.obs_model.observation_function, (x0, r0, p0, t0))
        
        # compile `f_tape` and `g_tape` into more optimized representations
        compiled_f_tape = ReverseDiff.compile(f_tape)
        compiled_h_tape = ReverseDiff.compile(h_tape)
        global Jresults_x, Jresults_y 
        global compiled_f_tape, compiled_h_tape

        if problem.m == 1 
            m = mean(problem.state_model.init) 
            S = cov(problem.state_model.init)#reshape([problem.state_model.init.σ], 1, 1)
        else 
            m = mean(problem.state_model.init)#problem.state_model.init.μ       
            S = convert(Matrix,cholesky(cov(problem.state_model.init)).L)
            #P = problem.state_model.init.Σ#convert(Matrix,cholesky().L)
        end
        new{typeof(m), typeof(S)}(m, S)
    end
end

struct SREKF{FP<:AbstractFilteringProblem} <: AbstractFilteringAlgorithm
    filt_prob::FP
end
    
######################
### MAIN ALGORITHM ###
######################

function propagate(filter_state::SREKFState, filt_prob::FilteringProblem, inputs, t1, t2, δx)
    propagate_state(filter_state, state_model(filt_prob), inputs, t1, t2, δx)
end

function update!(filter_state::SREKFState, filt_prob::FilteringProblem, inputs, t, δy, y)
    update_state!(filter_state, obs_model(filt_prob), inputs, t, δy, y)
end

function propagate_state(filter_state::SREKFState, model::DiffusionStateModel, p, t1, t2, δx)

    m = filter_state.mean;
    S = filter_state.sd;

    D = length(m);

    Q = function(t)
       Matrix(I, D, D)
    end

    Φ = function(M)
        for i in 1:size(M, 1)
            for j in 1:size(M, 2)
                M[i,j] *= (i==j ? 0.5 : (i<j ? 0.0 : 1.0))
            end
        end
        M
    end

    function MDE!(du, u, p, t)
        m = u.x[1]      # Vector
        S = u.x[2]      # Matrix

        du_m = du.x[1]  # Vector to fill
        du_S = du.x[2]  # Matrix to fill

        # 1) drift
        drift = drift_function(model, t)
        du_m .= drift(m, p, t) 
        
        # 2) jacobian and diffusion
        gg = diffusion_function(model, t)
        args = (reshape(m, D, 1), reshape(p, length(p), 1), [t])
        ReverseDiff.jacobian!(Jresults_x, compiled_f_tape, args)
        Fx = Jresults_x[1]  
        A = S \ (Fx * S)
        B = S \ gg(m, p, t)
        B = B * Q(t) * B'

        du_S .= S * Φ(A + A' + B)
    end

    u0 = ArrayPartition(m, S)

    tvec = t1:δx:t2;
    tspan = (tvec[1], tvec[end]);
    prob = ODEProblem(MDE!, u0, tspan, p)
    sol = solve(prob) #,abstol = 1e-14, reltol = 1e-14) 
    vals = sol(tvec[2:end]).u
    Nt = length(tvec)-1
    mm = Array{Float64}(undef, Nt, D);
    ss = Array{Float64}(undef, Nt, D, D); 

    for i in 1:Nt  
        mm[i, :] = vals[i].x[1]
        ss[i, :, :] = convert(Matrix, LowerTriangular(vals[i].x[2]))#
    end
    
    return mm, ss, filter_state

end

function update_state!(filter_state::SREKFState, model::UserDefinedDiscreteObservationModel,  p, t, δy, y)
 
    m = filter_state.mean;
    S = filter_state.sd;

    Dx = length(m);
    Dy = length(y);

    hh = observation_function(model, t);
    
    w_y = p[end-Dy+1:end]
    R = (1/δy)*Matrix(I, Dy, Dy) .* [w_y[i]^2 for i in 1:Dy]  ;    
    R05 = convert(Matrix, cholesky(R).L)

    # Compute Jacobian matrices of h with respect to state and noise
    # Hx(m) = dh(x,r)/dx|x=m,r=0
    # Hr(m) = dh(x,r)/dr|x=m,r=0
    args = (reshape(m, Dx, 1), zeros(Dy, 1), reshape(p, length(p), 1), [t])
    ReverseDiff.jacobian!(Jresults_y, compiled_h_tape, args)
    Hx = Jresults_y[1]
    Hr = Jresults_y[2]

    # update equations;
    residual2 = y - hh(m, zeros(Dy), p, t);
    
    AA = Matrix(blockdiag(sparse((Hr*R05)'), sparse(S')))
    AA[Dy+1:end, 1:Dy] = (Hx*S)'
    DD = qr(AA).R
    W = DD[1:Dy, Dy+1:end]'
    B = (DD[1:Dy, 1:Dy])'
    
    # Kalman Gain
    K = W*inv(B)                       
    S = (DD[Dy+1:end, Dy+1:end])'
    
    filter_state.sd = S
    filter_state.mean += K * residual2;

end

################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function SREKF(filt_prob::AbstractFilteringProblem, data)
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)
    
    return SREKF(st_mod, ob_mod, data)
end
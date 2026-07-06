@doc raw"""
    GeneralPolynomialBasis()

Approximated Poisson Equation
By polynomials evaluated via a Gram-Schmidt procedure
"""
mutable struct GeneralPolynomialBasis{P1, P2, P3, E1, S1, S2, S3, S4, S4, S5, S5, I1, L1} <: GainEstimationMethod
    P::P1
    ∂P::P2
    ∂³P::P3
    e₀::E1
    ϕ::S1 
    ∇ϕ::S2      
    L::S3
    ∇L::S4
    ∇³L::S4
    A::S5
    R::S5
    index_db::I1
    λ::L1
    function GeneralPolynomialBasis(Nᵦ::Int64, Nₑ::Int64, Nₓ::Int64, Nᵧ::Int64, P, ∂P, ∂³P; grid="full")

        # Potential and Gain initiation        
        ϕ        = Array{Float64}(undef, Nᵧ, Nₑ)
        ∇ϕ       = Array{Float64}(undef, Nₓ, Nₑ, Nᵧ)
            
        if grid=="hypercross"
            Nᵦcollection = hyperbolic_cross_indices(Nₓ,Nᵦ-1; drop_zero=true)
        elseif grid=="total"
            Nᵦcollection = total_degree_indices(Nₓ,Nᵦ-1,; drop_zero=true)
        else
            Nᵦcollection_tuple = vcat(collect(Iterators.product(Iterators.repeated(tuple(0:Nᵦ-1...), Nₓ)...))...)
            Nᵦcollection =  Nₓ == 1 ? collect(0:Nᵦ-1) : vcat([collect(Nᵦcollection_tuple[n, :][1])' for n in 1:length(Nᵦcollection_tuple)]...)
            Nᵦcollection = Nᵦcollection[end:-1:2,:] 
        end
    
        Nᵦexp = size(Nᵦcollection,1)

        e₀ = zeros(Nₓ)
        L = zeros(Nₑ, Nᵦexp)
        ∇L = [zeros(Nₓ, Nₑ) for _ in 1:Nᵦexp]
        ∇³L = [zeros(Nₓ, Nₑ) for _ in 1:Nᵦexp]
        A = zeros(Nᵦexp, Nᵦexp) # 1 extra to allocate boundary conditions
        R = zeros(Nᵦexp, Nᵦexp)
        λ = zeros(Nᵧ)
        new{typeof(P), typeof(∂P), typeof(∂³P), typeof(e₀), typeof(ϕ), typeof(∇ϕ),typeof(L),typeof(∇L),typeof(∇³L),typeof(A), typeof(R), typeof(Nᵦcollection),typeof(λ)}(P, ∂P, ∂³P, e₀, ϕ, ∇ϕ, L, ∇L, ∇³L, A, R, Nᵦcollection, λ)
    end
end


function weak_compositions(k::Int, n::Int)
    if n == 1
        return [[k]]
    end
    result = []
    for i in 0:k
        for tail in weak_compositions(k - i, n - 1)
            push!(result, [i; tail])
        end
    end
    return result
end

function index_grid(Nb, Nx)
    result = []
    for k in 0:Nb-1
        append!(result, weak_compositions(k, Nx))
    end
    return reduce(vcat, [permutedims(c) for c in result])
end


function solve!(eq::PoissonEquation, method::GeneralPolynomialBasis; gain=true)

    Nₓ, Nₑ, Nᵧ = size(eq.gain)
    M  = -(eq.L .- eq.mean_L)  
    Nᵦcollection = method.index_db
   
    if Nₓ == 1  Nᵦcollection = sort(Nᵦcollection, dims=1) end
    Nᵦ = maximum(Nᵦcollection)+1
    Nᵦexp = size(Nᵦcollection,1)
    x =  copy(eq.positions)

    if gain 
        P_vals =   [zeros(Nₓ, Nₑ) for _ in 1:Nᵦ]
        ∂P_vals =  [zeros(Nₓ, Nₑ) for _ in 1:Nᵦ]
        #∂³P_vals =  [zeros(Nₓ, Nₑ) for _ in 1:Nᵦ]
        method.e₀ .= mean(x, dims=2)
        #σ = std(x, dims=2)

        # Precompute P values for each dimension and element
        
        Base.Threads.@threads for nᵦ in 1:Nᵦ
            Pfun  = method.P[nᵦ]
            dPfun = method.∂P[nᵦ]
            #dP³fun = method.∂³P[nᵦ]
            for nᵢ in 1:Nₑ
                @inbounds for nₓ in 1:Nₓ    
                    xval = x[nₓ, nᵢ]
                    e0   = method.e₀[nₓ]
                    P_vals[nᵦ][nₓ, nᵢ]   = Pfun(xval, e0)
                    ∂P_vals[nᵦ][nₓ, nᵢ]  = dPfun(xval, e0)
                    #∂³P_vals[nᵦ][nₓ, nᵢ] = dP³fun(xval, e0)
                end
            end
        end

        # Fetching the auxiliary polynomials and evaluating them at the particle positions  
        E = zeros(Nₑ,Nᵦexp)
        Base.Threads.@threads for nid in 1:Nᵦexp #  @sync @distributed 
            n = Nᵦcollection[nid, :]
            E[:, nid] = [prod([P_vals[n[nₓ]+1][nₓ, nᵢ]  for nₓ in 1:Nₓ]) for nᵢ in 1:Nₑ]
        end 

        ∇E = [zeros(Nₓ, Nₑ) for _ in 1:Nᵦexp]
        #∇³E = [zeros(Nₓ, Nₑ) for _ in 1:Nᵦexp]
        
        # Fetching the gradient of auxiliary polynomials and evaluating them at the particle positions    
        if Nₓ == 1
            Base.Threads.@threads for nid in 1:Nᵦexp #@sync @distributed
                n                    = Nᵦcollection[nid, :]
                ∇E[nid][1, :]  = ∂P_vals[n[1]+1][1, :]
                #∇³E[nid][1, :]  = ∂³P_vals[n[1]+1][1, :]               
            end
        else
            Base.Threads.@threads for nid in 1:Nᵦexp
                n = @view Nᵦcollection[nid, :]
                    
                @inbounds for nₑ in 1:Nₑ

                    # compute all P values first
                    Ptmp = zeros(Nₓ)
                    for nₓ in 1:Nₓ
                        Ptmp[nₓ] = P_vals[n[nₓ] + 1][nₓ, nₑ]
                    end

                    # compute gradient safely
                    for nₓ in 1:Nₓ
                        prod = 1.0
                        for j in 1:Nₓ
                            if j != nₓ
                                prod *= Ptmp[j]
                            end
                        end
                        dp = ∂P_vals[n[nₓ] + 1][nₓ, nₑ]
                        ∇E[nid][nₓ, nₑ] = dp * prod
                    end
                end
            end
        end
    
        # Letting the mother basis be of L²α,0, so that the inner-product in H¹α,0 is equivalent to the traditional one 
        E .-= mean(E, dims=1)      
       
        # Update method.L, method.∇L, and method.∇³L
        method.L .= E
        method.∇L = ∇E
        #method.∇³L = ∇³E
        
        G = Matrix{Float64}(undef, Nₓ*Nₑ, Nᵦexp)
   
        
        for nid in 1:Nᵦexp
            @inbounds G[:, nid] = vec(method.∇L[nid])
     
        end

        mul!(method.A, G', G, 1/Nₑ, 0.0)

    end

    method.R = spdiagm(vec(sum(Nᵦcollection, dims=2)))

        
    for nᵧ in 1:Nᵧ
        mᵧ = M[nᵧ,:]
        b = [(1/Nₑ)*dot(mᵧ, method.L[:, nid]) for nid in 1:Nᵦexp]

       
        û = LinearSolve.solve(LinearProblem(method.A'method.A + method.λ[nᵧ] * method.R, method.A'b), KrylovJL_CG()).u#LinearSolve.solve(prob, KrylovJL_GMRES())
       
        eq.potential[nᵧ,:] .= sum([û[nid]*method.L[:, nid]' for nid in 1:Nᵦexp])[:];
        eq.gain[:,:,nᵧ] .= sum([û[nid]*method.∇L[nid] for nid in 1:Nᵦexp])
    end 

end


function safe_cholesky(G)
    jitter_levels = [1e-32, 1e-16, 1e-8]  # Define jitter levels
    Nᵦexp = size(G, 1)  # Assuming G is a square matrix
   
    for jitter in jitter_levels
        try
            L = cholesky(G + jitter * I(Nᵦexp)).L
            return L  # Return L if decomposition succeeds
        catch e
            println("Failed with jitter $(jitter): ", e)
        end
    end

    # If all attempts fail, you can choose to error out or return nothing:
    error("Cholesky decomposition failed for all jitter levels.")
end

#https://people.compute.dtu.dk/pcha/DIP/chap5.pdf
function optimise(eq::PoissonEquation, method::GeneralPolynomialBasis)

    Nₓ, Nₑ, Nᵧ = size(eq.gain)
    M  = -(eq.L .- eq.mean_L)  
    Nᵦcollection = method.index_db
    if Nₓ == 1  Nᵦcollection = sort(Nᵦcollection, dims=1) end
    Nᵦ = maximum(Nᵦcollection)+1
    Nᵦexp = size(Nᵦcollection,1)
    x =  copy(eq.positions)

    P_vals =   [zeros(Nₓ, Nₑ) for _ in 1:Nᵦ]
    ∂P_vals =  [zeros(Nₓ, Nₑ) for _ in 1:Nᵦ]
    #∂³P_vals =  [zeros(Nₓ, Nₑ) for _ in 1:Nᵦ]
    method.e₀ .= mean(x, dims=2)

    # Precompute P values for each dimension and element
    Base.Threads.@threads for nᵦ in 1:Nᵦ
        Pfun  = method.P[nᵦ]
        dPfun = method.∂P[nᵦ]
        dP³fun = method.∂³P[nᵦ]
        for nᵢ in 1:Nₑ
            @inbounds for nₓ in 1:Nₓ    
                xval = x[nₓ, nᵢ]
                e0   = method.e₀[nₓ]
                P_vals[nᵦ][nₓ, nᵢ]   = Pfun(xval, e0)
                ∂P_vals[nᵦ][nₓ, nᵢ]  = dPfun(xval, e0)
                #∂³P_vals[nᵦ][nₓ, nᵢ] = dP³fun(xval, e0)
            end
        end
    end

    # Fetching the auxiliary polynomials and evaluating them at the particle positions  
    E = zeros(Nₑ,Nᵦexp)
    Base.Threads.@threads for nid in 1:Nᵦexp #  @sync @distributed 
        n = Nᵦcollection[nid, :]
        E[:, nid] = [prod([P_vals[n[nₓ]+1][nₓ, nᵢ]  for nₓ in 1:Nₓ]) for nᵢ in 1:Nₑ]
    end 

    ∇E = [zeros(Nₓ, Nₑ) for _ in 1:Nᵦexp]
    #∇³E = [zeros(Nₓ, Nₑ) for _ in 1:Nᵦexp]
    
    # Fetching the gradient of auxiliary polynomials and evaluating them at the particle positions    
    if Nₓ == 1
        Base.Threads.@threads for nid in 1:Nᵦexp #@sync @distributed
            n                    = Nᵦcollection[nid, :]
            ∇E[nid][1, :]  = ∂P_vals[n[1]+1][1, :]
            #∇³E[nid][1, :]  = ∂³P_vals[n[1]+1][1, :]               
        end
    else
        Base.Threads.@threads for nid in 1:Nᵦexp
            n = @view Nᵦcollection[nid, :]
                
            @inbounds for nₑ in 1:Nₑ

                # compute all P values first
                Ptmp = zeros(Nₓ)
                for nₓ in 1:Nₓ
                    Ptmp[nₓ] = P_vals[n[nₓ] + 1][nₓ, nₑ]
                end

                # compute gradient safely
                for nₓ in 1:Nₓ
                    prod = 1.0
                    for j in 1:Nₓ
                        if j != nₓ
                            prod *= Ptmp[j]
                        end
                    end
                    dp = ∂P_vals[n[nₓ] + 1][nₓ, nₑ]
                    ∇E[nid][nₓ, nₑ] = dp * prod
                 end
            end

        end
    end
    
    # Letting the mother basis be of L²α,0, so that the inner-product in H¹α,0 is equivalent to the traditional one 
    E .-= mean(E, dims=1)      
    
    # Update method.L, method.∇L, and method.∇³L
    method.L .= E
    method.∇L = ∇E

    
    G = Matrix{Float64}(undef, Nₓ*Nₑ, Nᵦexp)

    for nid in 1:Nᵦexp
        @inbounds G[:, nid] = vec(method.∇L[nid])
   
    end

    mul!(method.A, G', G, 1/Nₑ, 0.0)


    v = vec(sum(Nᵦcollection, dims=2))
    method.R = spdiagm(v)

 
    λopt_vec = zeros(Nᵧ)
    for nᵧ in 1:Nᵧ
        mᵧ = M[nᵧ,:]
        b = [(1/Nₑ)*dot(mᵧ, method.L[:, nid]) for nid in 1:Nᵦexp]
        θ = -8:0.25:2
        #println("Theta": minimum(θ))

        function lcurve_curvature(log10λ; A, R, b)
            
            function XY(θ)
                λ = 10.0^θ
                AtA = A'A
                Atb = A'b
                
                x̂ = (AtA + λ * R) \ Atb

                res = norm(A * x̂ - b)

                sol = sqrt(dot(x̂, R*x̂))
                return log(res), log(sol)
            end

            # First derivative
            d1 = ForwardDiff.derivative(t -> XY(t)[1], log10λ)
            d2 = ForwardDiff.derivative(t -> XY(t)[2], log10λ)

            # Second derivative
            dd1 = ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> XY(s)[1], t), log10λ)
            dd2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> XY(s)[2], t), log10λ)

            numerator = abs(d1 * dd2 - d2 * dd1)
            denominator = (d1^2 + d2^2)^(3/2)

            return numerator / denominator
        end

        κ_vals = SharedArray(zeros(length(θ)))
        Base.Threads.@threads for i in 1:length(θ)            
            κ_vals[i] = lcurve_curvature(θ[i]; A=method.A, R=method.R, b=b)
        end
        #κ_vals = [lcurve_curvature(t; A=method.A, R=method.R, b=b) for t in θ]
        #Plots.plot(θ,κ_vals,xlabel=L"\lambda",ylabel="Curvature",label="")

        θ_opt = θ[argmax(κ_vals)]
        λopt_vec[nᵧ] = 10.0^θ_opt
    end

    return λopt_vec
end
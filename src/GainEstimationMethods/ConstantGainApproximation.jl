"""
    ConstantGainApproximation()

Represents an approximation of the gain by a constant (in the Euclidean sense) vector field, given by the covariance of the observation function and x under the particle distribution.
"""
struct ConstantGainApproximation <: GainEstimationMethod end

function solve!(eq::PoissonEquation, method::ConstantGainApproximation; gain=true)
    Dₓ, N, Dᵧ = size(eq.gain)
    M = m(eq)
    #massmatrix = eq.mass(eq.positions')
    #@inbounds for dᵧ in 1:Dᵧ 
    #    for dₓ in 1:Dₓ
    #        eq.gain[dₓ, :, dᵧ] .= dot(eq.positions[dₓ, :], M[dᵧ,:])/N
    #    end
    #end
    #eq.gain = SharedArray{Float64}(Dₓ, N, Dᵧ)

    for dᵧ in 1:Dᵧ #@sync @distributed 
        rhs = (1/N)*eq.positions*M[dᵧ,:]
        eq.gain[:, :, dᵧ] .= rhs
        eq.potential[dᵧ,:] .= (eq.positions.-mean(eq.positions,dims=2))'rhs
    end
    #[eq.gain[:, i, dᵧ] *= massmatrix[i] for i in 1:N, dᵧ in 1:Dᵧ];
    eq.gain
end
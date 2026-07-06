"""
    FilteringAlgorithm(alg_name; kwargs)

Specify a filtering algorithm.

Constraints: alg_name in (EKF)
This ensures that the filtering problem can be evaluated by one of the implemented filtering algorithm.
"""
mutable struct FilteringAlgorithm{A1, A2} <: AbstractFilteringAlgorithm
    name::A1
    params::A2
end
# Making params optional, default is empty, i.e. ()
FilteringAlgorithm(name; params = ()) = FilteringAlgorithm(name, params)                                                     
# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", alg::FilteringAlgorithm{A1}) where {A1}
    print(io, alg)
end

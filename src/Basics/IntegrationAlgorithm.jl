struct IntegrationAlgorithm{IA, M1} <: AbstractIntegrationAlgorithm{IA} 
    function IntegrationAlgorithm2(alg::OrdinaryDiffEqAlgorithm)
        return new{IA,typeof(alg)}(alg)
    end
end


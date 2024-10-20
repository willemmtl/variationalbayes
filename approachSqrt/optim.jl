using Distributions

include("../iGMRF.jl");

function KLOptim(F::iGMRF, Y::Vector{Vector{Float64}})
    α = Optim.minimizer(initializeValues(θ -> densityTarget(θ, F=F, Y=Y), [1, fill(0.0, m)...]))
    calculationSpace = createCalculationSpace(α, 1000)

    function KLDivergence(θ::Vector{<:Real})
        # Distribution d'approximation
        η = θ[1:m+1];
        Σ = diagm(θ[m+2:end].^2);
        distApprox = MvNormal(η, Σ);
        
        # Évaluation des densités
        p = vec(mapslices(x -> densityTarget(x, F=F, Y=Y), calculationSpace, dims=1))
        q = vec(mapslices(x -> logpdf(distApprox, x), calculationSpace, dims=1))
        return kldiv(q, p)
    end;

    η₀ = α;
    Σ₀ = sqrt.(diag(inv(computeFisherInformation(θ -> densityTarget(θ, F=F, Y=Y), α))));

    θ₀ = [η₀..., Σ₀...];
    res = optimize(KLDivergence, θ₀);

    return res
end;
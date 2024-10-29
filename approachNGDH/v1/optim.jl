using Distributions, Optim

include("../../iGMRF.jl");
include("../../approxFramework.jl");

function KLOptim(F::iGMRF, Y::Vector{Vector{Float64}})

    m = F.G.m₁ * F.G.m₂;

    mode = findMode(θ -> functionalFormPosterior(θ, F=F, Y=Y), [1, fill(0.0, m)...]);
    α = Optim.minimizer(mode);
    gap = Optim.minimum(mode);

    Fvar = inv(computeFisherInformation(θ -> densityTarget(θ, gap, F=F, Y=Y), α));

    calculationSpace = createCalculationSpace(α, Fvar, 1000);

    function KLDivergence(θ::Vector{<:Real})
        # Distribution d'approximation
        η = θ[1:m+1];
        Σ = diagm([θ[m+2]^2, fill(θ[m+3]^2, m)...]);
        logDensityApprox(x::DenseVector) = logpdf(MvNormal(η, Σ), x) + logpdf(Gamma(1, 100), x[1])
        
        # Évaluation des densités
        p = vec(mapslices(x -> densityTarget(x, gap, F=F, Y=Y), calculationSpace, dims=1))
        q = vec(mapslices(x -> logDensityApprox(x), calculationSpace, dims=1))
        return kldiv(p, q)
    end;

    η₀ = α;
    Σ₀ = sqrt.(diag(Fvar));

    θ₀ = [η₀..., Σ₀[1], mean(Σ₀[2:end])];
    res = optimize(KLDivergence, θ₀);

    return res
end;
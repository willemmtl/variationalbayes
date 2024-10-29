using Distributions

include("../../iGMRF.jl");

function KLOptim(F::iGMRF, Y::Vector{Vector{Float64}})

    m = F.G.m₁ * F.G.m₂;

    mode = findMode(θ -> functionalFormPosterior(θ, F=F, Y=Y), [1, fill(0.0, m)...]);
    α = Optim.minimizer(mode);
    gap = Optim.minimum(mode);

    Fvar = inv(computeFisherInformation(θ -> densityTarget(θ, gap, F=F, Y=Y), α));

    calculationSpace = createCalculationSpace(α, Fvar, 1000);

    function KLDivergence(θ::Vector{<:Real})
        # Distribution d'approximation
        η = θ[1:m];
        Σ = diagm(fill(θ[m+1]^2, m));
        a = θ[m+2]^2;
        b = θ[m+3]^2;
        logDensityApprox(x::DenseVector) = logpdf(MvNormal(η, Σ), x[2:m+1]) + logpdf(Gamma(a, b), x[1])
        
        # Évaluation des densités
        p = vec(mapslices(x -> densityTarget(x, gap, F=F, Y=Y), calculationSpace, dims=1))
        q = vec(mapslices(x -> logDensityApprox(x), calculationSpace, dims=1))
        return kldiv(p, q)
    end;

    η₀ = α[2:m+1];
    Σ₀ = sqrt.(diag(Fvar));

    θ₀ = [η₀..., mean(Σ₀[2:end]), 1, 10];
    res = optimize(KLDivergence, θ₀, Newton(); autodiff = :forward);

    return res
end;
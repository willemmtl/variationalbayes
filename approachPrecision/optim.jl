using Distributions, Optim

include("../iGMRF.jl");
include("../approxFramework.jl");

function KLOptim(F::iGMRF, Y::Vector{Vector{Float64}})

    m = F.G.m₁ * F.G.m₂;

    temps = @elapsed begin
        mode = findMode(θ -> functionalFormPosterior(θ, F=F, Y=Y), [1, fill(0.0, m)...]);
    end
    println("Temps pour trouver le mode : ", temps, " s")
    
    α = Optim.minimizer(mode);
    gap = Optim.minimum(mode);
    
    temps = @elapsed begin
        Fvar = inv(computeFisherInformation(θ -> densityTarget(θ, gap, F=F, Y=Y), α));
    end
    println("Temps pour calculer l'info de Fisher : ", temps, " s")
    
    calculationSpace = createCalculationSpace(α, Fvar, 1000);

    function KLDivergence(θ::Vector{<:Real})
        # Distribution d'approximation
        η = θ[1:m];
        a = θ[m+1]^2;
        b = θ[m+2]^2;
        logDensityApprox(x::DenseVector) = logpdf(MvNormal(η, I/x[1]), x[2:m+1]) + logpdf(Gamma(a, b), x[1])

        # Évaluation des densités
        p = vec(mapslices(x -> densityTarget(x, gap, F=F, Y=Y), calculationSpace, dims=1))
        q = vec(mapslices(x -> logDensityApprox(x), calculationSpace, dims=1))
        return kldiv(p, q)
    end;

    η₀ = α[2:end];

    θ₀ = [η₀..., 2, 10];
    
    temps = @elapsed begin
        res = optimize(KLDivergence, θ₀, Newton(), autodiff=:forward);
    end
    println("Temps pour l'optimisation : ", temps, " s")
    return res
end;
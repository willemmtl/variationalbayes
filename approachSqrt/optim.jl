using Distributions

include("../iGMRF.jl");

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
        L = buildLowerTriangular(θ[m+2:end], m);
        Σ = L*L';
        distApprox = MvNormal(η, Σ);
        
        # Évaluation des densités
        p = vec(mapslices(x -> densityTarget(x, gap, F=F, Y=Y), calculationSpace, dims=1))
        q = vec(mapslices(x -> logpdf(distApprox, x), calculationSpace, dims=1))
        return kldiv(q, p)
    end;

    η₀ = α;
    Σ₀ = round.(Fvar, digits=6);
    l₀ = extractCholeskyComponents(Σ₀)

    θ₀ = [η₀..., l₀...];
    res = optimize(KLDivergence, θ₀);

    return res
end;


"""
    extractCholeskyComponents(Σ)

Extracts components of the lower triangular side of a given matrix Σ.
Store components in a vector.

# Arguments
-`Σ::Matrix`: Covariance matrix.
"""
function extractCholeskyComponents(Σ::Matrix)
    L = cholesky(Σ).L;
    return [L[i, j] for i = 1:m+1 for j = 1:i]
end


"""
    buildLowerTriangular(components)

Build a lower triangular matrix from a vector of components.

# Arguments
- `components::DenseVector`: coefficients of the lower triangular matrix.
- `m::Integer`: 
"""
function buildLowerTriangular(components::DenseVector, m::Integer)
    L = zeros(m+1, m+1);
    ind = 1;
    for i = 1:m+1
        for j = 1:i
            L[i, j] = components[ind];
            ind += 1;
        end
    end
    
    return L
end
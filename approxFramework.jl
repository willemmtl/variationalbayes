using Optim

include("iGMRF.jl");


function densityTarget(θ::DenseVector, gap::Real; F::iGMRF, Y::Vector{Vector{Float64}})
    return functionalFormPosterior(θ, F=F, Y=Y) + gap
end;


function functionalFormPosterior(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    return (
        sum(loglikelihood.(Gumbel.(θ[2:end], 1), Y)) 
        + ((length(θ)-1) - F.r) * θ[1] / 2 
        - exp(θ[1]) * θ[2:end]' * F.G.W * θ[2:end] / 2
        + logpdf(Gamma(1, 100), exp(θ[1]))
    )
end;


function findMode(f::Function, θ₀::DenseVector)
    F(θ::DenseVector) = -f(θ)
    return optimize(F, θ₀, Newton(); autodiff = :forward)
end;


function computeFisherInformation(f::Function, θ̂::AbstractArray)
    return - ForwardDiff.hessian(f, θ̂)
end;


"""
    createCalculationSpace(α, N)

Define α's neighborhood.

# Arguments

- `Fvar::Matrix{<:Real}`: Fisher's covariance matrix.
- `N::Integer`: number of points of the calculation space.
"""
function createCalculationSpace(α::DenseVector, Fvar::Matrix{<:Real}, N::Integer)
    Σ = round.(Fvar .* .1 , digits=5);
    calculationSpace = rand(MvNormal(α, Σ), N);
    # On supprime les échantillons pour lesquels κ < 0
    return calculationSpace[:, calculationSpace[1, :] .> 0]
end;


function kldiv(p::DenseVector, q::DenseVector)
    return sum(exp.(p) .* (p .- q))
end;
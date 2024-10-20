include("iGMRF.jl");


function densityTarget(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    return (
        sum(loglikelihood.(Gumbel.(θ[2:end], 1), Y)) 
        + ((length(θ)-1) - F.r) * θ[1] / 2 
        - exp(θ[1]) * θ[2:end]' * F.G.W * θ[2:end] / 2
        + logpdf(Gamma(1, 100), exp(θ[1]))
    )
end;


function initializeValues(f::Function, θ₀::DenseVector)
    F(θ::DenseVector) = -f(θ)
    res = optimize(F, θ₀, Newton(); autodiff = :forward)
    return res # Optim.minimizer(res), I
end;


function computeFisherInformation(f::Function, θ̂::AbstractArray)
    return - ForwardDiff.hessian(f, θ̂)
end;


"""
    createCalculationSpace(α, N)

Define α's neighborhood.

# Arguments

- `α::DenseVector`: mode of the posterior
- `N::Integer`: number of points of the calculation space.
"""
function createCalculationSpace(α::DenseVector, N::Integer)
    Σ = round.(inv(computeFisherInformation(θ -> densityTarget(θ, F=F, Y=Y), α)), digits=4)
    return rand(MvNormal(α, Σ), N)
end;


function kldiv(p::DenseVector, q::DenseVector)
    return sum(exp.(p) .* (p .- q))
end;
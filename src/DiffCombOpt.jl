module DiffCombOpt

using Statistics

export randg, σ, gumbel_softmax, gumbel_sigmoid

"""sample random value from Gumbel(0, 1)"""
randg() = - log(-log(rand()))

"""sample a vector or array of values from Gumbel(0, 1)"""
randg(n::Int...) = - log.(-log.(rand(n...)))

exprandg(n::Int) = 1.0 ./ -log.(rand(n))

# normalize a vector with nonzero numbers
norm(a::Vector) = a / sum(a)
norm!(a::Vector) = (a ./= sum(a))

"""logistic map"""
function σ(x::Number)
    if x > 10.0
        return one(x)
    elseif x < - 10.0
        return zero(x)
    else 
        return inv(one(x) .+ exp(-x))
    end
end

"""softmax"""
σ(x::Vector) = exp.(x) |> norm

"""
    gumbel_softmax(lp::Vector; τ::Number=0.1)

Compute the Gumbel softmax approximation of sampling a one-hot-vector the logarithm
of an (unnormalized) probability vector. `τ` is the temperature parameter determining
the quality of the approximation.
"""
function gumbel_softmax(lp::Vector; τ::Number=0.1)
    z = lp .+ randg(length(lp))
    z = z .- mean(z)
    return exp.(z ./ τ) |> norm
end

"""
    gumbel_softmax(lp::Vector; τ::Number=0.1)

Compute the Gumbel softmax approximation of sampling a one-hot-vector the logarithm
of an (unnormalized) probability matric (row-wise by default). `τ` is the temperature
parameter determining the quality of the approximation.
"""
function gumbel_softmax(lp::Matrix; τ::Number=0.1, dims=2)
	Z = lp .+ randg(size(lp)...)
	Z = Z .- mean(Z; dims=dims)
    Y = exp.(Z ./ τ)
    return Y ./ sum(Y; dims=dims)
end

"""
gumbel_sigmoid(lp; τ::Number=1.0)

Compute the Gumbel approximation of sampling a binary vector from a vector
of an (unnormalized) log-probabilities. `τ` is the temperature parameter determining
the quality of the approximation.
"""
function gumbel_sigmoid(lp; τ::Number=1.0)
    n = length(lp)
    y = exp.((lp .+ randg(n)) ./ τ) 
    return y / sum(y .+ exp.(randg(n) ./ τ))
end

end
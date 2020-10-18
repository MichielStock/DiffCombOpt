

using Distributions

# sample random value from Gumbel(0, 1)
rgumb() = Gumbel() |> rand
rgumb() = - log(-log(rand()))

# sample n random values from Gumbel(0, 1)
rgumb(n) = rand(Gumbel(), n)
rgumb(n::Int) = - log.(-log.(rand(n)))

# normalize a vector with nonzero numbers
norm(a::Vector) = a / sum(a)
norm!(a::Vector) = (a ./= sum(a))

function σ(x)
    if x > 10.0
        return one(x)
    elseif x < - 10.0
        return zero(x)
    else 
        return inv(one(x) .+ exp(-x))
    end
end


function gumbel_softmax(p, τ=1.0)
    return exp.((log.(p) .+ rgumb(length(p))) ./ τ) |> norm
end

function gumbel_sigmoid(p, τ=1.0)
    y = exp.((log.(p) .+ rgumb(length(p))) ./ τ) 
    return y ./ (y .+ exp.((log.(1.0 .- p) .+ rgumb(length(p))) ./ τ) )
end

J = randn(100, 100)

# should be -1, 1 => FIXME
E(s) =  - s' * J * s + (1 .-s)' * J * (1 .-s)

p = rand(100)

loss(f; τ=1.0, λ=0.1) = E(gumbel_sigmoid(σ.(f), τ)) + λ * sum(f.^2.0)

f = randn(100)

loss(f)

using Zygote

∇loss(f) = loss'(f)
∇²loss(f) = Zygote.hessian(loss, f)

f .-= ∇²loss(f) \ ∇loss(f)
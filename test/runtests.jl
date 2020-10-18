using DiffCombOpt, Test

@testset "rand Gumbel" begin
    @test randg() isa Number
    @test randg(3) isa Vector{Float64}
    @test randg(3, 4) isa Matrix{Float64}
end

@testset "Gumbel softmax" begin
    p = gumbel_softmax([0, 1]; τ=0.01)
    @test maximum(p) ≈ 1.0
    @test sum(p) ≈ 1.0

    P = gumbel_softmax([0 1; 2 3]; τ=0.01)
    @test maximum(P) ≈ 1.0
    @test sum(P, dims=2) .≈ 1.0
end

@testset "Gumbel sigmoid" begin
    y = gumbel_sigmoid([5, -5, 0.1]; τ=0.01)
    @test y isa Vector
    @test y[1] ≈ 1
    @test y[2] < 1e-4
end
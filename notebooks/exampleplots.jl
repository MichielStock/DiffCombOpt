### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ fdaeccc8-139f-11eb-0474-ffb743e08fcf
using Plots, LinearAlgebra

# ╔═╡ 2a10283a-13a4-11eb-2598-bb10bb2e9c60
using DiffCombOpt, Distributions

# ╔═╡ f32ee0fc-1873-11eb-2b1b-05a124c7e005
using STMO.TestFuns

# ╔═╡ 03174924-13a0-11eb-1d1a-cb1e80fc18e6
P = 2*[20 -13; -13 16]

# ╔═╡ 118fb1b2-13a0-11eb-0bcf-85ac564610b0
q = [1, -100]

# ╔═╡ 36590a16-13a0-11eb-1f76-5d250cbe5b00
r = 2

# ╔═╡ 17f2ae4c-13a0-11eb-2690-4f89d04b3013
begin
	f(x) = 0.5x' * P * x + q ⋅ x + r
	f(x,y) = f([x,y])
end

# ╔═╡ f6fdfee8-13a0-11eb-2cb0-712a338819e4
n = 15

# ╔═╡ 4052280e-13a0-11eb-1093-c326a83ca963
xvals = -n:0.1:n

# ╔═╡ 4a16dbf0-13a0-11eb-2a13-e73a64cac7e3
yvals = -n:0.1:n

# ╔═╡ 6a89231a-13a1-11eb-3334-5f05ce41e5e4
xstar = - P \ q

# ╔═╡ b87fd91a-13a1-11eb-12a1-c33a3084d384
f(xstar)

# ╔═╡ b1b4b9c8-13a8-11eb-2329-29de0ed43225
x₀ = [-10, 10]

# ╔═╡ e7fef5ca-13a8-11eb-3b7a-877b8cffad7f
nsteps = 50

# ╔═╡ 4e4cdcce-13a0-11eb-325c-33d97947f187
begin
	pc = contour(xvals, yvals, f, xlabel="x1", ylabel="x2", color=:speed)
	scatter!([xstar[1]], [xstar[2]], m=:star, label="minimizer")
	savefig("../plots/real_valopt.pdf")
	pc
end

# ╔═╡ 944855be-13a0-11eb-0576-57af0f1ff502
function gradient_descent(x₀, ∇f; t=0.1, ϵ=1e-5)
	x = x₀
	while true
		x .-= t * ∇f(x)
		norm(∇f(x)) < ϵ && break
	end
	return x
end

# ╔═╡ 9385a9c4-13a0-11eb-2a59-cf5eaf5c7ee6
begin
	pint = contour(xvals, yvals, f, xlabel="x1", ylabel="x2", color=:speed)
	scatter!(pint, [i for i in -n:n for j in -n:n], [i for j in -n:n for i in -n:n], label="", ms=2, color=:red, markerstrokecolor=:red)
	savefig("../plots/int_valopt.pdf")
	pint
end

# ╔═╡ b1748150-13a3-11eb-280c-29992381ce5d
evals, evects = eigen(P)

# ╔═╡ b9607fe8-13a3-11eb-1dd2-d3b2b74d5fd4
t = 2 / (first(evals) + last(evals))

# ╔═╡ 11d945e6-160c-11eb-2339-a94c3acb028b
exp(-(10 - 11) / 4)

# ╔═╡ fea52b7c-13a4-11eb-1768-035fe22c0a2a
function simulated_annealing(x₀, f, neighbor, temperatures)
	x = x₀
	for T in temperatures
		xn = neighbor(x)
		if rand() < exp(-(f(xn) - f(x)) / T)
			x = xn
		end
	end
	return x
end

# ╔═╡ 1d3d4b48-160a-11eb-2ca3-efd22f33a988
function simulated_annealing_steps(x₀, f, neighbor, temperatures)
	x = x₀
	X = zeros(length(temperatures)+1, length(x))
	X[1,:] .= x
	t = 1
	for (i, T) in enumerate(temperatures)
		xn = neighbor(x)
		if rand() < exp(-(f(xn) - f(x)) / T)
			x = xn
		end
		X[i+1,:] .= x
	end
	return x, X
end

# ╔═╡ 5bfedb9c-13a5-11eb-098d-c7bae32477c4
neighbor(x) = clamp.(x .+ rand([-1,0,1], length(x)), -n, n)

# ╔═╡ 4f4b00b4-13a6-11eb-0af2-2df839938e28
temperatures = -20:0.1:5 .|> t->2^(-t)

# ╔═╡ eb4b3b10-13a5-11eb-2e30-3dcfc01f4e2b
simulated_annealing([-10, 10], f, neighbor, temperatures)

# ╔═╡ 666ba850-160a-11eb-0873-73e16ead74c8
_, Xsa = simulated_annealing_steps([-10, 10], f, neighbor, temperatures)

# ╔═╡ 9ed293f2-160a-11eb-04a2-9b300aaefc20
sa_steps = length(temperatures)

# ╔═╡ 8070492e-160a-11eb-18ab-3d9dc03ae0be
anim_sa = @animate for i ∈ 1:sa_steps-2
	contour(xvals, yvals, f, xlabel="x1", ylabel="x2", color=:speed)
	scatter!([i for i in -n:n for j in -n:n], [i for j in -n:n for i in -n:n], label="", ms=1, color=:red, markerstrokecolor=:red)
	scatter!(Xsa[[i, i+1, i+2], 1], Xsa[[i, i+1, i+2], 2], label="", alpha=[0.5 0.7 1], color=[:lightgreen :green :darkgreen], ms=5)
end

# ╔═╡ 3a2cbe90-160b-11eb-18de-2f5a8fe8c2b4
gif(anim_sa, "../plots/siman.gif", fps = 5)

# ╔═╡ d065fdea-13a4-11eb-3eef-0db1b487d889
∇f(x) = P * x + q

# ╔═╡ ad10c89e-13a8-11eb-144b-53862a35b086
begin
	Xgd = zeros(nsteps+1, 2)
	Xgd[1,:] .= x₀
	for i in 1:nsteps
		x = Xgd[i,:]
		Xgd[i+1,:] .= x .- t * ∇f(x)
	end
end

# ╔═╡ 1fcf2c5e-13a9-11eb-1fe8-41b780dbded4
anim = @animate for i ∈ 1:nsteps
    plot!(pc, Xgd[1:i, 1], Xgd[1:i, 2], color=:orange, label="")
end

# ╔═╡ 3ff07a88-13a9-11eb-0e1e-ed3b7191fe01
gif(anim, "../plots/grad_desc.gif", fps = 2)

# ╔═╡ 46c5a09c-13a7-11eb-31f6-c374ebc6354f
gradient_descent([-10.0, 10], ∇f; t)

# ╔═╡ ddcace7a-13a4-11eb-342b-01999075bd9f
gradient_descent([0.0, 0.0], ∇f; t)

# ╔═╡ 6395f1bc-16aa-11eb-3738-9333efb8c0e6
begin
plot(x -> pdf(Gumbel(), x), -3, 10, title="PDF of a standard Gumbel", xlabel="x", lw=2, label="")
	savefig("../plots/stand_gumbel.pdf")
end

# ╔═╡ 307c1bfc-16ab-11eb-008a-2b9da9464826
p = [0.05, 0.6, 0.15, 0.2, 0.5, 0.1, 0.2]

# ╔═╡ 6fda5d2c-16ab-11eb-1901-81f40b9506dc
p ./= sum(p)

# ╔═╡ 647aaf9a-16ab-11eb-3461-8ba3110d9651
pprobs = bar(p, label="", color=:green)

# ╔═╡ 3643af66-16b3-11eb-1927-b18462e93e1c
savefig(pprobs, "../plots/prob_vector.pdf")

# ╔═╡ 6a5fb270-16ab-11eb-3ae2-0bb30f9ca688
psample = plot((bar(gumbel_softmax(log.(p), τ=τ), label="", color=c) for rep in 1:3  for (τ, c) in zip([0.1, 0.5, 1, 10],[:darkred, :red, :orange, :yellow]))..., layout=(3, 4), yticks=[])

# ╔═╡ 464870ea-16b3-11eb-018f-b7b2b89ef35c
savefig(psample, "../plots/samples.pdf")

# ╔═╡ cc0bdf34-1873-11eb-2ed7-4fba79a84ea4
md"## Sampling vs optimization"

# ╔═╡ d36fa080-1873-11eb-114c-69cc679abe6d
g = Gumbel()

# ╔═╡ 02015c70-1876-11eb-032b-45d85a1aa9da
fsample = ackley

# ╔═╡ 1713a016-1876-11eb-3cda-0d9ffad51397


# ╔═╡ 55860cf8-1874-11eb-3be6-e71fd2d5d00b
begin 
	contourf(-10:.1:10, -10:0.1:10, fsample, color=:speed, xlabel="x1", ylabel="x2")
	#scatter!([i for i in -10:10 for j in -10:10], [i for j in -10:10 for i in -10:10], label="", ms=2, color=:red, markerstrokecolor=:red)
	scatter!([0],[0], label="minimizer", color=:red)
	savefig("../plots/minimizer.pdf")
end

# ╔═╡ 6f43fdf8-1874-11eb-14b5-5dc755d1fdd8
F = [fsample(x1, x2) for x1 in -10:10, x2 in -10:10]

# ╔═╡ 3131d47a-1876-11eb-36ae-e301b96283ae
X = [(x1, x2) for x1 in -10:10, x2 in -10:10]

# ╔═╡ 4ea3dc74-1876-11eb-008e-ef517f63b046
τ = 1

# ╔═╡ 55b57400-1876-11eb-1c05-dfdb56bfd0f8
nsamples = 50

# ╔═╡ 5b30916c-1876-11eb-1e82-1d20d8f172d8
function generate_samples()
	x, y = zeros(Int, nsamples), zeros(Int, nsamples)
	for i in 1:nsamples
		C = X[argmax(-F .+ τ * rand(g, size(F)...))]
		x[i], y[i] = Tuple(C)
	end
	return x, y
end

# ╔═╡ b58f626e-1876-11eb-1e8f-39d2cc5528d8
begin 
	contourf(-10:0.1:10, -10:0.1:10, fsample, color=:speed, xlabel="x1", ylabel="x2")
	#scatter!([i for i in -10:10 for j in -10:10], [i for j in -10:10 for i in -10:10], label="", ms=2, color=:red, markerstrokecolor=:red)
	scatter!(generate_samples()..., label="samples", color=:blue, alpha=0.5)
	savefig("../plots/sampling.pdf")
end

# ╔═╡ Cell order:
# ╠═fdaeccc8-139f-11eb-0474-ffb743e08fcf
# ╠═03174924-13a0-11eb-1d1a-cb1e80fc18e6
# ╠═118fb1b2-13a0-11eb-0bcf-85ac564610b0
# ╠═36590a16-13a0-11eb-1f76-5d250cbe5b00
# ╠═17f2ae4c-13a0-11eb-2690-4f89d04b3013
# ╠═f6fdfee8-13a0-11eb-2cb0-712a338819e4
# ╠═4052280e-13a0-11eb-1093-c326a83ca963
# ╠═4a16dbf0-13a0-11eb-2a13-e73a64cac7e3
# ╠═6a89231a-13a1-11eb-3334-5f05ce41e5e4
# ╠═b87fd91a-13a1-11eb-12a1-c33a3084d384
# ╠═b1b4b9c8-13a8-11eb-2329-29de0ed43225
# ╠═e7fef5ca-13a8-11eb-3b7a-877b8cffad7f
# ╠═ad10c89e-13a8-11eb-144b-53862a35b086
# ╠═1fcf2c5e-13a9-11eb-1fe8-41b780dbded4
# ╠═4e4cdcce-13a0-11eb-325c-33d97947f187
# ╠═3ff07a88-13a9-11eb-0e1e-ed3b7191fe01
# ╠═b9607fe8-13a3-11eb-1dd2-d3b2b74d5fd4
# ╠═944855be-13a0-11eb-0576-57af0f1ff502
# ╠═46c5a09c-13a7-11eb-31f6-c374ebc6354f
# ╠═9385a9c4-13a0-11eb-2a59-cf5eaf5c7ee6
# ╠═b1748150-13a3-11eb-280c-29992381ce5d
# ╠═11d945e6-160c-11eb-2339-a94c3acb028b
# ╠═fea52b7c-13a4-11eb-1768-035fe22c0a2a
# ╠═1d3d4b48-160a-11eb-2ca3-efd22f33a988
# ╠═5bfedb9c-13a5-11eb-098d-c7bae32477c4
# ╠═4f4b00b4-13a6-11eb-0af2-2df839938e28
# ╠═eb4b3b10-13a5-11eb-2e30-3dcfc01f4e2b
# ╠═666ba850-160a-11eb-0873-73e16ead74c8
# ╠═9ed293f2-160a-11eb-04a2-9b300aaefc20
# ╠═8070492e-160a-11eb-18ab-3d9dc03ae0be
# ╠═3a2cbe90-160b-11eb-18de-2f5a8fe8c2b4
# ╠═d065fdea-13a4-11eb-3eef-0db1b487d889
# ╠═ddcace7a-13a4-11eb-342b-01999075bd9f
# ╠═2a10283a-13a4-11eb-2598-bb10bb2e9c60
# ╠═6395f1bc-16aa-11eb-3738-9333efb8c0e6
# ╠═307c1bfc-16ab-11eb-008a-2b9da9464826
# ╠═6fda5d2c-16ab-11eb-1901-81f40b9506dc
# ╠═647aaf9a-16ab-11eb-3461-8ba3110d9651
# ╠═3643af66-16b3-11eb-1927-b18462e93e1c
# ╠═6a5fb270-16ab-11eb-3ae2-0bb30f9ca688
# ╠═464870ea-16b3-11eb-018f-b7b2b89ef35c
# ╠═cc0bdf34-1873-11eb-2ed7-4fba79a84ea4
# ╠═d36fa080-1873-11eb-114c-69cc679abe6d
# ╠═f32ee0fc-1873-11eb-2b1b-05a124c7e005
# ╠═02015c70-1876-11eb-032b-45d85a1aa9da
# ╠═1713a016-1876-11eb-3cda-0d9ffad51397
# ╠═55860cf8-1874-11eb-3be6-e71fd2d5d00b
# ╠═6f43fdf8-1874-11eb-14b5-5dc755d1fdd8
# ╠═3131d47a-1876-11eb-36ae-e301b96283ae
# ╠═4ea3dc74-1876-11eb-008e-ef517f63b046
# ╠═55b57400-1876-11eb-1c05-dfdb56bfd0f8
# ╠═5b30916c-1876-11eb-1e82-1d20d8f172d8
# ╠═b58f626e-1876-11eb-1e8f-39d2cc5528d8

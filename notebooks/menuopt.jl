### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 7287d96e-107a-11eb-2d57-07ef9c8a24c7
using DiffCombOpt, Plots, LinearAlgebra, Zygote, Statistics

# ╔═╡ 90104f3a-1079-11eb-16c2-7126d3ec1309
meals_with_properties = [("rice-lentils", 10.0, true, false),
		("dhal", 6.0, true, false),
		("spaghetti", 8.5, false, false),
		("chinese", 7.5, false, false),
		("fries", 8.0, true, true),
		("ribhouse", 9.5, false, true),
		("hamburger", 8.0, false, true),
		("pitta", 6.0, false, true),
		("noodles", 7.0, true, false),
		("chicken", 8.0, false, false),
		("curry", 7.5, true, false),
		("pizza (veg)", 8.0, true, false),
		("pizza", 8.0, true, true)
	]

# ╔═╡ 71380e76-107a-11eb-02b0-331cbf110379
#meals = first.(meals_with_properties)

# ╔═╡ 8c094cb0-107a-11eb-25bf-23be28912fdb
meals, appreciation, vegetarian, takeout = zip(meals_with_properties...) .|> tup -> [tup...]

# ╔═╡ 57013be2-107d-11eb-2c9e-47c46eff56e5
meat = .!vegetarian

# ╔═╡ b64c0f66-107b-11eb-341b-c7a4746f33b5
n_meals = length(meals)

# ╔═╡ b5a12ed0-107a-11eb-35b5-0585dd5d34e7
bar(appreciation, xticks=(1:n_meals, meals))

# ╔═╡ a8e854fa-107c-11eb-2af4-bded55801db2
n_days = 5

# ╔═╡ b7721dbc-107c-11eb-037e-e5897f39ce75
size_search_space = n_meals^n_days

# ╔═╡ fa5b5976-107b-11eb-2c73-ab0e70160663
pen_reuse = 100.0

# ╔═╡ 042a46f6-107c-11eb-2155-8f76ff3bbcb4
pen_takeout= 100.0

# ╔═╡ 197de258-107c-11eb-2875-953c484d7a3f
pen_meat = 100.0

# ╔═╡ 9853d1f0-1086-11eb-0b95-f981c7496592
pen_friday = 100.0

# ╔═╡ b374e5ec-107b-11eb-2957-ad3aad888818
adj_days(y) = y[2:end] ⋅ y[1:end-1]

# ╔═╡ 3bb8d18a-107b-11eb-194c-a927d1064eb5
obj(Y) = - 1sum(Y * appreciation) +
		pen_reuse * sum(sum(Y, dims=1).^2) +
		pen_meat * max(sum(Y * meat) - 2.0, 0.0) +
		pen_takeout * adj_days(Y * takeout) -
		pen_friday * Y[end,:] ⋅ takeout

# ╔═╡ 61631424-1083-11eb-0a43-3b96aa032b69
import Base.Iterators: product

# ╔═╡ 8936240e-107d-11eb-019a-3f9684a9e867
function find_best_Y()
	Y = zeros(Bool, n_days, n_meals)
	best_obj = Inf
	best_Y = copy(Y)
	for C in product(1:n_meals, 1:n_meals, 1:n_meals, 1:n_meals, 1:n_meals)
		Y .= false
		for (w, m) in enumerate(C)
			Y[w, m] = true
		end
		objective = obj(Y)
		if objective < best_obj
			best_obj = objective
			best_Y .= Y
		end
	end
	return best_Y, best_obj
end

# ╔═╡ 6be6c43e-1085-11eb-3b13-a9638744b2fc
Y, score = find_best_Y()

# ╔═╡ c64c866c-113d-11eb-0d05-1f7750b9a659
sum(Y, dims=1)

# ╔═╡ 46e705aa-113f-11eb-1af5-0f332c106053
τ = 5e-2

# ╔═╡ dcf5235c-1088-11eb-210c-b9c107ebd539
function grad_descent!(X, f; nsteps=500, t=1e-3, β=0.8)
	ΔX = zero(X)
	for i in 1:nsteps
		ΔX .= β .* ΔX .+ (1.0 - β) .* f'(X)
		X .-= t .* ΔX
	end
	return X
end

# ╔═╡ fb9ae93c-108a-11eb-2630-f331efec8ee5
randg(n::Int...) = - log.(-log.(rand(n...)))

# ╔═╡ b6f0c784-1089-11eb-21d7-bd352613c878
function gumbel_softmax(lp::Matrix; τ::Number=1.0)
	Z = lp .+ randg(size(lp)...)
	Z = Z .- mean(Z, dims=2)
    Y = exp.(Z ./ τ)
    return Y ./ sum(Y, dims=2)
end

# ╔═╡ e1d93150-1086-11eb-0162-2d7d62049773
function diff_obj(X; τ=τ, λ=1e-4, β=1e-1)
	Y = gumbel_softmax(X; τ)
	P = exp.(X)
	P = P ./ sum(P, dims=2)
	return obj(Y) + λ * sum(X.^2) + β * sum(P .* log.(P))
end

# ╔═╡ 3a50e38a-1087-11eb-3980-4f6c1a1cd452
diff_obj(0.1randn(n_days, n_meals))

# ╔═╡ c3f365d6-1087-11eb-0750-6fb0565c8fa9
diff_obj'(0.1randn(n_days, n_meals))

# ╔═╡ b465e910-108a-11eb-17d9-2318f0980796
X = grad_descent!(0.01 * zeros(n_days, n_meals), diff_obj)

# ╔═╡ e8b957c0-1089-11eb-0f53-0fdd87bab501
heatmap(X, xticks=(1:n_meals,meals))

# ╔═╡ 58e514d8-1140-11eb-1d54-abe68ad0e055
gumbel_softmax(randn(n_days, n_meals), τ=τ)

# ╔═╡ 48c0f538-1148-11eb-0bc0-25ef14d0fb3a
gumbel_softmax([5.0 4.5], τ=0.01)

# ╔═╡ db8a70de-1089-11eb-1c9e-dde90e49f6d5
sum(gumbel_softmax(randn(n_days, n_meals)), dims=2)

# ╔═╡ 358a3662-109b-11eb-16ce-83e489fa6f05
P = exp.(X) ./ sum(exp.(X), dims=2)

# ╔═╡ 1b4f2f9c-1147-11eb-370d-c1251d7556fa
gumbel_softmax(X;τ)

# ╔═╡ a19e979c-113f-11eb-1c32-8f49866d3383
sum(P, dims=2)

# ╔═╡ 76e1507a-113e-11eb-32ef-d1710f398e09
heatmap(P, xticks=(1:n_meals, meals))

# ╔═╡ 68f67324-1142-11eb-22fe-4d33beeff49d
Ŷ = gumbel_softmax(X, τ=τ)

# ╔═╡ 8530d0d6-1140-11eb-1d54-1dd7f74d5eea
obj(Ŷ.>0.6)

# ╔═╡ 9f62af3a-113e-11eb-3c32-83c93ff060f1
heatmap(Y, xticks=(1:n_meals,meals))

# ╔═╡ d44791c4-1144-11eb-2a69-9bf9425b2357
begin
	obj_values = [(gumbel_softmax(X, τ=1e-2).>0.5) |> obj for i in 1:1000]
histogram(obj_values, title="scores for random sampling")
	vline!([mean(obj_values)], label="mean")
end

# ╔═╡ 8cb61e4c-1140-11eb-29f0-f3efd96c3c42
obj(Y)

# ╔═╡ e73a7c8a-113d-11eb-16e3-4991495207a2
randg(10, 10)

# ╔═╡ db9ad198-1149-11eb-1a04-fd5ab64d47e0
mean((first(gumbel_softmax([log(0.2) log(1.8)], τ=0.02)) for i in 1:100000))

# ╔═╡ 57ffe040-114a-11eb-0e9d-ef0f167b7cb9
gumbel_softmax([100 100; 0 2], τ=0.02)

# ╔═╡ Cell order:
# ╠═7287d96e-107a-11eb-2d57-07ef9c8a24c7
# ╠═90104f3a-1079-11eb-16c2-7126d3ec1309
# ╠═71380e76-107a-11eb-02b0-331cbf110379
# ╠═8c094cb0-107a-11eb-25bf-23be28912fdb
# ╠═57013be2-107d-11eb-2c9e-47c46eff56e5
# ╠═b5a12ed0-107a-11eb-35b5-0585dd5d34e7
# ╠═b64c0f66-107b-11eb-341b-c7a4746f33b5
# ╠═a8e854fa-107c-11eb-2af4-bded55801db2
# ╠═b7721dbc-107c-11eb-037e-e5897f39ce75
# ╠═fa5b5976-107b-11eb-2c73-ab0e70160663
# ╠═042a46f6-107c-11eb-2155-8f76ff3bbcb4
# ╠═197de258-107c-11eb-2875-953c484d7a3f
# ╠═9853d1f0-1086-11eb-0b95-f981c7496592
# ╠═b374e5ec-107b-11eb-2957-ad3aad888818
# ╠═3bb8d18a-107b-11eb-194c-a927d1064eb5
# ╠═8936240e-107d-11eb-019a-3f9684a9e867
# ╠═6be6c43e-1085-11eb-3b13-a9638744b2fc
# ╠═c64c866c-113d-11eb-0d05-1f7750b9a659
# ╠═61631424-1083-11eb-0a43-3b96aa032b69
# ╠═46e705aa-113f-11eb-1af5-0f332c106053
# ╠═e1d93150-1086-11eb-0162-2d7d62049773
# ╠═3a50e38a-1087-11eb-3980-4f6c1a1cd452
# ╠═c3f365d6-1087-11eb-0750-6fb0565c8fa9
# ╠═dcf5235c-1088-11eb-210c-b9c107ebd539
# ╠═b6f0c784-1089-11eb-21d7-bd352613c878
# ╠═58e514d8-1140-11eb-1d54-abe68ad0e055
# ╠═48c0f538-1148-11eb-0bc0-25ef14d0fb3a
# ╠═db8a70de-1089-11eb-1c9e-dde90e49f6d5
# ╠═b465e910-108a-11eb-17d9-2318f0980796
# ╠═e8b957c0-1089-11eb-0f53-0fdd87bab501
# ╠═fb9ae93c-108a-11eb-2630-f331efec8ee5
# ╠═358a3662-109b-11eb-16ce-83e489fa6f05
# ╠═1b4f2f9c-1147-11eb-370d-c1251d7556fa
# ╠═a19e979c-113f-11eb-1c32-8f49866d3383
# ╠═76e1507a-113e-11eb-32ef-d1710f398e09
# ╠═68f67324-1142-11eb-22fe-4d33beeff49d
# ╠═8530d0d6-1140-11eb-1d54-1dd7f74d5eea
# ╠═9f62af3a-113e-11eb-3c32-83c93ff060f1
# ╠═d44791c4-1144-11eb-2a69-9bf9425b2357
# ╠═8cb61e4c-1140-11eb-29f0-f3efd96c3c42
# ╠═e73a7c8a-113d-11eb-16e3-4991495207a2
# ╠═db9ad198-1149-11eb-1a04-fd5ab64d47e0
# ╠═57ffe040-114a-11eb-0e9d-ef0f167b7cb9

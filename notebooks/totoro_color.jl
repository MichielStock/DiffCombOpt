### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ e0013b96-11e4-11eb-3f7a-cbe2f57f2dfb
using Plots, Colors, LinearAlgebra, Images

# ╔═╡ 43f829be-11ee-11eb-1fe0-21ce9202a574
using Zygote, DiffCombOpt

# ╔═╡ 4ddb56b0-11fe-11eb-2a34-6de65e7c30cb
using Distributions

# ╔═╡ 395f1dc6-120e-11eb-0107-f3390711a0a1
md"Load or generate a color palette."

# ╔═╡ 449559be-11ea-11eb-3b3e-b3f7aa754f42
randcolor() = RGB(rand(3)...)

# ╔═╡ 8f99bd60-1606-11eb-21cb-8f9fb2cc991b
spectrum = load("../data/raiders_lost_ark.png") .|> RGB

# ╔═╡ 781b2148-1209-11eb-0d02-91d61de5cd71
width_spectrum = size(spectrum, 2)

# ╔═╡ 56edab86-11ea-11eb-2b39-37ca8cb3bade
colors = [randcolor() for i in 1:20]
#colors = [spectrum[30,i] for i in 20:div(width_spectrum,21):width_spectrum-30]

# ╔═╡ 337becba-1606-11eb-2880-59e6430f8bf3
save("../data/totoro_col/colors.png", reshape(colors, 1, :))

# ╔═╡ 9c8cbe44-1211-11eb-2e3a-a55b9d440fc0
save("../data/totoro_col/colorscheme.png", hcat([[c for i in 1:50, j in 1:50] for c in colors]...))

# ╔═╡ 6b74628e-11ea-11eb-1ec3-8b07b4d6b83a
n_colors = length(colors)

# ╔═╡ 523eddf2-120e-11eb-14cd-953ef51504f5
md"Use colors and Totoro semantics to build constraints."

# ╔═╡ 8e323080-11ea-11eb-39ab-af273ebaf060
totoro_semantics = Dict(
	2 => "eyes",
	5 => "umbrella_light",
	6 => "body_light",
	7 => "nose",
	8 => "body_dark",
	10 => "markings",
	11 => "handle",
	12 => "umbrella_dark",
	13 => "leaf_dark",
	14 => "leaf_light",
	20 => "belly_dark",
	22 => "belly_light",
	)

# ╔═╡ f3c47e80-11ed-11eb-317d-41ceeb9131c2
totoro_segm_parts = Dict(p=>i for (i, p) in totoro_semantics)

# ╔═╡ 8657dfba-11eb-11eb-2d6e-d508d777d292
totoro_parts = ["eyes",
				"nose",
				"body_dark",
				"body_light",
				"belly_dark",
				"belly_light",
				"markings",
				"handle",
				"umbrella_dark",
				"umbrella_light",
				"leaf_dark",
				"leaf_light"]

# ╔═╡ a4183e48-11eb-11eb-1a9a-3b5d6a59ecae
n_parts = length(totoro_parts)

# ╔═╡ 469b6974-11ec-11eb-38b6-953cc9ac3c01
col_diffs = [colordiff(c1, c2) for c1 in colors, c2 in colors]; col_diffs

# ╔═╡ 87376ad0-120e-11eb-092f-0d9f3a78f81b
c_white = RGB(1,1,1)

# ╔═╡ 9bf81a98-11ec-11eb-2181-83b6e2abae43
col_brightness = -[HSV(c).s for c in colors]
#col_brightness = red.(colors) .+ green.(colors) .+ blue.(colors)

# ╔═╡ 7c252182-120c-11eb-1538-9de646aa7c3f
c_green = RGB(0, 1, 0)

# ╔═╡ e6aad380-11ee-11eb-0a2a-cf7eb01ae627
#col_greenish = green.(colors) .- red.(colors) .- blue.(colors)
col_greenish = -colordiff.(colors, c_green)

# ╔═╡ 588cfe1a-11ef-11eb-035a-9d865b2411c7
begin
	Cb = zeros(Int, 4, n_parts)
	Cb[1,3], Cb[1,4] = 1, -1  # body
	Cb[2,5], Cb[2,6] = 1, -1  # belly
	Cb[3,9], Cb[3,10] = 1, -1  # umbrella
	Cb[4,11], Cb[4,12] = 1, -1  # leaf
	Cb
end

# ╔═╡ c7697db0-11ed-11eb-0309-09fbf04e44c6
collect(enumerate(totoro_parts))

# ╔═╡ db8d4664-11ed-11eb-0704-e917b455437d
objective(Y) = 500sum((sum(Y, dims=1)).^2) + # all colors different
			-10(sum(Y[[11,12],:] * col_greenish)) +  # leaf is green
			#10sum(Y[[3,4,8],:] * col_brightness) + # dark body 
			-10sum(Y[[1,5,6],:] * col_brightness) + # light eyes and belly
			10sum(Cb * Y * col_brightness) +  # contrast shadow
			10tr((Y * col_diffs * Y')[[3,5,9,11,5],[4,6,10,12,7]]) + # colors parts similar
			- tr((Y * col_diffs * Y')[[9,3,4,5,5,6],[11,9,9,11,7,7]])  # contrasting colors

# ╔═╡ 4bc568d0-11f0-11eb-3abc-85cb0ea51106
function objective_diff(X; τ=0.1, λ=1e-5, γ=0)
	Y = gumbel_softmax(X; τ)
	P = exp.(X) ./ sum(exp.(X), dims=2)
	return objective(Y) + λ * sum(X.^2) + γ * sum(P .* log.(P))
end

# ╔═╡ 434012a2-11f1-11eb-2b42-d1913c7a7258
function grad_descent!(X, f; nsteps=500, t=1e-3, β=0.9, track=false)
	ΔX = zero(X)
	track && (bt = [])
	track && push!(bt, (f(X), copy(X)))
	for i in 1:nsteps
		ΔX .= β .* ΔX .+ (1.0 - β) .* f'(X)
		X .-= t .* ΔX
		track && push!(bt, (f(X), copy(X)))
	end
	track && return X, bt
	return X
end

# ╔═╡ 458b08c8-11f1-11eb-2cb5-ad4e385d65fe
X, bt = grad_descent!(zeros(n_parts, n_colors), objective_diff, track=true)

# ╔═╡ 196c5ea4-11f1-11eb-0940-6ff0c404efca
objective_diff(X)

# ╔═╡ 0047566a-1208-11eb-08dc-bb1eb7f1f653
objective_values = [obj for (obj, Xi) in bt]

# ╔═╡ 14b73110-1208-11eb-0470-f33c8a9fa44d
plot_obj = plot(objective_values, xlabel="iteration+1", label="objective")

# ╔═╡ 9d4cd6be-1213-11eb-0e72-332421b44f76
savefig(plot_obj, "../data/totoro_col/plot_obj.pdf")

# ╔═╡ 75bd4858-11f1-11eb-0bbd-2baf7465400d
P = exp.(X) ./ sum(exp.(X), dims=2)

# ╔═╡ 6c92ac46-11f1-11eb-2fc6-cbebc15b66a8
begin
	p = heatmap(P, yticks=(1:n_parts, totoro_parts), xlabel="colors")
	savefig("../data/totoro_col/probs.png")
	p
end

# ╔═╡ 1d985b2e-11fe-11eb-3aac-9d2393967ac7
function sample_solution(X)
	X += rand(Gumbel(), size(X)...)
	solution = Dict{String,RGB}()
	for (i, part) in enumerate(totoro_parts)
		bestv = -Inf
		bestc = RGB(0,0,0)
		for (j, col) in enumerate(colors)
			bestv, bestc = max((X[i,j], col), (bestv, bestc))
			solution[part] = bestc
		end
	end
	return solution
end

# ╔═╡ ee311974-11fe-11eb-1469-739a58a0e216
solution = sample_solution(X)

# ╔═╡ b1ed813a-1214-11eb-0407-05af8c8ad708
md"## Entropy-rich"

# ╔═╡ b977e544-1214-11eb-1ca4-53f73e9e3bd4
obj_entropy = X-> objective_diff(X; τ=0.1, λ=1e-5, γ=250)

# ╔═╡ cece26b8-1214-11eb-104c-d5bb6eb4c9a5
Xentropy = grad_descent!(zeros(n_parts, n_colors), obj_entropy, track=false)

# ╔═╡ dc798de2-1214-11eb-2d86-d3d33fd01679
Pentr = exp.(Xentropy) ./ sum(exp.(Xentropy), dims=2)

# ╔═╡ e5bc7cd2-1214-11eb-24c9-1f33fb13184d
begin
	pe = heatmap(Pentr, yticks=(1:n_parts, totoro_parts), xlabel="colors")
	savefig("../data/totoro_col/probs_entr.png")
	pe
end

# ╔═╡ cdba3502-1606-11eb-365b-b7df07cc5627
md"""## Gradient-descent

Find colors using weigths of colors.
"""

# ╔═╡ d5abdd72-1606-11eb-02ad-1546df5981e4
function determine_weigths!(W, f; nsteps=500, t=1e-3, β=0.5)
	ΔW = zero(W)
	
	for i in 1:nsteps
		ΔW .= β .* ΔW .+ (1.0 - β) .* f'(W)
		W .-= t .* ΔW
		W .= max.(W, 0.0)
		W ./= sum(W, dims=1)
	end
	return W
end

# ╔═╡ cb42ebd8-1607-11eb-33c8-ffae92a086cb
W = determine_weigths!(ones(n_parts, n_colors), objective, β=0.1)

# ╔═╡ 46453d72-1608-11eb-3489-e756eb47a01f
sum(W, dims=1)

# ╔═╡ c5739ade-1217-11eb-172e-2bc1d8c4df9b
md"## Structured sampling"

# ╔═╡ 9158d5ce-11f0-11eb-13d2-13a78ddc3bdb
segmentations = Dict(part=>load("../data/segmentation/segm_$(totoro_segm_parts[part]).png") for part in totoro_parts)

# ╔═╡ 82ad3916-1204-11eb-1b25-41783e692aec
contour = load("../data/segmentation/contour.png") .|> Gray

# ╔═╡ b1c5e042-1202-11eb-04e3-3f8f0d17f8ee
nose = segmentations["nose"]

# ╔═╡ c7fced74-1202-11eb-36e3-e3103bb38429
totoro = load("../data/totoro_nowm.png") .|> RGB

# ╔═╡ 080e24f0-1203-11eb-1b08-55fe34115453
function color_tororo(color_scheme)
	im = copy(totoro)
	for (part, col) in color_scheme
		seg = segmentations[part]
		im[seg.<0.1] .= col
	end
	im[contour.>0.7] .= RGB(0,0,0)
	return im
end

# ╔═╡ 82375814-1203-11eb-3f6c-e51cf32737c8
totoro_col = color_tororo(solution)

# ╔═╡ 6a2e769e-1208-11eb-290d-b5351245a1b1
save("../data/totoro_col/plain.png", totoro_col)

# ╔═╡ 14eac9b4-1215-11eb-202f-bf0eecacbc68
begin
anim = @animate for i in 1:50
	im = color_tororo(sample_solution(Xentropy))
	plot(im, xticks=[], yticks=[])
end
gif(anim, "../data/totoro_col/entropic_totoro.gif", fps=2.5)
end

# ╔═╡ Cell order:
# ╠═e0013b96-11e4-11eb-3f7a-cbe2f57f2dfb
# ╟─395f1dc6-120e-11eb-0107-f3390711a0a1
# ╠═449559be-11ea-11eb-3b3e-b3f7aa754f42
# ╠═8f99bd60-1606-11eb-21cb-8f9fb2cc991b
# ╠═781b2148-1209-11eb-0d02-91d61de5cd71
# ╠═56edab86-11ea-11eb-2b39-37ca8cb3bade
# ╠═337becba-1606-11eb-2880-59e6430f8bf3
# ╠═9c8cbe44-1211-11eb-2e3a-a55b9d440fc0
# ╠═6b74628e-11ea-11eb-1ec3-8b07b4d6b83a
# ╟─523eddf2-120e-11eb-14cd-953ef51504f5
# ╠═8e323080-11ea-11eb-39ab-af273ebaf060
# ╠═f3c47e80-11ed-11eb-317d-41ceeb9131c2
# ╠═8657dfba-11eb-11eb-2d6e-d508d777d292
# ╠═a4183e48-11eb-11eb-1a9a-3b5d6a59ecae
# ╠═469b6974-11ec-11eb-38b6-953cc9ac3c01
# ╠═87376ad0-120e-11eb-092f-0d9f3a78f81b
# ╠═9bf81a98-11ec-11eb-2181-83b6e2abae43
# ╠═7c252182-120c-11eb-1538-9de646aa7c3f
# ╠═e6aad380-11ee-11eb-0a2a-cf7eb01ae627
# ╠═588cfe1a-11ef-11eb-035a-9d865b2411c7
# ╠═c7697db0-11ed-11eb-0309-09fbf04e44c6
# ╠═db8d4664-11ed-11eb-0704-e917b455437d
# ╠═82375814-1203-11eb-3f6c-e51cf32737c8
# ╠═6a2e769e-1208-11eb-290d-b5351245a1b1
# ╠═43f829be-11ee-11eb-1fe0-21ce9202a574
# ╠═4bc568d0-11f0-11eb-3abc-85cb0ea51106
# ╠═196c5ea4-11f1-11eb-0940-6ff0c404efca
# ╠═434012a2-11f1-11eb-2b42-d1913c7a7258
# ╠═458b08c8-11f1-11eb-2cb5-ad4e385d65fe
# ╠═0047566a-1208-11eb-08dc-bb1eb7f1f653
# ╠═14b73110-1208-11eb-0470-f33c8a9fa44d
# ╠═9d4cd6be-1213-11eb-0e72-332421b44f76
# ╠═75bd4858-11f1-11eb-0bbd-2baf7465400d
# ╠═6c92ac46-11f1-11eb-2fc6-cbebc15b66a8
# ╠═4ddb56b0-11fe-11eb-2a34-6de65e7c30cb
# ╠═1d985b2e-11fe-11eb-3aac-9d2393967ac7
# ╠═ee311974-11fe-11eb-1469-739a58a0e216
# ╠═b1ed813a-1214-11eb-0407-05af8c8ad708
# ╠═b977e544-1214-11eb-1ca4-53f73e9e3bd4
# ╠═cece26b8-1214-11eb-104c-d5bb6eb4c9a5
# ╠═dc798de2-1214-11eb-2d86-d3d33fd01679
# ╠═e5bc7cd2-1214-11eb-24c9-1f33fb13184d
# ╠═14eac9b4-1215-11eb-202f-bf0eecacbc68
# ╠═cdba3502-1606-11eb-365b-b7df07cc5627
# ╠═d5abdd72-1606-11eb-02ad-1546df5981e4
# ╠═cb42ebd8-1607-11eb-33c8-ffae92a086cb
# ╠═46453d72-1608-11eb-3489-e756eb47a01f
# ╠═c5739ade-1217-11eb-172e-2bc1d8c4df9b
# ╠═9158d5ce-11f0-11eb-13d2-13a78ddc3bdb
# ╠═82ad3916-1204-11eb-1b25-41783e692aec
# ╠═b1c5e042-1202-11eb-04e3-3f8f0d17f8ee
# ╠═c7fced74-1202-11eb-36e3-e3103bb38429
# ╠═080e24f0-1203-11eb-1b08-55fe34115453

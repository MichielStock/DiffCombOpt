### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 7985ddf0-0d6a-11eb-1588-696559d4bcbf
using Images, ImageSegmentation, Plots

# ╔═╡ c8c27630-0e2a-11eb-3af3-3f0c9689beb2
using BSON

# ╔═╡ cb8d7972-0e33-11eb-0d3e-799eca9df60e
md"""
# Segmentation totoro

Segmenting the totoro image in meaningful regions.
"""

# ╔═╡ 89d6c692-0d6a-11eb-11f0-95e108281049
totoro = load("../data/totoro_nowm.png") .|> RGB

# ╔═╡ de4aa420-0d72-11eb-368b-257d2fb368c3
size(totoro)

# ╔═╡ b1789982-0d6a-11eb-27fa-bb5e0cd64ae4
segmentation_totoro = unseeded_region_growing(totoro, 0.165)

# ╔═╡ fcc245aa-0d6a-11eb-0398-c704efaf1969
segmentated_color_totoro = (map(i->segment_mean(segmentation_totoro,i), labels_map(segmentation_totoro)))

# ╔═╡ bd7edac4-0d70-11eb-0916-d116590a9863
segment_mean(segmentation_totoro)

# ╔═╡ 472e2546-0e2b-11eb-3c30-e59c6569d1cd
n_regions = segment_labels(segmentation_totoro) |> length

# ╔═╡ 300745c8-0e2b-11eb-0a3b-af6b3eb954cb
segmented_totoro = (map(i->Gray(i/n_regions), labels_map(segmentation_totoro)))

# ╔═╡ b104a18e-0d6b-11eb-01ae-43eacc51f000
grad_y, grad_x, mag, orient = imedge(totoro)

# ╔═╡ e27ede3c-0d6b-11eb-3edb-5f643ae6ea82
contours_totoro = Gray.(mag .> 0.175) |> closing

# ╔═╡ 44ace2f0-0e2c-11eb-1cce-8d73a77786bb
begin
	rm.([joinpath("../data/segmentation", file) for file in readdir("../data/segmentation/")])
	for i in segment_labels(segmentation_totoro)
		save("../data/segmentation/segm_$i.png", Gray.(labels_map(segmentation_totoro).!=i))
	end
	save("../data/segmentation/segmentation_color.png", segmentated_color_totoro)
	save("../data/segmentation/segmentation.png", segmented_totoro)
	save("../data/segmentation/contour.png", RGB.(contours_totoro))
end

# ╔═╡ f58ca1e8-0e2a-11eb-308d-1771b4873b6d
segment_mean(segmentation_totoro)

# ╔═╡ 92f71770-0e32-11eb-22d5-fb01c3cdd9a5
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

# ╔═╡ dbd13514-0e31-11eb-30a4-45d7b974f1db
BSON.@save "../data/totoro_segmentation.bson" totoro segmentation_totoro contours_totoro totoro_semantics

# ╔═╡ Cell order:
# ╟─cb8d7972-0e33-11eb-0d3e-799eca9df60e
# ╠═7985ddf0-0d6a-11eb-1588-696559d4bcbf
# ╠═89d6c692-0d6a-11eb-11f0-95e108281049
# ╠═de4aa420-0d72-11eb-368b-257d2fb368c3
# ╠═b1789982-0d6a-11eb-27fa-bb5e0cd64ae4
# ╠═fcc245aa-0d6a-11eb-0398-c704efaf1969
# ╠═bd7edac4-0d70-11eb-0916-d116590a9863
# ╠═472e2546-0e2b-11eb-3c30-e59c6569d1cd
# ╠═300745c8-0e2b-11eb-0a3b-af6b3eb954cb
# ╠═44ace2f0-0e2c-11eb-1cce-8d73a77786bb
# ╠═b104a18e-0d6b-11eb-01ae-43eacc51f000
# ╠═e27ede3c-0d6b-11eb-3edb-5f643ae6ea82
# ╠═f58ca1e8-0e2a-11eb-308d-1771b4873b6d
# ╠═92f71770-0e32-11eb-22d5-fb01c3cdd9a5
# ╠═c8c27630-0e2a-11eb-3af3-3f0c9689beb2
# ╠═dbd13514-0e31-11eb-30a4-45d7b974f1db

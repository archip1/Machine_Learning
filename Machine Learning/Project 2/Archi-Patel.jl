### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a7b272bc-e8f5-11eb-38c6-8f61fea9941c
using CSV, DataFrames, PlotlyJS, PlutoUI, Random, Statistics

# ╔═╡ 131466aa-d30d-4f75-a99f-13f47e1c7956
using LinearAlgebra: dot, norm, norm1, norm2

# ╔═╡ 7f8a82ea-1690-490c-a8bc-3c1f9556af2e
using Distributions: Uniform

# ╔═╡ 6fef79fe-aa3c-497b-92d3-6ddd87b6c26d
begin
	_check_complete(complete) = complete ? "✅" : "❌"
	
	md"""
	# Setup
	
	This section loads and installs all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
	"""
end

# ╔═╡ ac9b31de-a4f0-4cb6-82a7-890436ff1bed
import Colors: @colorant_str

# ╔═╡ d55b56d3-d50a-4e4e-a4ff-2f1be75dc44c
PlutoUI.TableOfContents()

# ╔═╡ 5924a9a5-88c7-4751-af77-4a217dfdc15f
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ 9d38b22c-5893-4587-9c33-82d846fd0728
student = (name="Archi Patel", email="archi1@ualberta.ca", ccid=" ", idnumber=1234567)

# ╔═╡ 72d6b53f-f20d-4c05-abd4-d6fb4c997795
let
	def_student = (name="NAME as in eclass", email="UofA Email", ccid="CCID", idnumber=0)
	if length(keys(def_student) ∩ keys(student)) != length(keys(def_student))
		md"You don't have all the right entries! Make sure you have `name`, `email`, `ccid`, `idnumber`. ❌"
	elseif any(getfield(def_student, k) == getfield(student, k) for k in keys(def_student))
		md"You haven't filled in all your details! ❌"
	elseif !all(typeof(getfield(def_student, k)) === typeof(getfield(student, k)) for k in keys(def_student))
		md"Your types seem to be off: `name::String`, `email::String`, `ccid::String`, `idnumber::Int`"
	else
		md"Welcome $(student.name)! ✅"
	end
end

# ╔═╡ e10ffb7d-ede9-4cec-8908-836cd1ee3ae1
md"""
Important Note: You should only write code in the cells that has: """


# ╔═╡ a095e895-cabb-4e3a-a027-ddd6aa356e41
#### BEGIN SOLUTION


#### END SOLUTION

# ╔═╡ c8ceaf7a-d740-4123-8b26-e652dd17a86e
md"""
# Distance Metrics

Here we define some functions for commonly used distance metrics, to later allows us to measure errors.

"""

# ╔═╡ b3edc600-6670-4ae6-af7a-337e0600390e
begin
	RMSE(x̂, x) = sqrt(mean((x̂ .- x).^2)) # abs2 is equivalent to squaring, but faster and better numerically. 
	l2_error(x̂, x) = norm2(x̂ .- x)
	l1_error(x̂, x) = norm1(x̂ .- x)
end

# ╔═╡ ac180a6e-009f-4db2-bdfc-a9011dc5eb7b
md"""
# Abstract type Regressor

This is the basic Regressor interface. For the methods below we will be specializing the `predict(reg::Regressor, x::Number)`, and `epoch!(reg::Regressor, args...)` functions. Notice the `!` character at the end of epoch. This is a common naming convention in Julia to indicate that this is a function that modifies its arguments.


"""

# ╔═╡ 999ce1f2-fd11-4584-9c2e-bb79585d11f7
"""
	Regressor

Abstract Type for regression algorithms. Interface includes `predict` and an `epoch!`. In this notebook, we will only be using single variate regression.
- `predict(reg::Regressor, X::Number)`: return a prediction of the target given the feature `x`.
- `epoch!(reg::Regressor, X::AbstractVector, Y::AbstractVector)`: trains using the features `X` and regression targets `Y`.
"""
abstract type Regressor end # assume linear regression

# ╔═╡ 80f4168d-c5a4-41ef-a542-4e6d7772aa80
predict(reg::Regressor, x::Number) = Nothing

# ╔═╡ 65505c23-5096-4621-b3d4-d8d5ed339ad5
predict(reg::Regressor, X::AbstractVector) = [predict(reg, x) for x in X]

# ╔═╡ 67a8db8a-365f-4912-a058-d61648ac096e
epoch!(reg::Regressor, X::AbstractVector, Y::AbstractVector) = nothing

# ╔═╡ 7e083787-7a16-4064-97d9-8e2fca6ed686
md"""
# Baselines

In this section we will define the:
- `MeanRegressor`: Predict the mean of the training set.
- `RandomRegressor`: Predict `w*x` where `w` is sampled from a random normal distribution.

All the following baselines assume one dimension
"""

# ╔═╡ 5d5fe2d8-98ea-4ee1-b2e2-354eefaf8f77
md"""
## MeanRegressor
"""

# ╔═╡ eb152f9f-908f-4e5b-9642-f7314eb66e09
begin
	"""
		MeanRegressor()
		
	Predicts the mean value of the regression targets passed in through `epoch!`.
	"""
	mutable struct MeanRegressor <: Regressor
		μ::Float64
	end
	MeanRegressor() = MeanRegressor(0.0)
	predict(reg::MeanRegressor, x::Number) = reg.μ
	epoch!(reg::MeanRegressor, X::AbstractVector, Y::AbstractVector) = reg.μ = mean(Y)
end

# ╔═╡ a7b7bdaf-f4fd-40fb-a704-a584a93920f2
md"""
## RandomRegressor
"""

# ╔═╡ 0a90ceec-fc50-41fb-bf8b-e96ed677a5c3
begin
	"""
		RandomRegressor
	
	Predicts `w*x` where `w` is sambled from a normal distribution.
	"""
	struct RandomRegressor <: Regressor # random weights
		w::Float64
	end
	RandomRegressor() = RandomRegressor(randn())
	predict(reg::RandomRegressor, x::Number) = reg.w*x
end

# ╔═╡ 76c5de40-a0de-4d90-82f0-4f6c941499d9
begin
	
	md"""# Gradient Descent Regressors"""
end

# ╔═╡ e8f3981d-545c-4e35-9066-69fa4f78dbce
md"""
In this section you will be implementing two gradient descent regressors, assuming $$p(y | x)$$ is Gaussian with the update given in the assignment pdf.  First we will create a Gaussian regressor, and then use this to build our two new GD regressors. You can test your algorithms in the [experiment section](#experiment).

All the Gaussian Regressors will have data:
- `w::Float64` which is the parameter we are learning.
"""

# ╔═╡ 6ef4d625-fdf6-4f11-81f1-b242b2195e8b
abstract type GaussianRegressor <: Regressor end

# ╔═╡ 579da3a3-0e4e-4ba2-8f44-2657229994e3
predict(reg::GaussianRegressor, x::Float64) = reg.w * x

# ╔═╡ 67274c5b-2ee3-4865-a5e1-799db543d0c7
predict(reg::GaussianRegressor, X::Vector{Float64}) = reg.w .* X

# ╔═╡ 57c4dbf9-b7ba-44e9-8120-0e1a85896976
# function probability(reg::GaussianRegressor, x, y)
# end

# ╔═╡ add25a22-7423-4077-b0b0-f71eed2d2e20
begin
	md"""
	The stochastic regressor will be implemented via the stochastic gradient rule
	
	```math
	\begin{align*}
	w_{t+1} = w_t - \eta (x_i w_t - y_i) x_i
	\end{align*}
	```
	
	by starting from an initial $$w_0 = 0$$ and doing multiple epochs over the dataset. Each epoch corresponds to iterating once over the entire dataset, in a random order. Here you only have to implement the function that does one epoch. It will be called multiple times in `train!`, which is a function provided for you described in the section on Training the Models. After you implement your algorithm, you should test it in the experiments section to answer Q3 b. Note: to obtain the length of an array `arr`, you can use `length(arr)`. If you see an error that says "MethodError: no method matching...", then this means you have an error in your implementation.
	"""
end

# ╔═╡ b0160dc0-35af-4750-9ac6-d9018ce89ea9
begin
	mutable struct StochasticRegressor <: GaussianRegressor
		w::Float64
		η::Float64
	end
	StochasticRegressor(η::Float64) = StochasticRegressor(0.0, η)
end

# ╔═╡ adedf31d-945f-491a-912a-bef9b03f6665
# Hint: Checkout the function randperm
function epoch!(reg::StochasticRegressor, 
		         X::AbstractVector{Float64}, 
		         Y::AbstractVector{Float64})
	# Your code here is for question 3(a)
	# For question 3(b), you will move to the experiment section. Remember to change the stepsize (η::Float64) 
	#### BEGIN SOLUTION
	shuffle(X)
	shuffle(Y)
	
	for i in 1:length(X)
		reg.w = reg.w - reg.η * (X[i] * reg.w - Y[i] ) * X[i]
	end
	
	return reg.w
	
	#### END SOLUTION
end

# ╔═╡ 9b69b182-ee8d-485e-ad6b-1f868874652c
md"""
The Batch regressor will be implemented via the gradient rule $$x_i \in X$$ and targets $$y_i \in Y$$.

```math
\begin{align*}
g_t &= \frac{1}{|\mathcal{I}|}\sum_{i\in \mathcal{I}} (x_i w_t - y_i) x_i \\
w_{t+1} &= w_t - \eta g_t.
\end{align*}
```

You need to implement the `epoch!` function, which will be called elsewhere by `train!`. 
"""

# ╔═╡ 4d422513-34e3-4d04-8ff1-c5165d953342
mutable struct BatchRegressor <: GaussianRegressor
	w::Float64
	η::Float64
	BatchRegressor(η) = new(0.0, η)
end

# ╔═╡ e5d78429-55f4-4a1a-831f-dbcb53c6a0f6
function epoch!(reg::BatchRegressor,
		        X::AbstractVector{Float64},
		        Y::AbstractVector{Float64})
	# Your code here is for question 3(c)
	# You will need to move to the experiment section to test your implementation
	#### BEGIN SOLUTION
	
	sum = 0.0 
	
	for i in 1:length(X)
		sum += ((X[i] * reg.w - Y[i] ) * X[i])
	end

	gt = (1/length(X)) * sum
	reg.w = reg.w - reg.η * gt
	
	println(reg.w)

	#### END SOLUTION
end

# ╔═╡ 48f10e3f-f39e-4872-ba5e-e6cca6247ce2
md"""
The Minibatch regressor will be implemented via the gradient rule for a minibatch `j` with indices for a batch defined by the set  $$\mathcal{I}$$

```math
\begin{align*}
g_t^j &= \frac{1}{|I_{j}|}\sum_{i\in \mathcal{I}_j} (x_i w_t - y_i) x_i \\
w_{t+1} &= w_t - \eta g^j_t.
\end{align*}
```

Once again you need to implement the `epoch!` function, which will be called elsewhere by `train!`. 
"""

# ╔═╡ 240e6eab-dad7-4f2d-8fcd-de54a4ec9008
mutable struct MiniBatchRegressor <: GaussianRegressor
	w::Float64
	η::Float64
	b::Int
	MiniBatchRegressor(η, b) = new(0.0, η, b)
end

# ╔═╡ 49093cd5-d01c-487e-8026-87585f21d4c2
function epoch!(reg::MiniBatchRegressor,
		        X::AbstractVector{Float64},
		        Y::AbstractVector{Float64})
	# Your code here is for question 3(c)
	# You will need to move to the experiment section to test your implementation
	#### BEGIN SOLUTION

	sum = 0.0 
	
	for i in 1:length(X)
		sum += ((X[i] * reg.w - Y[i] ) * X[i])
		gt = (1/length(X)) * sum
		reg.w = reg.w - reg.η * gt
	end
	
	println(reg.w)
	
	#### END SOLUTION
end

# ╔═╡ 2aa8bd30-4b65-4e89-9dd5-5333efbbda3f
md"""
# Stepsize Heuristic
"""

# ╔═╡ 51bc41b4-b27f-4e60-8dba-70783c60c1c2
begin
	mutable struct StochasticHeuristicRegressor <: GaussianRegressor
		w::Float64
	end
	StochasticHeuristicRegressor() = 	
		StochasticHeuristicRegressor(0.0)
end

# ╔═╡ 00fcafa5-308c-4895-ad52-961772348125
# Hint: Checkout the function randperm
function epoch!(reg::StochasticHeuristicRegressor, 
		         X::AbstractVector{Float64}, 
		         Y::AbstractVector{Float64})
	# Your code here is for 3(d)
	#### BEGIN SOLUTION

	sum = 0.0 
	n = 0.0
	
	for i in 1:length(X)
		sum += ((X[i] * reg.w - Y[i] ) * X[i])

		gt = (1/length(X)) * sum
		n = (1 + abs(gt))^(-1)
		
	end
	
	println(n)
	
	#### END SOLUTION
end

# ╔═╡ 7d6027af-6bfe-4cca-8b81-3e8fb12a6f79
begin
	mutable struct BatchHeuristicRegressor <: GaussianRegressor
		w::Float64
		BatchHeuristicRegressor() = new(0.0)
	end
	
end

# ╔═╡ 297b3bd1-29cc-4a96-bf2c-6305f8d375e4
function epoch!(reg::BatchHeuristicRegressor,
		        X::AbstractVector{Float64},
		        Y::AbstractVector{Float64})
	# Your code here is for question 3(e)
	#### BEGIN SOLUTION

	sum = 0.0 
	
	for i in 1:length(X)
		sum += ((X[i] * reg.w - Y[i] ) * X[i])
	end

	gt = (1/length(X)) * sum
	reg.η = (1 + abs(gt))^(-1)
	
	println(reg.η)

	#### END SOLUTION
end

# ╔═╡ 646109b0-ac77-4177-bb6d-c300876f76a4
mutable struct MiniBatchHeuristicRegressor <: GaussianRegressor
	w::Float64
	b::Int
	MiniBatchHeuristicRegressor(b) = new(0.0, b)
end

# ╔═╡ 22517997-db36-4a7b-891e-e689135d2fa3
function epoch!(reg::MiniBatchHeuristicRegressor,
		        X::AbstractVector{Float64},
		        Y::AbstractVector{Float64})
	# Your code here is for question 3(e)
	#### BEGIN SOLUTION
	

	#### END SOLUTION
end

# ╔═╡ a24deb81-9ba8-4bec-9ca7-e19e985d0d8d
begin # Test Case for Stochastic Regressor
	__check_stoch_reg, ret_md_stoch_reg = let
		sgr = StochasticRegressor(0.0, 0.1)
		epoch!(sgr, [1.0, 1.0], [1.0, 1.0])
		check_passed = predict(sgr, 1.0) ≈ 0.19

		ret_md = if check_passed
			md"""
			✅ You passed the test case!
			"""
		else
			md"""
			❌ Below is the check for the stochastic regressor:
			`predict(sgr, 1.0)=`$(predict(sgr, 1.0)) and should be about `0.19`.
			"""
		end
		check_passed, ret_md
	end
	ret_md_stoch_reg
end

# ╔═╡ 06be63f6-577c-4117-8dac-bba8bba8b9ce
begin
	md"## Q3 a: Stochastic $(_check_complete(__check_stoch_reg))"
end

# ╔═╡ 1705900b-cbb7-4e0b-a50e-73fb1d75ccf2
begin # Test Case for Batch Regressor
	__check_batch_reg, ret_md_batch_reg = let
		X, Y = [1.0, 2.0, 3.0], [1.0, 0.2, 0.1]
		bgr = BatchRegressor(0.1*3)
		epoch!(bgr, X, Y)
		check_passed = predict(bgr, 1.0) ≈ 0.17

		ret_md = if check_passed
			md"""
			✅ You passed the test case!
			"""
		else
			md"""
			❌ Below is the check for the stochastic regressor:
			`predict(bgr, 1.0)=`$(predict(bgr, 1.0)) and should be about `0.17`.
			"""
		end
		check_passed, ret_md
	end
	ret_md_batch_reg
end

# ╔═╡ debd1501-2fae-431c-85cf-bea549a4bba4
begin
	md"## Q3 c: Batch $(_check_complete(__check_batch_reg))"
end

# ╔═╡ ff91a0d2-5f17-4a3a-9f30-c6579bbb37ac
begin
	__check_mb, ret_md_mb = let
		X, Y = [1.0, 1.0, 1.0, 1.0,  1.0, 1.0], [0.32, 0.32, 0.32, 0.32, 0.32, 0.32]
		mbgr = MiniBatchRegressor(0.1*2, 2)
		epoch!(mbgr, X, Y)
		check_passed = predict(mbgr, 1.0) ≈ 0.15616

		ret_md = if check_passed
			md"""
			✅ You passed the test case!
			"""
		else
			md"""
			❌ Below is the check for the stochastic regressor:
			`predict(bgr, 1.0)=`$(predict(bgr, 1.0)) and should be about `0.15616`.
			"""
		end
		check_passed, ret_md
	end

	ret_md_mb
end

# ╔═╡ f3f20901-a36f-49a4-9f6d-4d71370a7a86
begin
	md"## Q3 c: MiniBatch $(_check_complete(__check_mb))"
end

# ╔═╡ 7b1d7d43-8613-46dc-9e92-00b91bfd5943
begin
	__check_shr = let
		sgr = StochasticHeuristicRegressor(0.0)
		epoch!(sgr, [1.0, 1.0], [1.0, 1.0])
		predict(sgr, 1.0) * 6 ≈ 5.0
	end
	md"## Q3 d: Stochastic $(_check_complete(__check_shr))"
end

# ╔═╡ 375f1a57-50f4-49ba-b089-a38f5315bd81
begin
	__check_bhr_full = let
		X, Y = [1.0, 2.0, 3.0], [1.0, 0.2, 0.1]
		bgr = BatchHeuristicRegressor()
		epoch!(bgr, X, Y)
		batch_test = predict(bgr, 1.0) ≈ 0.36170212765 
	end
		
	
	md"## Q3 e: Batch $(_check_complete(__check_bhr_full))"	
end

# ╔═╡ ca9fdc01-8e1b-4fbf-b1b6-5298787ab18b
begin
	__check_bhr_mini = let
		X, Y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.32, 0.32, 0.32, 0.32, 0.32, 0.32]
		mbgr = MiniBatchHeuristicRegressor(2)
		epoch!(mbgr, X, Y)
		mb_test = predict(mbgr, 1.0) ≈ 0.319968983713
	end
	md"## Q3 e: MiniBatch $(_check_complete(__check_bhr_mini))"
end

# ╔═╡ 753cc169-1579-4539-aaa8-7c514e2b3381
md"""
# Plotting Utilities

Below we define two plotting helper functions for using PlotlyJS. You can ignore these if you want. We use them below to compare the algorithms.
"""

# ╔═╡ f80a96ad-9bdf-4ddd-8591-d1a3006e4002
color_scheme = [
    colorant"#44AA99",
    colorant"#332288",
    colorant"#DDCC77",
    colorant"#999933",
    colorant"#CC6677",
    colorant"#AA4499",
    colorant"#117733",
    colorant"#882255",
    colorant"#1E90FF",
]

# ╔═╡ c61c4aaa-0ee7-44a8-8a4c-1119e0c22bde
function boxplot(
		names::Vector{String}, 
		data::Vector{<:AbstractVector};
		col=color_scheme,
		kwargs...)
	traces = GenericTrace{Dict{Symbol, Any}}[]
	for (idx, (name, datum)) in enumerate(zip(names, data))
		tr_bx = box(
			name=name, 
			y=datum, 
			jitter=0.3, 
			marker_color=col[idx])
		push!(traces, tr_bx)
	end
	layout = Layout(; showlegend=false, kwargs...)
	plt = Plot(traces, layout)
end

# ╔═╡ 0825e8cd-26f0-48a3-a146-d6e542f5dd2e
function learning_curve(names, data; err, col = color_scheme, kwargs...)
	trcs = GenericTrace{Dict{Symbol, Any}}[]
	for (idx, (n, d)) in enumerate(zip(names, data))
		terr = scatter(;
			x = vcat(1:length(d), length(d):-1:1),
			y = vcat(d.-err[idx], d.+err[idx]),
			fill="tozerox", 
			fillcolor="rgba($(col[idx].r), $(col[idx].g), $(col[idx].b), 0.2)",
			line_color="transparent")
		t = scatter(;x = 1:length(d), 
				y=d, 
				line_color=col[idx],
				name=n)
		push!(trcs, t)
	end
	layout = Layout(;kwargs...)
	Plot(trcs, layout)
end

# ╔═╡ 5aba1286-42de-4241-ba74-565385043f0b
md"""
# Data

Next we will be looking at the `height_weight.csv` dataset found in the data directory. This dataset provides three features `[sex, height, weight]`. In the following regression task we will be using `height` to predict `weight`, ignoring the `sex` feature.

The next few cells:
- Loads the dataset
- Plots distributions for the `height` and `weight` features seperated by `sex`
- Standardize the set so both `height` and `weight` conform to a standard normal.
- Defines `splitdataframe` which will be used to split the dataframe into training and testing sets.
"""

# ╔═╡ fdca1cde-48ba-4542-b762-b470ee80b1d3
# Read the data from the file in "data/height_weight.csv". DO NOT CHANGE THIS VALUE!
df_height_weight = DataFrame(
	CSV.File(joinpath(@__DIR__, "data/height_weight.csv"), 
	header=["sex", "height", "weight"]));

# ╔═╡ 210f224f-a6aa-4ba7-b560-84ddc18823bf
try 
	identity(df_height_weight)
	md"Successfully loaded dataset ✅"
catch
	Markdown.parse("""Please place datset at `$(joinpath(@__DIR__, "data/height_weight.csv"))`""")
end
	

# ╔═╡ ce1bc6dc-70db-4ad7-b70c-f5c7f6c1129e
df_hw_norm = let
	df = copy(df_height_weight)
	σ_height = sqrt(var(df[!, :height]))
	μ_height = mean(df[!, :height])

	
	df[:, :height] .= (df[!, :height] .- μ_height) ./ σ_height
	
	σ_weight = sqrt(var(df[!, :weight]))
	μ_weight = mean(df[!, :weight])
	df[:, :weight] .= (df[!, :weight] .- μ_weight) ./ σ_weight
	
	df
end

# ╔═╡ 6f4005da-57ff-4161-b4a5-437a1c072dd9
md"""
#### Plot data
Plot a boxplot and violin plot of the height and weight. This can be with the classes `male` and `female` combined or with them separate. 
"""

# ╔═╡ 79dc3541-a75a-4f50-ab0c-2f6522a32eba
plt_hw = let
	df = df_height_weight # For convenience in the bellow code
	nothing
	# plt1 = plot(xlabel="Sex", ylabel="Height", legend=nothing)
	sex_names = ["Male", "Female"]
	get_attr = (sex, attr) -> df[df.sex .== sex, :][!, attr]
	p1 = boxplot(sex_names, 
			get_attr.(sex_names, :height), 
		    yaxis=attr(title="Height"))

	p2 = boxplot(sex_names, 
				 get_attr.(sex_names, :weight), 
				 yaxis=attr(title="Weight"))
	
	p = [p1 p2]
	PlotlyJS.relayout(p, height=400, showlegend=false)

end

# ╔═╡ 4c764fe7-e407-4ed9-9f5a-e2740b9863f6
"""
	splitdataframe(split_to_X_Y::Function, df::DataFrame, test_perc; shuffle = false)
	splitdataframe(df::DataFrame, test_perc; shuffle = false)

Splits a dataframe into test and train sets. Optionally takes a function as the first parameter to split the dataframe into X and Y components for training. This defaults to the `identity` function.
"""
function splitdataframe(split_to_X_Y::Function, df::DataFrame, test_perc, rng=Random.GLOBAL_RNG; 
		                shuffle = false)
	#= shuffle dataframe. 
	This is innefficient as it makes an entire new dataframe, 
	but fine for the small dataset we have in this notebook. 
	Consider shuffling inplace before calling this function.
	=#
	
	df_shuffle = if shuffle == true
		df[randperm(rng, nrow(df)), :]
	else
		df
	end
	
	# Get train size with percentage of test data.
	train_size = Int(round(size(df,1) * (1 - test_perc)))
	
	dftrain = df_shuffle[1:train_size, :]
	dftest = df_shuffle[(train_size+1):end, :]
	
	split_to_X_Y(dftrain), split_to_X_Y(dftest)
end

# ╔═╡ 790b9f50-7a3b-480c-92f4-33edd967b73d
splitdataframe(df::DataFrame, test_perc, rng=Random.GLOBAL_RNG; kwargs...) = 
	splitdataframe(identity, df, test_perc, rng; kwargs...)

# ╔═╡ 97dc155f-ef8a-4e55-960c-07fd0f5a6dde
let
	#=
		A do block creates an anonymous function and passes this to the first parameter of the function the do block is decorating.
	=#
	trainset, testset =
		splitdataframe(df_hw_norm, 0.1; shuffle=true) do df
			(X=df[!, :height], Y=df[!, :weight]) # create namedtuple from dataframes
		end
end

# ╔═╡ 168ef51d-dbcd-44c9-8d2e-33e98471962d
md"""
# Training the Models

The following functions are defined as utilities to train and evaluate our models. While hidden below, you can expand these blocks to uncover what is happening. `run_experiment!` is the main function used below in "**Using and Analyzing you Algorithms**".
"""

# ╔═╡ ae23765a-33fd-4a6b-a18c-3081c4382b14
function evaluate(err_func::Function, reg::Regressor, X, Y)
	err_func(predict(reg, X), Y)
end

# ╔═╡ 30177ca7-f9a5-4b97-8967-ffc154a509b0
begin
	evaluate_l1(reg::Regressor, X, Y) = evaluate(l1_error, reg, X, Y)
	evaluate_l2(reg::Regressor, X, Y) = evaluate(l2_error, reg, X, Y)
	evaluate_l∞(reg::Regressor, X, Y) = evaluate(reg, X, Y) do ŷ, y
		maximum(abs.(ŷ .- y))
	end

end

# ╔═╡ 532ed065-1f61-4eb0-9b4f-a6c236ab334d
function train!(ev_func::Function, reg::Regressor, X, Y, num_epochs)
	
	start_error = ev_func(reg, X, Y)
	ret = zeros(typeof(start_error), num_epochs+1)
	ret[1] = start_error
	
	for epoch in 1:num_epochs
		epoch!(reg, X, Y)
		ret[epoch+1] = ev_func(reg, X, Y)
	end
	
	ret
end

# ╔═╡ fbb14b46-51a2-4041-860b-8259f85ef2b7
train!(reg::Regressor, X, Y, num_epochs) = train!(evaluate_l2, reg::Regressor, X, Y, num_epochs)

# ╔═╡ 74b7458e-c654-4b85-9a31-ea575e0aa548
function run_experiment!(reg::Regressor, 
		               	 trainset, 
		                 testset, 
		                 num_epochs)
	
	train_err = train!(reg, trainset.X, trainset.Y, num_epochs)
	test_err = evaluate(reg, testset.X, testset.Y) do p, t
		(p .- t).^2
	end

	(regressor=reg, train_error=train_err, test_error=test_err)
end

# ╔═╡ 071b7886-ac26-4999-b20d-d63848331ebe
function run_experiment(regressors::Dict{String, Function}, 
						num_epochs; 
						seed=10392)

	ret = Dict{String, Any}()
	for (k, reg_init) in regressors
		ret[k] = begin
			# Random.seed!(seed+r)
			rng = Random.Xoshiro(42)
			trainset, testset = splitdataframe(df_hw_norm, 0.1, rng; shuffle=true) do df
				(X=df[!, :height], Y=df[!, :weight]) # create named tuple from DF
			end
			run_experiment!(reg_init(), trainset, testset, num_epochs)
		end
	end

	ret
end

# ╔═╡ 6f2a0d1e-e76b-4326-a350-af49a8fd30e6
html"""<h1 id="experiment"> Using and Analyzing your Algorithms </h1>"""

# ╔═╡ 76d0599d-b893-4cb2-b94d-0e787fd39a74
begin
	
	__s_η = 0.01
	__b_η = 0.01
	__mb_η = 0.01
	__mb_n = 100
	
	Markdown.parse("""

	In this section we will be running and analyzing a small experiment. The goal is to get familiar with analyzing data, plotting learning curves, and comparing different methods. Below we've provided a start with the baselines. Add new initilizors for a Batch update `(η = $(__b_η))`, a Minibatch update `(η = $(__mb_η), n = $(__mb_n))`, and a Stochastic update `(η = $(__s_η))`. Also add their heuristic counterparts. 
		
	As a point of reference: running 
	```julia
	results = run_experiment(regressor_init, 30)
	```
	in the cell below takes roughly `8 seconds` on my machine.

	""")
end

# ╔═╡ 7b80a6fa-4b92-4ade-9c14-785f1a9334d1
md"""
**To run these experiments and draw plots use** $(@bind __run_q3_experiments PlutoUI.CheckBox())
"""

# ╔═╡ 468b1077-6cf4-45d4-a4a6-41b134a6d3d7
regressor_init = Dict(
	# use the keys "Batch", "Stochastic", and "Minibatch", "StochasticHeuristic" "BatchHeuristic", and "MinibatchHeuristic" 
	# This is the actual experiment section
	"Mean"=>()->MeanRegressor(),
	"Random"=>()->RandomRegressor(),
	"Batch"=>()->BatchRegressor(0.01),
	"Minibatch"=>()->MiniBatchRegressor(0.01, 100),
	"Stochastic"=>()->StochasticRegressor(0.01),
	"BatchHeuristic"=>()->BatchHeuristicRegressor(),
	"MinibatchHeuristic"=>()->MiniBatchHeuristicRegressor(100),
	"StochasticHeuristic"=>()->StochasticHeuristicRegressor()
)

# ╔═╡ 3a0a1a93-0ea1-4570-9be6-aa892b515c7b
md"""
The results dictionary is the resulting data from the experiment we run using `regressor_init` as the intializors. You will see the same keys used as in the `regressor_init` dictionary. For each run the experiment returns the final regressor, the training error vector, and the final test error. You can get one of these components for a particular methd using `getindex` and broadcasting:
```julia
getindex.(results["Mean"], :test_error)
```

"""

# ╔═╡ e3224f78-4cfb-45e7-a1f7-6640051afcd2
md"""
## Learning Curves

Plot the average learning curve with the standard error calculated as

```math
\sigma_{err}(\mathbf{x}) = \sqrt{\frac{\text{Var}(x)}{|x|}}
```

Note that $$\mathbf{x}$$ is a vector over runs, not over epochs.

_Note:_ if you notice one method is dominating the plot, change the axis limits to make sure the methods we are most concerned with (i.e. Stochastic, Batch, and Minibatch) are visible.
"""

# ╔═╡ 4f22d201-69bb-4ee1-9393-5058eaffd3d1
md"""
## Final Errors

Finally, we want to compare the final test errors of the different methods. One way to do this is through box plots. See [this great resource](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51) to learn how to compare data using a box and whisker plot. In this plot you can ignore the `Random` baseline.
"""

# ╔═╡ b5180603-9617-444f-ad4e-07ae38530f8f
@bind rerun_random Button("Re-Run Random")

# ╔═╡ ed20ad2c-87ab-4833-ae78-0c1906aa90a6
begin
	rerun_random
	results = run_experiment(regressor_init, 30)
end

# ╔═╡ 42ab7e2a-d390-4528-80cd-0e30a1eb2133
begin
	if __run_q3_experiments
		__results_checks = Dict{String, Union{Vector{Bool}, Bool}}(
			"Stochastic"=>false,
			"Minibatch"=>[false, false],
			"Batch"=>false
		)
		
		
		local a = """
		Experiment ran for:
			
		"""
		algs = [
			"Mean",
			"Random", 
			"Stochastic", 
			"Batch", 
			"Minibatch", 
			"StochasticHeuristic", 
			"BatchHeuristic", 
			"MinibatchHeuristic"]
		for k in algs
			if k in keys(results)
				if k == "Batch"
					a *= "- ✅ `$(k)`: "
					__results_checks[k] = results[k].regressor.η == __b_η
					a *= """with stepsize= `$(results[k].regressor.η)` $(results[k].regressor.η == __b_η ? "✅" : "❌" )"""
				elseif k == "Stochastic"
					a *= "- ✅ `$(k)`: "
					__results_checks[k] = results[k].regressor.η == __s_η
					a *= """with stepsize= `$(results[k].regressor.η)` $(results[k].regressor.η == __s_η ? "✅" : "❌" )"""
				elseif k == "Minibatch"
					a *= "- ✅ `$(k)`: "
					__results_checks[k] = [results[k].regressor.η == __mb_η, 	
						   				   results[k].regressor.b == __mb_n]
					a *= """with stepsize= `$(results[k].regressor.η)` $(results[k].regressor.η == __mb_η ? "✅" : "❌" ) and batch size = `$(results[k].regressor.b)` $(results[k].regressor.b == __mb_n ? "✅" : "❌" )"""
				elseif k == "MinibatchHeuristic"
					a *= "- ✅ `$(k)`: "	
					__results_checks[k] = [results[k].regressor.b == __mb_n]
					a *= """with batch size = `$(results[k].regressor.b)` $(results[k].regressor.b == __mb_n ? "✅" : "❌" )"""
				else
					a *= "- ✅ `$(k)`"
				end
			else
				a *= "- ❌ `$(k)`"
			end
			
			a *= "\n\n"
		end
		Markdown.parse(a)
	else
		md"**Run Experiments by clicking above check box.**"
	end
end

# ╔═╡ ac703837-ed65-49c6-97ad-5b54e30a680e

let
	if __run_q3_experiments
		# Play with data here! You can explore how to get different values.
		results["Mean"].test_error
	end
end

# ╔═╡ 6f26e6f8-088c-4578-bf8b-07d38ff53d00
let
	if __run_q3_experiments
		plt = plot()
		ms = collect(keys(results))
		get_train_mean = 
			(d)->d.train_error
		μs = [get_train_mean(results[m]) for m in ms]
		# σs = [get_train_std(results[m]) for m in ms]
		learning_curve(
			ms, 
			μs, 
			err= zero.(μs),
			height=400,
			xaxis_range=[0, 31])
	end
end

# ╔═╡ 7bc3d05f-9af6-4163-8a31-4143c9606b5b
let
	if __run_q3_experiments
		ms = filter((k)->k ∉ ["Random"], keys(results))
		p1 = boxplot(
			collect(ms), 
			[getindex(results[m], :test_error) for m in ms], 
			height=400,
			yaxis=attr(title="Test Error"))
		ms = ["Random"]
		p2 = boxplot(
			collect(ms), 
			[getindex(results[m], :test_error) for m in ms], 
			col=color_scheme[end-1:end],
			height=400)
		
		p = [p1 p2]
		PlotlyJS.relayout(p, height=400, showlegend=false)
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.10.4"
Colors = "~0.12.8"
DataFrames = "~1.3.4"
Distributions = "~0.25.67"
PlotlyJS = "~0.18.8"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "5cd870b1af36cf17401ca93a6888ae9fcaa757ed"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Blink]]
deps = ["Base64", "Distributed", "HTTP", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Pkg", "Reexport", "Sockets", "WebIO"]
git-tree-sha1 = "bc93511973d1f949d45b0ea17878e6cb0ad484a1"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.9"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"

[[deps.ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "799b25ca3a8a24936ae7b5c52ad194685fc3e6ef"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "3101c32aab536e7a27b1763c0797dba151b899ad"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.113"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Test"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "PDMats", "SparseArrays", "Statistics"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "6c4d6a1babbbee6f283b3da64ac895f0a3bfbc96"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.11"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Dates", "Test"]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "3b2db451a872b20519ebb0cec759d3d81a1c6bcb"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.20"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "MbedTLS", "Pkg", "Sockets"]
git-tree-sha1 = "7295d849103ac4fcbe3b2e439f229c5cc77b9b69"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "PlotlyKaleido", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "e415b25fdec06e57590a7d5ac8e0cf662fa317e2"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.15"

[[deps.PlotlyKaleido]]
deps = ["Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "3210de4d88af7ca5de9e26305758a59aabc48aac"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.2.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "d0553ce4031a081cc42387a9b9c8441b7d99f32d"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.7"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "0eef0765186f7452e52236fa42ca8c9b3c11c6e3"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.21"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "4162e95e05e79922e44b9952ccbc262832e4ad07"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.6.0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─6fef79fe-aa3c-497b-92d3-6ddd87b6c26d
# ╠═a7b272bc-e8f5-11eb-38c6-8f61fea9941c
# ╠═ac9b31de-a4f0-4cb6-82a7-890436ff1bed
# ╠═131466aa-d30d-4f75-a99f-13f47e1c7956
# ╠═7f8a82ea-1690-490c-a8bc-3c1f9556af2e
# ╠═d55b56d3-d50a-4e4e-a4ff-2f1be75dc44c
# ╟─5924a9a5-88c7-4751-af77-4a217dfdc15f
# ╟─72d6b53f-f20d-4c05-abd4-d6fb4c997795
# ╠═9d38b22c-5893-4587-9c33-82d846fd0728
# ╟─e10ffb7d-ede9-4cec-8908-836cd1ee3ae1
# ╠═a095e895-cabb-4e3a-a027-ddd6aa356e41
# ╟─c8ceaf7a-d740-4123-8b26-e652dd17a86e
# ╠═b3edc600-6670-4ae6-af7a-337e0600390e
# ╟─ac180a6e-009f-4db2-bdfc-a9011dc5eb7b
# ╟─999ce1f2-fd11-4584-9c2e-bb79585d11f7
# ╠═80f4168d-c5a4-41ef-a542-4e6d7772aa80
# ╠═65505c23-5096-4621-b3d4-d8d5ed339ad5
# ╠═67a8db8a-365f-4912-a058-d61648ac096e
# ╟─7e083787-7a16-4064-97d9-8e2fca6ed686
# ╟─5d5fe2d8-98ea-4ee1-b2e2-354eefaf8f77
# ╠═eb152f9f-908f-4e5b-9642-f7314eb66e09
# ╟─a7b7bdaf-f4fd-40fb-a704-a584a93920f2
# ╠═0a90ceec-fc50-41fb-bf8b-e96ed677a5c3
# ╠═76c5de40-a0de-4d90-82f0-4f6c941499d9
# ╟─e8f3981d-545c-4e35-9066-69fa4f78dbce
# ╠═6ef4d625-fdf6-4f11-81f1-b242b2195e8b
# ╠═579da3a3-0e4e-4ba2-8f44-2657229994e3
# ╠═67274c5b-2ee3-4865-a5e1-799db543d0c7
# ╠═57c4dbf9-b7ba-44e9-8120-0e1a85896976
# ╟─06be63f6-577c-4117-8dac-bba8bba8b9ce
# ╟─add25a22-7423-4077-b0b0-f71eed2d2e20
# ╠═b0160dc0-35af-4750-9ac6-d9018ce89ea9
# ╠═adedf31d-945f-491a-912a-bef9b03f6665
# ╟─a24deb81-9ba8-4bec-9ca7-e19e985d0d8d
# ╠═debd1501-2fae-431c-85cf-bea549a4bba4
# ╟─9b69b182-ee8d-485e-ad6b-1f868874652c
# ╠═4d422513-34e3-4d04-8ff1-c5165d953342
# ╠═e5d78429-55f4-4a1a-831f-dbcb53c6a0f6
# ╟─1705900b-cbb7-4e0b-a50e-73fb1d75ccf2
# ╠═f3f20901-a36f-49a4-9f6d-4d71370a7a86
# ╟─48f10e3f-f39e-4872-ba5e-e6cca6247ce2
# ╠═240e6eab-dad7-4f2d-8fcd-de54a4ec9008
# ╠═49093cd5-d01c-487e-8026-87585f21d4c2
# ╟─ff91a0d2-5f17-4a3a-9f30-c6579bbb37ac
# ╟─2aa8bd30-4b65-4e89-9dd5-5333efbbda3f
# ╠═7b1d7d43-8613-46dc-9e92-00b91bfd5943
# ╠═51bc41b4-b27f-4e60-8dba-70783c60c1c2
# ╠═00fcafa5-308c-4895-ad52-961772348125
# ╠═375f1a57-50f4-49ba-b089-a38f5315bd81
# ╠═7d6027af-6bfe-4cca-8b81-3e8fb12a6f79
# ╠═297b3bd1-29cc-4a96-bf2c-6305f8d375e4
# ╟─ca9fdc01-8e1b-4fbf-b1b6-5298787ab18b
# ╠═646109b0-ac77-4177-bb6d-c300876f76a4
# ╠═22517997-db36-4a7b-891e-e689135d2fa3
# ╟─753cc169-1579-4539-aaa8-7c514e2b3381
# ╟─f80a96ad-9bdf-4ddd-8591-d1a3006e4002
# ╠═c61c4aaa-0ee7-44a8-8a4c-1119e0c22bde
# ╠═0825e8cd-26f0-48a3-a146-d6e542f5dd2e
# ╟─5aba1286-42de-4241-ba74-565385043f0b
# ╠═fdca1cde-48ba-4542-b762-b470ee80b1d3
# ╟─210f224f-a6aa-4ba7-b560-84ddc18823bf
# ╟─ce1bc6dc-70db-4ad7-b70c-f5c7f6c1129e
# ╟─6f4005da-57ff-4161-b4a5-437a1c072dd9
# ╠═79dc3541-a75a-4f50-ab0c-2f6522a32eba
# ╠═4c764fe7-e407-4ed9-9f5a-e2740b9863f6
# ╠═790b9f50-7a3b-480c-92f4-33edd967b73d
# ╠═97dc155f-ef8a-4e55-960c-07fd0f5a6dde
# ╟─168ef51d-dbcd-44c9-8d2e-33e98471962d
# ╠═ae23765a-33fd-4a6b-a18c-3081c4382b14
# ╠═30177ca7-f9a5-4b97-8967-ffc154a509b0
# ╠═532ed065-1f61-4eb0-9b4f-a6c236ab334d
# ╠═fbb14b46-51a2-4041-860b-8259f85ef2b7
# ╠═74b7458e-c654-4b85-9a31-ea575e0aa548
# ╠═071b7886-ac26-4999-b20d-d63848331ebe
# ╟─6f2a0d1e-e76b-4326-a350-af49a8fd30e6
# ╟─76d0599d-b893-4cb2-b94d-0e787fd39a74
# ╟─7b80a6fa-4b92-4ade-9c14-785f1a9334d1
# ╟─42ab7e2a-d390-4528-80cd-0e30a1eb2133
# ╠═468b1077-6cf4-45d4-a4a6-41b134a6d3d7
# ╠═ed20ad2c-87ab-4833-ae78-0c1906aa90a6
# ╟─3a0a1a93-0ea1-4570-9be6-aa892b515c7b
# ╠═ac703837-ed65-49c6-97ad-5b54e30a680e
# ╟─e3224f78-4cfb-45e7-a1f7-6640051afcd2
# ╠═6f26e6f8-088c-4578-bf8b-07d38ff53d00
# ╟─4f22d201-69bb-4ee1-9393-5058eaffd3d1
# ╟─b5180603-9617-444f-ad4e-07ae38530f8f
# ╟─7bc3d05f-9af6-4163-8a31-4143c9606b5b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

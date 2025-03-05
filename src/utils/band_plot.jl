using LinearAlgebra, SparseArrays
using KrylovKit
using Plots

const Identity_Matrix = [[1 0; 0 1]]
const Pauli_Matrices = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]

# export plot_band


"""
Parse a kPath as a List of `nk_turning` and a List of kPoints Coordinates
---
Here `kPath` is input as a list of turning points coordinates (can either be in `k_crys` or `k_cart`)
"""
function parse_k_path(; kpath::Vector{<:Vector}, nk::Int=50)::Tuple{Vector{Int64},Vector{Vector{Float64}}}
    n_kpath = length(kpath) - 1
    @assert n_kpath >= 1 # check if there are at least two kpaths

    # determine `nk` for each kpath based on their lengths
    kpath_length_list = [norm(kpath[i+1] - kpath[i]) for i in 1:n_kpath]
    nk_list::Vector{Int64} = [ceil(nk * kpath_length / kpath_length_list[1]) |> Int64 for kpath_length in kpath_length_list]
    nk_turning_list = [1; cumsum(nk_list) .+ 1] # add initial line to `nk_turning_list`; shift by 1 for each kpath

    kpath_kpoint_list = let kpath_directional_vec_list = [(kpath[i+1] - kpath[i]) / nk_list[i] for i in 1:n_kpath]
        # note: avoid nearby duplicates!
        [[kpath[i] + n * kpath_directional_vec_list[i] for n in range(0, nk_list[i] - 1)] for i in 1:n_kpath]
    end
    # flatten `kpath_kpoint_list` to get the full list of kpoints
    full_kpoint_list = reduce(vcat, kpath_kpoint_list)
    push!(full_kpoint_list, kpath[end]) # manually add final point

    return (nk_turning_list, full_kpoint_list)
end


function gen_eigensys_list_along_k_path(hk::Function; kpath::Vector{<:Vector}, nk::Int=50, nband::Int64=10)::Tuple{Vector{Vector{Float64}},Vector{Matrix{Complex{Float64}}}}
    (_, full_kpoint_list) = parse_k_path(; kpath=kpath, nk=nk)

    dim = length(kpath[1])
    dim_H = size(hk(zeros(dim)))[1]
    @assert nband <= dim_H

    eigvals_list = Vector{Vector{Float64}}(undef, length(full_kpoint_list))
    eigvecs_list = Vector{Matrix{Complex{Float64}}}(undef, length(full_kpoint_list))

    @show is_sparse::Bool = hk(zeros(dim)) isa AbstractSparseArray
    if is_sparse
        # note: `eigsolve` return eigvecs as a list of vectors rather than a matrix
        sparse_eigsys_list = [KrylovKit.eigsolve(hk(k), nband, :SR; ishermitian=true) for k in full_kpoint_list] # usage of specifying `ClosestTo(λ)` can be found in `https://jutho.github.io/KrylovKit.jl/v0.1/man/eig/`

        eigvals_list = [sparse_eigsys[1][1:nband] for sparse_eigsys in sparse_eigsys_list]
        eigvecs_list = [hcat(sparse_eigsys[2]...)[:, 1:nband] for sparse_eigsys in sparse_eigsys_list]
    else
        kpath_eigsys_list = [eigen(hk(k)) for k in full_kpoint_list]
        (eigvals_list, eigvecs_list) = ([eigen.values[1:nband] for eigen in kpath_eigsys_list], [eigen.vectors[:, 1:nband] for eigen in kpath_eigsys_list])
    end

    # eigsys_list = [eigen(Hermitian(hk(k)) |> Matrix) for k in full_kpoint_list]
    # (eigvals_list, eigvecs_list) = ([eigsys.values[1:nband] for eigsys in eigsys_list], [eigsys.vectors[:, 1:nband] for eigsys in eigsys_list])

    return (eigvals_list, eigvecs_list)
end


"""
Plot Band for a Given Hamiltonian Function and a kPath within BZ
---
Note: here `hk` and `kpath` can either be `k_crys` or `k_cart`. But the input must be consistent!

Arguments:
- `hk::Function`: Hamiltonian matrix (either `k_crys` or `k_cart`).
- `kpath::Vector{Vector{Float64}}`: list of turning kpoint coordinates within the BZ (either `k_crys` or `k_cart`). Crystal Coordinates Example: `[[0.0, 0.0], [π, 0.0], [π, π], [0.0, 0.0]]`

Named Arguments:
- `nk::Int`: the number of kpoints of the first kpath. The `nk` of other kpaths will be scaled by their lengths.
- `nband::Int64`: the number of bands to plot (default: 10).
- `save_plot::Bool`: flag to save the plot as a PDF file (default: false).
- `save_plot_dir::String`: the directory to save the plot (default: `mkpath ./figure` at current working directory).
"""
function plot_band(hk::Function; kpath::Vector{<:Vector}, nk::Int=50, nband::Int64=10, ylims::Tuple{Number,Number}=(-5, 5), aspect_ratio::Number=1, save_plot::Bool=false, save_plot_dir::String="")::Tuple{Vector{Vector{Float64}},Vector{Matrix{Complex{Float64}}},Plots.Plot}
    (nk_turning_list, full_kpoint_list) = parse_k_path(; kpath=kpath, nk=nk)
    (eigvals_list, eigvecs_list) = gen_eigensys_list_along_k_path(hk; kpath=kpath, nk=nk, nband=nband)

    # initialize figure
    fig = plot(;
        # ylabel="E",
        xlims=(-1, nk_turning_list[end] + 2),
        ylims=ylims,
        xticks=false,
        legend=false,
        framestyle=:box,
        aspect_ratio=aspect_ratio * nk_turning_list[end] / (ylims[2] - ylims[1]), # 1:1 
    )
    # add vertical line signifying high symmetry points from `nk_turning_list`
    for kpoint in nk_turning_list
        vline!(fig, [kpoint], color=:black, alpha=0.32, lw=4)
    end

    # convert to appropriate data matrix for plotting
    let kpath_eigval_data = hcat(eigvals_list...) |> transpose
        # scatter!(fig, kpath_eigval_data,
        #     color=collect(1:nband) |> transpose, # band color in default order
        #     # color=:black,
        #     markersize=1.5,
        # )
        plot!(fig, kpath_eigval_data,
            color=collect(1:nband) |> transpose, # band color in default order
            # color=:black,
            lw=1.5,
        )
    end

    if save_plot
        if save_plot_dir == ""
            save_plot_dir = joinpath(pwd(), "figure") # if not specified, save to `pwd()/figure` by default
        end
        mkpath(save_plot_dir)

        fig_path = joinpath(save_plot_dir, "band_plot.pdf")
        savefig(fig, fig_path)
    else
        display(fig) # display only if not saving
    end
    return (eigvals_list, eigvecs_list, fig)
end





function test()
    t = 0.2
    μ = 0.5
    hk = k_crys -> sum(k_crys .* Pauli_Matrices[1:2]) |> sparse # a two-band model

    k_crys_path = [[0.0, 0.0], [0.0, π], [π, π], [0.0, 0.0]]

    @show "Hello, World!"

    plot_band(hk; kpath=k_crys_path, nband=2, save_plot=true, save_plot_dir="/home/hxd/Dropbox/Julia_Projects/CMP_Utils/src/figure")
end



# end # module `Band_Plot`
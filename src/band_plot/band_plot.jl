module Band_Plot

using LinearAlgebra, SparseArrays
using Plots

const Identity_Matrix = [[1 0; 0 1]]
const Pauli_Matrices = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]

export band_plot, test
export Identity_Matrix, Pauli_Matrices


"""
Band Plot `band_plot` for a Given Hamiltonian and a kPath in BZ
---
Arguments:
- `h_k_crys::Function`: function from `k_crys` to the Hamiltonian matrix.
- `kpath_list::Vector{Vector{Float64}}`: the vector of kpaths of the form `[k_crys_start, k_crys_end]` within the BZ. Example: `[[0.0, 0.0], [π, 0.0], [π, π], [0.0, 0.0]]`

Named Arguments:
- `kpoints::Int`: the number of kpoints of the first kpath. The kpoints of other kpaths will be automatically scaled by their lengths.
- `nbands::Int64`: the number of bands to plot (default: 10).
- `save_plot::Bool`: flag to save the plot as a PDF file (default: false).
- `save_plot_dir::String`: the directory to save the plot (default: `mkpath ./figure` at current working directory).
"""
function band_plot(h_k_crys::Function, kpath_list::Vector{Vector{Float64}}; kpoints::Int=30, nbands::Int64=10, save_plot::Bool=false, save_plot_dir::String="")::Tuple{Vector{Vector{Float64}},Vector{Matrix{Complex{Float64}}},Plots.Plot}
    @assert length(kpath_list) > 1 # check if there are at least two kpaths
    @assert [length(kpath) == 2 for kpath in kpath_list] |> all # check if each kpath is legal: `[kpath_start, kpath_end]`

    kpath_length_list = [norm(kpath_list[i+1] - kpath_list[i]) for i in 1:(length(kpath_list)-1)]
    kpoints_list::Vector{Int64} = [ceil(kpoints * kpath_length / kpath_length_list[1]) |> Int64 for kpath_length in kpath_length_list]
    kpath_turning_points = [1; cumsum(kpoints_list) .+ 1] # add initial line to `kpath_turning_points`; shift by 1 for each kpath

    kpath_k_crys_list = let kpath_step_list = [(kpath_list[i+1] - kpath_list[i]) / kpoints_list[i] for i in 1:(length(kpath_list)-1)]
        # note: avoid nearby duplicates!
        [[kpath_list[i] + n * kpath_step_list[i] for n in range(0, kpoints_list[i] - 1)] for i in 1:(length(kpath_list)-1)]
    end
    # flatten `kpath_k_crys_list` to list of vectors
    kpath_k_crys_list = reduce(vcat, kpath_k_crys_list)
    push!(kpath_k_crys_list, kpath_list[end]) # manually add final point

    kpath_eigen_list = [eigen(Hermitian(h_k_crys(k_crys))) for k_crys in kpath_k_crys_list]
    kpath_eigval_list = [eigen.values for eigen in kpath_eigen_list]
    kpath_eigvec_list = [eigen.vectors for eigen in kpath_eigen_list]
    # truncate bands if too many
    if nbands < length(kpath_eigval_list[1])
        kpath_eigval_list = [eigval[1:nbands] for eigval in kpath_eigval_list]
        kpath_eigvec_list = [eigvec[:, 1:nbands] for eigvec in kpath_eigvec_list]
    end

    # initialize figure
    fig = plot(;
        # ylabel="E",
        xlims=(-1, kpath_turning_points[end] + 2),
        xticks=false,
        legend=false,
        framestyle=:box
    )
    # add vertical line signifying high symmetry points from `kpoints_list`
    for kpoint in kpath_turning_points
        vline!(fig, [kpoint], color=:black, alpha=0.32, lw=4)
    end

    # convert to appropriate data matrix for plotting
    let kpath_eigval_data = hcat(kpath_eigval_list...) |> transpose
        scatter!(fig, kpath_eigval_data,
            color=collect(1:nbands) |> transpose # band color in default order
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
    return (kpath_eigval_list, kpath_eigvec_list, fig)
end



function test()
    t = 0.2
    μ = 0.5
    hk = k_crys -> sum(k_crys .* Pauli_Matrices[1:2]) # a two-band model

    k_crys_path = [[0.0, 0.0], [0.0, π], [π, π], [0.0, 0.0]]

    @show "Hello, World!"

    band_plot(hk, k_crys_path; save_plot=true, save_plot_dir="/home/hxd/Dropbox/Julia_Projects/CMP_Utils/src/figure")
end



end # module Band_Plot
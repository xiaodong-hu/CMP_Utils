module Lattice

using LinearAlgebra
using MLStyle
using Plots

const Identity_Matrix = [[1 0; 0 1]]
const Pauli_Matrices = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]


export RealSpaceLattice
export initialize_lattice, show, add_hopping_term!, update_full_hopping_Hamiltonian!


struct RealSpaceLattice
    lattice_name::String
    sample_size::Vector{Int64}
    brav_vecs::Vector{Vector{Float64}}
    dim::Int64
    sub_crys_vecs::Vector{Vector{Float64}}
    nsub::Int64
    sub_name_list::Vector{String}
    cell_volume::Float64
    cell_ind_list::Vector{Vector{Int64}} # `cell_ind` as `[i,j,k]`
    atom_pos_list::Vector{Tuple{Vector{Int64},Int64}} # `atom_pos` as `(cell_ind, atom_type)`
    atom_cart_list::Vector{Vector{Float64}} # `atom_cart_pos` in cartesian coordinates
end



"""
Struct Initialization of `RealSpaceLattice`
---
Named Arguments:
- `brav_vecs::Vector{Vector{Float64}}`: real-space bravias vectors as `[a1,a2,...]`
- `sample_size::Vector{Int64}`: sample size as `[N1,N2,...]`
- `sub_crys_vecs::Vector{Vector{Float64}}`: sublattice atom positions in *crystal* coordinates
- `lattice_name::String`: lattice name
- `origin_to_lattice_center::Bool`: flag to shift the origin to the center of the lattice
"""
function initialize_lattice(; brav_vecs::Vector{Vector{Float64}}, sample_size::Vector{Int64}, sub_crys_vecs::Vector{Vector{Float64}}, lattice_name::String="", origin_to_lattice_center::Bool=false)::RealSpaceLattice
    dim = length(brav_vecs)
    @assert dim == 2 || dim == 3
    @assert length(brav_vecs) == length(sample_size)
    nsub = length(sub_crys_vecs)
    @assert all(length.(sub_crys_vecs) == [dim for _ in 1:nsub])
    sub_name_list = [string("A", i) for i in 1:nsub]

    cell_ind_list = Iterators.map(x -> collect(x), Iterators.product([0:(N-1) for N in sample_size]...)) |> collect |> vec
    if !origin_to_lattice_center
        cell_ind_list = [cell_ind .+ (sample_size .÷ 2) for cell_ind in cell_ind_list] # shift the origin to the center of the lattice
    end
    atom_pos_list = [(cell_ind, atom_type) for cell_ind in cell_ind_list for atom_type in 1:nsub]
    atom_cart_list = [sum(brav_vecs .* (cell_ind + sub_crys_vecs[atom_type])) for (cell_ind, atom_type) in atom_pos_list]

    cell_volume = if dim == 2
        a1 = push!(deepcopy(brav_vecs[1]), 1.0)
        a2 = push!(deepcopy(brav_vecs[2]), 1.0)
        a3 = [0.0, 0.0, 1.0]
        det([a1 a2 a3])
    elseif dim == 3
        det(hcat(brav_vecs...))
    end

    return RealSpaceLattice(
        lattice_name,
        sample_size,
        brav_vecs,
        dim,
        sub_crys_vecs,
        nsub,
        sub_name_list,
        cell_volume,
        cell_ind_list,
        atom_pos_list,
        atom_cart_list
    )
end


"""
Example Struct Initialization of `RealSpaceLattice`
---
add dispatch to include example lattices of `lattice_name = square, honeycomb, kagome, Lieb...`
"""
function initialize_lattice(lattice_name::String, sample_size::Vector{Int64}; origin_to_lattice_center::Bool=false)::RealSpaceLattice
    (brav_vecs, sub_crys_vecs) = @match lattice_name begin
        "square" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]])
        "honeycomb" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 3, 1 / 3]])
        "kagome" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        "Lieb" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        _ => error("`lattice_name` $lattice_name not defined!")
    end
    return initialize_lattice(; brav_vecs=brav_vecs, sample_size=sample_size, sub_crys_vecs=sub_crys_vecs, lattice_name=lattice_name, origin_to_lattice_center=origin_to_lattice_center)
end


"plot a vector of `RealSpaceLattice`"
function plot_lattice(lattice_list::Vector{RealSpaceLattice}; save_plot::Bool=false, save_plot_dir::String="")::Plots.Plot
    fig = plot()
    for l in lattice_list
        atom_cart_list_along_each_direction = [[atom_cart[i] for atom_cart in l.atom_cart_list] for i in 1:l.dim]
        plot!(fig, atom_cart_list_along_each_direction...;
            seriestype=:scatter,
            aspect_ratio=:equal,
            legend=false,
            framestyle=:none,
            markersize=l.cell_volume * sqrt(reduce(*, l.sample_size)) / length(l.atom_cart_list) * 100, # adjust markersize with the number of atoms in the plot
            ticks=false,
        )
    end

    if save_plot
        if save_plot_dir == ""
            save_plot_dir = joinpath(pwd(), "figure") # if not specified, save to `pwd()/figure` by default
        end
        mkpath(save_plot_dir)

        fig_path = joinpath(save_plot_dir, l.lattice_name * ".pdf")
        savefig(fig, fig_path)
    else
        display(fig) # display only if not saving
    end
    return fig
end

"add dispatch to `plot_lattice` to allow single `RealSpaceLattice` plot"
function plot_lattice(lattice::RealSpaceLattice; save_plot::Bool=false, save_plot_dir::String="")::Plots.Plot
    return plot_lattice([lattice]; save_plot=save_plot, save_plot_dir=save_plot_dir)
end


end # end module

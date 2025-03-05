module Lattice

using LinearAlgebra
using MLStyle
using Plots

const Identity_Matrix = [[1 0; 0 1]]
const Pauli_Matrices = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]


export RealSpaceLattice
export initialize_lattice, plot_lattice


"""
Struct `RealSpaceLattice`
---
- Fields:
    - `lattice_name::String`: lattice name
    - `sample_size::Vector{Int64}`: sample size as `[N1,N2,...]`

"""
mutable struct RealSpaceLattice
    lattice_name::String
    sample_size::Vector{Int64}
    brav_vec_list::Vector{<:Vector}
    dim::Int64
    sub_crys_vec_list::Vector{<:Vector}
    sub_name_list::Vector{String}
    nsub::Int64

    cell_volume::Number
    cell_ind_list::Vector{Vector{Int64}} # cell indices such as `[[i,j,k],...]`
    atom_pos_list::Vector{Tuple{Vector{Int64},Int64}} # atom positions in each cell as `[([i,j,k], atom_type),...]`
    atom_cart_list::Vector{<:Vector} # Union{Vector{Vector{T}},Vector{Vector{U}}} # `atom_cart_pos` in cartesian coordinates
end



"""
Struct Initialization of `RealSpaceLattice{T,U}`
---
where even *symbolic* geometry setup are also supported.

Named Arguments:
- `brav_vec_list::Vector{Vector{T}}`: real-space bravias vectors as `[a1,a2,...]`
- `sample_size::Vector{Int64}`: sample size as `[N1,N2,...]`
- `sub_crys_vec_list::Vector{Vector{U}}`: sublattice atom positions in *crystal* coordinates
- `lattice_name::String`: lattice name
- `shift_origin_to_lattice_center::Bool`: flag to shift the origin to the center of the lattice
"""
function initialize_lattice(; brav_vec_list::Vector{<:Vector}, sample_size::Vector{Int64}, sub_crys_vec_list::Vector{<:Vector}, lattice_name::String="", shift_origin_to_lattice_center::Bool=false)::RealSpaceLattice
    dim = length(brav_vec_list)
    @assert dim == 2 || dim == 3
    @assert length(brav_vec_list) == length(sample_size)
    nsub = length(sub_crys_vec_list)
    @assert all(length.(sub_crys_vec_list) == [dim for _ in 1:nsub])
    sub_name_list = [string("A", i) for i in 1:nsub]

    cell_ind_list = Iterators.map(x -> collect(x), Iterators.product([0:(N-1) for N in sample_size]...)) |> collect |> vec
    if shift_origin_to_lattice_center
        cell_ind_list = [cell_ind .- (sample_size .÷ 2) for cell_ind in cell_ind_list] # shift the origin to the center of the lattice
    end
    atom_pos_list = [(cell_ind, atom_type) for cell_ind in cell_ind_list for atom_type in 1:nsub]
    atom_cart_list = [sum(brav_vec_list .* (cell_ind + sub_crys_vec_list[atom_type])) for (cell_ind, atom_type) in atom_pos_list]

    cell_volume = if dim == 2
        # a1 = push!(deepcopy(brav_vec_list[1]), 1)
        # a2 = push!(deepcopy(brav_vec_list[2]), 1)
        # a3 = [0, 0, 1]
        # det([a1 a2 a3])

        # element-wise determinant of 2x2 matrix (note: symbolic `det` is not defined for `SymEngine.Basic`)
        sum(brav_vec_list[1][i] * brav_vec_list[2][j] - brav_vec_list[1][j] * brav_vec_list[2][i] for i in 1:2, j in 1:2)
    elseif dim == 3
        # det(hcat(brav_vec_list...))
        # element-wise determinant of 3x3 matrix (note: symbolic `det` is not defined for `SymEngine.Basic`)
        sum(brav_vec_list[1][i] * (brav_vec_list[2][j] * brav_vec_list[3][k] - brav_vec_list[2][k] * brav_vec_list[3][j]) for i in 1:3, j in 1:3, k in 1:3)
    end

    return RealSpaceLattice(
        lattice_name,
        sample_size,
        brav_vec_list,
        dim,
        sub_crys_vec_list,
        sub_name_list,
        nsub,
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
function initialize_lattice(lattice_name::String, sample_size::Vector{Int64}; shift_origin_to_lattice_center::Bool=false)::RealSpaceLattice
    (brav_vec_list, sub_crys_vec_list) = @match lattice_name begin
        "square" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]])
        "honeycomb" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 3, 1 / 3]])
        "kagome" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        "Lieb" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        _ => error("`lattice_name` $lattice_name not defined!")
    end
    return initialize_lattice(; brav_vec_list=brav_vec_list, sample_size=sample_size, sub_crys_vec_list=sub_crys_vec_list, lattice_name=lattice_name, shift_origin_to_lattice_center=shift_origin_to_lattice_center)
end


"plot a vector of `RealSpaceLattice`"
function plot_lattice(lattice_list::Vector{RealSpaceLattice}; save_plot::Bool=false, save_plot_dir::String="")::Plots.Plot
    fig = plot()
    for l in lattice_list
        convert(Vector{Vector{Float64}}, l.atom_cart_list)

        @show plot_range = [extrema([atom[i] for atom in l.atom_cart_list]) for i in 1:l.dim]
        # @show 1 / 3 * minimum(norm.(l.brav_vec_list)) / l.nsub / reduce(*, l.sample_size)
        plot!(fig, Tuple.(l.atom_cart_list);
            seriestype=:scatter,
            aspect_ratio=:equal,
            legend=false,
            framestyle=:box,
            # grid=true,
            markersize=50 / sqrt(length(l.atom_cart_list) / l.nsub), # adjust markersize with the number of atoms in the plot
            # ticks=false,
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

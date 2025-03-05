
module LatticeModule

# export types
export AbstractLattice, Lattice
# export methods
export initialize_lattice_sample, _to_k_cart, _to_k_crys


using MLStyle, LinearAlgebra



abstract type AbstractLattice end

"""
Struct of discrete lattice `Lattice{T} <: AbstractLattice`
---
where `T` is the promoted data type based on the given input data (such as `brav_vec_list` and `sub_crys_vec_list`) and internal computed data (such as reciprocal vector) (so *symbolic geometry* is also supported).

- Fields:
    - `lattice_name::String`: lattice name
    - `sample_size::Vector{Int64}`: sample size as `[N1,N2,...]`
    - `n_cell::Int64`: number of cells
    - `dim::Int64`: k_crys_listdimension of the lattice
    - `brav_vec_list::Vector{Vector{T}}`: real-space bravias vectors as `[a1,a2,...]`
    - `cell_volume::T`: volume of the unit cell
    - `sub_crys_vec_list::Vector{Vector{T}}`: sublattice atom positions in *crystal* coordinates
    - `sub_name_list::Vector{String}`: sublattice atom names
    - `n_sub::Int64`: number of sublattices
    - `cell_list::Vector{Vector{Int64}}`: integer list of real-space lattice such as `[[i,j,k],...]` for looping over cells
    - `cell_sub_tuple_list::Vector{Tuple{Vector{Int64},Int64}}`: tuple list of `(cell, sub)` such as `[([i,j,k], sub_type),...]` for further construction of `r_crys_list` and `r_cart_list`
    - `r_crys_list::Vector{Vector{T}}`: site positions in crystal coordinates
    - `r_cart_list::Vector{Vector{T}}`: site positions in cartesian coordinates
    - `n_site::Int64`: number of sites
    - `reciprocal_vec_list::Vector{Vector{T}}`: reciprocal vectors as `[b1,b2,...]`
    - `k_sample_size::Vector{Int64}`: sample size in k-space
    - `n_k::Int64`: number of k-points
    - `k_int_list::Vector{Vector{Int64}}`: integer list of reciprocal lattice for looping over the momentum-space and for generation of `k_crys_list` etc
    - `k_crys_list::Vector{<:Vector}`: k-points position in crystal coordinates
    - `k_cart_list::Vector{<:Vector}`: k-points position in cartesian coordinates
"""
struct Lattice{T} <: AbstractLattice
    lattice_name::String
    sample_size::Vector{Int64}
    n_cell::Int64
    dim::Int64

    # real-space data
    brav_vec_list::Vector{Vector{T}}
    cell_volume::T

    sub_crys_vec_list::Vector{Vector{T}}
    sub_name_list::Vector{String}
    n_sub::Int64

    cell_list::Vector{Vector{Int64}} # integer list for real-space lattice such as `[[i,j,k],...]` for looping over cells
    cell_sub_tuple_list::Vector{Tuple{Vector{Int64},Int64}} # list of tuples `(cell, sub)` such as `[([i,j,k], sub_type),...]` for further construction of `r_crys_list` and `r_cart_list`
    r_crys_list::Vector{Vector{T}} # site position list in crystal coordinates
    r_cart_list::Vector{Vector{T}} # site position list in cartesian coordinates
    n_site::Int64 # number of sites

    # momentum-space data
    reciprocal_vec_list::Vector{Vector{T}} # reciprocal vectors as `[b1,b2,...]`
    n_k::Int64 # number of k-points
    k_int_list::Vector{Vector{Int64}} # integer list of reciprocal lattice for looping over the momentum-space and for generation of `k_crys_list` etc
    k_int_to_ik_dict::Dict{Vector{Int64},Int64} # dictionary for mapping from integer k-point to index
    k_crys_list::Vector{<:Vector} # k-points position in crystal coordinates
    k_crys_to_ik_dict::Dict{Vector{Float64},Int64} # dictionary for mapping from crystal k-point to index
    k_cart_list::Vector{<:Vector} # k-points position in cartesian coordinates
end

"""
initialization of the struct `Lattice{T}`
---
with data type `T`. Note: even *symbolic* geometry setup (of type `T`) are also supported.

- Named Arguments:
    - `brav_vec_list::Vector{Vector{T}}`: real-space bravias vectors as `[a1,a2,...]`
    - `sample_size::Vector{Int64}`: sample size as `[N1,N2,...]`
    - `sub_crys_vec_list::Vector{Vector{T}}`: sublattice atom positions in *crystal* coordinates
    - `lattice_name::String`: lattice name
"""
function initialize_lattice_sample(; brav_vec_list::Vector{<:Vector}, sample_size::Vector{Int64}, sub_crys_vec_list::Vector{<:Vector}, lattice_name::String="")
    n_cell = reduce(*, sample_size)
    dim = length(sample_size)
    n_sub = length(sub_crys_vec_list)
    sub_name_list = [string("A", i) for i in 1:n_sub] # set the default names as ["A1", "A2", ...]

    # make sure the input bravias vector and sublattice vectors have consistent dimensions
    @assert length(brav_vec_list) == dim
    @assert all(length.(brav_vec_list) == [dim for _ in 1:dim])
    @assert all(length.(sub_crys_vec_list) == [dim for _ in 1:n_sub])

    # generate the real-space cell list, site position list, and site cartesian list
    cell_list = Iterators.map(x -> collect(x), Iterators.product([0:(N-1) for N in sample_size]...)) |> collect |> vec
    cell_sub_tuple_list = [(cell, sub_type) for cell in cell_list for sub_type in 1:n_sub]
    r_crys_list = [cell + sub_crys_vec_list[sub_type] for (cell, sub_type) in cell_sub_tuple_list]
    r_cart_list = [sum(brav_vec_list .* (cell + sub_crys_vec_list[sub_type])) for (cell, sub_type) in cell_sub_tuple_list]
    n_site = n_cell * n_sub

    # element-wise determinant of either a 2*2 matrix or a 3*3 matrix (note: symbolic `det` is not defined for `SymEngine.Basic`)
    reciprocal_vec_list, cell_volume = if dim == 2
        a1 = push!(deepcopy(brav_vec_list[1]), 1)
        a2 = push!(deepcopy(brav_vec_list[2]), 1)
        a3 = [0, 0, 1]
        cell_volume = abs(det([a1 a2 a3]))

        b1 = cross(a2, a3)[1:2]
        b2 = cross(a3, a1)[1:2]
        # b3 = cross(a1, a2)

        2 * pi * [b1, b2] / cell_volume, cell_volume
    elseif dim == 3
        a1 = push!(deepcopy(brav_vec_list[1]), 1)
        a2 = push!(deepcopy(brav_vec_list[2]), 1)
        a3 = push!(deepcopy(brav_vec_list[3]), 1)
        cell_volume = abs(det([a1 a2 a3]))

        b1 = cross(a2, a3)
        b2 = cross(a3, a1)
        b3 = cross(a1, a2)

        2 * pi * [b1, b2, b3] / cell_volume, cell_volume
    end

    # generate the momentum-space k-data
    k_int_list = Iterators.map(x -> collect(x), Iterators.product([0:(N-1) for N in sample_size]...)) |> collect |> vec # the number of k-points along each direction in the reciprocal space just equal tot the number of cells along each direction in real-space
    k_int_to_ik_dict = Dict{Vector{Int64},Int64}((k_int, ik) for (ik, k_int) in enumerate(k_int_list))
    n_k = length(k_int_list)
    k_crys_list = [k_int ./ sample_size for k_int in k_int_list]
    k_crys_to_ik_dict = Dict{Vector{Float64},Int64}((k_crys, ik) for (ik, k_crys) in enumerate(k_crys_list))
    k_cart_list = [sum(reciprocal_vec_list .* k_crys) for k_crys in k_crys_list]

    # determine the appropriate data type in consideration of input data and computed internal data (for support of symbolic lattice)
    promote_data_type = @match (brav_vec_list, sub_crys_vec_list, reciprocal_vec_list) begin
        (_::Vector{Vector{T}} where {T}, _::Vector{Vector{U}} where {U}, _::Vector{Vector{S}} where {S}) => promote_type(T, U, S) # type pattern match from `MLStyle.jl`
    end


    return Lattice{promote_data_type}(
        lattice_name,
        sample_size,
        n_cell,
        dim,
        brav_vec_list,
        cell_volume,
        sub_crys_vec_list,
        sub_name_list,
        n_sub,
        cell_list,
        cell_sub_tuple_list,
        r_crys_list,
        r_cart_list,
        n_site,
        reciprocal_vec_list,
        n_k,
        k_int_list,
        k_int_to_ik_dict,
        k_crys_list,
        k_crys_to_ik_dict,
        k_cart_list
    )
end


"convert `k_crys` to `k_cart` for a given `Lattice{T}`"
function _to_k_cart(k_crys::Vector{Float64}, l::Lattice{T}) where {T}
    return sum(l.reciprocal_vec_list .* k_crys)
end



"convert `k_cart` to `k_crys` for a given `Lattice{T}`"
function _to_k_crys(k_cart::Vector{Float64}, l::Lattice{T}) where {T}
    # first, convert to matrix form for the dot product
    reciprocal_vec_mat = hcat(l.reciprocal_vec_list...)' # then each row is a reciprocal vector
    return reciprocal_vec_mat \ k_cart # linear solve from `k_cart = reciprocal_vec_mat * k_crys`
end



end # module LatticeModule
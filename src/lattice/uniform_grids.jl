module UniformGrids

export Graph, Uniform_Grid
export initialize_uniform_grid_from_site_cart_list, initialize_uniform_grid_from_site_int_list, get_sample_size

using LinearAlgebra, StaticArrays
using MLStyle


abstract type Graph end


"""
General Struct of Uniform Grid `Uniform_Grid{D} <: Graph`
---
- Fields:
    - `sample_size`: The size of the lattice in each direction. Note: this entry becomes meaningless after truncation
    - `basis_vec_list::Vector{<:Vector{<:Number}}`: bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
    - `basis_vec_mat::SMatrix{D,D,<:Number}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *rows* (we choose to store in rows for the convenience of matrix-vector multiplication)
    - `cell_volume::Number`: volume of the unit cell
    - `n_site::Integer`: number of sites in the lattice
    - `site_int_list::Vector{<:SVector{D,<:Integer}}`: list of site indices (integer for each direction)
    - `site_crys_list::Vector{<:SVector{D,<:Number}}`: list of site positions in crystal coordinates
    - `site_cart_list::Vector{<:SVector{D,<:Number}}`: list of site positions in cartesian coordinates
    - `site_int_to_index_map::Dict{<:SVector{D,<:Integer},<:Integer}`: hashmap `site_int -> index` for the stored site list
"""
struct Uniform_Grid{D} <: Graph where {D<:Integer}
    sample_size::Vector{<:Integer}
    basis_vec_list::Vector{<:Vector{<:Number}} # bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
    basis_vec_mat::SMatrix{D,D,<:Number} # bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *rows* (we choose to store in rows for the convenience of matrix-vector multiplication)
    cell_volume::Number

    n_site::Integer
    site_int_list::Vector{<:SVector{D,<:Integer}}
    site_crys_list::Vector{<:SVector{D,<:Number}}
    site_cart_list::Vector{<:SVector{D,<:Number}}
    site_int_to_index_map::Dict{<:SVector{D,<:Integer},<:Integer}
end

"""
Constructor of Struct `Uniform_Grid{D}`
---
- Named Args:
    - `sample_size::Vector{<:Integer}`: The size of the lattice in each direction.
    - `basis_vec_list::Union{Matrix{<:Number},Vector{<:Vector{<:Number}}}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *columns*
"""
function Uniform_Grid(;
    sample_size::Vector{<:Integer},
    basis_vec_list::Vector{<:Vector{<:Number}}
)
    D = length(sample_size)

    basis_vec_mat = SMatrix{D,D}(reduce(hcat, basis_vec_list) |> transpose) # transpose to store basis vectors in rows
    cell_volume = abs(det(basis_vec_mat))

    site_int_list = SVector{D}.(
        reduce(Iterators.product, UnitRange.(zero(sample_size), sample_size .- 1))
    ) |> vec # force it to be a vector of SVectors
    n_site = length(site_int_list)
    site_crys_list = [SVector{D}(site_int ./ sample_size) for site_int in site_int_list]
    site_cart_list = [SVector{D}(basis_vec_mat * site_crys) for site_crys in site_crys_list] # note that `basis_vec_mat` is already store in rows
    site_int_to_index_map = Dict(site_int => index for (index, site_int) in enumerate(site_int_list))

    return Uniform_Grid{D}(
        sample_size,
        basis_vec_list,
        basis_vec_mat,
        cell_volume,
        n_site,
        site_int_list,
        site_crys_list,
        site_cart_list,
        site_int_to_index_map
    )
end

function get_sample_size(ug::Uniform_Grid{D}) where {D}
    site_int_max_for_each_dimension = [maximum([site_int[i] for site_int in ug.site_int_list]) for i in 1:D]
    site_int_min_for_each_dimension = [minimum([site_int[i] for site_int in ug.site_int_list]) for i in 1:D]

    sample_size = site_int_max_for_each_dimension .- site_int_min_for_each_dimension .+ 1
    sample_size_product = reduce(*, sample_size)

    @assert sample_size_product == ug.n_site "The number of sites $(ug.n_site) does not equal to the product of the sample size $sample_size_product: `sample_size` is meaningful only for a regular lattice; the input Uniform_Grid may be truncated!"

    return sample_size
end

"""
Alternative Constructor of Struct `Uniform_Grid{D}`
---
from `site_cart_list::Vector{<:Vector{D,<:Number}}` with optional `truncate_radius::Float64`
- Args:
    - `site_cart_list::Vector{<:Vector{<:Number}}`: list of site positions in cartesian coordinates
- Named Args:
    - `sample_size::Vector{<:Integer}`: The size of the lattice in each direction. Note: this entry becomes meaningless after truncation
    - `basis_vec::Union{Matrix{<:Number},Vector{<:Vector{<:Number}}}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *columns*
    - `truncate_radius::Float64=0.0`: truncate the sites with cartesian norm larger than `truncate_radius`
"""
function initialize_uniform_grid_from_site_cart_list(site_cart_list::Vector{<:Vector{<:Number}}; sample_size::Vector{<:Integer}, basis_vec_list::Vector{<:Vector{<:Number}}, truncate_radius::Float64=0.0)::Uniform_Grid
    D = length(sample_size)
    # sample_size = SVector{D}(sample_size)
    # basis_vec_list = basis_vec
    basis_vec_mat = SMatrix{D,D}(reduce(hcat, basis_vec_list) |> transpose) # transpose to store basis vectors in rows
    cell_volume = abs(det(basis_vec_mat))

    if truncate_radius > 0.0
        site_cart_list = [site_cart for site_cart in site_cart_list if norm(site_cart) < truncate_radius]
    end
    p = sortperm(norm.(site_cart_list)) # sort by norm
    site_cart_list = SVector{D}.(site_cart_list[p])

    site_crys_list = [SVector(inv(basis_vec_mat) * site_cart) for site_cart in site_cart_list] # note that `basis_vec_mat` is already store in rows
    site_int_list = [SVector{D}(Int.(round.(site_crys .* sample_size, digits=12))) for site_crys in site_crys_list]
    n_site = length(site_int_list)
    site_int_to_index_map = Dict(site_int => index for (index, site_int) in enumerate(site_int_list))

    return Uniform_Grid{D}(
        sample_size,
        basis_vec_list,
        basis_vec_mat,
        cell_volume,
        n_site,
        site_int_list,
        site_crys_list,
        site_cart_list,
        site_int_to_index_map
    )
end


"""
Alternative Constructor of Struct `Uniform_Grid{D}`
---
from `site_int_list::Vector{<:Vector{<:Integer}}` with optional `truncate_radius::Float64`
- Args:
    - `site_int_list::Vector{<:Vector{<:Integer}}`: list of site indices (integer for each direction)
- Named Args:
    - `sample_size::Vector{<:Integer}`: The size of the lattice in each direction. Note: this entry becomes meaningless after truncation
    - `basis_vec::Union{Matrix{<:Number},Vector{<:Vector{<:Number}}}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *columns*
    - `truncate_radius::Float64=0.0`: truncate the sites with cartesian norm larger than `truncate_radius`
"""
function initialize_uniform_grid_from_site_int_list(site_int_list::Vector{<:Vector{<:Integer}}; sample_size::Vector{<:Integer}, basis_vec_list::Vector{<:Vector{<:Number}}, truncate_radius::Float64=0.0)::Uniform_Grid
    basis_vec_mat = reduce(hcat, basis_vec_list) |> transpose # transpose to store basis vectors in rows
    site_cart_list::Vector{<:Vector{<:Number}} = [basis_vec_mat * (site_int ./ sample_size) for site_int in site_int_list] # note that `basis_vec_mat` is already store in rows

    return initialize_uniform_grid_from_site_cart_list(site_cart_list; sample_size=sample_size, basis_vec_list=basis_vec_list, truncate_radius=truncate_radius)
end





end
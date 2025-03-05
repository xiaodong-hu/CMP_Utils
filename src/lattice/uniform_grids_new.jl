module UniformGrids_Module


using LinearAlgebra


abstract type Abstract_Uniform_Grids end


export Uniform_Grids, Abstract_Uniform_Grids
export dual_basis_vec_mat, dual_basis_vec_list, _k_cart_to_k_crys, _k_cart_to_k_int

"""
Struct `Uniform_Grids{D,T} <: Abstract_Uniform_Grids`
---
of a `D<:Int`-dimensional uniform grids with type-`T` element basis vectors.
- Fields:
    - `dim::Int`: dimension of the lattice
    - `sample_size::Vector{Int}`: The size of the lattice in each direction.
    - `basis_vec_list::Vector{<:Vector{<:T}}`: bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
    - `basis_vec_mat::Matrix{<:T}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *rows* for convenience of matrix-vector multiplications, so that a general vector would be simply `basis_vec_mat * coef_vec`
    - `cell_volume::T`: volume of the unit cell
    - `n_site::Int`: number of sites in the lattice
    - `site_int_list::Vector{<:Vector{Int}}`: list of site indices (integer for each direction)
    - `twisted_phases_over_2π::Vector{T}`: twisted phases in units of `2π`
    - `site_crys_list::Vector{<:Vector{<:T}}`: list of site positions in crystal coordinates
    - `site_cart_list::Vector{<:Vector{<:T}}`: list of site positions in cartesian coordinates
    - `site_complex_cart_list::Vector{Complex}`: list of site positions in complex cartesian coordinates. This can be convenient when studying rotation transformations
    - `site_int_to_index_map::Dict{<:Vector{Int},Int}`: hashmap `site_int -> index` for the stored site list
"""
struct Uniform_Grids{D,T} <: Abstract_Uniform_Grids
    dim::Int
    sample_size::Vector{Int}
    basis_vec_list::Vector{<:Vector{<:T}} # bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
    basis_vec_mat::Matrix{<:T} # bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *rows* for convenience of matrix-vector multiplications, so that a general vector would be simply `basis_vec_mat * coef_vec`
    cell_volume::T

    n_site::Int
    site_int_list::Vector{<:Vector{Int}}
    twisted_phases_over_2π::Vector{T}
    site_crys_list::Vector{<:Vector{<:T}}
    site_cart_list::Vector{<:Vector{<:T}}
    site_complex_cart_list::Vector{Complex}
    site_int_to_index_map::Dict{<:Vector{Int},Int}
end



"""
Constructor of Struct `Uniform_Grids{T}`
---
- Named Args:
    - `sample_size::Vector{<:Integer}`: The size of the lattice in each direction.
    - `basis_vec_list::Union{Matrix{<:Number},Vector{<:Vector{<:Number}}}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *columns*
"""
function Uniform_Grids(;
    sample_size::Vector{Int},
    basis_vec_list::Vector{<:Vector{T}},
    twisted_phases_over_2π::Vector{T}
) where {T<:Number}
    D = length(sample_size)
    @assert length(basis_vec_list) == D
    @assert all(length(basis_vec) == D for basis_vec in basis_vec_list)
    @assert length(twisted_phases_over_2π) == D

    basis_vec_mat = Matrix(reduce(hcat, basis_vec_list) |> transpose) # note that `hcat()` method forces the basis vectors to be stored in columns `[v1 v2 ...]`, so we need to transpose it to store in rows (for the convenience of matrix-vector multiplication below)
    cell_volume = abs(det(basis_vec_mat))

    site_int_list = collect.(
        reduce(Iterators.product, UnitRange.(zero(sample_size), sample_size .- 1))
    ) |> vec

    n_site = length(site_int_list)
    site_crys_list = [((site_int + twisted_phases_over_2π) ./ sample_size) for site_int in site_int_list]
    site_cart_list = [sum(site_crys .* basis_vec_list) for site_crys in site_crys_list] # equivalently, if `basis_vec_mat` is stored in rows, we can also obtain the cartesian coordinates using `site_cart = basis_vec_mat * site_crys`
    site_complex_cart_list = [(site_cart[1] + im * site_cart[2]) for site_cart in site_cart_list]
    site_int_to_index_map = Dict(site_int => index for (index, site_int) in enumerate(site_int_list))

    return Uniform_Grids{D,T}(
        D,
        sample_size,
        basis_vec_list,
        basis_vec_mat,
        cell_volume,
        n_site,
        site_int_list,
        twisted_phases_over_2π,
        site_crys_list,
        site_cart_list,
        site_complex_cart_list,
        site_int_to_index_map
    )
end



"""
Get Dual Basis Vector Matrix (in *rows*) from a Given Basis Vector Matrix (in *rows*)
---
Here the input can either be real-space or momentum-space basis vectors.
- Args:
    - `basis_vec_mat::Matrix{<:Number}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice
"""
function dual_basis_vec_mat(basis_vec_mat::Matrix{<:Number})
    return 2 * pi * inv(basis_vec_mat) |> transpose # transpose is necessary here because every matrix is stored in rows by default. This ensures that `dot(vec_list[i], dual_vec_list[j]) = 2pi * δ_ij`
end

"""
Get Dual-basis Vector List from a Given Basis Vector List
---
Here the input can either be real-space or momentum-space basis vectors.
- Args:
    - `basis_vec_list::Vector{<:Vector{<:Number}}`: bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
"""
function dual_basis_vec_list(basis_vec_list::Vector{<:Vector{<:Number}})
    basis_vec_mat = reduce(hcat, basis_vec_list)
    return (eachcol(dual_basis_vec_mat(basis_vec_mat)) .|> collect)
end


function _k_cart_to_k_crys(k_cart::Vector{<:Number}, k_data::Uniform_Grids{D,T}) where {D,T}
    # We use `site_cart ≡ basis_vec_mat * site_crys` to solve out `site_crys`
    k_crys = k_data.basis_vec_mat \ k_cart
    return k_crys
end

function _k_cart_to_k_int(k_cart::Vector{<:Number}, k_data::Uniform_Grids{D,T}) where {D,T}
    site_crys = _k_cart_to_k_crys(k_cart, k_data)
    k_int = round.(Int, site_crys .* sample_size)
    return k_int
end





end # module
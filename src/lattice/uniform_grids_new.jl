module UniformGrids_Module


using LinearAlgebra


abstract type Abstract_Uniform_Grids end


export Uniform_Grids, Abstract_Uniform_Grids
export dual_basis_vec_mat, dual_basis_vec_list, _k_cart_to_k_crys, _k_cart_to_k_int

"""
Struct `Uniform_Grids <: Abstract_Uniform_Grids`
---
- Fields:
    - `dim::Int`: dimension of the lattice
    - `sample_size::Vector{Int}`: The size of the lattice in each direction.
    - `basis_vec_list::Vector{<:Vector{<:Number}}`: bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
    - `basis_vec_mat::Matrix{<:Number}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *rows* for convenience of matrix-vector multiplications, so that a general vector would be simply `basis_vec_mat * coef_vec`
    - `cell_volume::Float64`: volume of the unit cell
    - `nsite::Int`: number of sites in the lattice
    - `site_int_list::Vector{<:Vector{Int}}`: list of site indices (integer for each direction)
    - `twisted_phases_over_2π::Vector{<:Number}`: twisted phases in units of `2π`
    - `site_crys_list::Vector{<:Vector{<:Number}}`: list of site positions in crystal coordinates
    - `site_cart_list::Vector{<:Vector{<:Number}}`: list of site positions in cartesian coordinates
    - `site_complex_cart_list::Vector{Complex}`: list of site positions in complex cartesian coordinates. This can be convenient when studying rotation transformations
    - `site_int_to_index_map::Dict{<:Vector{Int},Int}`: hashmap `site_int -> index` for the stored site list
"""
struct Uniform_Grids <: Abstract_Uniform_Grids
    dim::Int
    sample_size::Vector{Int}
    basis_vec_list::Vector{<:Vector{<:Number}} # bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
    basis_vec_mat::Matrix{<:Number} # bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice. Note: both are stored in *rows* for convenience of matrix-vector multiplications, so that a general vector would be simply `basis_vec_mat * coef_vec`
    cell_volume::T where {T<:Number}

    nsite::Int
    site_int_list::Vector{<:Vector{Int}}
    twisted_phases_over_2π::Vector{<:Number}
    site_crys_list::Vector{<:Vector{<:Number}}
    site_cart_list::Vector{<:Vector{<:Number}}
    site_complex_cart_list::Vector{Complex{<:Real}}
    site_int_to_index_map::Dict{Vector{Int},Int}
end



"""
Constructor of Struct `Uniform_Grids`
---
- Named Args:
    - `sample_size::Vector{Int}`: The size of the lattice in each direction.
    - `basis_vec_list::Vector{<:Vector{T}}`: bravias vectors
"""
function Uniform_Grids(;
    sample_size::Vector{Int},
    basis_vec_list::Vector{<:Vector{<:Number}},
    twisted_phases_over_2π::Vector{<:Number},
    filter_cart_radius::Float64=Inf,
    extended_multiples::Int=1,
    centrosymmetric_to_origin::Bool=false
)
    dim = length(sample_size)
    @assert length(basis_vec_list) == dim
    @assert all(length.(basis_vec_list) .== dim)
    @assert length(twisted_phases_over_2π) == dim

    basis_vec_mat = Matrix(reduce(hcat, basis_vec_list) |> transpose) # note that `hcat()` method forces the basis vectors to be stored in columns `[v1 v2 ...]`, so we need to transpose it to store in rows (for the convenience of matrix-vector multiplication below)
    cell_volume = abs(det(basis_vec_mat))

    site_int_list = collect.(
        if !centrosymmetric_to_origin
            reduce(Iterators.product, UnitRange.(zero(sample_size), extended_multiples * sample_size .- 1))
        else
            reduce(Iterators.product, UnitRange.(-extended_multiples * sample_size .+ 1, extended_multiples * sample_size .- 1))
        end
    ) |> vec

    nsite = length(site_int_list)
    site_crys_list = [((site_int + twisted_phases_over_2π) ./ sample_size) for site_int in site_int_list]
    site_cart_list = [sum(site_crys .* basis_vec_list) for site_crys in site_crys_list] # equivalently, if `basis_vec_mat` is stored in rows, we can also obtain the cartesian coordinates using `site_cart = basis_vec_mat * site_crys`

    if !isinf(filter_cart_radius)
        filtered_site_cart_list = filter(site_cart -> norm(site_cart) < filter_cart_radius, site_cart_list)

        filtered_site_ind_list = [findfirst(x -> x == site_cart, site_cart_list) for site_cart in filtered_site_cart_list]

        site_int_list = site_int_list[filtered_site_ind_list] # reduced `site_int_list`
        site_crys_list = site_crys_list[filtered_site_ind_list] # reduced `site_crys_list`
        site_cart_list = filtered_site_cart_list # reduced `site_cart_list`
        nsite = length(site_int_list) # reduced `nsite`
    end

    site_complex_cart_list = [(site_cart[1] + im * site_cart[2]) for site_cart in site_cart_list]
    site_int_to_index_map = Dict(site_int => index for (index, site_int) in enumerate(site_int_list))

    return Uniform_Grids(
        dim,
        sample_size,
        basis_vec_list,
        basis_vec_mat,
        cell_volume,
        nsite,
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
    - `basis_vec_mat::Matrix{T}`: bravias vectors as matrix for real-space lattice, or reciprocal vectors as matrix for k-space lattice
"""
function dual_basis_vec_mat(basis_vec_mat::Matrix{T}) where {T}
    return 2 * pi * inv(basis_vec_mat) |> transpose # transpose is necessary here because every matrix is stored in rows by default. This ensures that `dot(vec_list[i], dual_vec_list[j]) = 2pi * δ_ij`
end

"""
Get Dual-basis Vector List from a Given Basis Vector List
---
Here the input can either be real-space or momentum-space basis vectors.
- Args:
    - `basis_vec_list::Vector{Vector{T}}`: bravias vectors for real-space lattice, or reciprocal vectors for k-space lattice
"""
function dual_basis_vec_list(basis_vec_list::Vector{Vector{T}})::Vector{Vector{T}} where {T}
    basis_vec_mat = reduce(hcat, basis_vec_list)
    return (eachcol(dual_basis_vec_mat(basis_vec_mat)) .|> collect)
end


function _k_cart_to_k_crys(k_cart::Vector{T}, k_data::Uniform_Grids)::Vector{T} where {T}
    # We use `site_cart ≡ basis_vec_mat * site_crys` to solve out `site_crys`
    k_crys = k_data.basis_vec_mat \ k_cart
    return k_crys
end

function _k_cart_to_k_int(k_cart::Vector{T}, k_data::Uniform_Grids)::Vector{Int} where {T}
    site_crys = _k_cart_to_k_crys(k_cart, k_data)
    k_int = round.(Int, site_crys .* sample_size)
    return k_int
end





end # module
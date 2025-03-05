module CMP_Utils
# dependencies: Revise, LinearAlgebra, SparseArrays, Plots
using Revise # update submodule without restarting Julia REPL
using MLStyle, LinearAlgebra

include("utils/band_plot.jl") # where method `plot_band` is defined
include("utils/reciprocal_vec.jl") # where method `reciprocal_vec` is defined
include("lattice/lattice_new.jl") # where struct `Lattice_Sample` and relevant methods are defined
using .LatticeModule # escape submodule namespace `LatticeModule`
export Lattice # export structs
export initialize_lattice_sample, _to_k_cart, _to_k_crys # export functions


include("lattice/uniform_grids.jl") # where struct `Uniform_Grid` and relevant methods are defined
using .UniformGrids # escape submodule namespace `UniformGrids`
export Graph, Uniform_Grid # export structs and constructors
export initialize_uniform_grid_from_site_cart_list, initialize_uniform_grid_from_site_int_list, get_sample_size # export functions (alternative constructors)

include("lattice/uniform_grids_new.jl") # where struct `Lattice` and relevant methods are defined
using .UniformGrids_Module # escape submodule namespace `Lattice_Module`
export Uniform_Grids, Abstract_Uniform_Grids # export structs and constructors
export dual_basis_vec_mat, dual_basis_vec_list, _k_cart_to_k_crys, _k_cart_to_k_int # export functions

include("single_particle_topology/chern_number.jl") # where method `Chern_number_with_wilson_loop_method` is defined

# include("lattice/complex_lattice.jl") # where struct `Complex_2D_Lattice`, `Complex_2D_Lattice_with_Flux` and relevant methods are defined
# using .Complex_2D_Lattice_Module # escape submodule namespace `Complex_2D_Lattice_Module`
# export Complex_2D_Lattice, Complex_2D_Lattice_with_Flux # export structs and constructors


include("second_quantization/quantum_operators.jl")
using .QuantumOperatorsModule # escape submodule namespace `QuantumOperatorsModule`

export AbstractOp, Fermionic_Creation, Fermionic_Annihilation, Quantum_Expectation # export quantum operators
export is_normal_ordered # export functions

export Pauli_Matrices, Identity_Matrix # export constants

export plot_band, plot_lattice, reciprocal_vec # export functions

end # module CMP_Utils

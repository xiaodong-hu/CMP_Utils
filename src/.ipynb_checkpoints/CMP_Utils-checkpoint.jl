module CMP_Utils
# dependencies: Revise, LinearAlgebra, SparseArrays, Plots
using Revise # update submodule without restarting Julia REPL
using MLStyle, LinearAlgebra

include("utils/band_plot.jl") # where method `plot_band` is defined
include("utils/reciprocal_vec.jl") # where method `reciprocal_vec` is defined
include("lattice/lattice_new.jl") # where struct `Lattice_Sample` and relevant methods are defined
# include("lattice/lattice.jl") # where struct `RealSpaceLattice` and relevant methods are defined
using .LatticeModule # escape submodule namespace `LatticeModule`
export Lattice # export structs
export initialize_lattice_sample, _to_k_cart, _to_k_crys # export functions

include("second_quantization/quantum_operators.jl")
using .QuantumOperatorsModule # escape submodule namespace `QuantumOperatorsModule`

export AbstractOp, Fermionic_Creation, Fermionic_Annihilation, Quantum_Expectation # export quantum operators
export is_normal_ordered # export functions

export Pauli_Matrices, Identity_Matrix # export constants

export plot_band, plot_lattice, reciprocal_vec # export functions

end # module CMP_Utils

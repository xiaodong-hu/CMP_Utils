module CMP_Utils
# dependencies: Revise, LinearAlgebra, SparseArrays, Plots
using Revise # update submodule without restarting Julia REPL


include("utils/band_plot.jl") # where method `plot_band` is defined
include("lattice/lattice.jl") # where struct `RealSpaceLattice` and relevant methods are defined
using .Lattice # escape submodule namespace `Lattice`

export Pauli_Matrices, Identity_Matrix # export constants
export RealSpaceLattice # export structs
export plot_band, initialize_lattice, plot_lattice # export functions

end # module CMP_Utils

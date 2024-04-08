module CMP_Utils
# dependencies: Revise, LinearAlgebra, SparseArrays, Plots
using Revise # update submodule without restarting Julia REPL


include("band_plot/band_plot.jl") # module `Band_Plot`
using .Band_Plot

export band_plot
export Identity_Matrix, Pauli_Matrices

end # module CMP_Utils

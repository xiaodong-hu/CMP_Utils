module QuantumOperatorsModule

export AbstractOp, Fermionic_Creation, Fermionic_Annihilation, Quantum_Expectation
export is_normal_ordered
# export C, C_dg

using MLStyle




"""
Abstract Type for ALL Second-quantized Operators
"""
abstract type AbstractOp end

# todo! generalized to bosonic case
struct Fermionic_Creation{D} <: AbstractOp where {D<:Int64}
    state_idx_tuple::NTuple{D,Int64}
end
struct Fermionic_Annihilation{D} <: AbstractOp where {D<:Int64}
    state_idx_tuple::NTuple{D,Int64}
end

Base.show(io::IO, op::Fermionic_Creation) = print(io, "c†_{$(op.state_idx_tuple)}")
Base.show(io::IO, op::Fermionic_Annihilation) = print(io, "c_{$(op.state_idx_tuple)}")
Base.show(io::IO, op_vec::Vector{AbstractOp}) = [print(io, op) for op in op_vec]
Base.show(io::IO, op_tuple::NTuple{N,AbstractOp}) where {N} = [print(io, op) for op in op_tuple]

# define multiplication for `AbstractOp` for construction of operator expression
Base.:*(op1::AbstractOp, op2::AbstractOp) = [op1, op2]
Base.:*(op::AbstractOp, ops::Vector{AbstractOp}) = [op, ops...]
Base.:*(ops::Vector{<:AbstractOp}, op::AbstractOp) = [ops..., op]
Base.:*(ops_lhs::Vector{<:AbstractOp}, ops_rhs::Vector{<:AbstractOp}) = [ops_lhs..., ops_rhs...]

# define equality operator for `AbstractOp` for comparison and inclusion/outside-colletion checking
Base.:(==)(op1::AbstractOp, op2::AbstractOp) = begin
    @match (op1, op2) begin
        (op1::T1 where {T1}, op2::T2 where {T2}) => begin
            if T1 != T2
                return false
            else
                return op1.state_idx_tuple == op2.state_idx_tuple
            end
        end
    end
end


# "constructor method for `Fermionic_Creation`"
# function C(state_idx_tuple::NTuple{D,Int64}) where {D<:Int64}
#     return Fermionic_Creation{D}(state_idx_tuple)
# end

# "constructor function for `Fermionic_Annihilation`"
# function C_dg(state_idx_tuple::NTuple{D,Int64}) where {D<:Int64}
#     return Fermionic_Annihilation{D}(state_idx_tuple)
# end


"determine if a given expression of operators (as a `Vector{AbstractOp}`) is normal ordered"
function is_normal_ordered(op_vec::Vector{<:AbstractOp})::Bool
    N = length(op_vec)
    if isodd(N)
        return false
    else
        for i in 1:(N-1)
            op1 = op_vec[i]
            op2 = op_vec[i+1]
            if op1 isa Fermionic_Annihilation && op2 isa Fermionic_Creation
                return false
            end
        end
        true
    end
end
"add a method for single `AbstractOp`: always return `false`"
is_normal_ordered(op::AbstractOp)::Bool = false



struct Quantum_Expectation <: Number
    ops::Vector{AbstractOp}
end
Base.show(io::IO, qe::Quantum_Expectation) = print(io, "⟨", qe.ops, "⟩")
# also define equality operator for `Quantum_Expectation` for comparison and inclusion/outside-colletion checking (inherit from equality checking from `AbstractOp`)
# Base.:(==)(qm_exp1::Quantum_Expectation, qm_exp2::Quantum_Expectation) = qm_exp1.ops == qm_exp2.ops
Base.:(==)(qm_exp_set1::Set{Quantum_Expectation}, qm_exp_set2::Set{Quantum_Expectation}) = begin
    @show "using my method!"
    if length(qm_exp_set1) != length(qm_exp_set2)
        return false
    else
        return collect(qm_exp_set1) == collect(qm_exp_set2)
    end
end


begin
    # C1_dg = Fermionic_Creation((:i, :a))
    # C2 = Fermionic_Annihilation((:j, :a))
    # C3_dg = Fermionic_Creation((:k, :a))
    # C4 = Fermionic_Annihilation((:l, :a))

    # @show is_normal_ordered(C1_dg * C3_dg)
    # @show is_normal_ordered(C1_dg * C2)
    # @show is_normal_ordered(C2 * C1_dg)

    # @show C1_dg * C2 == [C1_dg, C2]
    # @show Set([C1_dg * C2, C3_dg * C4]) == Set([C3_dg * C4, C1_dg * C2])
    # @show [Quantum_Expectation(C1_dg * C2), Quantum_Expectation(C3_dg * C4)] == [Quantum_Expectation(C3_dg * C4), Quantum_Expectation(C1_dg * C2)]
    # @show typeof(Set([Quantum_Expectation(C1_dg * C2), Quantum_Expectation(C3_dg * C4)]))

    # Base.:(==)(qm_exp_set1::Set{Quantum_Expectation}, qm_exp_set2::Set{Quantum_Expectation}) = begin
    #     @show state_idx_tuple_set1 = Set([[op.state_idx_tuple for op in qm_exp.ops] for qm_exp in qm_exp_set1])
    #     @show state_idx_tuple_set2 = Set([[op.state_idx_tuple for op in qm_exp.ops] for qm_exp in qm_exp_set2])
    #     return state_idx_tuple_set1 == state_idx_tuple_set2
    # end

    # @show Set([Quantum_Expectation(C1_dg * C2), Quantum_Expectation(C3_dg * C4)]) == Set([Quantum_Expectation(C3_dg * C4), Quantum_Expectation(C1_dg * C2)])
    # # @show permutations([Quantum_Expectation(C1_dg * C2), Quantum_Expectation(C3_dg * C4)], 2) |> collect |> sort
    # # @show permutations([Quantum_Expectation(C3_dg * C4), Quantum_Expectation(C1_dg * C2)], 2) |> collect |> sort
end


end # end module
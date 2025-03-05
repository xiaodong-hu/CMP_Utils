"""
Wilson-loop Method for Computing the Chern Number of 2D Bloch States
---
Basically we sum over all fluxes enclosed by each plaquette within BZ. `∑_k U_x(k)U_y(k+Δk_x)U_x(k+Δk_x+Δk_y)U_y(k+Δk_y)` where `U_x(k) = ⟨ψ(k)|ψ(k+Δk_x)⟩` and `U_y(k) = ⟨ψ(k)|ψ(k+Δk_y)⟩`.

- Args:
    - `ψ_list::Vector{Vector{ComplexF64}}`: list of Bloch states
    - `k_int_list::Vector{Vector{Int}}`: list of integer-valued momentum points
- Named Args:
    - `sample_size::Vector{Int}`: sample size of the lattice
    - `show_each_loop_flux::Bool=false`: whether to show the flux of each Wilson loop

"""
function Chern_number_with_wilson_loop_method(ψ_list::Vector{Vector{ComplexF64}}, k_int_list::Vector{Vector{Int}}; sample_size::Vector{Int}, show_each_loop_flux::Bool=false, show_info::Bool=false)::Float64
    @assert length(unique(length.(ψ_list))) == 1 "Check Input: all Bloch states should have the same length to allow the Wilson-loop computation of Chern number"

    if any(sample_size .<= 1)
        @error "Check Input: sample size should be no less than 1 to allow the Wilson-loop computation of Chern number"
    end

    U_link = (k_int::AbstractArray{Int,1}, direction::Symbol) -> begin
        k_ind = findfirst(==(mod.(k_int, sample_size)), k_int_list)
        ψ_k = ψ_list[k_ind]

        k_plus_Δk_int = if direction == :x
            mod.(k_int .+ [1, 0], sample_size)  # periodic boundary condition
        elseif direction == :y
            mod.(k_int .+ [0, 1], sample_size)  # periodic boundary condition
        end

        k_plus_Δk_ind = findfirst(==(k_plus_Δk_int), k_int_list)
        ψ_k_plus_Δk = ψ_list[k_plus_Δk_ind]

        dot(ψ_k, ψ_k_plus_Δk) / abs(dot(ψ_k, ψ_k_plus_Δk))
    end

    # Wilson loop method
    loop_flux = 0.0
    for k_int in k_int_list
        # wilson loop along the path: k -> k+Δk_x -> k+Δk_x+Δk_y -> k+Δk_y -> k
        expr = U_link(k_int, :x) * U_link(k_int + [1, 0], :y) * U_link(k_int + [0, 1], :x)' * U_link(k_int, :y)'

        current_flux = log(expr)
        loop_flux += current_flux
        if show_each_loop_flux
            println("\t current loop flux: ", current_flux)
        end
    end

    C = loop_flux / (2π * im) |> real

    if show_info
        @info "Wilson-loop computation of Chern number: `C = $C"
    end
    return C
end


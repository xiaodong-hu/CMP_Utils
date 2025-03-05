module Complex_2D_Lattice_Module

include("lattice_new_new.jl") # include module `Lattice_Module`

export Complex_2D_Lattice, Complex_2D_Lattice_with_Flux


struct Torus_Sample
    ℓ::Float64 # magnetic length satisfying `2πℓ^2 * Nϕ = |ω1∧ω2|`
    Nϕ::Int # number of flux quantum per unit cell
    Ns::Int # total number of flux quantum piercing through the entire sample, equal to `Ns = Nϕ * lattice.n_site`
    twisted_phases_over_2π::Vector{Float64} # twisted phases along two directions `[φ1,φ2]` in units of `2π`
end

"""
Struct `Complex_2D_Lattice` Embeded on Torus
---
- Fields:
    - `lattice::Lattice_Module.Lattice{2,Float64}`: 2D real lattice
    - `complex_periods::Vector{ComplexF64}`: `ω1`, `ω2`. Note: we differ by a factor of two from many math literatures, where they usually use *half* periods instead of the *full* periods
    - `τ::ComplexF64`: modular parameter `τ = ω2 / ω1`
"""
struct Complex_2D_Lattice <: Lattice_Module.AbstractLattice
    lattice::Lattice_Module.Lattice{2,Float64} # 2D real lattice

    complex_periods::Vector{ComplexF64} # `ω1`, `ω2`. Note: we differ by a factor of two from many math literatures, where they usually use *half* periods instead of the *full* periods
    τ::ComplexF64 # modular parameter `τ = ω2 / ω1`   
end

"""
Constructor of `Complex_2D_Lattice`
---
- Named Args:
    - `sample_size::Vector{Int64}`: The size of the lattice in each direction.
    - `complex_periods::Vector{ComplexF64}`: `ω1`, `ω2`. Note: we differ by a factor of two from many math literatures, where they usually use *half* periods instead of the *full* periods
"""
function Complex_2D_Lattice(; sample_size::Vector{Int64}, complex_periods::Vector{ComplexF64}, twisted_phases_over_2π::Vector{Float64}=[0.0, 0.0])
    @assert length(sample_size) == 2
    @assert length(complex_periods) == 2
    basis_vec_list = collect.(reim.(complex_periods))
    lattice = Lattice_Module.Lattice(; sample_size=sample_size, basis_vec_list=basis_vec_list, twisted_phases_over_2π=twisted_phases_over_2π)

    (z1, z2) = complex_periods
    τ = z2 / z1
    @assert abs(imag(z1 * conj(z2))) == lattice.cell_volume # volume of the fundamental parallelogram formed by two complex numbers `z1=a+bi` and `z2=c+di`: `A=|ad-bc|=|z1*conj(z2)|`
    return Complex_2D_Lattice(
        lattice,
        complex_periods,
        τ
    )
end

"""
Struct `Complex_2D_Lattice_with_Flux` Embeded on Torus
---
- Fields:
    - `lattice::Lattice_Module.Lattice{2,Float64}`: 2D real lattice
    - `complex_periods::Vector{ComplexF64}`: `ω1`, `ω2`. Note: we differ by a factor of two from many math literatures, where they usually use *half* periods instead of the *full* periods
    - `τ::ComplexF64`: modular parameter `τ = ω2 / ω1`
    - `Nϕ::Int`: number of flux quantum per unit cell
    - `Ns::Int`: total number of flux quantum piercing through the entire sample, equal to `Ns = Nϕ * lattice.n_site`
    - `ℓ::Float64`: magnetic length satisfying the Dirac quantization condition `2πℓ^2Nϕ = |ω1∧ω2|`
    - `magnetic_translation_units::Vector{ComplexF64}`: the allowed *minimal* intervals of real-space translations along two directions `[δ1,δ2]` such that each minimal rectangular contains only a single quantum flux `|δ1∧L2|=2πℓ^2` or `|δ2∧L1|=2πℓ^2`
"""
struct Complex_2D_Lattice_with_Flux <: Lattice_Module.AbstractLattice
    lattice::Lattice_Module.Lattice{2,Float64} # 2D real lattice

    complex_periods::Vector{ComplexF64} # `ω1`, `ω2`. Note: we differ by a factor of two from many math literatures, where they usually use *half* periods instead of the *full* periods
    τ::ComplexF64 # modular parameter `τ = ω2 / ω1`

    Nϕ::Int # number of flux quantum per unit cell
    Ns::Int # total number of flux quantum piercing through the entire sample, equal to `Ns = Nϕ * lattice.n_site`
    ℓ::Float64 # magnetic length satisfying the Dirac quantization condition `2πℓ^2Nϕ = |ω1∧ω2|`
    magnetic_translation_units::Vector{ComplexF64} # the allowed *minimal* intervals of real-space translations along two directions `[δ1,δ2]` such that each minimal rectangular contains only a single quantum flux `|δ1∧L2|=2πℓ^2` or `|δ2∧L1|=2πℓ^2`
end


"""
Constructor of `Complex_2D_Lattice_with_Flux`
---
- Named Args:
    - `sample_size::Vector{Int64}`: The size of the lattice in each direction.
    - `complex_periods::Vector{ComplexF64}`: `ω1`, `ω2`. Note: we differ by a factor of two from many math literatures, where they usually use *half* periods instead of the *full* periods
    - `Nϕ::Int=1`: number of flux quantum per unit cell
    - `twisted_phases_over_2π::Vector{Float64}=[0.0, 0.0]`: twisted phases along two directions `[φ1,φ2]` in units of `2π`
"""
function Complex_2D_Lattice_with_Flux(; sample_size::Vector{Int64}, complex_periods::Vector{ComplexF64}, Nϕ::Int=1, twisted_phases_over_2π::Vector{Float64}=[0.0, 0.0])
    @assert length(sample_size) == 2
    @assert length(complex_periods) == 2
    basis_vec_list = collect.(reim.(complex_periods))
    lattice = Lattice_Module.Lattice(; sample_size=sample_size, basis_vec_list=basis_vec_list, twisted_phases_over_2π=twisted_phases_over_2π)

    (z1, z2) = complex_periods
    τ = z2 / z1

    Ns = reduce(*, sample_size) * Nϕ
    ℓ = sqrt(lattice.cell_volume / (2π * Nϕ)) # Dirac quantization condition `2πℓ^2Nϕ = |ω1∧ω2|`
    magnetic_translation_units = complex_periods ./ reverse(sample_size) # only when `δ1=Nϕ/N2 * ω1` and `δ2=Nϕ/N1 * ω2` would we have the desired quantization condition `|δ1∧L2|=2πℓ^2` or `|δ2∧L1|=2πℓ^2`

    return Complex_2D_Lattice_with_Flux(
        lattice,
        complex_periods,
        τ, Nϕ,
        Ns,
        ℓ,
        magnetic_translation_units
    )
end

end # module
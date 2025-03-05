using CMP_Utils
using LinearAlgebra
using Test


@testset verbose = true showtiming = true "Module `UniformGrids_Module`" begin
    sample_size = [8, 7]
    basis_vec_list = [[1.0, 0.1], [0.2, 2.0]]
    twisted_phases_over_2π = [0.1, 0.2]

    r_data = CMP_Utils.Uniform_Grids(; sample_size=sample_size, basis_vec_list=basis_vec_list, twisted_phases_over_2π=[0.0, 0.0]) # real space lattice will never been shifted by the twisted phases
    k_data = CMP_Utils.Uniform_Grids(; sample_size=sample_size, basis_vec_list=CMP_Utils.dual_basis_vec_list(basis_vec_list), twisted_phases_over_2π=twisted_phases_over_2π)

    @testset "`a_i ⋅ b_j = δ_{ij}`" begin
        @test dot(r_data.basis_vec_list[1], k_data.basis_vec_list[2]) ≈ 0.0
        @test dot(r_data.basis_vec_list[1], k_data.basis_vec_list[1]) ≈ 2π
    end


    @testset "twisted boundary condition" begin
        for k_cart in k_data.site_cart_list
            L1_cart = r_data.sample_size[1] * r_data.basis_vec_list[1]
            L2_cart = r_data.sample_size[2] * r_data.basis_vec_list[2]
            @test exp(im * dot(k_cart, L1_cart)) ≈ exp(im * 2π * twisted_phases_over_2π[1])
            @test exp(im * dot(k_cart, L2_cart)) ≈ exp(im * 2π * twisted_phases_over_2π[2])
        end
    end
end



"""
Compute Reciprocal Vector from the given Bravais Vector
---
it works for both 2D and 3D case
"""
function reciprocal_vec(brav_vec_list::Vector{<:Vector})
    dim = length(brav_vec_list[1])
    @assert all(brav_vec -> length(brav_vec) == dim, brav_vec_list)

    @match dim begin
        3 => begin
            a1 = push!(deepcopy(brav_vec_list[1]), 1)
            a2 = push!(deepcopy(brav_vec_list[2]), 1)
            a3 = push!(deepcopy(brav_vec_list[3]), 1)
            cell_volume = det([a1 a2 a3])

            b1 = cross(a2, a3)
            b2 = cross(a3, a1)
            b3 = cross(a1, a2)

            2 * pi * [b1, b2, b3] / cell_volume
        end
        2 => begin
            a1 = push!(deepcopy(brav_vec_list[1]), 1)
            a2 = push!(deepcopy(brav_vec_list[2]), 1)
            a3 = [0, 0, 1]
            cell_volume = det([a1 a2 a3])

            b1 = cross(a2, a3)[1:2]
            b2 = cross(a3, a1)[1:2]
            # b3 = cross(a1, a2)

            2 * pi * [b1, b2] / cell_volume
        end
    end
end
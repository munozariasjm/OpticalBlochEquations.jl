using LinearAlgebra
using BenchmarkTools

dgg = zeros(4, 4)

dge = [
    0. 0 1 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0
]

dee = [
    -1. 0 0 0;
    0 -1 0 0;
    0 0 -1 0;
    0 0 0 -1
]

Z = zeros(4, 4)

d  = [dgg Z; Z dee]
d′ = [Z dge; Z Z]

ρ = [
    1. 1 1 1 1 1 0 1;
    1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1;
    0 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 0
]
ρ₀ = similar(ρ); ρ₀ .= 0
ρ_tmp = similar(ρ); ρ_tmp .= 0

D = (d' * d)::Array{Float64, 2}
d′t = transpose(d)

function test1!(ρ₀, ρ, ρ_tmp, d, d′, d′t, D)
    mul!(ρ_tmp, ρ, d′t)
    mul!(ρ₀, d′, ρ_tmp)
    mul!(ρ₀, D, ρ, -1/2, 0)
    mul!(ρ₀, ρ, D, -1/2, 0)
    return nothing
end

function test2!(ρ₀, ρ, ρ_tmp, d, d′, d′t, D)
    BLAS.gemm!('N', 'T', 1., ρ,  d′, 1., ρ_tmp)
    BLAS.gemm!('N', 'N', 1., ρ₀, d′, 1., ρ_tmp)
    BLAS.gemm!('N', 'N', -1/2, D, ρ, 1., ρ₀)
    BLAS.gemm!('N', 'N', -1/2, ρ, D, 1., ρ₀)
    return nothing
end

@benchmark test1!($ρ₀, $ρ, $ρ_tmp, $d, $d′, $d′t, $D) evals=1
@benchmark test2!($ρ₀, $ρ, $ρ_tmp, $d, $d′, $d′t, $D) evals=1

test2!(ρ₀, ρ, ρ_tmp, d, d′, d′t, D); ρ₀

function test3!(t, d, ρ)

    @inbounds for ga in 1:4
        @inbounds for eb in 5:8
            t[ga, eb] = -ρ[ga, eb] / 2
            t[eb, ga] = t[ga, eb]
        end
    end

    @inbounds for ea in 5:8
        @inbounds for eb in 5:8
            t[ea, eb] = -ρ[ea, eb]
        end
    end

    @inbounds for ga in 1:4
        @inbounds for gb in 1:4
            @inbounds for ec′ in 5:8
                @inbounds for ec′′ in 5:8
                    t[ga, gb] += d[ec′, ga] * d[gb, ec′′] * ρ[ec′, ec′′]
                end
            end
        end
    end

    return nothing
end

@benchmark test3!(t_, $d, $ρ) setup=(t_ = zeros(8, 8)) evals=1

@time test3!(t, d, ρ)

const cart2sph = @SMatrix [
    +1/√2 +im/√2 0;
    0 0 1
    -1/√2 +im/√2 0;
]
export cart2sph

const sph2cart = inv(cart2sph)
export sph2cart

const x̂ = @SVector [1,0,0]
const ŷ = @SVector [0,1,0]
const ẑ = @SVector [0,0,1]
export x̂, ŷ, ẑ
export x̂, ŷ, ẑ

const ϵ₊ = SVector{3, ComplexF64}(-1/√2, -im/√2, 0) # in Cartesian representation
const ϵ₋ = SVector{3, ComplexF64}(+1/√2, -im/√2, 0)
const ϵ₀ = SVector{3, ComplexF64}(0.0, 0.0, 1.0)
const ϵ = [ϵ₋, ϵ₀, ϵ₊]
export ϵ₊, ϵ₋, ϵ₀
export ϵ

const σ⁻ = @SVector [1.0, 0.0, 0.0]
const σ⁺ = @SVector [0.0, 0.0, 1.0]
export σ⁻, σ⁺

const qs = @SVector [-1, 0, 1]
export qs
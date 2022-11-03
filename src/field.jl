using LoopVectorization: @turbo
using StructArrays: StructVector

"""
    Field([])
"""
struct Field{T,F}
    k::SVector{3, T}                        # k-vector
    ϵ::F                                    # update function to update `ϵ` according to the time `t`
    ϵ_val::SVector{3, Complex{T}}           # polarization
    ω::T                                    # frequency
    s::T                                    # saturation parameter
    re::T                                   
    im::T
    kr::T                                   # current value of `k ⋅ r`
    E::SVector{3, Complex{T}}               # current value of the field
end
function Field(k, ϵ, ω, s)
    _zero = zero(Float64)
    Field{Float64, typeof(ϵ)}(k, ϵ, SVector(_zero,_zero,_zero), ω, s, _zero, _zero, _zero, SVector(_zero,_zero,_zero))
end
function Field(T, k, ϵ, ω, s)
    _zero = zero(T)
    Field{T, typeof(ϵ)}(k, ϵ, SVector(_zero,_zero,_zero), ω, s, _zero, _zero, _zero, SVector(_zero,_zero,_zero))
end
export Field

function update_fields!(fields::StructVector{Field{T,F}}, r, t) where {T,F}
    # Fields are represented as ϵ_q * exp(i(kr - ωt)), where ϵ_q is in spherical coordinates
    for i ∈ eachindex(fields)
        k = fields.k[i]
        fields.kr[i] = k ⋅ r
        fields.ϵ_val[i] = fields.ϵ[i](t)
    end
    @turbo for i ∈ eachindex(fields)
        fields.im[i], fields.re[i] = sincos(- fields.kr[i] - fields.ω[i] * t)
    end
    for i ∈ eachindex(fields)
        val = (fields.re[i] + im * fields.im[i]) .* fields.ϵ_val[i]
        fields.E[i] = val
    end
    return nothing
end
export update_fields!
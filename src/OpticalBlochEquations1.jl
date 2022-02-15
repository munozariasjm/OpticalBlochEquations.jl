module OpticalBlochEquations

using StaticArrays
using StructArrays
using Parameters
using HalfIntegers
using Unitful
using LinearAlgebra
using WignerSymbols
using LoopVectorization

macro with_unit(arg1, arg2)
    arg2 = @eval @u_str $arg2
    return convert(Float64, upreferred(eval(arg1) .* arg2).val)
end
export @with_unit

macro params(fields_tuple)
    fields = fields_tuple.args
    esc(
        quote
            NamedTuple{Tuple($fields)}(($fields_tuple))
        end)
end
export @params

const sph2cart = @SMatrix [
    -1/√2 +1/√2 0;
    +im/√2 +im/√2 0;
    0 0 1;
]
export sph2cart

const ϵ₊ = @SVector [-1/√2, -im/√2, 0]
const ϵ₋ = @SVector [ 1/√2, -im/√2, 0]
const ϵ₀ = @SVector [    0,      0, 1]
const ϵ = [ϵ₋, ϵ₀, ϵ₊]
export ϵ₊, ϵ₋, ϵ₀
export ϵ

const qs = @SVector [-1, 0, 1]
export qs

x̂ = @SVector [1,0,0]
ŷ = @SVector [0,1,0]
ẑ = @SVector [0,0,1]
#export x̂, ŷ, ẑ

# function make_hamiltonian(states::Vector(State), lasers::Vector(Laser))
#     H = Hamiltonian(; states)
#     return H
# end

@with_kw struct Laser1
    k::SVector{3, Float64}      # k-vector
    e::SVector{3, ComplexF64}   # polarization
    ω::Float64                  # frequency
    s::Float64                  # saturation parameter
    fre::SVector{3, Float64}
    fim::SVector{3, Float64}
    Hq::Vector{Array{Float64, 2}}
end

@with_kw struct Field
    k::SVector{3, Float64}      # k-vector
    e::SVector{3, ComplexF64}   # polarization
    ω::Float64                  # frequency
    s::Float64                  # saturation parameter
    fre::SVector{3, Float64}
    fim::SVector{3, Float64}
    Hqm::Array{Float64, 2}
    Hq0::Array{Float64, 2}
    Hqp::Array{Float64, 2}
end

@with_kw struct State
    F::HalfInt                          # angular quantum number
    m::HalfInt                          # projection of angular quantum number
    ω::Float64                          # frequency
    μ::Float64                          # magnetic moment
    Γ::Union{Nothing, Float64}=nothing  # linewidth (defaults to `nothing` for ground states)
end
export State

@with_kw struct Hamiltonian12
    states::Vector{State}
    H::StructArray{<:Complex} = StructArray(zeros(ComplexF64, length(states), length(states)))
    Hsubs::Vector{StructArray{<:Complex}} = []
end
export Hamiltonian12

function define_laser(k, e, ω, s)
    fre = zeros(Float64, 3)
    fim = zeros(Float64, 3)
    for q in eachindex(qs)
        dotted = e ⋅ ϵ[q]
        fre[q] = real(dotted)
        fim[q] = imag(dotted)
    end
    Hq = [zeros(Float64, (12, 12)) for _ in qs]
    return Laser1(k, e, ω, s, fre, fim, Hq)
end
export define_laser

function define_field(k, e, ω, s)
    fre = zeros(Float64, 3)
    fim = zeros(Float64, 3)
    for q in eachindex(qs)
        dotted = e ⋅ ϵ[q]
        fre[q] = real(dotted)
        fim[q] = imag(dotted)
    end
    Hqm = zeros(Float64, (3, 3))
    Hq0 = zeros(Float64, (3, 3))
    Hqp = zeros(Float64, (3, 3))
    return Field(k, e, ω, s, fre, fim, Hqm, Hq0, Hqp)
end
export define_field

@with_kw struct Manifold
    F::HalfInt                          # angular quantum number
    states::Vector{State}
end
manifold(; F, ω, μ, Γ=nothing) = Manifold(F, [State(F=F, m=m, ω=ω, μ=μ, Γ=Γ) for m in -F:F])
export manifold

# Structure for quantum jumps from state `s` to state `s′` with rate `r`
struct Jump
    s ::Int64
    s′::Int64
    r ::Float64
end

roundmult(val, prec) = (inv_prec = 1 / prec; round(val * inv_prec) / inv_prec)
function round_freq(ω, Γ)
    ω_min = Γ * 1e-2
    return roundmult(ω, ω_min)
end

function schrödinger(states, lasers, d, f, p)

    n_states = length(states)
    n_lasers = length(lasers)

    lasers = StructArray(lasers)

    r = @SVector [0.,0.,0.]
    v = @SVector [0.,0.,0.]

    type_complex = ComplexF64
    type_real    = Float64

    H = StructArray( zeros(type_complex, n_states, n_states) )

    # Define the optical Hamiltonian; it has dimensions (k, q)
    for i ∈ 1:n_states, j ∈ i:n_states
        for k in eachindex(lasers)
            lasers[k].Hqm[i,j] += d[1,i,j] * (lasers[k].e ⋅ ϵ[1])
            lasers[k].Hq0[i,j] += d[2,i,j] * (lasers[k].e ⋅ ϵ[2])
            lasers[k].Hqp[i,j] += d[3,i,j] * (lasers[k].e ⋅ ϵ[3])
            lasers[k].Hqm[j,i] += d[1,i,j] * (lasers[k].e ⋅ ϵ[1])
            lasers[k].Hq0[j,i] += d[2,i,j] * (lasers[k].e ⋅ ϵ[2])
            lasers[k].Hqp[j,i] += d[3,i,j] * (lasers[k].e ⋅ ϵ[3])
        end
    end

    ψ = zeros(type_complex, n_states)
    dψ = deepcopy(ψ)
    ψ[1] = 1
    ψ_soa = StructArray(ψ)
    dψ_soa = StructArray(dψ)

    p = @params (H, ψ_soa, dψ_soa, states, lasers, r, v, f, p)

    return (dψ, ψ, p)
end
export schrödinger

function obe(states, lasers, d, ρ)

    n_states = length(states)
    n_lasers = length(lasers)

    lasers = StructArray(lasers)

    r = @SVector [0., 0., 0.]
    v = @SVector [0., 0., 0.]

    type_complex = ComplexF64
    type_real = Float64

    H       = StructArray( zeros(type_complex, n_states, n_states) )
    H_adj   = StructArray( zeros(type_complex, n_states, n_states) )
    # HJ      = zeros(type_real, n_states, n_states)
    HJ      = zeros(Float64, n_states)
    Hₒ      = [zeros(type_real, n_states, n_states) for l in lasers, q in qs]

    # Define the optical Hamiltonian; it has dimensions (k, q)
    for k in eachindex(lasers)
        for s in eachindex(states), s′ in s:n_states
            q = Int64(states[s′].m - states[s].m)
            if abs(q) <= 1
                if !iszero(lasers[k].e ⋅ ϵ[q+2])
                    # Hₒ[k, q+2][s, s′] = d[s, s′, q+2]
                    # Hₒ[k, q+2][s′, s] = d[s, s′, q+2]
                    lasers[k].Hq[q+2][s, s′] = d[s, s′, q+2]
                    lasers[k].Hq[q+2][s′, s] = d[s, s′, q+2]
                end
            end
        end
    end

    # Define the magnetic Hamiltonian; it has dimensions (q)
    # μ = (s, s′, q) -> - s.g * (-1)^(s.F - s′.m) * sqrt(s.F * (s.F + 1) * (2s.F + 1))
    #     * wigner3j(s.F, 1, s′.F, -s.m, q, s′.m)
    # for s in eachindex(states), s′ in eachindex(states), q in qs
    #     if states[s].F == states[s].F # States only mix if they belong to the same F state
    #         Hₘ[q+2][s, s′] = im * (-1)^q * μ(states[s], states[s′], -q)
    #     end
    # end

    # Construct an array containing all jump operators, as defined by `d`
    Js = Array{Jump}(undef, 0)
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s, s′, q+2]
        if dme != 0 & (states[s′].ω < states[s].ω) # only energy-allowed jumps are generated
            J = Jump(s, s′, dme)
            push!(Js, J)
        end
    end

    for J in Js
        # Adds the term - (iħ / 2) ∑ᵢ Jᵢ† Jᵢ
        # We assume jumps take the form Jᵢ = sqrt(Γ)|g⟩⟨e| such that Jᵢ† Jᵢ = Γ|e⟩⟨e|
        HJ[J.s] -= 0.5 * J.r^2
    end

    ω = [s.ω for s in states]
    eiωt  = StructArray(zeros(type_complex, n_states))

    dρ = deepcopy(ρ)
    ρ_soa = StructArray(ρ)
    dρ_soa = StructArray(dρ)

    # B(r) = [0,0,0]

    Γs = [s.Γ for s in states]
    Γ = maximum(filter(x -> x != nothing, Γs))

    conj_mat = ones(Float64, n_states, n_states)
    for i in 1:n_states, j in 1:n_states
        if j < i
            conj_mat[i,j] = -1
        end
    end

    # Allocate some temporary arrays
    A12 = zeros(type_real, n_states, n_states)
    B12 = zeros(type_real, n_states, n_states)
    T1 = zeros(type_real, n_states, n_states)
    T2 = zeros(type_real, n_states, n_states)
    tmp1 = StructArray(zeros(ComplexF64, n_states, n_states))
    tmp2 = StructArray(zeros(ComplexF64, n_states, n_states))

    p = @params (H, HJ, ρ_soa, dρ_soa, Js, ω, eiωt, states, lasers, r, v, Γ, conj_mat, Hₒ, A12, B12, T1, T2, tmp1, tmp2)

    return (dρ, ρ, p)
end
export obe

function update_H_schrödinger!(τ, v, lasers, H)

    @turbo for l in eachindex(lasers)
        s = lasers.s[l]
        Ho_laser_m = lasers.Hqm[l]
        Ho_laser_0 = lasers.Hq0[l]
        Ho_laser_p = lasers.Hqp[l]
        for i in eachindex(H)
            H.re[i] += s * (Ho_laser_m[i] + Ho_laser_0[i] + Ho_laser_p[i])
        end
    end

    return nothing
end

# function force(lasers, Γ)
#
#     x = - h * Γ / (2 * √2 )
#     for l in eachindex(lasers)


function update_H!(τ, v, Γ, lasers, H, conj_mat)

    @turbo H.re .= 0
    @turbo H.im .= 0
    r = v .* (τ / Γ)

    for l in eachindex(lasers)
        s = lasers.s[l]
        k = lasers.k[l]
        e = lasers.e[l]
        ω = lasers.ω[l]
        fre = lasers.fre[l]
        fim = lasers.fim[l]
        Ho_laser = lasers.Hq[l]
        c, v = sincos(k ⋅ r - ω * τ)
        x = sqrt(s) / (2 * √2)
        for q in eachindex(qs)
            freq = x * (fre[q] * v - fim[q] * c)
            fimq = x * (fre[q] * c + fim[q] * v)
            Ho_laser_q = Ho_laser[q]
            if (freq > 1e-10) || (fimq > 1e-10) || (freq < -1e-10) || (freq < -1e-10)
                @turbo for i in eachindex(H)
                    a = Ho_laser_q[i]
                    H.re[i] += freq * a
                    H.im[i] += conj_mat[i] * fimq * a
                end
            end
        end
    end

    # @inbounds @simd for q in qs
    #     β = (μB / (Γ * ħ)) * (B(r) ⋅ ϵ[q+2])
    #     axpy!(β, Hₘ[q+2], H)
    # end

    return nothing
end

function soa_to_base!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @turbo for i in eachindex(ρ, ρ_soa)
        ρ[i] = ρ_soa.re[i] + im * ρ_soa.im[i]
    end
    return nothing
end

function base_to_soa!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @turbo for i in eachindex(ρ, ρ_soa)
        ρ_soa.re[i] = real(ρ[i])
        ρ_soa.im[i] = imag(ρ[i])
    end
    return nothing
end

# function base_to_base!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
#     @turbo for i in eachindex(ρ, ρ_soa)
#         ρ[i] = ρ_soa.re[i] + im * ρ_soa.im[i]
#     end
#     return nothing
# end
#
# function soa_to_base_mult!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
#     @turbo for i in eachindex(ρ, ρ_soa)
#         ρ[i] *= ρ_soa.re[i] + im * ρ_soa.im[i]
#     end
#     return nothing
# end

function update_eiωt!(eiωt::StructArray{<:Complex}, ω::Array{<:Real}, τ::Real)
    @turbo for i ∈ 1:size(ω, 1)
        eiωt.im[i], eiωt.re[i] = sincos( ω[i] * τ )
    end
    return nothing
end

function Heisenberg!(ρ::StructArray{<:Complex}, eiωt::StructArray{<:Complex}, im_factor=1)
    @turbo for j ∈ 1:size(ρ, 2)
        jre = eiωt.re[j]
        jim = eiωt.im[j]
        for i ∈ 1:size(ρ, 1)
            ire = eiωt.re[i]
            iim = eiωt.im[i]
            cisim = im_factor * (iim * jre - ire * jim)
            cisre = ire * jre + iim * jim
            ρre_i = ρ.re[i,j]
            ρim_i = ρ.im[i,j]
            ρ.re[i,j] = ρre_i * cisre - ρim_i * cisim
            ρ.im[i,j] = ρre_i * cisim + ρim_i * cisre
        end
    end
    return nothing
end
export Heisenberg!

#
# function mul_times!(A::StructArray{<:Complex}, B::StructArray{<:Complex}, im_factor=1)
#     @turbo for i ∈ eachindex(A)
#         A.re[i] *= B.re[i]
#         A.im[i] *= im_factor * B.im[i]
#     end
# end

function im_commutator!(C, A, B, tmp1, tmp2, A_diag)
    @turbo for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cre = 0.0
        Cim = 0.0
        for k ∈ 1:size(A,2)
            Aik_re = A.re[i,k]
            Aik_im = A.im[i,k]
            Bkj_re = B.re[k,j]
            Bkj_im = B.im[k,j]
            # Multiply by -1
            Cre -= Aik_re * Bkj_re - Aik_im * Bkj_im
            Cim -= Aik_re * Bkj_im + Aik_im * Bkj_re
        end
        C.re[i,j] = Cre
        C.im[i,j] = Cim
    end

    adjoint!(tmp1, C)
    C_add_A!(C, tmp1, -1)

    mul_by_im!(C)

    mul_diagonal!(tmp1, B, A_diag)
    adjoint!(tmp2, tmp1)

    C_add_AplusB!(C, tmp1, tmp2, 1, 1)

end
export im_commutator!

# function im_commutator_old!(C::StructArray{<:Complex}, A::StructArray{<:Complex}, B::StructArray{<:Complex}, A12::Array{<:Real}, B12::Array{<:Real}, T1::Array{<:Real}, T2::Array{<:Real}, A_diag::Array{<:Real}, tmp1::StructArray{<:Complex}, tmp2::StructArray{<:Complex})
#     """
#     Computes `-im * [A, B] = -im * (A * B - B * A)` in-place by overwriting `C`.
#
#     3m complex multiplication (includes a factor -1):
#     -C_re = T2 - T1
#     -C_im = T1 + T2 - (A1 + A2)(B1 + B2)
#
#     (Should also try via a single matrix multiplication defined on a StructArray type.)
#     """
#
#     update_T₁T₂!(T1, T2, A, B)
#     C_copy_AplusB!(A12, A.re, A.im)
#     C_copy_AplusB!(B12, B.re, B.im)
#
#     C_copy_T₁T₂!(C, T1, T2)
#     #jmul!(C.im, A12, B12, -1, 1)
#
#     adjoint!(tmp1, C)
#     C_add_A!(C, tmp1, -1)
#
#     mul_by_im!(C)
#
#     mul_diagonal!(tmp1, B, A_diag)
#     adjoint!(tmp2, tmp1)
#
#     C_add_AplusB!(C, tmp1, tmp2, 1, 1)
#
#     return nothing
# end

# function jgemvavx!(y, A, x)
#     @turbo for i ∈ eachindex(y)
#         yi_re = 0.0
#         yi_im = 0.0
#         for j ∈ eachindex(x)
#             xj_re = x.re[j]
#             xj_im = x.im[j]
#             Aij_re = A.re[i,j]
#             Aij_im = A.im[i,j]
#             yi_re += Aij_re * xj_re - Aij_im * xj_im
#             yi_im -= Aij_re * xj_im + Aij_im * xj_re
#         end
#         y.re[i] = yi_im
#         y.im[i] = yi_re
#     end
#     return y
# end

function ψ!(dψ, ψ, p, τ)

    # @avx p.H .= 0
    # @avx p.ψ_soa .= ψ
    base_to_soa!(ψ, p.ψ_soa)
    # Update the Hamiltonian according to the new time τ
    p.f(p.H, τ, p.p)
    # update_H_schrödinger!(τ, p.v, p.lasers, p.H)
    jgemvavx!(p.dψ_soa, p.H, p.ψ_soa)
    soa_to_base!(dψ, p.dψ_soa)

    return nothing
end
export ψ!

function ρ!(dρ, ρ, p, τ)

    base_to_soa!(ρ, p.ρ_soa)
    #p.ρ_soa .= ρ

    # Update the Hamiltonian according to the new time τ
    update_H!(τ, p.v, p.Γ, p.lasers, p.H, p.conj_mat)

    # Apply a transformation to go to the Heisenberg picture
    update_eiωt!(p.eiωt, p.ω, τ)
    Heisenberg!(p.ρ_soa, p.eiωt)

    # Compute coherent evolution terms
    # im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.A12, p.B12, p.T1, p.T2, p.HJ, p.tmp1, p.tmp2)
    im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.tmp1, p.tmp2, p.HJ)

    # Add the terms ∑ᵢ Jᵢ ρ Jᵢ†
    # We assume jumps take the form Jᵢ = sqrt(Γ)|g⟩⟨e| such that JᵢρJᵢ† = Γ^2|g⟩⟨g|ρₑₑ
    @inbounds for i in eachindex(p.Js)
        J = p.Js[i]
        p.dρ_soa.re[J.s′, J.s′] += J.r^2 * p.ρ_soa.re[J.s, J.s]
    end

    # The left-hand side also needs to be transformed into the Heisenberg picture
    # To do this, we require the transpose of the `ω` matrix
    # Heisenberg!(p.dρ_soa, p.ω_trans, τ)
    Heisenberg!(p.dρ_soa, p.eiωt, -1)
    soa_to_base!(dρ, p.dρ_soa)

    return nothing
end
export ρ!

function C_copy_AplusB!(C::Array{<:Real}, A::Array{<:Real}, B::Array{<:Real}, α=1, β=1)
    @turbo for i in eachindex(A, B, C)
        C[i] = α * A[i] + β * B[i]
    end
end

function C_add_AplusB!(C::Array{<:Real}, A::Array{<:Real}, B::Array{<:Real}, α=1, β=1)
    @turbo for i in eachindex(A, B, C)
        C[i] += α * A[i] + β * B[i]
    end
end

function C_add_AplusB!(C::StructArray{<:Complex}, A::StructArray{<:Complex}, B::StructArray{<:Complex},
    α=1, β=1)
    @turbo for i in eachindex(A, B, C)
        C.re[i] += α * A.re[i] + β * B.re[i]
        C.im[i] += α * A.im[i] + β * B.im[i]
    end
end

function C_add_A!(C::StructArray{<:Complex}, A::StructArray{<:Complex}, factor=1)
    @turbo for i in eachindex(C, A)
        C.re[i] += factor * A.re[i]
        C.im[i] += factor * A.im[i]
    end
end

function C_copy_T₁T₂!(C::StructArray{<:Complex}, T1::Array{<:Real}, T2::Array{<:Real})
    @turbo for i ∈ eachindex(C)
        C.re[i] = T2[i] - T1[i]
        C.im[i] = T2[i] + T1[i]
    end
end

function mul_by_im!(C::StructArray{<:Complex})
    @turbo for i ∈ eachindex(C)
        a = C.re[i]
        C.re[i] = -C.im[i]
        C.im[i] = a
    end
end

function mul_diagonal!(C::StructArray{<:Complex}, A::StructArray{<:Complex}, B::Array{<:Real})
    """
    Computes A × B, where B is diagonal and real.
    """
    @turbo for j in axes(A,2)
        d = B[j]
        for i in axes(A,1)
            C.re[i,j] = d * A.re[i,j]
            C.im[i,j] = d * A.im[i,j]
        end
    end
end

function update_T₁T₂!(T1::Array{<:Real}, T2::Array{<:Real}, A::StructArray{<:Complex}, B::StructArray{<:Complex})
    @turbo for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        C1 = zero(eltype(A))
        C2 = zero(eltype(A))
        for k ∈ 1:size(A,2)
            C1 += A.re[i,k] * B.re[k,j]
            C2 += A.im[i,k] * B.im[k,j]
        end
        T1[i,j] = C1
        T2[i,j] = C2
    end
end

end

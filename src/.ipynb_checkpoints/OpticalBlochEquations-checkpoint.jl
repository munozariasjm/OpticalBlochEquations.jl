module OpticalBlochEquations

using StaticArrays
using StructArrays
using Parameters
using HalfIntegers
using Unitful
using LinearAlgebra
using WignerSymbols
using LoopVectorization
using PhysicalConstants.CODATA2018

const h = PlanckConstant.val
const ħ = h / 2π
const c = SpeedOfLightInVacuum.val
export h, ħ, c

macro with_unit(arg1, arg2)
    arg2 = @eval @u_str $arg2
    return convert(Float64, upreferred(eval(arg1) .* arg2).val)
end
export @with_unit

const cart2sph = @SMatrix [
    +1/√2 +im/√2 0;
    0 0 1
    -1/√2 +im/√2 0;
]
export cart2sph

const sph2cart = inv(cart2sph)
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

# @with_kw struct Hamiltonian12
#     states::Vector{State}
#     H::StructArray{<:Complex} = StructArray(zeros(ComplexF64, length(states), length(states)))
#     Hsubs::Vector{StructArray{<:Complex}} = []
# end
# export Hamiltonian12

# function make_hamiltonian(states::Vector(State), lasers::Vector(Laser))
#     H = Hamiltonian(; states)
#     return H
# end

@with_kw struct Laser
    k::SVector{3, Float64}         # k-vector
    ϵ_re::SVector{3, Float64}      # real part of polarization
    ϵ_im::SVector{3, Float64}      # imaginary part of polarization
    ω::Float64                     # frequency
    s::Float64                     # saturation parameter
    kr::Float64                    # value of k ⋅ r, defaults to 0
    E::MVector{3, ComplexF64}      # f = exp(i(kr - ωt))
    re::Float64
    im::Float64
    Laser(k, ϵ, ω, s) = new(k, real.(ϵ), imag.(ϵ), ω, s, 0.0, MVector(0.0, 0, 0), 1.0, 0.0)
end
export Laser

evaluate_field(field::Laser, k, r, ω, t) = (field.ϵ + im * field.ϵ_im) * cis(k * r - ω * t)
export evaluate_field

@with_kw struct Field{T<:Function}
    f::T                                                # function for the field
    ω::Float64                                          # angular frequency of field
    s::Float64                                          # saturation parameter
    ϵ::SVector{3, Float64}                              # polarization vector
    k::SVector{3, Float64} = zeros(Float64, 3)          # k-vector
    E::MVector{3, ComplexF64} = zeros(ComplexF64, 3)    # the actual field components
end
export Field

# @with_kw struct Field_Array1{T<:Function}
#     f::T                                            # function for the field
#     E::Vector{Float64} = zeros(ComplexF64, 3)   # the actual field components
# end
# export Field_Array1

@with_kw struct State
    F::HalfInt                          # angular quantum number
    m::HalfInt                          # projection of angular quantum number
    E::Float64                          # frequency
    μ::Float64                          # magnetic moment
    Γ::Union{Nothing, Float64}=nothing  # linewidth (defaults to `nothing` for ground states)
end
# export State

# function define_field(k, e, ω, s)
#     fre = zeros(Float64, 3)
#     fim = zeros(Float64, 3)
#     for q in eachindex(qs)
#         dotted = e ⋅ ϵ[q]
#         fre[q] = real(dotted)
#         fim[q] = imag(dotted)
#     end
#     Hqm = zeros(Float64, (3, 3))
#     Hq0 = zeros(Float64, (3, 3))
#     Hqp = zeros(Float64, (3, 3))
#     return Field(k, e, ω, s, fre, fim, Hqm, Hq0, Hqp)
# end
# export define_field

@with_kw struct Manifold
    F::HalfInt                          # angular quantum number
    states::Vector{State}
end
manifold(; F, E, μ, Γ=nothing) = Manifold(F, [State(F=F, m=m, E=E, μ=μ, Γ=Γ) for m in -F:F])
export manifold

# Structure for quantum jumps from state `s` to state `s′` with rate `r`
struct Jump
    s ::Int64
    s′::Int64
    r ::Float64
end

# Round `val` to the nearest multiple of `prec`
round_to_mult(val, prec) = (inv_prec = 1 / prec; round.(val * inv_prec) / inv_prec)

function round_freq(ω, Γ, freq_res)
    ω_min = freq_res * Γ
    return round_to_mult(ω, ω_min)
end
export round_freq

function round_vel(v, λ, Γ, freq_res)
    v_min = freq_res * Γ * λ / 2π
    return round_to_mult(v, v_min)
end
export round_vel

function schrödinger(states, fields, d, Γ, freq_res)

    n_states = length(states)
    n_lasers = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)
    states, fields = round_freqs(states, fields, Γ, freq_res)
    
    r = @SVector [0.,0.,0.]
    v = @SVector [0.,0.,0.]

    type_complex = ComplexF64
    type_real    = Float64

    H       = StructArray( zeros(type_real, n_states, n_states) )
    H_adj   = StructArray( zeros(type_real, n_states, n_states) )
    # HJ      = zeros(type_real, n_states, n_states)
    HJ      = zeros(Float64, n_states)
    Hₒ      = [zeros(type_real, n_states, n_states) for l in fields, q in qs]
    
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

    p = (H = H, ψ_soa = ψ_soa, dψ_soa = dψ_soa, states = states, lasers = lasers, r = r, v = v, f = f, p = p)

    return (dψ, ψ, p)
end
export schrödinger

mutable struct Particle
    r0::SVector{3, Float64}
    r::SVector{3, Float64}
    v::SVector{3, Float64}
end

function round_freqs(states, fields, Γ, freq_res)
    """
    Rounds frequencies of state energies and fields by a common denominator.
    
    freq_res::Float: all frequencies are rounded by this value (in units of Γ)
    """

    for i in eachindex(fields)
        fields.ω[i] = round_freq(fields.ω[i], Γ, freq_res)
        fields.ω[i] /= Γ
    end
    for i in eachindex(states)
        states.E[i] = round_freq(states.E[i], Γ, freq_res)
        states.E[i] /= Γ
    end
    
    return (states, fields)
end

function obe(states, fields, d, d_mag, ρ, round_freqs, include_jumps; λ=nothing, Γ=nothing, freq_res=1e-2)

    # Γs = [s.Γ for s in states]
    # Γ = maximum(filter(x -> x != nothing, Γs))

    # λs = [c / s.E for s in states]
    # λs_nonzero = [c / s.E for s in states] .!= Inf
    # λ = 2π * minimum(λs[λs_nonzero])

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    if n_fields > 0
        fields = StructArray(fields)
    end
    
    if round_freqs
        states, fields = round_freqs(states, fields, Γ, freq_res)
    end
    
    # Convert to angular frequencies
    for i ∈ eachindex(states)
        states.E[i] *= 2π
    end

    type_complex = ComplexF64
    type_real = Float64

    H       = StructArray( zeros(type_complex, n_states, n_states) )
    H_adj   = StructArray( zeros(type_complex, n_states, n_states) )
    # HJ      = zeros(type_real, n_states, n_states)
    HJ      = zeros(Float64, n_states)
    Hₒ      = [zeros(type_real, n_states, n_states) for _ in fields, _ in qs]

    # Define the optical Hamiltonian; it has dimensions (k, q)
    # for k in eachindex(lasers)
    #     for s in eachindex(states), s′ in s:n_states
    #         q = Int64(states[s′].m - states[s].m)
    #         if abs(q) <= 1
    #             laser_ϵ = lasers[k].ϵ_re + im * lasers[k].ϵ_im
    #             print(laser_ϵ)
    #             print(ϵ[q+2])
    #             if !iszero(laser_ϵ ⋅ ϵ[q+2])
    #                 # Hₒ[k, q+2][s, s′] = d[s, s′, q+2]
    #                 # Hₒ[k, q+2][s′, s] = d[s, s′, q+2]
    #                 lasers[k].Hq[q+2][s, s′] = d[s, s′, q+2]
    #                 lasers[k].Hq[q+2][s′, s] = d[s, s′, q+2]
    #             end
    #         end
    #     end
    # end

    # Define the magnetic Hamiltonian; it has dimensions (q)
    # μ = (s, s′, q) -> - s.g * (-1)^(s.F - s′.m) * sqrt(s.F * (s.F + 1) * (2s.F + 1))
    #     * wigner3j(s.F, 1, s′.F, -s.m, q, s′.m)
    # H_magnetic = zeros(Float64, n_states, n_states)
    # for s in eachindex(states), s′ in eachindex(states), q in qs
    #     if states[s].F == states[s].F # States only mix if they belong to the same F state
    #         H_magnetic[q+2][s, s′] = im * (-1)^q * μ(states[s], states[s′], -q)
    #     end
    # end

    # Construct an array containing all jump operators, as defined by `d`
    Js = Array{Jump}(undef, 0)
    if include_jumps
        for s′ in eachindex(states), s in s′:n_states, q in qs
            dme = d[s, s′, q+2]
            # println(dme)
            if dme != 0 & (states[s′].E < states[s].E) # only energy-allowed jumps are generated
                J = Jump(s, s′, dme)
                push!(Js, J)
            end
        end

        for J in Js
            # Adds the term - (iħ / 2) ∑ᵢ Jᵢ† Jᵢ
            # We assume jumps take the form Jᵢ = sqrt(Γ)|g⟩⟨e| such that Jᵢ† Jᵢ = Γ|e⟩⟨e|
            HJ[J.s] -= 0.5 * J.r^2
        end
    end

    ω = [s.E for s in states]
    eiωt  = StructArray(zeros(type_complex, n_states))

    dρ = deepcopy(ρ)
    ρ_soa = StructArray(ρ)
    dρ_soa = StructArray(dρ)

    # B(r) = [0,0,0]

    conj_mat = ones(Float64, n_states, n_states)
    for i in 1:n_states, j in 1:n_states
        if j < i
            conj_mat[i,j] = -1
        end
    end

    # Allocate some temporary arrays
    tmp1 = StructArray(zeros(ComplexF64, n_states, n_states))
    tmp2 = StructArray(zeros(ComplexF64, n_states, n_states))

     # Compute indices to indicate nonzero values in d
    d_nnz_m = Int64[]
    d_nnz_0 = Int64[]
    d_nnz_p = Int64[]
    d_m = @view d[:,:,1]
    d_0 = @view d[:,:,2]
    d_p = @view d[:,:,3]
    for i in 1:n_states^2
        if d_m[i] != 0
            push!(d_nnz_m, i)
        end
        if d_0[i] != 0
            push!(d_nnz_0, i)
        end
        if d_p[i] != 0
            push!(d_nnz_p, i)
        end
    end
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    particle = Particle([0.,0,0], [0.,0,0], [0.,0,0])

    t0 = [0.0, 0.0]

    p = (H=H, HJ=HJ, ρ=ρ, dρ=dρ, ρ_soa=ρ_soa, dρ_soa=dρ_soa, Js=Js, ω=ω, eiωt=eiωt, 
         states=states, fields=fields, particle=particle, Γ=Γ, conj_mat=conj_mat, Hₒ=Hₒ, tmp1=tmp1, tmp2=tmp2, d=d, d_nnz=d_nnz, λ=λ, t0=t0, d_m=d_mag)

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

function derivative_force(p, ρ, τ)

    @unpack ρ_soa, lasers, Γ, d, d_nnz = p

    r = p.particle.v .* τ
    update_lasers!(r, lasers, τ)

    F = SVector(0, 0, 0)

    for q in 1:3

        ampl = SVector(0, 0, 0)
        @inbounds for i in 1:length(lasers)
            s = lasers.s[i]
            k = lasers.k[i]
            ω = lasers.ω[i]
            x = h * Γ * s / (4π * √2)
            ampl += k * x * (im * lasers.f_re_q[i][q] - lasers.f_im_q[i][q])
        end

        d_q = @view d[:,:,q]
        d_nnz_q = d_nnz[q]
        @inbounds for i in d_nnz_q
            F += ampl * d_q[i] * ρ_soa[i] + conj(ampl * d_q[i] * ρ_soa[i])
            # F += ampl * d_q[i] * ρ[i] + conj(ampl * d_q[i] * ρ[i])
        end
    end

    return real(F[1])
end
export derivative_force

function force(p, ρ, τ)

    @unpack fields, Γ, d, d_nnz = p

    r = p.particle.r0 + p.particle.v .* τ
    # r = p.particle.v .* τ
    update_fields!(fields, r, τ)

    base_to_soa!(ρ, p.ρ_soa)
    update_eiωt!(p.eiωt, p.ω, τ)
    Heisenberg!(p.ρ_soa, p.eiωt)

    F = SVector(0, 0, 0)

    for q in 1:3

        ampl = SVector(0, 0, 0)
        @inbounds for i in 1:length(fields)
            s = fields.s[i]
            k = fields.k[i]
            ω = fields.ω[i]
            # With SI units, should be x = -h * Γ * sqrt(s) / (2 * √2 * p.λ)
            x = sqrt(s) / (2 * √2)
            # x = h * Γ * sqrt(s) / (2 * √2 * p.λ)
            ampl += k * x * im * fields.E[i][q]
        end
        # Note h * Γ / λ = 2π ħ * Γ / Λ = ħ * k * Γ, so this has units units of ħ k Γ
        # println(ampl)
        d_q = @view d[:,:,q]
        d_nnz_q = d_nnz[q]
        # print(d_nnz_q)
        @inbounds for i in d_nnz_q
            # print(ampl * d_q[i] * p.ρ_soa[i] + conj(ampl * d_q[i] * p.ρ_soa[i]))
            F += ampl * d_q[i] * p.ρ_soa[i] + conj(ampl * d_q[i] * p.ρ_soa[i])
            # F += ampl * d_q[i] * ρ[i] + conj(ampl * d_q[i] * ρ[i])
        end
    end

    v = p.particle.v
    proj_F = real.(F) ⋅ v / norm(v)

#     return real(F[3])
    return proj_F
end
export force

function update_fields!(fields::StructVector{Laser}, r, t)
    for i in 1:length(fields)
        k = fields.k[i]
        fields.kr[i] = k ⋅ r
    end
    @turbo for i in 1:length(fields)
        fields.im[i], fields.re[i] = sincos(fields.kr[i] - fields.ω[i] * t)
    end
    for i in 1:length(fields)
        ϵ_re = fields.ϵ_re[i]
        ϵ_im = fields.ϵ_im[i]
        real_ = fields.re[i]
        imag_ = fields.im[i]
        real_part = ϵ_re * real_ - ϵ_im * imag_
        imag_part = ϵ_re * imag_ + ϵ_im * real_
        fields.E[i] .= real_part + im * imag_part
    end
    return nothing
end

# function update_fields!(fields::StructVector{Laser1}, r, t)
#     for i in 1:length(fields)
#         kr = fields.k[i] ⋅ r
#         ω = fields.ω[i]
#         fields.E[i] .= (fields.ϵ_re[i] + im * fields.ϵ_im[i]) * cis(kr) #* cos(ω * t)
#     end
#     return nothing
# end

function update_fields!(fields::StructVector{Field{T}}, r, t) where T
    """
    Fields must be specified as one of the following types:
    """
    for i in eachindex(fields)
        fields.E[i] .= fields.f[i](fields.ω[i], r, t)
    end
    return nothing
end
export update_fields!

function update_H!(τ, r, fields, H, conj_mat, d, d_nnz, d_m)
    """
    Anything uncommented in-line is from previous version.
    """

    update_fields!(fields, r, τ)

    @turbo for i in eachindex(H)
        H.re[i] = 0
        H.im[i] = 0
    end

    @inbounds for i in 1:length(fields)
        s = fields.s[i]
        x = sqrt(s) / (2 * √2)

        @inbounds for q in 1:3
            field_val = fields.E[i][q]
            freq = real(field_val) #lasers.f_re_q[i][q]
            fimq = imag(field_val) #lasers.f_im_q[i][q]
            if (freq > 1e-10) || (freq < -1e-10) || (fimq > 1e-10) || (fimq < -1e-10)
                # @turbo for i in 1:size(H,1), j in 1:size(H,1)
                #     H.re[i,j] += x * freq * d[i,j,q]
                #     H.im[i,j] += x * fimq * d[i,j,q] * conj_mat[i,j]
                # end
                # The approach below which iterates only over nonzero dipole moments is a bit faster
                d_nnz_q = d_nnz[q]
                d_q = @view d[:,:,q]
                @inbounds @simd for i in d_nnz_q
                    H.re[i] += x * freq * d_q[i]
                    H.im[i] += x * fimq * d_q[i] * conj_mat[i]
                end
            end
        end
    end

    # Add Zeeman interaction terms, need to optimize this
    for q ∈ 1:3
        for i ∈ axes(H,1), j ∈ axes(H,2)
            H[i,j] += d_m[i,j,q]
        end
    end

    # @inbounds @simd for q in qs
    #     β = (μB / (Γ * ħ)) * (B(r) ⋅ ϵ[q+2])
    #     axpy!(β, Hₘ[q+2], H)
    # end

    return nothing
end

function soa_to_base!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @inbounds for i in eachindex(ρ, ρ_soa)
        ρ[i] = ρ_soa.re[i] + im * ρ_soa.im[i]
    end
    return nothing
end

function base_to_soa!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @inbounds for i in eachindex(ρ, ρ_soa)
        ρ_soa.re[i] = real(ρ[i])
        ρ_soa.im[i] = imag(ρ[i])
    end
    return nothing
end

function update_eiωt!(eiωt::StructArray{<:Complex}, ω::Array{<:Real}, τ::Real)
    @turbo for i ∈ 1:size(ω, 1)
        eiωt.im[i], eiωt.re[i] = sincos( ω[i] * τ )
    end
    return nothing
end

function Heisenberg!(ρ::StructArray{<:Complex}, eiωt::StructArray{<:Complex}, im_factor=1)
    @inbounds for j ∈ 1:size(ρ, 2)
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

function ψ!(dψ, ψ, p, τ)

    # @avx p.H .= 0
    # @avx p.ψ_soa .= ψ
    base_to_soa!(ψ, p.ψ_soa)
    
    # Update the Hamiltonian according to the new time τ
    p.f(p.H, τ, p.p)
    
    jgemvavx!(p.dψ_soa, p.H, p.ψ_soa)
    
    soa_to_base!(dψ, p.dψ_soa)

    return nothing
end
export ψ!

function ρ!(dρ, ρ, p, τ)

    p.particle.r = p.particle.r0 + p.particle.v .* τ

    base_to_soa!(ρ, p.ρ_soa)
    #p.ρ_soa .= ρ

    # Update the Hamiltonian according to the new time τ
    update_H!(τ, p.particle.r, p.fields, p.H, p.conj_mat, p.d, p.d_nnz, p.d_m)

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
    # p.dρ .= dρ

    return nothing
end
export ρ!

function mat_to_vec!(ρ, ρ_vec)
    @turbo for i in eachindex(ρ)
        ρ_vec[i] = ρ[i]
    end
    return nothing
end
export mat_to_vec!

function mat_to_vec_minus1!(ρ, ρ_vec)
    @turbo for i in 1:(length(ρ)-1)
        ρ_vec[i] = ρ[i]
    end
    return nothing
end
export mat_to_vec_minus1!

# function ρ_and_force!(du, u, p, τ)

#     p.particle.r = p.particle.v .* τ

#     mat_to_vec_minus1!(u, p.ρ)
#     base_to_soa!(p.ρ, p.ρ_soa)
#     #p.ρ_soa .= ρ

#     # Update the Hamiltonian according to the new time τ
#     update_H!(τ, p.particle.r, p.lasers, p.H, p.conj_mat, p.d, p.d_nnz)

#     # Apply a transformation to go to the Heisenberg picture
#     update_eiωt!(p.eiωt, p.ω, τ)
#     Heisenberg!(p.ρ_soa, p.eiωt)

#     # Compute coherent evolution terms
#     # im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.A12, p.B12, p.T1, p.T2, p.HJ, p.tmp1, p.tmp2)
#     im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.tmp1, p.tmp2, p.HJ)

#     # Add the terms ∑ᵢ Jᵢ ρ Jᵢ†
#     # We assume jumps take the form Jᵢ = sqrt(Γ)|g⟩⟨e| such that JᵢρJᵢ† = Γ^2|g⟩⟨g|ρₑₑ
#     @inbounds for i in eachindex(p.Js)
#         J = p.Js[i]
#         p.dρ_soa.re[J.s′, J.s′] += J.r^2 * p.ρ_soa.re[J.s, J.s]
#     end

#     # The left-hand side also needs to be transformed into the Heisenberg picture
#     # To do this, we require the transpose of the `ω` matrix
#     # Heisenberg!(p.dρ_soa, p.ω_trans, τ)
#     Heisenberg!(p.dρ_soa, p.eiωt, -1)
#     soa_to_base!(p.dρ, p.dρ_soa)

#     mat_to_vec!(p.dρ, du)
#     du[end] = derivative_force(p.ρ, p, τ)
#     # u[end] = force(p.ρ, p, τ)

#     return nothing
# end
# export ρ_and_force!

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
import StaticArrays: SVector, MVector
import StructArrays: StructArray
import Parameters: @with_kw
import LoopVectorization: @turbo
import DifferentialEquations: ODEProblem, solve

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

@with_kw struct Laser
    k::SVector{3, Float64}         # k-vector
    ϵ_re::SVector{3, Float64}      # real part of polarization
    ϵ_im::SVector{3, Float64}      # imaginary part of polarization
    ϵ_cart::SVector{3, ComplexF64} # polarization in cartesian coordinates 
    ω::Float64                     # frequency
    s::Float64                     # saturation parameter
    kr::Float64                    # value of k ⋅ r, defaults to 0
    E::MVector{3, ComplexF64}      # f = exp(i(kr - ωt))
    re::Float64
    im::Float64
    Laser(k, ϵ, ω, s) = new(k, real.(ϵ), imag.(ϵ), sph2cart * ϵ, ω, s, 0.0, MVector(0.0, 0.0, 0.0), 1.0, 0.0)
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

# Structure for quantum jumps from state `s` to state `s′` with rate `r`
struct Jump
    s ::Int64
    s′::Int64
    q::Int64
    r ::ComplexF64
end

function schrödinger(particle, states, H₀, fields, d, d_mag, ψ, should_round_freqs; λ=1.0, freq_res=1e-2)

    period = 2π / freq_res

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    if n_fields > 0
        fields = StructArray(fields)
    end

    k = 2π / λ
    particle.r0 *= 2π
    # particle.v /= (Γ / k) # velocity is in units of Γ/k
    # Convert to angular frequencies
    for i ∈ eachindex(states)
        states.E[i] *= 2π
    end

    if should_round_freqs
        round_freqs!(states, fields, freq_res)
        particle.v = round_vel(particle.v, freq_res)
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = ComplexF64

    B = MVector(0.0, 0.0, 0.0)

    H = StructArray( zeros(type_complex, n_states, n_states) )

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    dψ = deepcopy(ψ)
    ψ_soa = StructArray(ψ)
    dψ_soa = StructArray(dψ)

    H₀ = StructArray(H₀)

    p = MutableNamedTuple(
        H=H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Jump[],
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, λ=λ, d_m=d_mag,
        period=period, B=B, k=k, freq_res=freq_res, H₀=H₀)

    return (dψ, ψ, p)
end
export schrödinger

@with_kw mutable struct Particle
    r0::MVector{3, Float64} = MVector(0.0, 0.0, 0.0)
    r::MVector{3, Float64}  = MVector(0.0, 0.0, 0.0)
    v::MVector{3, Float64}  = MVector(0.0, 0.0, 0.0)
end
export Particle

# Round `val` to the nearest multiple of `prec`
round_to_mult(val, prec) = (inv_prec = 1 / prec; round.(val * inv_prec) / inv_prec)

function round_freq(ω, freq_res)
    ω_min = freq_res
    return round_to_mult(ω, ω_min)
end
export round_freq

function round_vel(v, freq_res)
    v_min = freq_res
    return round_to_mult(v, v_min)
end
export round_vel

function round_v_by_norm(v, digits)
    v_norm = norm(v)
    return v ./ (v_norm / round(v_norm, digits=digits))
end

function round_freqs!(states, fields, freq_res)
    """
    Rounds frequencies of state energies and fields by a common denominator.
    
    freq_res::Float: all frequencies are rounded by this value (in units of Γ)
    """
    for i in eachindex(fields)
        fields.ω[i] = round_freq(fields.ω[i], freq_res)
    end
    for i in eachindex(states)
        states.E[i] = round_freq(states.E[i], freq_res)
    end
    return nothing
end

function round_params(p)
    round_freqs!(p.states, p.fields, p.freq_res)
    p.v .= round_vel(p.v, p.freq_res)
end
export round_params

function obe(particle, states, fields, d, d_m, should_round_freqs, include_jumps; λ=1.0, Γ=2π, freq_res=1e-2)

    period = 2π / freq_res

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    if n_fields > 0
        fields = StructArray(fields)
    end

    k = 2π / λ
    particle.r0 *= 2π #(1 / k)  # `r` is in units of 1/k
    particle.v /= (Γ / k) #(Γ / k)   # velocity is in units of Γ/k
    # Convert to angular frequencies
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    if should_round_freqs
        round_freqs!(states, fields, freq_res)
        particle.v = round_vel(particle.v, freq_res)
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = ComplexF64

    B = MVector(0.0, 0.0, 0.0)
    H = StructArray( zeros(type_complex, n_states, n_states) )

    # Construct an array containing all jump operators, as defined by `d`
    Js = Array{Jump}(undef, 0)
    if include_jumps
        for s′ in eachindex(states), s in s′:n_states, q in qs
            dme = d[s′, s, q+2]
            # println(dme)
            if dme != 0 & (states[s′].E < states[s].E) # only energy-allowed jumps are generated
                J = Jump(s, s′, q, dme)
                push!(Js, J)
            end
        end
    end

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    ρ_soa = StructArray(zeros(ComplexF64, n_states, n_states))
    dρ_soa = deepcopy(ρ_soa)

    # Allocate some temporary arrays
    H₀ = deepcopy(ρ_soa)
    tmp = deepcopy(ρ_soa)

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    p = (H=H, ρ_soa=ρ_soa, dρ_soa=dρ_soa, Js=Js, ω=ω, eiωt=eiωt,
         states=states, fields=fields, r0=r0, r=r, v=v, Γ=Γ, tmp=tmp, d=d, d_nnz=d_nnz, λ=λ, d_m=d_m,
         period=period, B=B, k=k, freq_res=freq_res, H₀=H₀)

    return p
end
export obe

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
        @inbounds for i ∈ d_nnz_q
            F += ampl * d_q[i] * ρ_soa[i] + conj(ampl * d_q[i] * ρ_soa[i])
        end
    end

    return real(F[1])
end
export derivative_force

function update_fields!(fields::StructVector{Laser}, r, t)
    # Fields are represented as ϵ_q * exp(i(kr - ωt)), where ϵ_q is in spherical coordinates
    for i ∈ eachindex(fields)
        k = fields.k[i]
        fields.kr[i] = k ⋅ r
    end
    @turbo for i ∈ eachindex(fields)
        # Compute exp(i(kr - ωt))
        fields.im[i], fields.re[i] = sincos(- fields.kr[i] - fields.ω[i] * t) # second term needed for Heisenberg picture
    end
    @simd for i ∈ eachindex(fields)
        field_value = fields.re[i] + im * fields.im[i]
        fields.E[i] .= field_value .* (fields.ϵ_re[i] .+ im .* fields.ϵ_im[i])
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

function update_H!(τ, r, H₀, fields, H, d, d_nnz, B, d_m, Js, ω, Γ)
    """
    Anything uncommented in-line is from previous version.
    """
    update_fields!(fields, r, τ)

    @turbo for i in eachindex(H)
        H.re[i] = H₀.re[i]
        H.im[i] = H₀.im[i]
    end

    @inbounds for i ∈ eachindex(fields)
        s = fields.s[i]
        ω_field = fields.ω[i]
        x = sqrt(s) / (2 * √2)
        @inbounds for q ∈ 1:3
            E_i_q = fields.E[i][q]
            if norm(E_i_q) > 1e-10
                d_nnz_q = d_nnz[q]
                d_q = @view d[:,:,q]
                @inbounds @simd for cart_idx ∈ d_nnz_q
                    m, n = cart_idx.I
                    if abs(ω_field - (ω[n] - ω[m])) < 100 # Do not include very off-resonant terms (terms detuned by >10Γ)
                        val = x * E_i_q * d_q[m,n]
                        H[m,n] += val
                        H[n,m] += conj(val)
                    end
                end
            end
        end
    end

    for J ∈ Js
        rate = im * J.r^2 / 2
        H[J.s, J.s] -= rate
    end

    # Add Zeeman interaction terms, need to optimize this
    for q ∈ 1:3
        B_q = 2π * norm(B ⋅ ϵ[q]) # the amount of B-field projected into each spherical component, which are already written in a Cartesian basis
        for j ∈ axes(H,2), i ∈ axes(H,1)
            val = B_q * d_m[i,j,4-q]
            if j > i
                H[i,j] -= val
                H[j,i] -= conj(val)
            elseif i == j
                H[i,j] -= val
            end
        end
    end

    return nothing
end
export update_H!

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

function Heisenberg_ψ!(ψ::StructArray{<:Complex}, eiωt::StructArray{<:Complex}, im_factor=1)
    @turbo for i ∈ 1:size(ψ, 1)
        cisre = eiωt.re[i]
        cisim = im_factor * eiωt.im[i]
        ψ_i_re = ψ.re[i]
        ψ_i_im = ψ.im[i]
        ψ.re[i] = ψ_i_re * cisre - ψ_i_im * cisim
        ψ.im[i] = ψ_i_re * cisim + ψ_i_im * cisre
    end
    return nothing
end
export Heisenberg_ψ!

function im_commutator!(C, A, B, tmp)
    # Multiply C = A * B -- (H * ρ)
    # Add adjoint C = A * B + B† * A†
    # Multiply by "i"
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
        C.im[i,j] = Cim # not sure about this minus sign...
    end

    # Add adjoint, which is equivalent to ρ * H, then compute [H, ρ]
    adjoint!(tmp, C)
    C_add_A!(C, tmp, -1)
    mul_by_im!(C)

    # C .= -im * (A * B - B * A')

    # Add diagonal (jump) elements (ρ * H_jumps), add adjoint (H_jumps * ρ) as well
    # mul_diagonal!(tmp1, B, A_diag)
    # adjoint!(tmp2, tmp1)

    # Add everything together
    # C_add_AplusB!(C, tmp1, tmp2, 1, 1)

end
export im_commutator!

function ψ!(dψ, ψ, p, τ)

    base_to_soa!(ψ, p.ψ_soa)
    
    update_H!(τ, p.r, p.H₀, p.fields, p.H, p.d, p.d_nnz, p.B, p.d_m, p.Js, p.ω, nothing)
    
    update_eiωt!(p.eiωt, p.ω, τ)
    # Heisenberg_ψ!(p.ψ_soa, p.eiωt, -1)
    
    Heisenberg!(p.H, p.eiωt, -1)
    mul_by_im_minus!(p.ψ_soa)
    mul_turbo!(p.dψ_soa, p.H, p.ψ_soa)
    
    # Heisenberg_ψ!(p.dψ_soa, p.eiωt, -1)
    soa_to_base!(dψ, p.dψ_soa)

    return nothing
end
export ψ!

# Note that we specialize to the type of the parameter argument, `p`
function ρ!(dρ, ρ, p, τ)

    p.r .= p.r0 + p.v .* τ

    base_to_soa!(ρ, p.ρ_soa)

    # Update the Hamiltonian according to the new time τ
    update_H!(τ, p.r, p.H₀, p.fields, p.H, p.d, p.d_nnz, p.B, p.d_m, p.Js, p.ω, p.Γ)

    # Apply a transformation to go to the Heisenberg picture
    update_eiωt!(p.eiωt, p.ω, τ)
    Heisenberg!(p.ρ_soa, p.eiωt)

    # Compute coherent evolution terms
    im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.tmp)

    # Add the terms ∑ᵢ Jᵢ ρ Jᵢ†
    # We assume jumps take the form Jᵢ = sqrt(Γ)|g⟩⟨e| such that JᵢρJᵢ† = Γ^2|g⟩⟨g|ρₑₑ
    @inbounds for i ∈ eachindex(p.Js)
        J = p.Js[i]
        p.dρ_soa[J.s′, J.s′] += J.r^2 * p.ρ_soa[J.s, J.s]
        @inbounds for j ∈ (i+1):length(p.Js)
            J′ = p.Js[j]
            if J.q == J′.q
                val = J.r * J′.r * p.ρ_soa[J.s, J′.s]
                p.dρ_soa[J.s′, J′.s′] += val
                p.dρ_soa[J′.s′, J.s′] += conj(val)
            end
        end
    end

    # The left-hand side also needs to be transformed into the Heisenberg picture
    # To do this, we require the transpose of the `ω` matrix
    Heisenberg!(p.dρ_soa, p.eiωt, -1)
    soa_to_base!(dρ, p.dρ_soa)

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

function mul_by_im_minus!(C::StructArray{<:Complex})
    @turbo for i ∈ eachindex(C)
        a = C.re[i]
        C.re[i] = C.im[i]
        C.im[i] = -a
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
export mul_diagonal!

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

# Wigner D-matrix to rotate polarization vector
# D(cosβ, sinβ, α, γ) = [
#     (1/2)*(1 + cosβ)*exp(-im*(α + γ)) -(1/√2)*sinβ*exp(-im*α) (1/2)*(1 - cosβ)*exp(-im*(α - γ));
#     (1/√2)*sinβ*exp(-im*γ) cosβ -(1/√2)*sinβ*exp(im*γ);
#     (1/2)*(1 - cosβ)*exp(im*(α - γ)) (1/√2)*sinβ*exp(im*α) (1/2)*(1 + cosβ)*exp(im*(α + γ))
# ]
function D(cosβ, sinβ, α, γ)
    γ = -γ
    # Sign convention is different than the matrix above 
    return [
        (1/2)*(1 + cosβ)*exp(-im*(α + γ)) -(1/√2)*sinβ*exp(-im*α) (1/2)*(1 - cosβ)*exp(-im*(α - γ));
        (1/√2)*sinβ*exp(-im*γ) cosβ -(1/√2)*sinβ*exp(im*γ);
        (1/2)*(1 - cosβ)*exp(im*(α - γ)) (1/√2)*sinβ*exp(im*α) (1/2)*(1 + cosβ)*exp(im*(α + γ))
    ]
end

function rotate_pol(pol, k)
    # Rotates polarization `pol` onto the quantization axis `k`
    k = k / norm(k)
    cosβ = k[3]
    sinβ = sqrt(1 - cosβ^2)
    α = 0.0
    if abs(cosβ) < 1
        γ = atan(k[2], k[1])
    else
        γ = 0.0
    end
    return inv(D(cosβ, sinβ, α, γ)) * pol
end
export rotate_pol

# Multiplication using `@turbo` from LoopVectorization
function mul_turbo!(C, A, B)
    @turbo for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
        Cmn_re = 0.0
        Cmn_im = 0.0
        for k ∈ 1:size(A,2)
            A_mk_re = A.re[m,k]
            A_mk_im = A.im[m,k]
            B_kn_re = B.re[k,n]
            B_kn_im = B.im[k,n]
            Cmn_re += A_mk_re * B_kn_re - A_mk_im * B_kn_im
            Cmn_im += A_mk_re * B_kn_im + A_mk_im * B_kn_re
        end
        C.re[m,n] = Cmn_re
        C.im[m,n] = Cmn_im
    end
end

end
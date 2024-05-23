function normalize!(p, ψ, norm_factor=1)
    ψ_norm_squared = zero(Float64)
    for i ∈ 1:p.n_states
        ψ_norm_squared += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm_squared)
    for i ∈ 1:p.n_states
        ψ[i] *= (norm_factor / ψ_norm)
    end
    return ψ_norm
end

function recoil_kick(ψ, p, m)
    
    dp = sample_direction(m)
    dv = dp ./ p.mass
    ψ[p.n_states + p.n_excited + 4] += dv[1]
    ψ[p.n_states + p.n_excited + 5] += dv[2]
    ψ[p.n_states + p.n_excited + 6] += dv[3]

    return nothing
end

function recoil_kick_1D(dψ, ψ, p, m, i)
    
    sgn = sign(0.5 - rand())
    ψ[p.n_states + p.n_excited + 3 + i] += sgn * (m / p.mass)

    return nothing
end

"""
    Calculate ⟨f²⟩ - ⟨f⟩²

    ⟨f²⟩ = ⟨ψ|ff|ψ⟩
    ⟨f⟩² = ⟨ψ|f|ψ⟩²
"""
function calculate_force_twotimecorrelation(p_diffusion, ψs, ts, τ_idx, t_idx)

    @unpack p, χ = p_diffusion

    ψ_t = ψs[t_idx]

    # update fields so that the force evaluated is f(τ), using ψ(τ)
    τ = ts[τ_idx]
    ψ_τ = ψs[τ_idx]
    update_fields_for_diffusion!(p, ψ_τ, τ)
    
    # evaluate f(τ)|ψ(t)⟩
    for i ∈ eachindex(χ)
        χ[i] = ψ_t[i]
    end
    a = 1.0
    fψ_to_χ!(p, ψ_τ, χ, a, 1)

    # evaluate ⟨ψ|f(τ)|ψ⟩
    ftau = force_stochastic_ψ1ψ2(p, χ, χ, 1)

    # update fields so that the force evaluated is f(t), using ψ(t)
    t = ts[t_idx]
    r_t = SVector(
        real(ψ_t[p.n_states + p.n_excited + 1]),
        real(ψ_t[p.n_states + p.n_excited + 2]),
        real(ψ_t[p.n_states + p.n_excited + 3])
        )
    update_fields_for_diffusion!(p, r_t, t)

    # evaluate ⟨ψ|f(t)f(τ)|ψ⟩
    f_squared_exp = force_stochastic_ψ1ψ2(p, ψ_t, χ, 1)

    # evaluate ⟨ψ|f(t)|ψ⟩
    ft = force_stochastic_ψ1ψ2(p, ψ_t, ψ_t, 1)

    # evaluate ⟨ψ|f(t)|ψ⟩ * ⟨ψ|f(τ)|ψ⟩
    f_exp_squared = ft * ftau

    f_σ2 = f_squared_exp - f_exp_squared

    return f_σ2
end
export calculate_force_twotimecorrelation

function overlap(p, ψ1, ψ2)
    _overlap = zero(Float64)
    for i ∈ 1:p.n_states
        _overlap += conj(ψ1[i]) * ψ2[i]
    end
    return _overlap
end

function convert_to_schrodinger(p, ψ, ψ_s)
    for i ∈ 1:p.n_states
        ψ_s[i] = ψ[i] * p.eiωt[i]
    end
    return nothing
end

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve_with_diffusion!(i, p_diffusion, ts, ψs, dψ, ψ, δt)

    @unpack p, ϕ, dϕ, fϕ, dfϕ = p_diffusion

    t_idx = i
    t = ts[t_idx]
    ψs[t_idx] .= ψ

    # first check for a quantum jump, with δp = δt * [excited state population]
    δp = zero(Float64)
    for i ∈ (p.n_states - p.n_excited + 1):p.n_states
        δp += δt * norm(ψ[i])^2
    end

    # decay or evolve the state
    if rand() < δp

        # collapse!(ψ, p)
        # recoil_kick(ψ, p, 1)
        # _ = normalize!(p, ψ)

        # dp = sqrt(2 * real(p_diffusion.D) * p_diffusion.Δt) / p.mass
        # println(p_diffusion.Δt, " ", dp)

        # recoil_kick_1D(dψ, ψ, p, dp, 1)
        # recoil_kick_1D(dψ, ψ, p, dp, 2)
        # recoil_kick_1D(dψ, ψ, p, dp, 3)

        # calculate |ϕ(0)⟩ = f(0)|ψ(0)⟩, which will evolve over time
        # update fields so that the force evaluated is f(0), using ψ(0)
        t0_idx = p_diffusion.t0_idx
        t0 = ts[t0_idx]
        ψ0 = ψs[t0_idx]
        ψt = ψs[t_idx-1]

        for i ∈ eachindex(ϕ)
            ϕ[i] = ψ0[i]
        end

        fτ_sum = zero(Float64)
        p_diffusion.D = zero(Float64)
        for idx ∈ t0_idx:(t_idx-1)

            τ = ts[idx]
            ψτ = ψs[idx]

            # calculate f(0)|τ⟩, add to total
            # update_fields_for_diffusion!(p, ψ0, t0)
            update_fields_for_diffusion!(p, ψτ, τ)
            fψ_addto_χ!(p, ψτ, fϕ, 1, 1)

            # evolve the total, no normalization
            evolve_no_jumps!(p, dfϕ, fϕ, τ, δt)

            # update_fields_for_diffusion!(p, ψτ, τ)
            fτ_sum += force_stochastic_ψ1ψ2(p, ψτ, ψτ, 1)
            # println(force_stochastic_ψ1ψ2(p, ψτ, ψτ, 1))

        end

        # println(ψt[1:16])
        # println(fϕ[1:16])
        # println(" ")
        # println(force_stochastic_ψ1ψ2(p, ψt, fϕ, 1))

        # println(force_stochastic_ψ1ψ2(p, ψt, fϕ, 1) * δt)
        
        # update_fields_for_diffusion!(p, ψ0, t0)
        update_fields_for_diffusion!(p, ψt, t)
        p_diffusion.D += force_stochastic_ψ1ψ2(p, ψt, fϕ, 1) * δt
        println(force_stochastic_ψ1ψ2(p, ψt, fϕ, 1) * δt)

        # update_fields_for_diffusion!(p, ψt, t)
        p_diffusion.D -= fτ_sum * force_stochastic_ψ1ψ2(p, ψt, ψt, 1) * δt

        p_diffusion.Δt = t - t0

        p_diffusion.t0_idx = t_idx

        ### FOR CONTINUOUS EVALUATION OF D ###
        # set up variables for next evaluation of diffusion
        # note that this happens after the collapse

        # # calculate |ϕ(0)⟩ = f(0)|ψ(0)⟩, which will evolve over time
        # # update fields so that the force evaluated is f(0), using ψ(0)
        # t = ts[t_idx]
        # update_fields_for_diffusion!(p, ψ, t)
        # fψ_to_χ!(p, ψ, ϕ, 1, 1)

        # ϕ_norm_squared = zero(Float64)
        # for i ∈ 1:p.n_states
        #     ϕ_norm_squared += norm(ϕ[i])^2
        # end
        # p_diffusion.fϕ_norm = normalize!(p, ϕ)

        # for idx ∈ t0_idx:t_idx
            
        # end

        # p_diffusion.D = zero(Float64)
        # p_diffusion.t0_idx = t_idx

        p_diffusion.F_sum = 0.0
        for i ∈ eachindex(fϕ)
            fϕ[i] = 0.0
        end

    else

        # update dψ
        update_dψ!(dψ, ψ, p, t)
        #println(force_stochastic(p.E_k, p.ds, p.ds_state1, p.ds_state2, p.ψ_soa, p.eiωt)[1])
        for i ∈ eachindex(ψ)
            ψ[i] += dψ[i] * δt
        end
        _ = normalize!(p, ψ)
        
        p_diffusion.F_sum += dψ[p.n_states + p.n_excited + 4] * p.mass
        # println(force_stochastic(p.E_k, p.ds, p.ds_state1, p.ds_state2, p.ψ_soa, p.eiωt)[1])

        ### FOR CONTINUOUS EVALUATION OF D ###
        # convert_to_schrodinger(p, ψ, ψ_s)

        # # evolve ϕ, normalize, and add to the sum ∫₀ᵗ |ϕ(τ)⟩ dτ
        # update_dψ!(dϕ, ϕ, p, t)
        # for i ∈ eachindex(ϕ)
        #     ϕ[i] += dϕ[i] * δt
        # end
        # _ = normalize!(p, ϕ)
        # convert_to_schrodinger(p, ϕ, ϕ_s)

        # # add to ∫₀ᵗ |ϕ(τ)⟩ dτ
        # for i ∈ eachindex(ϕ_sum_t)
        #     ϕ_sum_t[i] += ϕ_s[i] * δt
        # end

        # # println(force_stochastic_ψ1ψ2(p, ψ_s, ϕ_sum_t, 1) * δt * p_diffusion.fϕ_norm)

        # # evaluate ∫₀ᵗ ⟨f(t)f(τ)⟩ dτ = ⟨ψ(t)|f ∫₀ᵗ |ϕ(τ)⟩ dτ and add it to diffusion
        # p_diffusion.D += force_stochastic_ψ1ψ2(p, ψ_s, ϕ_sum_t, 1) * δt * p_diffusion.fϕ_norm
        # # println(p_diffusion.D)

        # # evaluate ∫₀ᵗ ⟨f(t)⟩⟨f(τ)⟩ dτ = ⟨f(t)⟩ ∫₀ᵗ ⟨f(τ)⟩ dτ and subtract it from the diffusion
        # # f_t = force_stochastic_ψ1ψ2(p, ψ_s, ψ_s, 1)
        # # p_diffusion.fϕ_sum_t += f_t * δt
        # # p_diffusion.D -= f_t * p_diffusion.fϕ_sum_t * δt

        # # println(p_diffusion.D)

    end

    return nothing
end
export evolve_with_diffusion!

function update_fields_for_diffusion!(p, ψ, t)
    @unpack fields, E_k, eiωt, ω = p

    r = SVector(
        real(ψ[p.n_states + p.n_excited + 1]),
        real(ψ[p.n_states + p.n_excited + 2]),
        real(ψ[p.n_states + p.n_excited + 3])
        )

    update_fields!(fields, r, t)

    # Set summed fields to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # Sum updated fields
    @inbounds @simd for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    update_eiωt!(eiωt, ω, t)

    return nothing
end

function update_dψ!(dψ, ψ, p, t)

    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))
    
    base_to_soa!(ψ, ψ_soa)
    
    update_H!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in schrodinger picutre
    
    update_eiωt!(eiωt, ω, t)
    Heisenberg!(H, eiωt)  # molecule-light Hamiltonian in interation picture
    
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT hamiltonian in schrodinger picutre
    Heisenberg!(H₀, eiωt) # Zeeman and ODT Hamiltonian in interaction picture
    
    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end
    
    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt) # force due to lasers

    H₀_expectation = operator_matrix_expectation(H₀, ψ_soa)
    f += ∇H .* (-H₀_expectation) # force due to conservative potential

    # # add gravity to the force
    # g = -9.81 / (Γ^2/k)
    # f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
        dψ[n_states + n_excited + 6 + i] = f[i] # update INTEGRATED force
    end

    # # set the instantaneous force
    # p_diffusion.Fx = f[1]
    # p_diffusion.Fy = f[2]
    # p_diffusion.Fz = f[3]
    
    return nothing
end
export update_dψ!

function collapse!(ψ, p)

    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    
    # A photon is observed.
    # Measure the polarization of the photon along z.
    p⁺ = 0.0
    p⁰ = 0.0
    p⁻ = 0.0
    
    for i ∈ 1:n_excited
        ψ_pop = norm(ψ[n_ground + i])^2
        for j ∈ 1:n_ground
            p⁺ += ψ_pop * norm(d[j,n_ground+i,1])^2
            p⁰ += ψ_pop * norm(d[j,n_ground+i,2])^2
            p⁻ += ψ_pop * norm(d[j,n_ground+i,3])^2
        end
        # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
        # whereas the polarization of the emitted photon is m_g - m_e
    end
    
    # set all ground state amplitudes to zero
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end
    
    for i in 1:n_ground
        for j in (n_ground+1):n_states
            ψ[i] += ψ[j] * d[i,j,pol]
        end
    end
    
    # zero excited state amplitudes
    for i ∈ (n_ground + 1):n_states
        ψ[i] = 0.0
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        ψ[i] = 0.0
    end

    return nothing
end
export collapse!

"""
    Evaluate a * fψ, where f is the force operator, and place the result in χ.
"""
function fψ_to_χ!(p, ψ, χ, a, k)

    @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

    for i ∈ eachindex(ψ)
        χ[i] = ψ[i]
    end

    @inbounds @fastmath for q ∈ 1:3

        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]

        E_kq = E_k[k][q]
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)

        for j ∈ eachindex(ds_q)
            m = ds_state1_q[j] # excited state
            n = ds_state2_q[j] # ground state

            # find the density matrix element
            c_m = conj(eiωt[m]) # exp(-iωt) factor to transform to interaction picture
            c_n = conj(eiωt[n]) # exp(-iωt) factor to transform to interaction picture

            ρ_mn = conj(c_m) * c_n

            ρ_re = real(ρ_mn)
            ρ_im = imag(ρ_mn)
            
            d_re = ds_q_re[j]
            d_im = ds_q_im[j]

            a1 = d_re * ρ_re - d_im * ρ_im
            a2 = d_re * ρ_im + d_im * ρ_re
            F_k_re = E_kq_re * a1 - E_kq_im * a2
            F_k_im = E_kq_im * a1 + E_kq_re * a2
            
            # note the factor -i
            val = -im * (F_k_re + im * F_k_im)

            χ[m] += a * val * ψ[n]
            χ[n] += a * conj(val) * ψ[m]

        end
    end
    return nothing
end
export fψ_to_χ!

"""
    Evaluate a * fψ, where f is the force operator, and place the result in χ.
"""
function fψ_addto_χ!(p, ψ, χ, a, k)

    @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

    @inbounds @fastmath for q ∈ 1:3

        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]

        E_kq = E_k[k][q]
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)

        for j ∈ eachindex(ds_q)
            m = ds_state1_q[j] # excited state
            n = ds_state2_q[j] # ground state

            # find the density matrix element
            c_m = conj(eiωt[m]) # exp(-iωt) factor to transform to interaction picture
            c_n = conj(eiωt[n]) # exp(-iωt) factor to transform to interaction picture

            ρ_mn = c_m * conj(c_n)

            ρ_re = real(ρ_mn)
            ρ_im = imag(ρ_mn)
            
            d_re = ds_q_re[j]
            d_im = ds_q_im[j]

            a1 = d_re * ρ_re - d_im * ρ_im
            a2 = d_re * ρ_im + d_im * ρ_re
            F_k_re = E_kq_re * a1 - E_kq_im * a2
            F_k_im = E_kq_im * a1 + E_kq_re * a2
            
            # note the factor -i
            val = -im * (F_k_re + im * F_k_im)

            χ[m] += a * val * ψ[n] #* eiωt[n]
            χ[n] += a * conj(val) * ψ[m] #* eiωt[m]

        end
    end
    return nothing
end
export fψ_addto_χ!

# """
#     Evaluate a * ⟨ψ|f, where f is the force operator, and place the result in χ.
# """
# function ψf_to_χ!(p, ψ, χ, a, k)

#     @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

#     for i ∈ eachindex(ψ)
#         χ[i] = ψ[i]
#     end

#     @inbounds @fastmath for q ∈ 1:3

#         ds_q = ds[q]
#         ds_q_re = ds_q.re
#         ds_q_im = ds_q.im
#         ds_state1_q = ds_state1[q]
#         ds_state2_q = ds_state2[q]

#         E_kq = E_k[k][q]
#         E_kq_re = real(E_kq)
#         E_kq_im = imag(E_kq)

#         for j ∈ eachindex(ds_q)
#             m = ds_state1_q[j] # excited state
#             n = ds_state2_q[j] # ground state

#             # find the density matrix element
#             c_m = 1 #conj(eiωt[m]) # exp(-iωt) factor to transform to interaction picture
#             c_n = 1 #conj(eiωt[n]) # exp(-iωt) factor to transform to interaction picture

#             ρ_mn = conj(c_m) * c_n

#             ρ_re = real(ρ_mn)
#             ρ_im = imag(ρ_mn)

#             d_re = ds_q_re[j]
#             d_im = ds_q_im[j]

#             a1 = d_re * ρ_re - d_im * ρ_im
#             a2 = d_re * ρ_im + d_im * ρ_re
#             F_k_re = E_kq_re * a1 - E_kq_im * a2
#             F_k_im = E_kq_im * a1 + E_kq_re * a2
            
#             # note the factor -i
#             val = -im * (F_k_re + im * F_k_im)

#             χ[m] += a * conj(val) * conj(ψ[n])
#             χ[n] += a * val * conj(ψ[m])

#         end
#     end
#     return nothing
# end

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve!(p, dψ, ψ, t, δt)

    # update dψ
    update_dψ!(dψ, ψ, p, t)

    # first check for a quantum jump, with δp = δt * [excited state population]
    δp = zero(Float64)
    for i ∈ (p.n_states - p.n_excited + 1):p.n_states
        δp += δt * norm(ψ[i])^2
    end
    if rand() < δp
        # whenever a collapse happens, we also evaluate the diffusion coefficient
        collapse!(ψ, p)
    end

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    _ = normalize!(p, ψ)

    return nothing
end
export evolve!

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve_no_jumps!(p, dψ, ψ, t, δt)

    # update dψ
    update_dψ!(dψ, ψ, p, t)

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    return nothing
end
export evolve_no_jumps!

"""
    Evaluate ⟨ψ₁|fₖ|ψ₂⟩
"""
function force_stochastic_ψ1ψ2(p, ψ1, ψ2, k)
    @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

    F = zero(Float64)

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        E_kq = E_k[k][q]
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)
        F_k_re = 0.0
        F_k_im = 0.0
        for j ∈ eachindex(ds_q)
            m = ds_state1_q[j] # excited state
            n = ds_state2_q[j] # ground state
            
            # construct ρ_mn = c_m c_n^*
            # ρ_mn = ψ2[m]*eiωt[m] * conj(ψ1[n]*eiωt[n]) 

            # the terms evaluated are ⟨e (e^(-iω₁t)|e⟩⟨g|e^(iω₂t)) g⟩
            c_m = ψ1[m] * conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
            c_n = ψ2[n] * conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

            ρ_mn = c_m * conj(c_n)

            ρ_re = real(ρ_mn)
            ρ_im = imag(ρ_mn)
            
            d_re = ds_q_re[j]
            d_im = ds_q_im[j]

            a1 = d_re * ρ_re - d_im * ρ_im
            a2 = d_re * ρ_im + d_im * ρ_re
            F_k_re += E_kq_re * a1 - E_kq_im * a2
            F_k_im += E_kq_im * a1 + E_kq_re * a2    
             
        end
        F -= (im * F_k_re - F_k_im) # multiply by -im
    end
    F += conj(F)

    return F
end
export force_stochastic_ψ1ψ2

"""
    Update the matrix representing the operator fₖ
"""
function update_fk_matrix(p, k, fk)
    @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

    F = zero(Float64)

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        E_kq = E_k[k][q]
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)
        F_k_re = 0.0
        F_k_im = 0.0
        for j ∈ eachindex(ds_q)
            m = ds_state1_q[j] # excited state
            n = ds_state2_q[j] # ground state
            
            # construct ρ_mn = c_m c_n^*
            # ρ_mn = ψ2[m]*eiωt[m] * conj(ψ1[n]*eiωt[n]) 

            # the terms evaluated are ⟨e (e^(-iω₁t)|e⟩⟨g|e^(iω₂t)) g⟩
            c_m = conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
            c_n = conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

            ρ_mn = conj(c_m) * c_n * conj(ψ1[m]) * ψ2[n]

            ρ_re = real(ρ_mn)
            ρ_im = imag(ρ_mn)
            
            d_re = ds_q_re[j]
            d_im = ds_q_im[j]

            a1 = d_re * ρ_re - d_im * ρ_im
            a2 = d_re * ρ_im + d_im * ρ_re
            F_k_re += E_kq_re * a1 - E_kq_im * a2
            F_k_im += E_kq_im * a1 + E_kq_re * a2    
             
            fk[m,n] = -im * (F_k_re + im * F_k_im)
            fk[n,m] = conj(fk[m,n])
        end
    end

    return nothing
end
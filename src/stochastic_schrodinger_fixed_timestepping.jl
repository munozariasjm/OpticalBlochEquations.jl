"""
    Fixed timestepping
"""

function normalize!(p, ψ)
    ψ_norm_squared = zero(Float64)
    for i ∈ 1:p.n_states
        ψ_norm_squared += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm_squared)
    for i ∈ 1:p.n_states
        ψ[i] /= ψ_norm
    end
    return ψ_norm
end
export normalize!

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve_fixed_timestep!(p, dψ, ψ, t, δt)

    # first check for a quantum jump, with δp = δt * [excited state population]
    δp = zero(Float64)
    for i ∈ (p.n_states - p.n_excited + 1):p.n_states
        δp += δt * norm(ψ[i])^2
    end

    # collapse the wavefunction
    if rand() < δp
        collapse!(ψ, p)
    end

    # update dψ
    update_dψ!(dψ, ψ, p, t)

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    _ = normalize!(p, ψ)

    # diffusion
    for i ∈ 1:3
        ψ[p.n_states + p.n_excited + 3 + i] += rand((-1,1)) * sqrt( 2p.diffusion_constant * δt ) / p.mass
    end
    p.time_before_decay += δt

    return nothing
end
export evolve_fixed_timestep!

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

    # H₀_expectation = operator_matrix_expectation(H₀, ψ_soa)
    # f += ∇H .* (-H₀_expectation) # force due to conservative potential

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
    end
    
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
    
    for i ∈ 1:n_ground
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
    
    dp = sample_direction(1)
    dv = dp ./ p.mass
    ψ[n_states + n_excited + 4] += dv[1]
    ψ[n_states + n_excited + 5] += dv[2]
    ψ[n_states + n_excited + 6] += dv[3]

    # # diffusion
    # for i ∈ 1:3
    #     ψ[p.n_states + p.n_excited + 3 + i] += rand((-1,1)) * sqrt( 2p.diffusion_constant * p.time_before_decay ) / p.mass
    # end
    # p.time_before_decay = 0.0

    return nothing
end
export collapse!
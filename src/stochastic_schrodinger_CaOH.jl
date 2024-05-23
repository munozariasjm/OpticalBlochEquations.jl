using Distributions, StatsBase

function SE_collapse_pol_always_CaOH!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
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
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
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
    
    # is this required?
    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        integrator.u[i] = 0.0
    end

    # add a repumper photon kick along the x-axis
    repump_states_prob = 1/150 # 1/150 photons are repumped with a laser that is not retroreflected
    if rand() < repump_states_prob
        integrator.u[n_states + n_excited + 4] += 2 / p.mass # 2 photons are required to repump on average
    end

    repump_states_prob = 1/20 # 1/20 photons are not (000), and on average two photon scatters are required to repump
    if rand() < repump_states_prob
        dp = sample_direction(1)
        dv = dp ./ p.mass
        integrator.u[n_states + n_excited + 4] += dv[1]
        integrator.u[n_states + n_excited + 5] += dv[2]
        integrator.u[n_states + n_excited + 6] += dv[3]
        
        dp = sample_direction(1)
        dv = dp ./ p.mass
        integrator.u[n_states + n_excited + 4] += dv[1]
        integrator.u[n_states + n_excited + 5] += dv[2]
        integrator.u[n_states + n_excited + 6] += dv[3]
    end

     # reset excited state population accumulation
    # integrator.u[n_states + 1:n_states + n_excited] .= 0
    
#     for i in 1:n_states
#         integrator.p.populations[i] = norm(integrator.u[i])^2
#     end
    
    # add two photon recoils
    dp = sample_direction(1)
    dv = dp ./ p.mass
    integrator.u[n_states + n_excited + 4] += 2dv[1]
    integrator.u[n_states + n_excited + 5] += 2dv[2]
    integrator.u[n_states + n_excited + 6] += 2dv[3]

    # dp = sample_direction(1)
    # dv = dp ./ p.mass
    # integrator.u[n_states + n_excited + 4] += dv[1]
    # integrator.u[n_states + n_excited + 5] += dv[2]
    # integrator.u[n_states + n_excited + 6] += dv[3]
    
    p.time_to_decay = rand(p.decay_dist)

    return nothing
end
export SE_collapse_pol_always!

function ψ_stochastic!(dψ, ψ, p, τ)
    @unpack ψ_soa, dψ_soa, r, H₀, ω, fields, H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    base_to_soa!(ψ, ψ_soa)
    
    update_H!(p, τ, r, H₀, fields, H, E_k, ds, ds_state1, ds_state2, Js)
    
    update_eiωt!(eiωt, ω, τ)
    Heisenberg!(H, eiωt)

    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    # calculate force
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i]/mass # update velocity
    end

    # update force
    ψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= f
    dψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0
    
    return nothing
end
export ψ_stochastic!

function force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq = E_k[k][q]
            E_kq_re = real(E_kq)
            E_kq_im = imag(E_kq)
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                
                # construct ρ_mn = c_m c_n^*
                # ρ_mn = conj(ψ_soa[n]*eiωt[n]) * ψ_soa[m]*eiωt[m]

                c_m = ψ_soa[m] * conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
                c_n = ψ_soa[n] * conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

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
            F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end
    F += conj(F)

    return real.(F)
end

function force_stochastic_v2(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq = E_k[k][q]
            E_kq_re = real(E_kq)
            E_kq_im = imag(E_kq)
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                d_re = ds_q_re[j]
                d_im = ds_q_im[j]

                c_m_re = ψ_soa.re[m]
                c_m_im = ψ_soa.im[m]
                c_n_re = ψ_soa.re[n]
                c_n_im = -ψ_soa.im[n] # taking the conjugate 
                
                ρ_re = c_m_re * c_n_re - c_m_im * c_n_im
                ρ_im = c_m_re * c_n_im + c_m_im * c_n_re

                a1 = d_re * ρ_re - d_im * ρ_im
                a2 = d_re * ρ_im + d_im * ρ_re
                F_k_re += E_kq_re * a1 - E_kq_im * a2
                F_k_im += E_kq_im * a1 + E_kq_re * a2     
            end
            F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end
    F += conj(F)

    return real.(F)
end

function force_stochastic_nocomplex(n_states, E_k_re, E_k_im, ds, ds_state1, ds_state2, ψ, eiωt_re, eiωt_im)
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq_re = E_k_re[k][q]
            E_kq_im = E_k_im[k][q]
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                
                # construct ρ_mn = c_m c_n^*

                c_m = (ψ[m] + im * ψ[n_states+m]) * conj(eiωt_re[m] + im * eiωt_im[m]) # exp(-iωt) factor to transform to Heisenberg picture
                c_n = (ψ[n] + im * ψ[n_states+n]) * conj(eiωt_re[n] + im * eiωt_im[n]) # exp(-iωt) factor to transform to Heisenberg picture

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
            F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end
    F += conj(F)
    return real.(F)
end

function force_stochastic_nocomplex_v0(n_states, E_k, ds, ds_state1, ds_state2, ψ, eiωt)
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq = E_k[k][q]
            E_kq_re = real(E_kq)
            E_kq_im = imag(E_kq)
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                
                # construct ρ_mn = c_m c_n^*

                c_m = (ψ[m] + im * ψ[n_states+m]) * conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
                c_n = (ψ[n] + im * ψ[n_states+n]) * conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

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
            F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end
    F += conj(F)
    return real.(F)
end

function ψ_stochastic_nocomplex_v0!(dψ, ψ, p, τ)

    @unpack ω, fields, H, H₀, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[2n_states + n_excited + 1]), real(ψ[2n_states + n_excited + 2]), real(ψ[2n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += ψ[i]^2 + ψ[n_states+i]^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:2n_states
        ψ[i] /= ψ_norm
    end

    update_H!(p, τ, r, H₀, fields, H, E_k, ds, ds_state1, ds_state2, Js)
    
    update_eiωt!(eiωt, ω, τ)
    Heisenberg!(H, eiωt)

    # multiply ψ by -im
    @turbo for i ∈ 1:n_states
        ψ_re = ψ[i]
        ψ[i] = ψ[n_states+i]
        ψ[n_states+i] = -ψ_re
    end
    
    # mul_turbo_nocomplex_v0!(n_states, dψ, H, ψ)
    mul_turbo_nocomplex_v00!(n_states, dψ, H, ψ)
    
    # calculate force
    f = force_stochastic_nocomplex_v0(n_states, E_k, ds, ds_state1, ds_state2, ψ, eiωt)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[2n_states + i] = ψ[n_states-n_excited+i]^2 + ψ[2n_states-n_excited+i]^2
        # println(ψ[n_states-n_excited+i]^2 + ψ[2n_states-n_excited+i]^2)
    end
    
    for i ∈ 1:3
        dψ[2n_states + n_excited + i] = ψ[2n_states + n_excited + i + 3] # update position
        dψ[2n_states + n_excited + i + 3] = f[i]/mass # update velocity
    end

    ψ[end-2:end] .= f
    dψ[end-2:end] .= 0
    
    return nothing
end
export ψ_stochastic_nocomplex_v0!

function schrodinger_stochastic_nocomplex_v0(
    particle, states, fields, d, ψ₀, mass, n_excited;
    extra_p=nothing, λ=1.0, Γ=2π, update_H=update_H)

    n_states = length(states)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = Complex{Float64}

    H = StructArray( zeros(type_complex, n_states, n_states) )
    H₀ = deepcopy(H)

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    # Define jumps
    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]

    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    # ψ contains the state vector (real and imaginary separated), accumulated excited state populations, position, velocity.
    ψ = zeros(Float64, 2n_states + n_excited + 6 + 3)

    ψ[1:n_states] .= ψ₀
    ψ[2n_states + n_excited + 1:2n_states + n_excited + 3] .= r
    ψ[2n_states + n_excited + 4:2n_states + n_excited + 6] .= v

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        ψ=ψ,
        H=H,
        H₀=H₀,
        ω=ω, 
        eiωt=eiωt,
        Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, λ=λ,
        k=k,
        E=E,
        E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        extra_p=extra_p, mass = mass, update_H = update_H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{Float64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist)
        )

    return p
end
export schrodinger_stochastic_nocomplex_v0

# p should have:
# a "dark state" lifetime distribution extra_p.dark_time_dist
# p.dark_time, save sampled value from the distribution
# a boolean variable p.is_dark
# p.dark_t0
# p.FC_mainline

function schrodinger_stochastic_repump(
    particle, states, fields, d, ψ₀, mass, n_excited;
    extra_p=nothing, λ=1.0, Γ=2π, update_H=update_H, dark_lifetime=0.0, FC_mainline = 1.0)
    """
    dark_lifetime = time the molecule spends in a dark (to mainline laser) state
    FC_mainline = Frank-Condon factor of mainline transtion
    
    extra_p should contain n_excited
    
    ψ in the output will be of the following format:
    the first n_states indicies will be the coefficients of the current state;
    the next n_excited indicies is the time-integrated excited state population (reset by callbacks);
    the next 3 indicies are the current position;
    the next 3 indicies are the current velocity;
    the last 3 indicies are the current force.
    """

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = ComplexF64

    H = StructArray( zeros(type_complex, n_states, n_states) )
    ∇H = SVector{3, ComplexF64}(0,0,0)

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    ψ_soa = StructArray(ψ₀)
    dψ_soa = StructArray(ψ₀)
    
    # ψ contains the state vector, accumulated excited state populations, position, velocity, force
    ψ = zeros(ComplexF64, n_states + n_excited + 3 + 3 + 3)
    ψ[1:n_states] .= ψ₀
    ψ[n_states + n_excited + 1: n_states + n_excited + 3] .= r
    ψ[n_states + n_excited + 4: n_states + n_excited + 6] .= v
    dψ = deepcopy(ψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)
    dark_time_dist = Exponential(dark_lifetime * Γ)

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, ∇H=∇H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, λ=λ,
        k=k, 
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        extra_p=extra_p, mass = mass, update_H_and_∇H=update_H_and_∇H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{ComplexF64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        dark_time_dist = dark_time_dist,
        dark_time = rand(dark_time_dist),
        dark_time_t0 = 0.0,
        is_dark = false,
        FC_mainline = FC_mainline
        )

    return p
end

function ψ_stochastic_repump!(dψ, ψ, p, τ)
      
    if p.is_dark == false
        ψ_stochastic_potential!(dψ, ψ, p, τ)
        return nothing
    else
       @unpack ψ_soa, dψ_soa, r, H₀, ω, fields, H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
        
        n_states = length(states)
        n_excited = p.n_excited

        r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))
        
        for i ∈ 1:n_excited + n_states
            dψ[i] = 0.0
        end
        # force = 0 if state is outside the cycling Hilbert space
        for i ∈ 1:3
            dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
            dψ[n_states + n_excited + 3 + i] = 0.0 # update velocity
        end

        # update force
        ψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0
        dψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0

        return nothing 
    end
end
export schrodinger_stochastic_repump;
export ψ_stochastic_repump!;

function SE_collapse_repump!(integrator)
    # go back to bright state
    if integrator.p.is_dark == true
        integrator.p.is_dark = false
        n_states = length(integrator.p.states)
        n_excited = integrator.p.n_excited
        i = Int(floor(rand()*n_excited)) + 1
        for i in 1:n_states
            integrator.u[i] = 0.0
        end
        integrator.u[n_states - n_excited + i] = 1.0
    else
        # scatter
        rn = rand()
        if rn < integrator.p.FC_mainline # decay back to a cycling state
            SE_collapse_pol_always!(integrator)
        else # decay into a dark state and wait to be repumped
            SE_collapse_pol_always!(integrator) # give a momentum kick
            integrator.p.is_dark = true
            integrator.p.dark_time = rand(integrator.p.dark_time_dist)
            integrator.p.dark_time_t0 = integrator.t
        end
    end
end
export SE_collapse_repump!
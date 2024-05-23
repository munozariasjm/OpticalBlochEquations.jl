function schrodinger_stochastic_diffusion(
    particle, states, fields, d, ψ₀, mass, n_excited;
    sim_params=nothing, extra_data=nothing, λ=1.0, Γ=2π, update_H_and_∇H=update_H_and_∇H)
    """
    extra_p should contain n_excited
    
    ψ in the output will be of the following format:
    the first n_states indicies will be the coefficients of the current state;
    the next n_excited indicies is the time-integrated excited state population (reset by callbacks);
    the next 3 indicies are the current position;
    the next 3 indicies are the current velocity;
    the last 3 indicies are the current force.
    """
    dt = sim_params.dt
    
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

    H = StructArray( zeros(type_complex, n_states, n_states) ) # light-molecule (dipole) Hamiltonian
    H₀ = deepcopy(H) # Zeeman and ODT Hamiltonian
    U_t = deepcopy(H) # time-evolution operator
    for i in 1:n_states
        U_t[i,i] = 1.0
    end
    P_x = StructArray( zeros(type_complex, n_states, n_states) ) # momentum operator
    Px_sq = StructArray( zeros(type_complex, n_states, n_states) ) # momentum operator squared
    P_y = StructArray( zeros(type_complex, n_states, n_states) ) # momentum operator
    Py_sq = StructArray( zeros(type_complex, n_states, n_states) ) # momentum operator squared
    P_z = StructArray( zeros(type_complex, n_states, n_states) ) # momentum operator
    Pz_sq = StructArray( zeros(type_complex, n_states, n_states) ) # momentum operator squared
    for i in 1:n_states
        P_x[i,i] = v[1]*mass
        P_y[i,i] = v[2]*mass
        P_z[i,i] = v[3]*mass
        Px_sq[i,i] = (v[1]*mass)^2
        Py_sq[i,i] = (v[2]*mass)^2
        Pz_sq[i,i] = (v[3]*mass)^2
    end
    
    mat_aux = deepcopy(H) # an auxiliary matrix
    

    ∇H_x = deepcopy(H) # gradient of dipole Hamiltonian (operator!)
    ∇H_y = deepcopy(H)
    ∇H_z = deepcopy(H)
    
    ∇H = SVector{3, ComplexF64}(0,0,0) # gradient of the ODT Hamiltonian = ∇H * H_ODT. ∇H is just a 3-vector

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))
    U0_dt = deepcopy(H)
    for i in 1:n_states
        U0_dt[i,i] = exp(-im*dt*ω[i])
    end
    

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

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, H₀=H₀, ∇H=∇H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, 
        λ=λ, k=k, Γ=Γ,
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        sim_params=sim_params, extra_data=extra_data, mass = mass, update_H_and_∇H=update_H_and_∇H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{ComplexF64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        ∇H_x = ∇H_x, ∇H_y = ∇H_y, ∇H_z = ∇H_z,
        U_t = U_t, U_t_dagger = deepcopy(U_t), 
        P_x = P_x, P_y = P_y, P_z = P_z, 
        Px_sq = Px_sq, Py_sq =Py_sq, Pz_sq = Pz_sq,
        mat_aux = mat_aux,
        mat_aux1 = deepcopy(mat_aux),
        mat_aux2 = deepcopy(mat_aux),
        ψ_prev = deepcopy(ψ_soa),
        U0_dt = U0_dt,
        last_decay_time=0.0,
        dt = sim_params.dt,
        sum_diffusion_x = 0.0,
        sum_diffusion_y = 0.0,
        sum_diffusion_z = 0.0
        )

    return p
end
export schrodinger_stochastic_diffusion

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve_fixed_timestep_diffusion!(p, dψ, ψ, t, δt)

    # first check for a quantum jump, with δp = δt * [excited state population]
    δp = zero(Float64)
    for i ∈ (p.n_states - p.n_excited + 1):p.n_states
        δp += δt * norm(ψ[i])^2
    end

    # collapse the wavefunction
    if rand() < δp
        SE_collapse_pol_diffusion!(ψ, p, t)
    end

    # update dψ
    ψ_stochastic_diffusion!(dψ, ψ, p, t)

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    _ = normalize!(p, ψ)

    return nothing
end
export evolve_fixed_timestep_diffusion!

function SE_collapse_pol_diffusion!(ψ, p, t)

    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    
    mul_turbo!(p.Px_sq, p.P_x, p.P_x)
    σ_px = operator_matrix_expectation_complex(p.Px_sq, p.ψ_prev)[1] - ((operator_matrix_expectation_complex(p.P_x, p.ψ_prev)[1]))^2 
    
    mul_turbo!(p.Py_sq, p.P_y, p.P_y)
    σ_py = operator_matrix_expectation_complex(p.Py_sq, p.ψ_prev)[1] - ((operator_matrix_expectation_complex(p.P_y, p.ψ_prev)[1]))^2
    
    mul_turbo!(p.Pz_sq, p.P_z, p.P_z)
    σ_pz = operator_matrix_expectation_complex(p.Pz_sq, p.ψ_prev)[1] - ((operator_matrix_expectation_complex(p.P_z, p.ψ_prev)[1]))^2
    
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
        ψ[i] = 0.0
    end

    time_before_decay = t - p.last_decay_time

    p.sum_diffusion_x += abs(σ_px)/time_before_decay/2
    p.sum_diffusion_y += abs(σ_py)/time_before_decay/2
    p.sum_diffusion_z += abs(σ_pz)/time_before_decay/2
    
    p.last_decay_time = t

    dp = sample_direction(1)
    dv = dp ./ p.mass
    ψ[n_states + n_excited + 4] += dv[1]
    ψ[n_states + n_excited + 5] += dv[2]
    ψ[n_states + n_excited + 6] += dv[3]
    
    ψ[n_states + n_excited + 4] += rand((-1,1)) * sqrt(abs(σ_px)) / p.mass
    ψ[n_states + n_excited + 5] += rand((-1,1)) * sqrt(abs(σ_py)) / p.mass
    ψ[n_states + n_excited + 6] += rand((-1,1)) * sqrt(abs(σ_pz)) / p.mass

    p.time_to_decay = rand(p.decay_dist)
    
    # reset P, P_sq, U, U dagger, psi_prev
    reset_operator_diagonal!(p.P_x, ψ[n_states + n_excited + 4] * p.mass)
    reset_operator_diagonal!(p.P_y, ψ[n_states + n_excited + 5] * p.mass)
    reset_operator_diagonal!(p.P_z, ψ[n_states + n_excited + 6] * p.mass)
    
    reset_operator_diagonal!(p.Px_sq, (ψ[n_states + n_excited + 4] * p.mass)^2)
    reset_operator_diagonal!(p.Py_sq, (ψ[n_states + n_excited + 5] * p.mass)^2)
    reset_operator_diagonal!(p.Pz_sq, (ψ[n_states + n_excited + 6] * p.mass)^2)
    
    reset_operator_diagonal!(p.U_t, 1)
    reset_operator_diagonal!(p.U_t_dagger, 1)
    
    for i ∈ eachindex(p.ψ_prev)
        p.ψ_prev[i] = ψ[i] * conj(p.eiωt[i])
    end

    return nothing
end
export SE_collapse_pol_diffusion!
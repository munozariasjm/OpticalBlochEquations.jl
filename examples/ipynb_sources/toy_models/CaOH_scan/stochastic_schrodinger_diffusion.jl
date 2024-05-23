

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
        sum_diffusion_z = 0.0,
        diffusion_record  = []
        )

    return p
end
export schrodinger_stochastic_diffusion


function ψ_stochastic_diffusion!(dψ, ψ, p, t)
    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p

    
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
    
    update_H_dipole!(p, t,r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in schrodinger picutre
    
    update_eiωt!(eiωt, ω, t)
    
    
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT hamiltonian in schrodinger picutre
    
    # add the Zeeman and ODT Hamiltonian to dipole Hamiltonian
    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end 
    
    
    update_evolution_operator!(p)
    
    update_momentum_operator!(p, 1) 
    
    update_momentum_operator!(p, 2)
    
    update_momentum_operator!(p, 3)
    

    
    Heisenberg!(H, eiωt)
  
    # average dipole force
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt) # force due to lasers
    
    ## Another way to calculate the average force is: Heisenberg!(p.∇H_x, eiωt), ..., then
    #     f_x = -1*operator_matrix_expectation_complex(p.∇H_x, ψ_soa)[1]
    #     f_y = -1*operator_matrix_expectation_complex(p.∇H_y, ψ_soa)[1]
    #     f_z = -1*operator_matrix_expectation_complex(p.∇H_z, ψ_soa)[1]
    ## they produce the same answer.
    

    # add force due to conservative potential
    H₀_expectation = operator_matrix_expectation_complex(H₀, ψ_soa)[1]
    f += ∇H .* (-H₀_expectation)  

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    ##  Calculate change in the state vector ## 
    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    soa_to_base!(dψ, dψ_soa)
    
    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
    end

    # update force
    # ψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= f/ψ_norm
    # dψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0
    
    return nothing
end
export ψ_stochastic_diffusion!;


function update_evolution_operator!(p)
    """ Update p.U_t and p.U_t_dagger. 
    This must be called after p.H is updated, and before transforming p.H to interaction picture.
    """
    dt = p.dt
    
    
    # evolution operator of this step U_dt = 1 - i H dt
    # p.mat_aux1 will be set to U_dt
    reset_operator_diagonal!(p.mat_aux1, 1.0+0*im)
    for i ∈ eachindex(p.mat_aux1)
        p.mat_aux1.im[i] += -1 * p.H.re[i] * dt
    #         p.mat_aux1.re[i] +=  p.H.im[i] * dt
    end
    
    
    mul_turbo!(p.mat_aux, p.mat_aux1, p.U0_dt)
    
    mul_turbo!(p.mat_aux2, p.mat_aux, p.U_t)
    for i ∈ eachindex(p.U_t)
        p.U_t.re[i] = p.mat_aux2.re[i]
        p.U_t.im[i] = p.mat_aux2.im[i]
    end
    
    # evolution operator of this step U_dt_dagger = 1 + i H dt
    # p.mat_aux1 will be set to U_dt_dagger
     reset_operator_diagonal!(p.mat_aux1, 1.0+0*im)
    for i ∈ eachindex(p.mat_aux1)
        p.mat_aux1.im[i] += 1 * p.H.re[i] * dt
        #         p.mat_aux1.re[i] += p.H.im[i] * dt
    end
    
    mul_turbo_conjA!(p.mat_aux, p.U0_dt, p.mat_aux1)
    mul_turbo!(p.mat_aux2, p.U_t_dagger, p.mat_aux)
    set_H_zero!(p.U_t_dagger)
    for i ∈ eachindex(p.U_t)
        p.U_t_dagger.re[i] += p.mat_aux2.re[i]
        p.U_t_dagger.im[i] += p.mat_aux2.im[i]
    end  
    
    #     # normalize the determinant of U and U dagger
    n_states = length(p.states)
    rescale_det = norm(det(p.U_t))^(1/n_states)
    
    for i ∈ eachindex(p.U_t)
        p.U_t.re[i] /= rescale_det
        p.U_t.im[i] /= rescale_det
        p.U_t_dagger.re[i] /= rescale_det
        p.U_t_dagger.im[i] /= rescale_det
    end


end



function update_momentum_operator!(p, k)
    """ Update the (Heisenberg picutre) momentum operator """
    dt = p.dt
    P_k = p.mat_aux
    f_k = p.mat_aux1
    if k == 1
        P_k = p.P_x
        f_k = p.∇H_x
    elseif k == 2
        P_k = p.P_y
        f_k = p.∇H_y
    elseif k == 3
        P_k = p.P_z
        f_k = p.∇H_z
    end
    
    # calculate force operator in heisenberg picture. mat_aux2 = U_dagger f U = f(t)
    mul_turbo!(p.mat_aux, f_k, p.U_t)
    mul_turbo!(p.mat_aux2, p.U_t_dagger, p.mat_aux)
    
    # P(t+dt) = P(t) + f(t) dt
    for i ∈ eachindex(p.mat_aux2)
        P_k[i] -= p.mat_aux2[i] * dt      
    end

end


function SE_collapse_pol_diffusion!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
    
    
    
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
        integrator.u[i] = 0.0
    end

    time_before_decay = integrator.t - p.last_decay_time

    p.sum_diffusion_x += abs(σ_px)/time_before_decay
    p.sum_diffusion_y += abs(σ_py)/time_before_decay         
    p.sum_diffusion_z += abs(σ_pz)/time_before_decay

    push!(p.diffusion_record, σ_pz/time_before_decay)
    
    p.last_decay_time = integrator.t

    dp = sample_direction(1)
    dv = dp ./ p.mass
    integrator.u[n_states + n_excited + 4] += dv[1]
    integrator.u[n_states + n_excited + 5] += dv[2]
    integrator.u[n_states + n_excited + 6] += dv[3]
    
    rand1 = rand()
    diffusion_direction = 1.0
    if rand1 < 0.5
        diffusion_direction= -1
    end
    integrator.u[n_states + n_excited + 4] += sqrt(abs(σ_px))/p.mass * diffusion_direction
    
    rand1 = rand()
    diffusion_direction = 1.0
    if rand1 < 0.5
        diffusion_direction= -1
    end
    integrator.u[n_states + n_excited + 5] += sqrt(abs(σ_py))/p.mass * diffusion_direction
    
    rand1 = rand()
    diffusion_direction = 1.0
    if rand1 < 0.5
        diffusion_direction= -1
    end
    integrator.u[n_states + n_excited + 6] += sqrt(abs(σ_pz))/p.mass * diffusion_direction
    
    p.time_to_decay = rand(p.decay_dist)
    
    # reset P, P_sq, U, U dagger, psi_prev
    reset_operator_diagonal!(p.P_x, integrator.u[n_states + n_excited + 4] * p.mass)
    reset_operator_diagonal!(p.P_y, integrator.u[n_states + n_excited + 5] * p.mass)
    reset_operator_diagonal!(p.P_z, integrator.u[n_states + n_excited + 6] * p.mass)
    
    reset_operator_diagonal!(p.Px_sq, (integrator.u[n_states + n_excited + 4] * p.mass)^2)
    reset_operator_diagonal!(p.Py_sq, (integrator.u[n_states + n_excited + 5] * p.mass)^2)
    reset_operator_diagonal!(p.Pz_sq, (integrator.u[n_states + n_excited + 6] * p.mass)^2)
    
    reset_operator_diagonal!(p.U_t, 1)
    reset_operator_diagonal!(p.U_t_dagger, 1)
    
    for i ∈ eachindex(p.ψ_prev)
        p.ψ_prev[i] = integrator.u[i] * conj(p.eiωt[i])
    end

    return nothing
end
export SE_collapse_pol_diffusion!;

function reset_operator_diagonal!(O, val)
    set_H_zero!(O)
    for i in 1:size(O,1)
        O.re[i,i] = real(val)
        O.im[i,i] = imag(val)
    end
end
export reset_operator_diagonal!


function update_H_dipole!(p, τ,r, fields, H, E_k, ds, ds_state1, ds_state2, Js)
    # unpack some variables from p
    
    # reset the matrices
    set_H_zero!(H)
    set_H_zero!(p.∇H_x)
    set_H_zero!(p.∇H_y)
    set_H_zero!(p.∇H_z)


    # Reset total E field and E dot k to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # update each laser at the current time and position
    update_fields!(fields, r, τ)
    
    # Calculate total E field and total E dot k
    @inbounds @simd for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    # calculate dipole Hamiltonian matrix elements
    @inbounds @fastmath for q ∈ 1:3
        E_q = p.E[q]
        E_q_re = real(E_q)
        E_q_im = imag(E_q)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            H.im[n,m] += -val_im
            H.re[m,n] += -val_re
            H.im[m,n] -= -val_im
        end
    end

    # add the anti-Hermitian term
    @inbounds @fastmath for J ∈ Js
        H.im[J.s, J.s] -= J.r # note that this is different from OBE calcs because we already converted to J.r = Γ^2/2
    end
    
    
    # calculate matrix elements of the gradient of the dipole Hamiltonian
    
    calculate_grad_H!(p.∇H_x, 1, p, E_k, ds, ds_state1, ds_state2)
    calculate_grad_H!(p.∇H_y, 2, p, E_k, ds, ds_state1, ds_state2)
    calculate_grad_H!(p.∇H_z, 3, p, E_k, ds, ds_state1, ds_state2)

    return nothing
end
export update_H_dipole!


function calculate_grad_H!(∇H_k, k, p, E_k, ds, ds_state1, ds_state2)
    @inbounds @fastmath for q ∈ 1:3
        E_kq =  -im * E_k[k][q] # E_kq = im * sum_{field i} (wave vector i in direction k) * (E field i in spherical component q)
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            val_re = E_kq_re * d_re - E_kq_im * d_im
            val_im = E_kq_re * d_im + E_kq_im * d_re
            ∇H_k.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            ∇H_k.im[n,m] += -val_im
            ∇H_k.re[m,n] += -val_re
            ∇H_k.im[m,n] -= -val_im
        end
        
    end
    
end
export calculate_grad_H!;





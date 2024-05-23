
function schrodinger_stochastic_constant_diffusion(
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

    
    ∇H = SVector{3, ComplexF64}(0,0,0) # gradient of the ODT Hamiltonian = ∇H * H_ODT. ∇H is just a 3-vector

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
        last_decay_time=0.0,
        diffusion_constant = sim_params.diffusion_constant
        )

    return p
end
export schrodinger_stochastic_constant_diffusion


function SE_collapse_pol_constant_diffusion!(integrator)

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

    dp = sample_direction(1)
    dv = dp ./ p.mass
    integrator.u[n_states + n_excited + 4] += dv[1]
    integrator.u[n_states + n_excited + 5] += dv[2]
    integrator.u[n_states + n_excited + 6] += dv[3]
    
    time_before_decay = integrator.t - p.last_decay_time

    for i ∈ 1:3
        integrator.u[p.n_states + p.n_excited + 3 + i] += rand((-1,1)) * sqrt( 2p.diffusion_constant * time_before_decay ) / p.mass
    end

    p.last_decay_time = integrator.t
    
    p.time_to_decay = rand(p.decay_dist)
    return nothing
end
export SE_collapse_pol_constant_diffusion!;


function state_overlap(state1, state2)
    """ Helper function that returns <state1|state2> """
    overlap = 0.0 *im
    @turbo for i ∈ eachindex(state1)
        state1_re = state1.re[i]
        state1_im = state1.im[i]
        state2_re = state2.re[i]
        state2_im = state2.im[i]
        overlap += state1_re*state2_re + state1_im*state2_im + im*(state1_re*state2_im - state2_re*state1_im)
    end
    return overlap
end

function operator_matrix_expectation_complex(O, state)
    """ Helper function that returns <state|O|state> """
    O_re = 0.0
    O_im = 0.0
    @turbo for i ∈ eachindex(state)
        re_i = state.re[i]
        im_i = state.im[i]
        for j ∈ eachindex(state)
            re_j = state.re[j]
            im_j = state.im[j]
            cicj_re = re_i * re_j + im_i * im_j # real part of ci* * cj
            cicj_im = re_i * im_j - im_i * re_j
            O_re += O.re[i,j] * cicj_re - O.im[i,j] * cicj_im
            O_im += O.re[i,j] * cicj_im + O.im[i,j] * cicj_re
        end
    end
    return (O_re, O_im)
end

function operator_matrix_expectation_complex_v2(O, state)
    """ Helper function that returns <state|O|state> """
    O_re = 0.0
    O_im = 0.0
    @turbo for i ∈ eachindex(state)
        re_i = state.re[i]
        im_i = state.im[i]
        for j ∈ eachindex(state)
            re_j = state.re[j]
            im_j = state.im[j]
            cicj_re = re_i * re_j + im_i * im_j # real part of ci* * cj
            cicj_im = re_i * im_j - im_i * re_j
            O_re += O.re[i,j] * cicj_re - O.im[i,j] * cicj_im
            O_im += O.re[i,j] * cicj_im + O.im[i,j] * cicj_re
        end
    end
    return O_re + im * O_im
end
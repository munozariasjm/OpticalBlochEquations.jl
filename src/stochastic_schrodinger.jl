using Distributions, StatsBase

function SE_collapse!(integrator)
    """ Periodically called by the solver. Random spontaneous emission. Restart excited state population averaging. """
   
    rn = rand()  # sample from random variable uniform([0,1]).
    
    n_states = length(integrator.p.states)
    n_excited = integrator.p.extra_p.n_excited
    
    # excited state population integrated over dT  
    lower_baseline = 0.0
    upper_baseline = 0.0
    
    for i in 1:n_excited
        upper_baseline += norm(integrator.u[n_states + i])
        if lower_baseline <= rn < upper_baseline

            i_ground, δm = spontaneous_emission_event(integrator.p, i+n_states-n_excited)
            
            # reset state to i_ground
            for i in 1:n_states
                integrator.u[i] = 0.0 #-= integrator.u[i]
            end
            
            integrator.u[i_ground] = 1

            dp = sample_direction(1)
            dv = dp ./ integrator.p.mass
            integrator.u[n_states + n_excited + 4] += dv[1]
            integrator.u[n_states + n_excited + 5] += dv[2]
            integrator.u[n_states + n_excited + 6] += dv[3]
            
            integrator.p.n_scatters += 1
            break
        end
        lower_baseline += norm(integrator.u[n_states + i])
    end
    
    # reset excited state population accumulation
    integrator.u[n_states + 1: n_states + n_excited] .= 0
    
    for i in 1:n_states
        integrator.p.populations[i] = norm(integrator.u[i])^2
    end
end
export SE_collapse! 
    
function spontaneous_emission_event(p, i_excited)
    """ 
        Excited state i_excited sponatneously emits. Randomly sample which ground state it decays into, 
        return the ground state index and change in m_F (which is relevant in determining the direction of
        momentum kick).
        
    """
    n_states = length(p.states)
    
    transition_probs = norm.(p.d[:,i_excited,:]).^2
    w = weights(transition_probs)
    i = sample(w)
    δm = -((i-1)÷n_states - 2)
    i_ground = (i-1) % n_states + 1
    # @printf("decay from %i to %i", i_excited, i_ground)
    # println()

    return (i_ground, δm)
end

uniform_dist = Uniform(0, 2π)
function sample_direction(r=1.0)
    θ = 2π * rand()
    z = rand() * 2 - 1
    return (r * sqrt(1 - z^2) * cos(θ), r * sqrt(1 - z^2) * sin(θ), r * z)
end

function schrodinger_stochastic(
    particle, states, fields, d, ψ₀, mass; 
    extra_p=nothing, λ=1.0, Γ=2π, update_H=update_H)
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
    if n_fields > 0
        fields = StructArray(fields)
    end
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
    H₀ = deepcopy(H)

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
            J = Jump(s, s′, q, dme)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    ψ_soa = StructArray(ψ₀)
    dψ_soa = StructArray(ψ₀)
    
    n_excited = extra_p.n_excited
    
    # ψ contains the state vector, accumulated excited state populations, position, velocity.
    ψ = zeros(ComplexF64, n_states + n_excited + 6+3)
    ψ[1:n_states] .= ψ₀
    ψ[n_states + n_excited + 1: n_states + n_excited + 3] .= r
    ψ[n_states + n_excited + 4: n_states + n_excited + 6] .= v
    dψ = deepcopy(ψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, λ=λ,
        k=k, H₀=H₀, 
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        extra_p=extra_p, mass = mass, update_H = update_H, populations = zeros(Float64, n_states),
        n_scatters = 0)

    return p
end
export schrodinger_stochastic

function ψ_stochastic!(dψ, ψ, p, τ)
    @unpack ψ_soa, dψ_soa, r, H₀, ω, fields, H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
    
    n_states = length(states)
    n_excited = extra_p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]),real(ψ[n_states + n_excited + 2]),real(ψ[n_states + n_excited + 3]))

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
    Heisenberg!(H, eiωt, -1)

    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    # calculate force
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)
    
    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states-n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + i + 3] = f[i]/mass # update velocity
    end

    ψ[end-2:end] .= f
    dψ[end-2:end] .= 0
    
    return nothing
end
export ψ_stochastic!

function force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa,eiωt)
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds for q ∈ 1:3
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
                
                ρ_mn = conj(ψ_soa[n]*eiωt[n]) * ψ_soa[m]*eiωt[m]
                ρ_re = real(ρ_mn)
                ρ_im = imag(ρ_mn)
                
                d_re = ds_q_re[j]
                d_im = ds_q_im[j]
                a1 = d_re * ρ_re - d_im * ρ_im
                a2 = d_re * ρ_im + d_im * ρ_re
                F_k_re += E_kq_re * a1 - E_kq_im * a2
                F_k_im += E_kq_im * a1 + E_kq_re * a2     
                
            end
           
            F += (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end

    F += conj(F)
    
    return real.(F)
end
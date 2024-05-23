function schrodinger_stochastic_nocomplex(
    particle, states, fields, d, ψ₀, mass, n_excited;
    sim_params=nothing, extra_data=nothing, λ=1.0, Γ=2π, update_H_and_∇H=update_H_and_∇H)

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
    H₀ = deepcopy(H)
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
        time_to_decay=rand(decay_dist)
        )

    return p
end
export schrodinger_stochastic_nocomplex

function ψ_stochastic_nocomplex!(du, u, p, t)
    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ, ψ, dψ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    ψ_norm = 0.0
    for i ∈ 1:2n_states
        ψ_norm += norm(u[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:2n_states
        u[i] /= ψ_norm
    end

    # convert u to a complex vector ψ
    for i ∈ 1:n_states
        ψ[i] = u[2i-1] + im * u[2i]
    end
    for i ∈ (n_states+1):length(ψ)
        j = 2n_states + (i - n_states)
        ψ[i] = u[j]
    end

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

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i]/mass # update velocity
    end

    # convert dψ to du
    for i ∈ 1:n_states
        du[2i-1] = real(dψ[i])
        du[2i] = imag(dψ[i])
    end
    for i ∈ (n_states+1):length(ψ)
        j = 2n_states + (i - n_states)
        du[j] = real(dψ[i])
    end

    return nothing
end
export ψ_stochastic_nocomplex!

function condition_nocomplex_discrete(u,t,integrator)
    p = integrator.p
    ψ = p.ψ
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[2p.n_states+i])
    end
    _condition = integrated_excited_pop - p.time_to_decay

    # terminate if the particle is more than 20mm from the centre
    # r = 0.0
    # for i ∈ 1:3
    #     r += norm(u[p.n_states + p.n_excited + i])^2
    # end
    # r = sqrt(r)
    # if r >= 5e-3 * integrator.p.k
    #    terminate!(integrator)
    # end
    
    return _condition > 0
end
export condition_nocomplex_discrete

function condition_nocomplex(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[2p.n_states+i])
    end
    _condition = integrated_excited_pop - p.time_to_decay

    # terminate if the particle is more than 20mm from the centre
    # r = 0.0
    # for i ∈ 1:3
    #     r += norm(u[p.n_states + p.n_excited + i])^2
    # end
    # r = sqrt(r)
    # if r >= 5e-3 * integrator.p.k
    #    terminate!(integrator)
    # end
    
    return _condition
end
export condition_nocomplex

function SE_collapse_pol_always_nocomplex!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = p.ψ
    
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

     # reset excited state population accumulation
    # integrator.u[n_states + 1:n_states + n_excited] .= 0
    
#     for i in 1:n_states
#         integrator.p.populations[i] = norm(integrator.u[i])^2
#     end
    
    dp = sample_direction(1)
    dv = dp ./ p.mass
    ψ[n_states + n_excited + 4] += 2 * dv[1]
    ψ[n_states + n_excited + 5] += 2 * dv[2]
    ψ[n_states + n_excited + 6] += 2 * dv[3]

    p.time_to_decay = rand(p.decay_dist)

    # convert ψ to u
    for i ∈ 1:n_states
        integrator.u[2i-1] = real(ψ[i])
        integrator.u[2i] = imag(ψ[i])
    end
    for i ∈ (n_states+1):length(ψ)
        j = 2n_states + (i - n_states)
        integrator.u[j] = real(ψ[i])
    end

    return nothing
end
export SE_collapse_pol_always_nocomplex!

# p = schrodinger_stochastic(particle, states, lasers, d, ψ₀, m/(ħ*k^2/Γ), n_excited; sim_params=sim_params, extra_data=extra_data, λ=λ, Γ=Γ, update_H_and_∇H=update_H_and_∇H)
# u = zeros(Float64, 2n_states + n_excited + 3 + 3 + 3)
# u[1] = 1.0

# prob_nocomplex = ODEProblem(ψ_stochastic_nocomplex!, u, t_span, p)

# # create a manifold callback
# function g(resid, u, p, t)
#     population = zero(Float64)
#     for i ∈ eachindex(resid)
#         resid[i] = 0.0
#     end
#     for i ∈ 1:32
#         population += norm(u[i])^2
#     end
#     resid[1] = population - 1
#     return nothing
# end
        
# cb_manifold = ManifoldProjection(g)
# cb = ContinuousCallback(condition_nocomplex, SE_collapse_pol_always_nocomplex!, nothing, save_positions=(false,false))
# # cb = DiscreteCallback(condition_nocomplex_discrete, SE_collapse_pol_always_nocomplex!, save_positions=(false,false))

# cb_set = CallbackSet(cb_manifold, cb)

# @time sol_nocomplex = DifferentialEquations.solve(prob_nocomplex, alg=DP5(), reltol=5e-4, callback=cb_set, save_on=false, dense=false)
# ;

# N = 20
# A = rand(Float64, N, N)
# B = rand(Float64, N)
# C = rand(Float64, N, N)
# @btime A * B

# A_static = SMatrix{N,N}(A)
# B_static = SVector{N}(B)
# C_static = SMatrix{N,N}(C)
# @btime C_static = A_static * B_static

# A_mutable = MMatrix{N,N}(A)
# B_mutable = MVector{N}(B)
# C_mutable = MMatrix{N,N}(C)
# @btime C_mutable = A_mutable * B_mutable

# # A_soa = StructArray(A)
# # B_soa = StructArray(B)
# # C_soa = StructArray(C)
# # @btime A_soa * B_soa
# ;

# ψ_static = SA[p.ψ...]

# prob = ODEProblem(ψ_stochastic_oop!, ψ_static, t_span, p)

# @time sol = DifferentialEquations.solve(prob, alg=DP5(), reltol=5e-4, callback=cb, saveat=1000, maxiters=200000000, progress=true, progress_steps=2000000)
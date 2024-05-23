

using
    QuantumStates,
    OpticalBlochEquations,
    DifferentialEquations,
    UnitsToValue,
    LinearAlgebra,
    Printf,
    Plots,
    # DiffEqNoiseProcess,
    Random,
    StatsBase
;

using Distributions

import MutableNamedTuples: MutableNamedTuple
import StructArrays: StructArray, StructVector
import StaticArrays: @SVector, SVector
import LinearAlgebra: norm, ⋅, adjoint!, diag
import LoopVectorization: @turbo
using BenchmarkTools
using Parameters
using LsqFit
import OpticalBlochEquations: extend_operator

import ProgressMeter: Progress, next!

function ψ_stochastic_continuous_diffusion!(dψ, ψ, p, t)
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
         rand1 = rand()
        diffusion_direction = 1.0
        if rand1 < 0.5
            diffusion_direction= -1
        end
        dψ[n_states + n_excited + 3 + i] = f[i] / mass + diffusion_direction / sqrt(p.dt * p.diffusion_constant * 2)  / mass # update velocity
    end

    # update force
    # ψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= f/ψ_norm
    # dψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0
    
    return nothing
end

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
        diffusion_constant = sim_params.diffusion_constant,
        dt = sim_params.dt
        )

    return p
end



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
    
    # rand1 = rand()
    # diffusion_direction = 1.0
    # if rand1 < 0.5
    #     diffusion_direction= -1
    # end
    # integrator.u[n_states + n_excited + 4] += p.diffusion_constant*sqrt(time_before_decay)/p.mass * diffusion_direction
    
    # rand1 = rand()
    # diffusion_direction = 1.0
    # if rand1 < 0.5
    #     diffusion_direction= -1
    # end
    # integrator.u[n_states + n_excited + 5] += p.diffusion_constant*sqrt(time_before_decay)/p.mass * diffusion_direction
    
    # rand1 = rand()
    # diffusion_direction = 1.0
    # if rand1 < 0.5
    #     diffusion_direction= -1
    # end
    # integrator.u[n_states + n_excited + 6] += p.diffusion_constant*sqrt(time_before_decay)/p.mass * diffusion_direction
    
    p.last_decay_time = integrator.t
    
    p.time_to_decay = rand(p.decay_dist)
    return nothing
end


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


function update_H_and_∇H(H, p, r, t)
    
    # Define a ramping magnetic field
    Zeeman_Hz = p.extra_data.Zeeman_Hz
    Zeeman_Hx = p.extra_data.Zeeman_Hx
    Zeeman_Hy = p.extra_data.Zeeman_Hy
    
    τ_bfield = p.sim_params.B_ramp_time 
    scalar = t/τ_bfield
    scalar = min(scalar, 1.0)
    
    gradient_x = -scalar * p.sim_params.B_gradient * 1e2 / k/2
    gradient_y = +scalar * p.sim_params.B_gradient * 1e2 / k/2
    gradient_z = -scalar * p.sim_params.B_gradient * 1e2 / k
    
    Bx = gradient_x * r[1] + p.sim_params.B_offset[1]
    By = gradient_y * r[2] + p.sim_params.B_offset[2]
    Bz = gradient_z * r[3] + p.sim_params.B_offset[3]
    
    @turbo for i in eachindex(H)
        H.re[i] = Bz * Zeeman_Hz.re[i] + Bx * Zeeman_Hx.re[i] + By * Zeeman_Hy.re[i]
        H.im[i] = Bz * Zeeman_Hz.im[i] + Bx * Zeeman_Hx.im[i] + By * Zeeman_Hy.im[i]
    end
    
    # Update the Hamiltonian for the molecule-ODT interaction
    H_ODT = p.extra_data.H_ODT_static
    
    ODT_size = p.sim_params.ODT_size .* p.k
    update_ODT_center!(p, t)
    ODT_x = p.extra_data.ODT_position[1] * p.k
    ODT_z = p.extra_data.ODT_position[2] * p.k
    
    scalar_ODT = exp(-2(r[1]-ODT_x)^2/ODT_size[1]^2) * exp(-2r[2]^2/ODT_size[2]^2) * exp(-2(r[3]-ODT_z)^2/ODT_size[3]^2)
    
    @turbo for i in eachindex(H)
        H.re[i] += H_ODT.re[i] * scalar_ODT
        H.im[i] += H_ODT.im[i] * scalar_ODT
    end
    
    # return SVector{3,ComplexF64}(0,0,0)
    return SVector{3,ComplexF64}((-4(r[1]-ODT_x) / ODT_size[1]^2) * scalar_ODT, (-4r[2] / ODT_size[2]^2) * scalar_ODT, (-4(r[3]-ODT_z) / ODT_size[3]^2) * scalar_ODT)
    
end


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
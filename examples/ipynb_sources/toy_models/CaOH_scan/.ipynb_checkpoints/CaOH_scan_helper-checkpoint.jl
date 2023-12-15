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

import ProgressMeter: Progress, next!

function get_CaOH_package()
    QN_bounds = (
        S = 1/2, 
        I = 1/2, 
        Λ = 0, 
        N = 0:3
    )
    X_state_basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

    X_state_operator = :(
        BX * Rotation + 
        DX * RotationDistortion + 
        γX * SpinRotation + 
        bFX * Hyperfine_IS + 
        cX * (Hyperfine_Dipolar/3)
    )

    X_state_parameters = QuantumStates.@params begin
        BX = 0.33441 * 299792458 * 1e-4
        DX = 0.3869e-6 * 299792458 * 1e-4
        γX = 0.001134 * 299792458 * 1e-4
        bFX = 2.602
        cX = 2.053
    end

    X_state_ham = Hamiltonian(basis=X_state_basis, operator=X_state_operator, parameters=X_state_parameters)

    # Add Zeeman terms
    Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1))/sqrt(2)
    Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1))/sqrt(2)
    Zeeman_z(state, state′) = Zeeman(state, state′, 0)

    X_state_ham = add_to_H(X_state_ham, :B_x, (gS * _μB * 1e-6) * Zeeman_x)
    X_state_ham = add_to_H(X_state_ham, :B_y, (gS * _μB * 1e-6) * Zeeman_y)
    X_state_ham = add_to_H(X_state_ham, :B_z, (gS * _μB * 1e-6) * Zeeman_z)
    X_state_ham.parameters.B_x = 0.
    X_state_ham.parameters.B_y = 0.
    X_state_ham.parameters.B_z = 0.

    evaluate!(X_state_ham)
    QuantumStates.solve!(X_state_ham)
    ;

    QN_bounds = (
        S = 1/2,
        I = 1/2,
        Λ = (-1,1),
        J = 1/2:5/2
    )
    A_state_basis = enumerate_states(HundsCaseA_Rot, QN_bounds)

    A_state_operator = :(
        T_A * DiagonalOperator +
        Be_A * Rotation + 
        Aso_A * SpinOrbit +
        q_A * ΛDoubling_q +
        p_A * ΛDoubling_p2q + q_A * (2ΛDoubling_p2q)
    )

    # Spectroscopic constants for CaOH, A state
    A_state_parameters = QuantumStates.@params begin
        T_A = 15998.122 * 299792458 * 1e-4
        Be_A = 0.3412200 * 299792458 * 1e-4
        Aso_A = 66.8181 * 299792458 * 1e-4
        p_A = -0.04287 * 299792458 * 1e-4
        q_A = -0.3257e-3 * 299792458 * 1e-4
    end

    A_state_ham = Hamiltonian(basis=A_state_basis, operator=A_state_operator, parameters=A_state_parameters)
    evaluate!(A_state_ham)
    QuantumStates.solve!(A_state_ham)
    ;

    A_state_J12_pos_parity_states = A_state_ham.states[5:8]

    QN_bounds = (
        S = 1/2, 
        I = 1/2, 
        Λ = (-1,1), 
        N = 0:3
    )
    A_state_caseB_basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

    ground_states = X_state_ham.states[5:16]
    excited_states = convert_basis(A_state_J12_pos_parity_states, A_state_caseB_basis)

    states = [ground_states; excited_states]
    n_excited = length(excited_states)

    for state ∈ states
        state.E *= 1e6
    end
    ;

    d = zeros(ComplexF64, 16, 16, 3)
    d_ge = zeros(ComplexF64, 12, 4, 3)

    basis_tdms = get_tdms_two_bases(X_state_ham.basis, A_state_caseB_basis, TDM)
    tdms_between_states!(d_ge, basis_tdms, ground_states, excited_states)
    d[1:12, 13:16, :] .= d_ge
    
    Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1))/sqrt(2)
    Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1))/sqrt(2)
    Zeeman_z(state, state′) = Zeeman(state, state′, 0)

    Zeeman_x_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_x, ground_states, excited_states) .* (2π*gS*_μB/Γ))
    Zeeman_y_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_y, ground_states, excited_states) .* (2π*gS*_μB/Γ))
    Zeeman_z_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_z, ground_states, excited_states) .* (2π*gS*_μB/Γ))

    package = MutableNamedTuple(states=states, n_excited=n_excited, d=d, Zeeman_x_mat=Zeeman_x_mat, Zeeman_y_mat=Zeeman_y_mat, Zeeman_z_mat=Zeeman_z_mat)
    return package
end;

function update_H(H, p, r, τ)
    Zeeman_Hz = p.extra_p.Zeeman_Hz
    Zeeman_Hx = p.extra_p.Zeeman_Hx
    Zeeman_Hy = p.extra_p.Zeeman_Hy
    
    gradient_z = p.extra_p.gradient_z
    gradient_x = p.extra_p.gradient_x
    gradient_y = p.extra_p.gradient_y
    @turbo for i in eachindex(H)
        H.re[i] = gradient_z * Zeeman_Hz.re[i] * r[3] + gradient_x * Zeeman_Hx.re[i] * r[1] + gradient_y * Zeeman_Hy.re[i] * r[2]
        H.im[i] = gradient_z * Zeeman_Hz.im[i] * r[3] + gradient_x * Zeeman_Hx.im[i] * r[1] + gradient_y * Zeeman_Hy.im[i] * r[2]
    end
    return nothing
end

function randomize_initial_vector!(p, r_dist, v_dist)
    n_excited = extra_p.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = rand(r_dist)*k
    p.ψ[n_states + n_excited + 2] = rand(r_dist)*k
    p.ψ[n_states + n_excited + 3] = rand(r_dist)*k
    p.ψ[n_states + n_excited + 4] = rand(v_dist)*k/Γ
    p.ψ[n_states + n_excited + 5] = rand(v_dist)*k/Γ
    p.ψ[n_states + n_excited + 6] = rand(v_dist)*k/Γ
end


function operator_to_matrix(A, states)
    """
    Write an operator as a matrix in basis {states}.
    """
    n_states = length(states)
    A_mat = zeros(ComplexF64, n_states, n_states)
    for i in 1:n_states
        for j in 1:n_states
            A_mat[i,j] = extend_operator(A, states[i], states[j])
        end
    end
    return A_mat
end

function operator_to_matrix_zero_padding2(OA, A_states, B_states)
    """
    OA is an operator on Hilbert space A (basis = A_states).
    We would like to extend A to the direct-sum space A ⨁ B by padding with zeros, i.e.
    <i|OAB|j> = 0 if i∉A or j∉A, <i|OAB|j> = <i|OA|j> if i∈A and j∈A.
    """
    n_A = length(A_states)
    n_B = length(B_states)
    OAB_mat = zeros(ComplexF64, n_A+n_B, n_A+n_B)
    OAB_mat[1:n_A, 1:n_A] .= operator_to_matrix(OA, A_states)
    return OAB_mat
end


function condition(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[p.n_states+i])
    end
    _condition = integrated_excited_pop - p.extra_p.time_to_decay
    
    n_states = length(integrator.p.states)
    n_excited = integrator.p.n_excited
    r = sqrt(sum(norm.(u[n_states + n_excited + 1: n_states + n_excited + 3]).^2))
    if r >= 4e-3*k # terminate if the particle is more than 4mm from the centre
        # println(u[n_states + n_excited + 1: n_states + n_excited + 3]./k)
       terminate!(integrator) 
    end
    return _condition
end

function init_isotropic_MOT_distribution(T, diameter)
    kB = 1.381e-23
    m = @with_unit 57 "u"
    σ = sqrt(kB * T / m)
    
    r = Normal(0, diameter)
    v = Normal(0, σ)
    return r, v
end

function init_MOT_distribution(T, diameter,displacement,kick)
    kB = 1.381e-23
    m = @with_unit 57 "u"
    σ = sqrt(kB * T / m)
    
    r = Normal(displacement, diameter)
    v = Normal(kick, σ)
    return r, v
end

function randomize_initial_vector!(p, x_dist, y_dist, z_dist, vx_dist, vy_dist, vz_dist)
    n_excited = p.extra_p.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = rand(x_dist)*k
    p.ψ[n_states + n_excited + 2] = rand(y_dist)*k
    p.ψ[n_states + n_excited + 3] = rand(z_dist)*k
    p.ψ[n_states + n_excited + 4] = rand(vx_dist)*k/Γ
    p.ψ[n_states + n_excited + 5] = rand(vy_dist)*k/Γ
    p.ψ[n_states + n_excited + 6] = rand(vz_dist)*k/Γ
end

function fixed_initial_vector!(p, x, y, z, vx, vy, vz)
     n_excited = extra_p.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = x*k
    p.ψ[n_states + n_excited + 2] = y*k
    p.ψ[n_states + n_excited + 3] = z*k
    p.ψ[n_states + n_excited + 4] = vx*k/Γ
    p.ψ[n_states + n_excited + 5] = vy*k/Γ
    p.ψ[n_states + n_excited + 6] = vz*k/Γ
end

function random_initial_state!(p)
    n_excited = p.extra_p.n_excited
    n_states = length(p.states)
   rn = rand() * (n_states - n_excited)
    i = Int(floor(rn))+1
    p.ψ[1:n_states].=0.0
    p.ψ[i] = 1.0
end

flip(ϵ) = (ϵ == σ⁻) ? σ⁺ : σ⁻

decay_dist = Exponential(1)

function make_problem_with_param(molecule_package, param)    
    t_end = param.t_end
    pol1_x, pol2_x, pol3_x, pol4_x = param.pol1_x, param.pol2_x, param.pol3_x, param.pol4_x

    s1, s2, s3, s4 = param.s1, param.s2, param.s3, param.s4
    Δ1, Δ2, Δ3, Δ4 = param.Δ1, param.Δ2, param.Δ3, param.Δ4
    B_gradient = param.B_gradient
    ramp_time = param.ramp_time
    displacement = param.displacement
    kick = param.kick
    
    x_dist, vx_dist = init_MOT_distribution(temp, diameter, displacement[1], kick[1])
    y_dist, vy_dist = init_MOT_distribution(temp, diameter, displacement[2], kick[2])
    z_dist, vz_dist = init_MOT_distribution(temp, diameter, displacement[3], kick[3])
  
    
   
    states = molecule_package.states
    n_excited = molecule_package.n_excited
    d = molecule_package.d
    Zeeman_x_mat = molecule_package.Zeeman_x_mat
    Zeeman_y_mat = molecule_package.Zeeman_y_mat
    Zeeman_z_mat = molecule_package.Zeeman_z_mat
    
    
    n_states = length(states)
    particle = OpticalBlochEquations.Particle()
    particle.r = (0.6e-3, 0.6e-3, 0.6e-3) ./ (1/k)
    particle.v = (0, 0, 0) ./ (Γ/k)
    ψ₀ = zeros(ComplexF64, n_states)
    ψ₀[1] = 1.0
    H₀ = zeros(ComplexF64, n_states, n_states)

    extra_p = MutableNamedTuple(
        Zeeman_Hx=Zeeman_x_mat,
        Zeeman_Hy=Zeeman_y_mat,
        Zeeman_Hz=Zeeman_z_mat,
        gradient_z=- B_gradient*1e2/k,
        gradient_x=- B_gradient*1e2/k/2,
        gradient_y=B_gradient * 1e2/k/2, # should be -1 but has wrong sign of Zeeman interaction
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        n_excited = n_excited,
         ramp_time = ramp_time*Γ
        )

    dT = 0.1
    save_every = 10000
   

    t_span = (0, t_end) ./ (1/Γ);
    
    ω1 = 2π * (energy(states[end]) - energy(states[1])) + Δ1
    ω2 = 2π * (energy(states[end]) - energy(states[1])) + Δ2
    ω3 = 2π * (energy(states[end]) - energy(states[5])) + Δ3
    ω4 = 2π * (energy(states[end]) - energy(states[5])) + Δ4 #Γ - 2π*1.5e6

    ϵ_(ϵ, f) = t -> ϵ
    s_func(s) = t -> s
    # ϵ_(ϵ, f) = t -> exp(-im*2π*f*t/50) .* ϵ
    ϕs = [exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand())]
    # ϕs = [1,1,1,1,1,1]
    
 
    

    
    
    
    k̂ = +x̂; ϵ1 = ϕs[1]*rotate_pol(pol1_x, k̂); ϵ_func1 = ϵ_(ϵ1, 1.00); laser1 = Field(k̂, ϵ_func1, ω1, s_func(s1))
    k̂ = -x̂; ϵ2 = ϕs[2]*rotate_pol(pol1_x, k̂); ϵ_func2 = ϵ_(ϵ2, 0.99); laser2 = Field(k̂, ϵ_func2, ω1, s_func(s1))
    k̂ = +ŷ; ϵ3 = ϕs[3]*rotate_pol(pol1_x, k̂); ϵ_func3 = ϵ_(ϵ3, 0.98); laser3 = Field(k̂, ϵ_func3, ω1, s_func(s1))
    k̂ = -ŷ; ϵ4 = ϕs[4]*rotate_pol(pol1_x, k̂); ϵ_func4 = ϵ_(ϵ4, 0.97); laser4 = Field(k̂, ϵ_func4, ω1, s_func(s1))
    k̂ = +ẑ; ϵ5 = ϕs[5]*rotate_pol(flip(pol1_x), k̂); ϵ_func5 = ϵ_(ϵ5, 0.96); laser5 = Field(k̂, ϵ_func5, ω1, s_func(s1))
    k̂ = -ẑ; ϵ6 = ϕs[6]*rotate_pol(flip(pol1_x), k̂); ϵ_func6 = ϵ_(ϵ6, 0.95); laser6 = Field(k̂, ϵ_func6, ω1, s_func(s1))

    lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]

    k̂ = +x̂; ϵ7 = ϕs[1]*rotate_pol(pol2_x, k̂); ϵ_func7 = ϵ_(ϵ7, 0.94); laser7 = Field(k̂, ϵ_func7, ω2, s_func(s2))
    k̂ = -x̂; ϵ8 = ϕs[2]*rotate_pol(pol2_x, k̂); ϵ_func8 = ϵ_(ϵ8, 0.93); laser8 = Field(k̂, ϵ_func8, ω2, s_func(s2))
    k̂ = +ŷ; ϵ9 = ϕs[3]*rotate_pol(pol2_x, k̂); ϵ_func9 = ϵ_(ϵ9, 0.92); laser9 = Field(k̂, ϵ_func9, ω2, s_func(s2))
    k̂ = -ŷ; ϵ10 = ϕs[4]*rotate_pol(pol2_x, k̂); ϵ_func10 = ϵ_(ϵ10, 0.91); laser10 = Field(k̂, ϵ_func10, ω2, s_func(s2))
    k̂ = +ẑ; ϵ11 = ϕs[5]*rotate_pol(flip(pol2_x), k̂); ϵ_func11 = ϵ_(ϵ11, 0.90); laser11 = Field(k̂, ϵ_func11, ω2, s_func(s2))
    k̂ = -ẑ; ϵ12 = ϕs[6]*rotate_pol(flip(pol2_x), k̂); ϵ_func12 = ϵ_(ϵ12, 0.89); laser12 = Field(k̂, ϵ_func12, ω2, s_func(s2))

    lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]

    k̂ = +x̂; ϵ13 = ϕs[1]*rotate_pol(pol3_x, k̂); ϵ_func13 = ϵ_(ϵ13, 0.88); laser13 = Field(k̂, ϵ_func13, ω3, s_func(s3))
    k̂ = -x̂; ϵ14 = ϕs[2]*rotate_pol(pol3_x, k̂); ϵ_func14 = ϵ_(ϵ14, 0.87); laser14 = Field(k̂, ϵ_func14, ω3, s_func(s3))
    k̂ = +ŷ; ϵ15 = ϕs[3]*rotate_pol(pol3_x, k̂); ϵ_func15 = ϵ_(ϵ15, 0.86); laser15 = Field(k̂, ϵ_func15, ω3, s_func(s3))
    k̂ = -ŷ; ϵ16 = ϕs[4]*rotate_pol(pol3_x, k̂); ϵ_func16 = ϵ_(ϵ16, 0.85); laser16 = Field(k̂, ϵ_func16, ω3, s_func(s3))
    k̂ = +ẑ; ϵ17 = ϕs[5]*rotate_pol(flip(pol3_x), k̂); ϵ_func17 = ϵ_(ϵ17, 0.84); laser17 = Field(k̂, ϵ_func17, ω3, s_func(s3))
    k̂ = -ẑ; ϵ18 = ϕs[6]*rotate_pol(flip(pol3_x), k̂); ϵ_func18 = ϵ_(ϵ18, 0.83); laser18 = Field(k̂, ϵ_func18, ω3, s_func(s3))

    lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]

    k̂ = +x̂; ϵ19 = ϕs[1]*rotate_pol(pol4_x, k̂); ϵ_func19 = ϵ_(ϵ19, 0.82); laser19 = Field(k̂, ϵ_func19, ω4, s_func(s4))
    k̂ = -x̂; ϵ20 = ϕs[2]*rotate_pol(pol4_x, k̂); ϵ_func20 = ϵ_(ϵ20, 0.81); laser20 = Field(k̂, ϵ_func20, ω4, s_func(s4))
    k̂ = +ŷ; ϵ21 = ϕs[3]*rotate_pol(pol4_x, k̂); ϵ_func21 = ϵ_(ϵ21, 0.80); laser21 = Field(k̂, ϵ_func21, ω4, s_func(s4))
    k̂ = -ŷ; ϵ22 = ϕs[4]*rotate_pol(pol4_x, k̂); ϵ_func22 = ϵ_(ϵ22, 0.79); laser22 = Field(k̂, ϵ_func22, ω4, s_func(s4))
    k̂ = +ẑ; ϵ23 = ϕs[5]*rotate_pol(flip(pol4_x), k̂); ϵ_func23 = ϵ_(ϵ23, 0.78); laser23 = Field(k̂, ϵ_func23, ω4, s_func(s4))
    k̂ = -ẑ; ϵ24 = ϕs[6]*rotate_pol(flip(pol4_x), k̂); ϵ_func24 = ϵ_(ϵ24, 0.77); laser24 = Field(k̂, ϵ_func24, ω4, s_func(s4))

    lasers_4 = [laser19, laser20, laser21, laser22, laser23, laser24]

    lasers = [lasers_1;lasers_2; lasers_3; lasers_4]
        
    p = schrodinger_stochastic(particle, states, lasers, d, ψ₀, m/(ħ*k^2/Γ), dT, save_every, n_excited;
    extra_p=extra_p, λ=λ, Γ=Γ, update_H=update_H)

    prob = ODEProblem(ψ_stochastic!, p.ψ, t_span, p)

    randomize_initial_vector!(prob.p, x_dist, y_dist, z_dist, vx_dist, vy_dist, vz_dist)
    # fixed_initial_vector!(prob.p, 1e-3, 1e-3, 1e-3, 0,0,0)
    random_initial_state!(prob.p)
    
    return prob
end

function simulate_particles(package, params, n_values)
    
    n_threads=Threads.nthreads()
    batch_size = fld(n_values, n_threads)
    remainder = n_values - batch_size * n_threads
    prog_bar = Progress(n_values)

    n_states = length(package.states)
    n_excited = package.n_excited

    x_trajectories = Array{Vector{Float64}}(undef, n_values) 
    y_trajectories = Array{Vector{Float64}}(undef, n_values) 
    z_trajectories = Array{Vector{Float64}}(undef, n_values) 
    x_velocities = Array{Vector{Float64}}(undef, n_values) 
    y_velocities = Array{Vector{Float64}}(undef, n_values) 
    z_velocities = Array{Vector{Float64}}(undef, n_values) 
    A_populations = Array{Vector{Float64}}(undef, n_values) 
    times = Array{Vector{Float64}}(undef, n_values) 

    prob = make_problem_with_param(package, params);
    cb = ContinuousCallback(condition, SE_collapse_pol_always!, nothing, save_positions=(false,false))

    Threads.@threads for i ∈ 1:n_threads
        cb_copy = deepcopy(cb)
    
        _batch_size = i <= remainder ? (batch_size + 1) : batch_size
        batch_start_idx = 1 + (i <= remainder ? (i - 1) : remainder) + batch_size * (i-1)

        for j ∈ batch_start_idx:(batch_start_idx + _batch_size - 1) 

            prob_copy = make_problem_with_param(package, params);

            sol = DifferentialEquations.solve(prob_copy, alg=DP5(), reltol=1e-3, callback=cb_copy, saveat=10000, maxiters=10000000)
            plot_us = sol.u
            plot_ts = sol.t

            x_trajectories[j] =  [real(u[n_states + n_excited + 1]) for u in plot_us]./k*1e3
            y_trajectories[j] = [real(u[n_states + n_excited + 2]) for u in plot_us]./k*1e3
            z_trajectories[j] = [real(u[n_states + n_excited + 3]) for u in plot_us]./k*1e3
            x_velocities[j] = [real(u[n_states + n_excited + 4]) for u in plot_us]./k*Γ
            y_velocities[j] = [real(u[n_states + n_excited + 5]) for u in plot_us]./k*Γ
            z_velocities[j] = [real(u[n_states + n_excited + 6]) for u in plot_us]./k*Γ
            times[j] =  plot_ts./Γ*1e3
            A_populations[j] = [sum(real.(norm.(u[n_states - n_excited + 1 : n_states]).^2)) for u in plot_us]

            next!(prog_bar)
        end
    end
    
    trapped_indicies = []
    for j in eachindex(y_trajectories)
        if sqrt(x_trajectories[j][end]^2 + y_trajectories[j][end]^2 + z_trajectories[j][end]^2) <= 1#mm
           push!(trapped_indicies, j) 
        end
    end

    common_length = 0
    plot_ts = []
    if length(trapped_indicies) > 0
        common_length = length(x_trajectories[trapped_indicies[1]])
        for j in trapped_indicies
           if length(times[j]) < common_length
                common_length = length(times[j])
            end
        end

        plot_ts = times[1]
        for j in trapped_indicies
           if length(times[j]) >= common_length
                plot_ts = times[j]
                break
            end
        end

        plot_ts = plot_ts[1:common_length] 
    end

    # @printf("Number molecules trapped: %i out of %i", length(trapped_indicies), n_values)
    
    results = MutableNamedTuple(x_trajectories = x_trajectories, y_trajectories= y_trajectories, z_trajectories=z_trajectories,
                                x_velocities = x_velocities, y_velocities=y_velocities, z_velocities=z_velocities,
                                times=times, A_populations=A_populations, trapped_indicies = trapped_indicies, 
                                common_length=common_length, plot_ts=plot_ts, n_values=n_values)
    return results
end
;
function plot_all_trajectories(results, direction)
    if direction == "x"
        plot(legend=false, title="x position", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.x_trajectories[j])
        end
    elseif direction == "y"
        plot(legend=false, title="y position", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.y_trajectories[j])
        end
    elseif direction == "z"
        plot(legend=false, title="z position", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.z_trajectories[j])
        end
    end    
end

function plot_trajectories(results, direction)
    if length(results.trapped_indicies) == 0
        println("No trapped particles.")
        return nothing
    end
    
    if direction == "x"
        plot(legend=false, title="x position", xlabel="time (ms)",ylabel="position (mm)")
        for j in results.trapped_indicies
           plot!(results.times[j], results.x_trajectories[j])
        end
    elseif direction == "y"
        plot(legend=false, title="y position", xlabel="time (ms)",ylabel="position (mm)")
        for j in results.trapped_indicies
           plot!(results.times[j], results.y_trajectories[j])
        end
    elseif direction == "z"
        plot(legend=false, title="z position", xlabel="time (ms)",ylabel="position (mm)")
        for j in results.trapped_indicies
           plot!(results.times[j], results.z_trajectories[j])
        end
    end
end

function plot_velocities(results, direction)
    if length(results.trapped_indicies) == 0
        println("No trapped particles.")
        return nothing
    end
    
   if direction == "x"
        plot(legend=false, title="x velocity", xlabel="time (ms)",ylabel="v (m/s)")
        v_end = Float64[]
        for j in results.trapped_indicies
           plot!(results.times[j], results.x_velocities[j]) 
            push!(v_end, results.x_velocities[j][results.common_length])
        end
        v2 = std(v_end)^2
        k_B = 1.381e-23
        T =  m*v2/k_B
        return T
    elseif direction == "y"
        plot(legend=false, title="y velocity", xlabel="time (ms)",ylabel="v (m/s)")
        v_end = Float64[]
        for j in results.trapped_indicies
           plot!(results.times[j], results.y_velocities[j]) 
            push!(v_end, results.y_velocities[j][results.common_length])
        end
        v2 = std(v_end)^2
        k_B = 1.381e-23
        T =  m*v2/k_B
        return T
        
    elseif direction == "z"
        
        plot(legend=false, title="z velocity", xlabel="time (ms)",ylabel="v (m/s)")
        v_end = Float64[]
        for j in results.trapped_indicies
           plot!(results.times[j], results.z_velocities[j]) 
            push!(v_end, results.z_velocities[j][results.common_length])
        end
        v2 = std(v_end)^2
        k_B = 1.381e-23
        T =  m*v2/k_B
        return T
    end
end

function plot_scattering_rate(results)
    if length(results.trapped_indicies) == 0
        # println("No trapped particles.")
        return 0.0
    end
    
   plot(legend=false, title="average scattering rate", xlabel="time (ms)",ylabel="Scattering Rate (MHz)")
    avg_scattering = Float64[]

    for i in 1:results.common_length
       push!(avg_scattering, mean([results.A_populations[j][i].*Γ for j in results.trapped_indicies]))
    end

    plot!(results.plot_ts, avg_scattering*1e-6) 
    return mean(avg_scattering) * results.plot_ts[end]*1e-3 # avg number of photons scattered
end


function plot_size(results, direction,do_plot=true)
    if length(results.trapped_indicies) == 0
        # println("No trapped particles.")
        return NaN
    end
    
   if direction == "x"
        σ_x = Float64[]

        for i in 1:results.common_length
           push!(σ_x, std([results.x_trajectories[j][i] for j in results.trapped_indicies]))
        end
        if do_plot
            plot(legend=false, title="x_size", xlabel="time (ms)",ylabel="size (mm)")
            plot!(results.plot_ts, σ_x)
        end
        return σ_x[end]
    elseif direction == "y"

        σ_y = Float64[]

        for i in 1:results.common_length
           push!(σ_y, std([results.y_trajectories[j][i] for j in results.trapped_indicies]))
        end
        if do_plot
            plot(legend=false, title="y_size", xlabel="time (ms)",ylabel="size (mm)")
            plot!(results.plot_ts, σ_y) 
        end
        return σ_y[end]
    elseif direction == "z"
        σ_z = Float64[]
        for i in 1:results.common_length
           push!(σ_z, std([results.z_trajectories[j][i] for j in results.trapped_indicies]))
        end
        if do_plot
            plot!(results.plot_ts, σ_z) 
        end
        return σ_z[end]
    elseif direction == "all"
        if do_plot
             
            
        end
        σ_x = Float64[]
        for i in 1:results.common_length
           push!(σ_x, std([results.x_trajectories[j][i] for j in results.trapped_indicies]))
        end
        
        σ_y = Float64[]
        for i in 1:results.common_length
           push!(σ_y, std([results.y_trajectories[j][i] for j in results.trapped_indicies]))
        end
        
        
        σ_z = Float64[]
        for i in 1:results.common_length
           push!(σ_z, std([results.z_trajectories[j][i] for j in results.trapped_indicies]))
        end
        
        if do_plot
            plot(legend=false, title="cloud size", xlabel="time (ms)",ylabel="size (mm)")
            plot!(results.plot_ts, σ_x, label="x")
            plot!(results.plot_ts, σ_y, label="y") 
            plot!(results.plot_ts, σ_z, label="z") 
        end
        
        return (σ_x[end],σ_y[end],σ_z[end])
        
    end
end


function plot_temperature(results, direction)
    if length(results.trapped_indicies) == 0
        # println("No trapped particles.")
        return NaN
    end
    
    k_B = 1.381e-23
   if direction == "x"
        plot(legend=false, title="x temperature", xlabel="time (ms)",ylabel="temperature (μK)")
        σ_v = []
        for i in 1:results.common_length
           push!(σ_v, std([results.x_velocities[j][i] for j in results.trapped_indicies]))
        end
        plot!(results.plot_ts, m*σ_v.^2/k_B*1e6)
        return m*σ_v[end].^2/k_B*1e6
    elseif direction == "y"
        plot(legend=false, title="y temperature", xlabel="time (ms)",ylabel="temperature (μK)")
        σ_v = []
        for i in 1:results.common_length
           push!(σ_v, std([results.y_velocities[j][i] for j in results.trapped_indicies]))
        end
        plot!(results.plot_ts, m*σ_v.^2/k_B*1e6)
        return m*σ_v[end].^2/k_B*1e6
    elseif direction == "z"
        plot(legend=false, title="z temperature", xlabel="time (ms)",ylabel="temperature (μK)")
        σ_v = []
        for i in 1:results.common_length
           push!(σ_v, std([results.z_velocities[j][i] for j in results.trapped_indicies]))
        end
        plot!(results.plot_ts, m*σ_v.^2/k_B*1e6)
        return  m*σ_v[end].^2/k_B*1e6
    end
end
;



using Serialization

function log_test_info(saving_dir, test_i, params)
    # make a new folder under saving_dir for this test
    folder = @sprintf("test%d", test_i)
    while isdir(joinpath(saving_dir, folder))
        test_i += 1
        folder = @sprintf("test%d", test_i)
    end
    folder_dir = joinpath(saving_dir, folder)
    mkdir(folder_dir)
    
    # save current parameters to .jl file
    serialize(joinpath(folder_dir, "params.jl"), params)
    
    write_test_info(saving_dir, test_i, params)
    
    return test_i
end

function load_test_params(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    params = deserialize(joinpath(folder_dir, "params.jl"))
    return params
end

function save_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    serialize(joinpath(folder_dir, "results.jl"), results)
end

function pol2str(pol)
    if pol == σ⁺
        return "+"
    elseif pol == σ⁻
        return "-"
    end
end

function display_test_info(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    params = deserialize(joinpath(folder_dir, "params.jl"))
    header = "-"^50
    println(header)
    @printf("test %d information", test_i)
    println()
    println(header)
    @printf("propagation time = %.2f ms", params.t_end*1e3)
    println()
    @printf("particle number = %d", params.n_values)
    println()
    println(header)
    
    @printf("Laser parameters:")
    println()
    @printf("Polarizations (+x beam): %s, %s, %s, %s", 
            pol2str(params.pol1_x), pol2str(params.pol2_x), pol2str(params.pol3_x), pol2str(params.pol4_x))
    println()
    @printf("Detunings (MHz): %.2f, %.2f, %.2f, %.2f", params.Δ1/(2π)/1e6, params.Δ2/(2π)/1e6, params.Δ3/(2π)/1e6, params.Δ4/(2π)/1e6)
    println()
    @printf("Saturations: %.2f, %.2f, %.2f, %.2f", params.s1, params.s2, params.s3, params.s4)
    println()
    println(header)
    
    @printf("max B field gradient: (%.2f, %.2f, %.2f) G/cm", -params.B_gradient/2, params.B_gradient/2, -params.B_gradient)
    println()
    @printf("B field ramp time: %.1f ms", params.ramp_time*1e3)
    println()
    println(header)
    
    println("Initial state:")
    @printf("Cloud radius = %.2f mm", params.diameter*1e3)
    println()
    @printf("Cloud temperature = %.2f mK", params.temp*1e3)
    println()
    @printf("Displacement from centre = (%.2f, %.2f, %.2f) mm", params.displacement[1],params.displacement[2],params.displacement[3])
    println()
    @printf("Centre of mass velocity = (%.2f, %.2f, %.2f) m/s", params.kick[1], params.kick[2], params.kick[3])
    println()
    println(header)
end
;


function write_summarized_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    
    open(joinpath(folder_dir, "results.txt"), "w") do file 
        header = ("-"^50)*"\n"
        write(file, header)
        
        write(file, @sprintf("test %d results \n", test_i))

        write(file, header)
        
        write(file, @sprintf("Molecules trapped: %i out of %i \n", length(results.trapped_indicies), results.n_values))
        
        x = plot_size(results, "x")
        y = plot_size(results, "y")
        z = plot_size(results, "z")
        write(file, @sprintf("Final cloud size: (%.2f, %.2f, %.2f) mm \n", x, y, z))
        
        Tx = plot_temperature(results, "x")
        Ty = plot_temperature(results, "y")
        Tz = plot_temperature(results, "z")
        write(file, @sprintf("Final temperature: (%.2f, %.2f, %.2f) μK \n", Tx, Ty, Tz))
        
        n_photon = plot_scattering_rate(results)
        write(file, @sprintf("Average photons scattered: %.0f \n", n_photon))
        if length(results.trapped_indicies)!= 0
            write(file, @sprintf("Average scattering rate: %.3f MHz \n", n_photon / (results.plot_ts[end]*1e-3) * 1e-6))
        end
        
        write(file, header)
    end
end


function save_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    serialize(joinpath(folder_dir, "results.jl"), results)
    write_summarized_results(saving_dir, test_i, results)
end

function load_results(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    results = deserialize(joinpath(folder_dir, "results.jl"))
    return results
end

function summarize_results(results)
    header = "-"^50
    println(header)
    @printf("Molecules trapped: %i out of %i", length(results.trapped_indicies), results.n_values)
    println()
    
    x = plot_size(results, "x")
    y = plot_size(results, "y")
    z = plot_size(results, "z")
    @printf("Final cloud size: (%.2f, %.2f, %.2f) mm", x, y, z)
    println()
    
    Tx = plot_temperature(results, "x")
    Ty = plot_temperature(results, "y")
    Tz = plot_temperature(results, "z")
    @printf("Final temperature: (%.2f, %.2f, %.2f) μK", Tx, Ty, Tz)
    println()
    
    n_photon = plot_scattering_rate(results)
    @printf("Average photons scattered: %i", n_photon)
    println()
    @printf("Average scattering rate: %.3f MHz", n_photon / (results.plot_ts[end]*1e-3) * 1e-6)
    println()
    
    println(header)
end


function summarize_results(saving_dir, test_i)
    header = "-"^50
    println(header)
    @printf("test %d results", test_i)
    println()
   results = load_results(saving_dir, test_i)
    summarize_results(results)
end

function make_scan_folder(lists, working_dir, scan_i, comments)
    folder = @sprintf("scan%d", scan_i)
    while isdir(joinpath(working_dir, folder))
        scan_i += 1
        folder = @sprintf("scan%d", scan_i)
    end
    folder_dir = joinpath(working_dir, folder)
    mkdir(folder_dir)
    
    serialize(joinpath(folder_dir, "lists.jl"), lists)
    
    open(joinpath(folder_dir, "comments.txt"), "w") do file
        write(file, comments)
    end;
    return folder_dir
end;

function write_test_info(saving_dir, test_i, params)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    
    open(joinpath(folder_dir, "info.txt"), "w") do file  
        header = ("-"^50)*"\n"
        write(file, header)

        write(file, @sprintf("test %d information \n", test_i))

        write(file, header)

        write(file, @sprintf("propagation time = %.2f ms \n", params.t_end*1e3))

        write(file, @sprintf("particle number = %d \n", params.n_values))

        write(file, header)
        
        write(file, "Laser parameters:\n")
        
        write(file, @sprintf("Polarizations (+x beam): %s, %s, %s, %s \n", 
            pol2str(params.pol1_x), pol2str(params.pol2_x), pol2str(params.pol3_x), pol2str(params.pol4_x)))
    
        write(file, @sprintf("Detunings (MHz): %.2f, %.2f, %.2f, %.2f \n", params.Δ1/(2π)/1e6, params.Δ2/(2π)/1e6, params.Δ3/(2π)/1e6, params.Δ4/(2π)/1e6))

        write(file, @sprintf("Saturations: %.2f, %.2f, %.2f, %.2f \n", params.s1, params.s2, params.s3, params.s4))

        write(file, header)
        
        write(file,  @sprintf("max B field gradient: (%.2f, %.2f, %.2f) G/cm \n", -params.B_gradient/2, params.B_gradient/2, -params.B_gradient))
    
        write(file, @sprintf("B field ramp time: %.1f ms \n", params.ramp_time*1e3))

        write(file, header)
        
        write(file, "Initial state: \n")
    
        write(file, @sprintf("Cloud radius = %.2f mm \n", params.diameter*1e3))
        
        write(file, @sprintf("Cloud temperature = %.2f mK \n", params.temp*1e3))
    
        write(file, @sprintf("Displacement from centre = (%.2f, %.2f, %.2f) mm \n", params.displacement[1],params.displacement[2],params.displacement[3]))
 
        write(file, @sprintf("Centre of mass velocity = (%.2f, %.2f, %.2f) m/s \n", params.kick[1], params.kick[2], params.kick[3]))
    
        write(file, header)
    end;
end



function reanalyze_results!(results, r_max)
    results.trapped_indicies = []
    for j in eachindex(results.y_trajectories)
        if sqrt(results.x_trajectories[j][end]^2 + results.y_trajectories[j][end]^2 + results.z_trajectories[j][end]^2) <= r_max#mm
           push!(results.trapped_indicies, j) 
        end
    end

    common_length = 0
    results.plot_ts = []
    if length(results.trapped_indicies) > 0
        common_length = length(results.x_trajectories[results.trapped_indicies[1]])
        for j in results.trapped_indicies
           if length(results.times[j]) < common_length
                common_length = length(results.times[j])
            end
        end

        results.plot_ts = results.times[1]
        for j in results.trapped_indicies
           if length(results.times[j]) >= common_length
                results.plot_ts = results.times[j]
                break
            end
        end

        results.plot_ts = results.plot_ts[1:common_length] 
    end
    
end
;
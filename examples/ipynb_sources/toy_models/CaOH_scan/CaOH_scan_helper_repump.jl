# -*- coding: utf-8 -*-

# Updates from v1:
# * Ramp the magnetic field
# * Add third frequency
# * Account for losses due to finite photon budget

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
        BX = 10023.0841
        DX = 1.154e-2
        γX = 34.7593
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
    
    τ_bfield = p.extra_p.ramp_time / (1/Γ)
    scalar = τ/τ_bfield
    scalar = min(scalar, 1.0)
    
    gradient_z = scalar * p.extra_p.gradient_z
    gradient_x = scalar * p.extra_p.gradient_x
    gradient_y = scalar * p.extra_p.gradient_y
    
    Bz = gradient_z * r[3] + p.extra_p.Bz_offset
    Bx = gradient_x * r[1] + p.extra_p.Bx_offset
    By = gradient_y * r[2] + p.extra_p.By_offset
    @turbo for i in eachindex(H)
        H.re[i] = Bz * Zeeman_Hz.re[i]  + Bx * Zeeman_Hx.re[i] + By * Zeeman_Hy.re[i]
        H.im[i] = Bz * Zeeman_Hz.im[i] + Bx * Zeeman_Hx.im[i] + By * Zeeman_Hy.im[i]
    end
    return nothing
end;

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
    if integrator.p.is_dark == true
        _condition = t-p.dark_time_t0 - p.dark_time
    else
        _condition = integrated_excited_pop - p.time_to_decay
    end
    
    r = 0.0
    for i ∈ 1:3
        r += norm(u[p.n_states + p.n_excited + i])^2
    end
    r = sqrt(r)
    if r >= 8e-3*k # terminate if the particle is more than 3 mm from the centre
       terminate!(integrator) 
    elseif integrator.p.n_scatters > integrator.p.extra_p.photon_budget # also terminate if too many photons have been scattered
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

# flip(ϵ) = (ϵ == σ⁻) ? σ⁺ : σ⁻
function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function gaussian_intensity_along_axes(r, axes, centers)
    """1/e^2 width = 5mm Gaussian beam """
    d2 = (r[axes[1]] - centers[1])^2 + (r[axes[2]] - centers[2])^2   
    return exp(-2*d2/(5e-3/(1/k))^2)
end




function make_problem_with_param(molecule_package, param; variable_budget=false)    
    t_end = param.t_end
    pol1_x, pol2_x, pol3_x, pol4_x = param.pol1_x, param.pol2_x, param.pol3_x, param.pol4_x

    s1, s2, s3, s4 = param.s1, param.s2, param.s3, param.s4
    s_ramp_time = param.s_ramp_time * Γ
    s_ramp_factor = param.s_ramp_to_factor
    
    Δ1, Δ2, Δ3, Δ4 = param.Δ1, param.Δ2, param.Δ3, param.Δ4
    B_gradient = param.B_gradient
    temp = param.temp
    diameter = param.diameter
    ramp_time = param.ramp_time
    displacement = param.displacement
    kick = param.kick
    
    imbalance = param.pol_imbalance
    sx_imbalance, sy_imbalance, sz_imbalance = param.s_imbalance[1], param.s_imbalance[2], param.s_imbalance[3] 
    x_center_y, x_center_z, y_center_x, y_center_z, z_center_x, z_center_y = param.off_center .* k
    retro_loss = param.retro_loss
    
    x_center_y *= rand()
    x_center_z *= rand()
    y_center_x *= rand()
    y_center_z *= rand()
    z_center_x *= rand()
    z_center_y *= rand()
    
    dark_lifetime = param.dark_lifetime
    FC_mainline = param.FC_mainline
    
    x_dist, vx_dist = init_MOT_distribution(temp, diameter, displacement[1], kick[1])
    y_dist, vy_dist = init_MOT_distribution(temp, diameter, displacement[2], kick[2])
    z_dist, vz_dist = init_MOT_distribution(temp, diameter, displacement[3], kick[3])
  
    states = molecule_package.states
    n_excited = molecule_package.n_excited
    d = molecule_package.d
    Zeeman_x_mat = molecule_package.Zeeman_x_mat
    Zeeman_y_mat = molecule_package.Zeeman_y_mat
    Zeeman_z_mat = molecule_package.Zeeman_z_mat
    
    kx = x̂ + [0, param.pointing_error[1],param.pointing_error[2]]
    kx = kx ./ sqrt(kx[1]^2 + kx[2]^2 + kx[3]^2)
    ky = ŷ + [param.pointing_error[3],0.0,param.pointing_error[4]]
    ky = ky ./ sqrt(ky[1]^2 + ky[2]^2 + ky[3]^2)
    kz = ẑ + [param.pointing_error[5],param.pointing_error[6],0.0]
    kz = kz / sqrt(kz[1]^2 + kz[2]^2 + kz[3]^2)
    
    photon_budget = 14000
    
    if variable_budget == true
        photon_budget = param.photon_budget
    end
    
    n_states = length(states)
    particle = OpticalBlochEquations.Particle()
    ψ₀ = zeros(ComplexF64, n_states)
    ψ₀[1] = 1.0
    H₀ = zeros(ComplexF64, n_states, n_states)
    
    extra_p = MutableNamedTuple(
        Zeeman_Hx=Zeeman_x_mat,
        Zeeman_Hy=Zeeman_y_mat,
        Zeeman_Hz=Zeeman_z_mat,
        gradient_z = -B_gradient*1e2/k,
        gradient_x = -B_gradient*1e2/k/2,
        gradient_y = B_gradient*1e2/k/2, # should be -1 but has wrong sign of Zeeman interaction
        n_excited = n_excited,
        ramp_time = ramp_time,
        photon_budget = rand(Exponential(photon_budget)),
        Bz_offset = param.Bz_offset,
        Bx_offset = param.Bx_offset,
        By_offset = param.By_offset
        )

    t_span = (0, t_end) ./ (1/Γ);
    
    ω1 = 2π * (energy(states[end]) - energy(states[1])) + Δ1
    ω2 = 2π * (energy(states[end]) - energy(states[1])) + Δ2
    ω3 = 2π * (energy(states[end]) - energy(states[5])) + Δ3
    ω4 = 2π * (energy(states[end]) - energy(states[5])) + Δ4

    ϵ_(ϵ, f) = t -> ϵ
    s_func(s) = (x,t) -> s
    s_gaussian(s, axes, centers) = (r,t) -> s * gaussian_intensity_along_axes(r, axes, centers)
    
    s_gaussian_ramp(s, factor, ramp_time, axes, centers) = (r,t) -> ((s*factor-s)/ramp_time * min(t, ramp_time) + s) * gaussian_intensity_along_axes(r, axes, centers)
    # ϵ_(ϵ, f) = t -> exp(-im*2π*f*t/500) .* ϵ
    
    rand1 = rand()
    pol1_x = pol1_x.*sqrt(1 - rand1*imbalance) + flip(pol1_x).*sqrt(rand1*imbalance)
    rand2 = rand()
    pol2_x = pol2_x.*sqrt(1 - rand2*imbalance) + flip(pol2_x).*sqrt(rand2*imbalance)
    rand3 = rand()
    pol3_x = pol3_x.*sqrt(1 - rand3*imbalance)  + flip(pol3_x).*sqrt(rand3*imbalance)
    rand4 = rand()
    pol4_x = pol4_x.*sqrt(1 - rand4*imbalance) + flip(pol4_x).*sqrt(rand4*imbalance)
    
    sx_rand = 1/2-rand()
    sy_rand = 1/2-rand()
    sz_rand = 1/2-rand()
    
    ϕs = [exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand())]
    s1x = s1 * (1+sx_imbalance*sx_rand)
    s1y = s1 * (1+sy_imbalance*sy_rand)
    s1z = s1 * (1+sz_imbalance*sz_rand)
    k̂ = kx; ϵ1 = ϕs[1]*rotate_pol(pol1_x, k̂); ϵ_func1 = ϵ_(ϵ1, 1); laser1 = Field(k̂, ϵ_func1, ω1,  s_gaussian_ramp(s1x, s_ramp_factor, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ2 = ϕs[2]*rotate_pol(pol1_x, k̂); ϵ_func2 = ϵ_(ϵ2, 2); laser2 = Field(k̂, ϵ_func2, ω1, s_gaussian_ramp(s1x*(1-retro_loss), s_ramp_factor, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
    k̂ = ky; ϵ3 = ϕs[3]*rotate_pol(pol1_x, k̂); ϵ_func3 = ϵ_(ϵ3, 3); laser3 = Field(k̂, ϵ_func3, ω1,  s_gaussian_ramp(s1y, s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ4 = ϕs[4]*rotate_pol(pol1_x, k̂); ϵ_func4 = ϵ_(ϵ4, 4); laser4 = Field(k̂, ϵ_func4, ω1,  s_gaussian_ramp(s1y*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ5 = ϕs[5]*rotate_pol(flip(pol1_x), k̂); ϵ_func5 = ϵ_(ϵ5, 5); laser5 = Field(k̂, ϵ_func5, ω1,  s_gaussian_ramp(s1z, s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ6 = ϕs[6]*rotate_pol(flip(pol1_x), k̂); ϵ_func6 = ϵ_(ϵ6, 6); laser6 = Field(k̂, ϵ_func6, ω1, s_gaussian_ramp(s1z*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]

    s2x = s2 * (1+sx_imbalance*sx_rand)
    s2y = s2 * (1+sy_imbalance*sy_rand)
    s2z = s2 * (1+sz_imbalance*sz_rand)
    k̂ = +kx; ϵ7 = ϕs[1]*rotate_pol(pol2_x, k̂); ϵ_func7 = ϵ_(ϵ7, 1); laser7 = Field(k̂, ϵ_func7, ω2, s_gaussian_ramp(s2x,s_ramp_factor, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ8 = ϕs[2]*rotate_pol(pol2_x, k̂); ϵ_func8 = ϵ_(ϵ8, 2); laser8 = Field(k̂, ϵ_func8, ω2, s_gaussian_ramp(s2x*(1-retro_loss), s_ramp_factor, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = +ky; ϵ9 = ϕs[3]*rotate_pol(pol2_x, k̂); ϵ_func9 = ϵ_(ϵ9, 3); laser9 = Field(k̂, ϵ_func9, ω2, s_gaussian_ramp(s2y, s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ10 = ϕs[4]*rotate_pol(pol2_x, k̂); ϵ_func10 = ϵ_(ϵ10, 4); laser10 = Field(k̂, ϵ_func10, ω2, s_gaussian_ramp(s2y*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ11 = ϕs[5]*rotate_pol(flip(pol2_x), k̂); ϵ_func11 = ϵ_(ϵ11, 5); laser11 = Field(k̂, ϵ_func11, ω2, s_gaussian_ramp(s2z, s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ12 = ϕs[6]*rotate_pol(flip(pol2_x), k̂); ϵ_func12 = ϵ_(ϵ12, 6); laser12 = Field(k̂, ϵ_func12, ω2, s_gaussian_ramp(s2z*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]

    s3x = s3 * (1+sx_imbalance*sx_rand)
    s3y = s3 * (1+sy_imbalance*sy_rand)
    s3z = s3 * (1+sz_imbalance*sz_rand)
    k̂ = +kx; ϵ13 = ϕs[1]*rotate_pol(pol3_x, k̂); ϵ_func13 = ϵ_(ϵ13, 1); laser13 = Field(k̂, ϵ_func13, ω3, s_gaussian_ramp(s3x,s_ramp_factor, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ14 = ϕs[2]*rotate_pol(pol3_x, k̂); ϵ_func14 = ϵ_(ϵ14, 2); laser14 = Field(k̂, ϵ_func14, ω3, s_gaussian_ramp(s3x*(1-retro_loss),s_ramp_factor, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
    k̂ = +ky; ϵ15 = ϕs[3]*rotate_pol(pol3_x, k̂); ϵ_func15 = ϵ_(ϵ15, 3); laser15 = Field(k̂, ϵ_func15, ω3, s_gaussian_ramp(s3y, s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ16 = ϕs[4]*rotate_pol(pol3_x, k̂); ϵ_func16 = ϵ_(ϵ16, 4); laser16 = Field(k̂, ϵ_func16, ω3, s_gaussian_ramp(s3y*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ17 = ϕs[5]*rotate_pol(flip(pol3_x), k̂); ϵ_func17 = ϵ_(ϵ17, 5); laser17 = Field(k̂, ϵ_func17, ω3, s_gaussian_ramp(s3z, s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ18 = ϕs[6]*rotate_pol(flip(pol3_x), k̂); ϵ_func18 = ϵ_(ϵ18, 6); laser18 = Field(k̂, ϵ_func18, ω3, s_gaussian_ramp(s3z*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]

    s4x = s4 * (1+sx_imbalance*sx_rand)
    s4y = s4 * (1+sy_imbalance*sy_rand)
    s4z = s4 * (1+sz_imbalance*sz_rand)
    k̂ = +kx; ϵ19 = ϕs[1]*rotate_pol(pol4_x, k̂); ϵ_func19 = ϵ_(ϵ19, 1); laser19 = Field(k̂, ϵ_func19, ω4,s_gaussian_ramp(s4x,s_ramp_factor, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ20 = ϕs[2]*rotate_pol(pol4_x, k̂); ϵ_func20 = ϵ_(ϵ20, 2); laser20 = Field(k̂, ϵ_func20, ω4, s_gaussian_ramp(s4x*(1-retro_loss), s_ramp_factor, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = +ky; ϵ21 = ϕs[3]*rotate_pol(pol4_x, k̂); ϵ_func21 = ϵ_(ϵ21, 3); laser21 = Field(k̂, ϵ_func21, ω4, s_gaussian_ramp(s4y, s_ramp_factor, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ22 = ϕs[4]*rotate_pol(pol4_x, k̂); ϵ_func22 = ϵ_(ϵ22, 4); laser22 = Field(k̂, ϵ_func22, ω4, s_gaussian_ramp(s4y*(1-retro_loss),s_ramp_factor, s_ramp_time,   (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ23 = ϕs[5]*rotate_pol(flip(pol4_x), k̂); ϵ_func23 = ϵ_(ϵ23, 5); laser23 = Field(k̂, ϵ_func23, ω4, s_gaussian_ramp(s4z,s_ramp_factor, s_ramp_time,   (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ24 = ϕs[6]*rotate_pol(flip(pol4_x), k̂); ϵ_func24 = ϵ_(ϵ24, 6); laser24 = Field(k̂, ϵ_func24, ω4, s_gaussian_ramp(s4z*(1-retro_loss), s_ramp_factor, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_4 = [laser19, laser20, laser21, laser22, laser23, laser24]

    lasers = [lasers_1;lasers_2; lasers_3; lasers_4]
        
    p = schrodinger_stochastic_repump(particle, states, lasers, d, ψ₀, m/(ħ*k^2/Γ), n_excited;
    extra_p=extra_p, λ=λ, Γ=Γ, update_H=update_H, dark_lifetime=dark_lifetime, FC_mainline = FC_mainline)

    prob = ODEProblem(ψ_stochastic_repump!, p.ψ, t_span, p)

    randomize_initial_vector!(prob.p, x_dist, y_dist, z_dist, vx_dist, vy_dist, vz_dist)
    random_initial_state!(prob.p)
    
    return prob
end

function simulate_particles_repump(package, params; variable_budget=false)
    
    n_values = params.n_values
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
    photons_scattered = Array{Float64}(undef, n_values)

    cb = ContinuousCallback(condition, SE_collapse_repump!, nothing, save_positions=(false,false))

    Threads.@threads for i ∈ 1:n_threads
    
        _batch_size = i <= remainder ? (batch_size + 1) : batch_size
        batch_start_idx = 1 + (i <= remainder ? (i - 1) : remainder) + batch_size * (i-1)

        for j ∈ batch_start_idx:(batch_start_idx + _batch_size - 1) 

            prob_copy = make_problem_with_param(package, params, variable_budget=variable_budget)
            cb_copy = deepcopy(cb)
            
            sol = DifferentialEquations.solve(prob_copy, alg=DP5(), reltol=1e-3, callback=cb_copy, saveat=4000, maxiters=10000000)
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
            photons_scattered[j] = prob_copy.p.n_scatters

            next!(prog_bar)
        end
    end
    
    results = MutableNamedTuple(x_trajectories = x_trajectories, y_trajectories= y_trajectories, z_trajectories=z_trajectories,
                                x_velocities = x_velocities, y_velocities=y_velocities, z_velocities=z_velocities,
                                times=times, A_populations=A_populations,
                                n_values=n_values, photons_scattered=photons_scattered)
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
    elseif direction == "all"
        plot(legend=false, title="distance from centre", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], sqrt.(results.z_trajectories[j].^2 + results.x_trajectories[j].^2 + results.y_trajectories[j].^2))
        end
    end    
end
                                            
                                            
function plot_all_velocities(results, direction)
    if direction == "x"
        plot(legend=false, title="x velcocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.x_velocities[j])
        end
    elseif direction == "y"
        plot(legend=false, title="y velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.y_velocities[j])
        end
    elseif direction == "z"
        plot(legend=false, title="z velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.z_velocities[j])
        end
    elseif direction == "all"
        plot(legend=false, title="speed", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], sqrt.(results.z_velocities[j].^2 + results.x_velocities[j].^2 + results.y_velocities[j].^2))
        end
    end    
end
                                                                    

function plot_scattering_rate(results)
    plot(legend=false, xlabel="time (ms)", ylabel="MHz")
    for p in 1:length(results.times)
       plot!(results.times[p], results.A_populations[p] .* Γ * 1e-6)
    end
    plot!(title="scattering rate")

    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end

    out = []
    for i in 1:max_t_id
       n_total = 0
        n_num = 0.01
        for p in 1:length(results.times)
            if length(results.times[p]) >= i
                n_total += results.A_populations[p][i]
                n_num += 1
            end
        end
        push!(out, n_total/n_num)
    end
    plot!(plot_ts, out.*Γ*1e-6, linewidth=3, color=:red)
    return mean(out.*Γ*1e-6)                                                                    
end
                                                                                                
                                                                                                
function scattering_rate_at_t(results, t)
     max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end                                                                                               
     dt = plot_ts[2]-plot_ts[1]
     i = Int(t ÷ dt) + 1
                                                                                                            
      n_total = 0
      n_num = 0.01
      for p in 1:length(results.times)
          if length(results.times[p]) >= i
              n_total += results.A_populations[p][i]
              n_num += 1
          end
      end                                                                                                       
      return n_total/n_num *Γ*1e-6                                                                                                    
end                                                                                                
                                                                        
                                                                        
function plot_photons_scattered(results)
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end 
    trapped = []       
    out= []                                                                                
    for i in 1:length(results.times)
         if length(results.times[i]) == max_t_id   
             push!(trapped, i)     
               push!(out, results.photons_scattered[i])                                                                             
         end                                                                                   
    end 
   scatter(trapped, out, linewidth=2,title="photon scattered for survived molecules",xlabel="trapped particle index", size=(500,300),dpi=300, legend=false)
   return mean(out)                                                                                                                                                                         
end
                                                                            
function plot_size(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
                                                                                                                         
    if direction == "x"
         plot(legend=false, title="x_size", xlabel="time (ms)",ylabel="size (mm)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_trajectories)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.x_trajectories[j][i])
                    end    
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, σs)                                                                                                                           
          return plot_ts, σs
     elseif direction == "y"
         plot(legend=false, title="y_size", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            y_at_t = Float64[]
            for j in 1:length(results.y_trajectories)
                if length(results.y_trajectories[j]) >= i    
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0
                        push!(y_at_t, results.y_trajectories[j][i])     
                    end
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(y_at_t))
         end
                                                                                                                                                     plot!(plot_ts, σs)
                                                                                                                                                     return plot_ts, σs
     elseif direction == "z"
         plot(legend=false, title="z_size", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            z_at_t = Float64[]
            for j in 1:length(results.z_trajectories)
                if length(results.z_trajectories[j]) >= i  
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0
                        push!(z_at_t, results.z_trajectories[j][i]) 
                    end
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(z_at_t))
         end
         plot!(plot_ts, σs)
         return plot_ts, σs
     elseif direction == "all"
         
         plot_ts, σx = plot_size(results, "x")
         ~,  σy = plot_size(results, "y")
         ~,  σz = plot_size(results, "z")
         plot(legend=true, title="cloud size", xlabel="time (ms)",ylabel="size (mm)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz
     end  
     
 end
                                                                                                                                                
                                                                                                                                            function plot_centre(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
                                                                                                                         
    if direction == "x"
         plot(legend=false, title="x centre", xlabel="time (ms)",ylabel="size (mm)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_trajectories)
                if length(results.x_trajectories[j]) >= i                                                                                                          push!(x_at_t, results.x_trajectories[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(x_at_t))
         end
          plot!(plot_ts, σs)                                                                                                                           
          return plot_ts, σs
     elseif direction == "y"
         plot(legend=false, title="y centre", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            y_at_t = Float64[]
            for j in 1:length(results.y_trajectories)
                if length(results.y_trajectories[j]) >= i                                                                                                          push!(y_at_t, results.y_trajectories[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(y_at_t))
         end
                                                                                                                                                     plot!(plot_ts, σs)
                                                                                                                                                     return plot_ts, σs
     elseif direction == "z"
         plot(legend=false, title="z centre", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            z_at_t = Float64[]
            for j in 1:length(results.z_trajectories)
                if length(results.z_trajectories[j]) >= i                                                                                                          push!(z_at_t, results.z_trajectories[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(z_at_t))
         end
         plot!(plot_ts, σs)
         return plot_ts, σs
     elseif direction == "all"
         
         plot_ts, σx = plot_centre(results, "x")
         ~,  σy = plot_centre(results, "y")
         ~,  σz = plot_centre(results, "z")
         plot(legend=true, title="cloud centre", xlabel="time (ms)",ylabel="size (mm)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz
     end  
     
 end
 
 
 
 
 
 function plot_temperature(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
    k_B = 1.381e-23                                                                                                                     
    if direction == "x"
         plot(legend=false, title="x temperature", xlabel="time (ms)",ylabel="temperature (μK)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_velocities)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.x_velocities[j][i])
                    end    
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, m*σs.^2/k_B*1e6)                                                                                                                        
          return plot_ts, m*σs.^2/k_B*1e6
     elseif direction == "y"
         plot(legend=false, title="y temperature", xlabel="time (ms)",ylabel="temperature (μK)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.y_velocities)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.y_velocities[j][i])
                    end    
               end                                                                                                                      
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, m*σs.^2/k_B*1e6)                                                                                                                        
          return plot_ts, m*σs.^2/k_B*1e6
     elseif direction == "z"
         plot(legend=false, title="z temperature", xlabel="time (ms)",ylabel="temperature (μK)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.z_velocities)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.z_velocities[j][i])
                    end    
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, m*σs.^2/k_B*1e6)                                                                                                                        
          return plot_ts, m*σs.^2/k_B*1e6
     else 
         return [],[]
         
      end
     
     
 end



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
    if isdir(folder_dir) == false
        @printf("%s is not found.", folder_dir)
        println()
       return nothing 
    end
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
    @printf("Polarization imbalance: %.3f", params.pol_imbalance)
    println()
    @printf("Detunings (MHz): %.2f, %.2f, %.2f, %.2f", params.Δ1/(2π)/1e6, params.Δ2/(2π)/1e6, params.Δ3/(2π)/1e6, params.Δ4/(2π)/1e6)
    println()
    @printf("Saturations: %.2f, %.2f, %.2f, %.2f", params.s1, params.s2, params.s3, params.s4)
    println()
    @printf("Power imbalance (x,y,z): %.3f, %.3f, %.3f", params.s_imbalance[1], params.s_imbalance[2], params.s_imbalance[3])
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
        
        ~, x = plot_size(results, "x")
        ~, y = plot_size(results, "y")
        ~, z = plot_size(results, "z")
        write(file, @sprintf("Final cloud size: (%.2f, %.2f, %.2f) mm \n", x[end], y[end], z[end]))
        
        ~, Tx = plot_temperature(results, "x")
        ~, Ty = plot_temperature(results, "y")
        ~, Tz = plot_temperature(results, "z")
        write(file, @sprintf("Final temperature: (%.2f, %.2f, %.2f) μK \n", Tx[end], Ty[end], Tz[end]))
        
        n_photon = plot_photons_scattered(results)
        write(file, @sprintf("Average photons scattered: %.0f \n", n_photon))
        rate = plot_scattering_rate(results)
        write(file, @sprintf("Average scattering rate: %.3f MHz \n", rate))
        
        write(file, header)
    end
end


function save_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    serialize(joinpath(folder_dir, "results.jl"), results)
    # write_summarized_results(saving_dir, test_i, results)
end

function load_results(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    if isdir(folder_dir)==false
        @printf("%s is not found.", folder_dir)
        println()
       return nothing 
    end
    results = deserialize(joinpath(folder_dir, "results.jl"))
    return results
end

function summarize_results(results)
    header = "-"^50
    println(header)
#     @printf("Molecules trapped: %i out of %i", length(results.trapped_indicies), results.n_values)
#     println()
    
    ~,x = plot_size(results, "x")
    ~,y = plot_size(results, "y")
    ~,z = plot_size(results, "z")
    @printf("Final cloud size: (%.2f, %.2f, %.2f) mm", x[end], y[end], z[end])
    println()
    
    ~,Tx = plot_temperature(results, "x")
    ~,Ty = plot_temperature(results, "y")
    ~,Tz = plot_temperature(results, "z")
    @printf("Final temperature: (%.2f, %.2f, %.2f) μK", Tx[end], Ty[end], Tz[end])
    println()
    
    n_photon = plot_photons_scattered(results)
    @printf("Average photons scattered: %i", n_photon)
    println()
    rate = plot_scattering_rate(results)
    @printf("Average scattering rate: %.3f MHz", rate)
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
            
        write(file, @sprintf("Polarization imbalance: %.3f \n", params.pol_imbalance))
    
        write(file, @sprintf("Detunings (MHz): %.2f, %.2f, %.2f, %.2f \n", params.Δ1/(2π)/1e6, params.Δ2/(2π)/1e6, params.Δ3/(2π)/1e6, params.Δ4/(2π)/1e6))

        write(file, @sprintf("Saturations: %.2f, %.2f, %.2f, %.2f \n", params.s1, params.s2, params.s3, params.s4))
        
        write(file, @sprintf("Power imbalance (x, y, z): %.3f, %.3f, %.3f \n", params.s_imbalance[1], params.s_imbalance[2], params.s_imbalance[3]))

        write(file, header)
        
        write(file,  @sprintf("max B field gradient: (%.2f, %.2f, %.2f) G/cm \n", -params.B_gradient/2, params.B_gradient/2, -params.B_gradient))
    
        write(file, @sprintf("B field ramp time: %.1f ms \n", params.ramp_time*1e3))

        write(file, header)
        
        write(file, "Initial state: \n")
    
        write(file, @sprintf("Cloud radius = %.2f mm \n", params.diameter*1e3))
        
        write(file, @sprintf("Cloud temperature = %.2f mK \n", params.temp*1e3))
    
        write(file, @sprintf("Displacement from centre = (%.2f, %.2f, %.2f) mm \n", params.displacement[1]*1e3,params.displacement[2]*1e3,params.displacement[3]*1e3))
 
        write(file, @sprintf("Centre of mass velocity = (%.2f, %.2f, %.2f) m/s \n", params.kick[1], params.kick[2], params.kick[3]))
    
        write(file, header)
    end;
end
;

# function survived(idx, times, trajectories)
#     _survived = Int64[]
#     for i ∈ eachindex(trajectories)
#         if length(times[i]) >= idx
#             push!(_survived, i)
#         end
#     end
#     return _survived
# end

# function cloud_size(trajectories, i)
#     std(trajectory[i] for trajectory ∈ trajectories)
# end

# function density_gaussian(idx, times, x_trajectories, y_trajectories, z_trajectories)
#     _survived = survived(idx, times, x_trajectories)
#     surviving_x_trajectories = x_trajectories[_survived]
#     surviving_y_trajectories = y_trajectories[_survived]
#     surviving_z_trajectories = z_trajectories[_survived]
    
#     n = length(_survived)
#     _density = Float64(n)
#     if n > 0
#         σ_x = cloud_size(surviving_x_trajectories, idx)
#         σ_y = cloud_size(surviving_y_trajectories, idx)
#         σ_z = cloud_size(surviving_z_trajectories, idx)
#         _density /= σ_x * σ_y * σ_z
#     end
#     return _density
# end
# ;

function get_points_from_results(results, it)
   points = []
    for i in 1:length(results.times)
        if it < length(results.times[i])
            push!(points, (results.x_trajectories[i][it], results.y_trajectories[i][it],results.z_trajectories[i][it]))
        end
    end
    return points
end;



function plot_survival(params, results; keep=false, label="")
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end

    dt = plot_ts[2] - plot_ts[1]
    time_plot = plot_ts

    survival = []

    for i in 1:Int(params.t_end*1e3 ÷ dt)
        points = get_points_from_results(results, i)
        num = 0
        for p in points
           if p[1]^2 + p[2]^2 + p[3]^2 < (2)^2
                num += 1
            end
        end
        push!(survival, length(points))
    end

    time_plot = LinRange(0, Int(params.t_end*1e3 ÷ dt)*dt, Int(params.t_end*1e3 ÷ dt))
    if keep
        plot!(time_plot,survival, linewidth=3, ylim=(0,survival[1]+5), label=label)
    else
        plot(time_plot,survival, linewidth=3,ylim=(0,survival[1]+5), label=label)
    end
    plot!(title="Survived molecules", xlabel="time (ms)", size = (400,300), legend = true, dpi=300)
    ;
    return time_plot, survival
end;


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
#         println()
#         println("Molecule out of jail.")
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
#             n_excited = integrator.p.n_excited
#             n_states = length(integrator.p.states)

#             @printf("Molecule put in jail at time %.1e", integrator.t / Γ)
#             println()
#             @printf("dark_time = %.1e", integrator.p.dark_time/Γ)
        end
    end
end;
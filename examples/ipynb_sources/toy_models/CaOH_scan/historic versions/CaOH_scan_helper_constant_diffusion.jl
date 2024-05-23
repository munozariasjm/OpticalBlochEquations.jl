# -*- coding: utf-8 -*-


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


function get_CaOH_package_12_A_states()
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

    A_state_J12_pos_parity_states = [A_state_ham.states[5:8];A_state_ham.states[9:16]] # J1/2 and J3/2

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
    n_ground = length(ground_states)
    n_excited = length(excited_states)

    for state ∈ states
        state.E *= 1e6
    end
    ;
    n_states = length(states)
    d = zeros(ComplexF64, n_states, n_states, 3)
    d_ge = zeros(ComplexF64, n_ground, n_excited, 3)

    basis_tdms = get_tdms_two_bases(X_state_ham.basis, A_state_caseB_basis, TDM)
    tdms_between_states!(d_ge, basis_tdms, ground_states, excited_states)
    d[1:n_ground, n_ground+1:n_states, :] .= d_ge
    
    package = MutableNamedTuple(states=states, n_excited=n_excited, d=d)
    return package
end;

# function ODT_radius(p, t)
#     t1 = min(t - p.ODT_motion_t_start*Γ,  p.ODT_motion_t_stop*Γ)
#     return p.ODT_rad * t1 / (p.τ_ODT / (1/Γ))
# end

# function ODT_center(p, t)
#     if t < p.ODT_motion_t_start*Γ
#         return 0.0, 0.0
#     end
#     t1 = min(t - p.ODT_motion_t_start*Γ,  p.ODT_motion_t_stop*Γ)
#     return ODT_radius(p, t1) * cos(2π * (p.ODT_freq / Γ) * t1), ODT_radius(p, t1) * sin(2π * (p.ODT_freq / Γ) * t1) # trace out a circle
# end

function update_ODT_center!(p1, t1)::Nothing
    p = p1.sim_params
    if t1 >= p.ODT_motion_t_start*Γ
        t = min(t1/Γ - p.ODT_motion_t_start,  p.ODT_motion_t_stop - p.ODT_motion_t_start )
        iτ = searchsortedfirst(p.interpolation_times,t)
        τ = p.interpolation_τs[iτ] + (p.interpolation_τs[iτ+1]-p.interpolation_τs[iτ])/(p.interpolation_times[iτ+1]-p.interpolation_times[iτ])*(t-p.interpolation_times[iτ])
        # τ = func_t_to_τ(t)
        p1.extra_data.ODT_position[1] = p.ODT_rmax * τ * cos(2*π*τ* p.ODT_revolutions)
        p1.extra_data.ODT_position[2] = p.ODT_rmax * τ * sin(2*π*τ* p.ODT_revolutions)
    end
    return nothing
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
;


function randomize_initial_vector!(p, r_dist, v_dist)
    n_excited = p.sim_params.n_excited
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
    
    _condition = integrated_excited_pop - p.time_to_decay
    
    r = 0.0
    for i ∈ 1:3
        r += norm(u[p.n_states + p.n_excited + i])^2
    end
    r = sqrt(r)
    if r >= 3e-3*k # terminate if the particle is more than 3 mm from the centre
       terminate!(integrator) 
    elseif integrator.p.n_scatters > integrator.p.sim_params.photon_budget # also terminate if too many photons have been scattered
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
    n_excited = p.sim_params.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = rand(x_dist)*k
    p.ψ[n_states + n_excited + 2] = rand(y_dist)*k
    p.ψ[n_states + n_excited + 3] = rand(z_dist)*k
    p.ψ[n_states + n_excited + 4] = rand(vx_dist)*k/Γ
    p.ψ[n_states + n_excited + 5] = rand(vy_dist)*k/Γ
    p.ψ[n_states + n_excited + 6] = rand(vz_dist)*k/Γ
end

function fixed_initial_vector!(p, x, y, z, vx, vy, vz)
     n_excited = p.sim_params.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = x*k
    p.ψ[n_states + n_excited + 2] = y*k
    p.ψ[n_states + n_excited + 3] = z*k
    p.ψ[n_states + n_excited + 4] = vx*k/Γ
    p.ψ[n_states + n_excited + 5] = vy*k/Γ
    p.ψ[n_states + n_excited + 6] = vz*k/Γ
end

function random_initial_state!(p)
    n_excited = p.sim_params.n_excited
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
                                                                                            
function plot_survival_velocities(results, direction)
     max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end                                                                                             
    if direction == "x"
        plot(legend=false, title="x velcocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           if length(results.times[j]) < max_t_id 
                continue
           end                                                                                                      
           plot!(results.times[j], results.x_velocities[j])
        end
    elseif direction == "y"
        plot(legend=false, title="y velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           if length(results.times[j]) < max_t_id 
                continue
           end                                                                                                             
               plot!(results.times[j], results.y_velocities[j])
        end
    elseif direction == "z"
        plot(legend=false, title="z velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           if length(results.times[j]) < max_t_id 
                continue
           end                                                                                                                      
           plot!(results.times[j], results.z_velocities[j])
        end
    elseif direction == "all"
        plot(legend=false, title="speed", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           if length(results.times[j] )< max_t_id
                continue
           end                                                                                                                              
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
     elseif direction == "all"
         plot_ts, σx = plot_temperature(results, "x")
         ~,  σy = plot_temperature(results, "y")
         ~,  σz = plot_temperature(results, "z")
         plot(legend=true, title="Temperature", xlabel="time (ms)",ylabel="T (uK)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz                                              
         
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
    @printf("B field ramp time: %.1f ms", params.B_ramp_time*1e3)
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
    
        write(file, @sprintf("B field ramp time: %.1f ms \n", params.B_ramp_time*1e3))

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

# function SE_collapse_repump!(integrator)
#     # go back to bright state
#     if integrator.p.is_dark == true
#         integrator.p.is_dark = false
#         n_states = length(integrator.p.states)
#         n_excited = integrator.p.n_excited
#         i = Int(floor(rand()*n_excited)) + 1
#         for i in 1:n_states
#             integrator.u[i] = 0.0
#         end
#         integrator.u[n_states - n_excited + i] = 1.0
#         println()
#         println("Molecule out of jail.")
#     else
#         # scatter
#        rn = rand()
#         if rn < integrator.p.FC_mainline # decay back to a cycling state
#             SE_collapse_pol_always!(integrator)
#         else # decay into a dark state and wait to be repumped
#             SE_collapse_pol_always!(integrator) # give a momentum kick
#             integrator.p.is_dark = true
#             integrator.p.dark_time = rand(integrator.p.dark_time_dist)
#             integrator.p.dark_time_t0 = integrator.t
# #             n_excited = integrator.p.n_excited
# #             n_states = length(integrator.p.states)

#             @printf("Molecule put in jail at time %.1e", integrator.t / Γ)
#             println()
#             @printf("dark_time = %.1e", integrator.p.dark_time/Γ)
#         end
#     end
# end;



function generate_sequence!(sequence, timing)
    t_last = 0.0
    i_last = 1
   for params in sequence
        params.t_start = t_last
        params.t_end = t_last + timing[i_last]
        t_last = t_last + timing[i_last]
        i_last += 1
    end
end;



function get_ODT_Hamiltonian_matrix(package, package_A, peak_intensity, pol, wavelength=1064e-9)
    Is = π*h*c*Γ/(3λ^3) / 10 # saturation intensity in mW/cm^2
    s = peak_intensity / Is
    f_ODT = c/wavelength 
    
    n_states = length(package.states)
    E0 = sqrt(s)/(2 * √2) # factor?
    H_ODT = zeros(ComplexF64, n_states, n_states)
    
    d = package_A.d
    fs = energy.(package_A.states)

    
    for q in 1:3
        for p in 1:3
            
            for i in 1:n_states
                for j in 1:n_states
                    for l in 1:length(package_A.states)
                    H_ODT[i,j] -= 2π * Γ * (E0^2/4 * d[min(i,l),max(i,l),q] * pol[q] * d[min(j,l),max(j,l),p] * pol[p]) * 
                                    (1/(fs[l]-fs[i] - f_ODT) + 1/(fs[l]-fs[i] + f_ODT))
                    end
                end
            end
            
        end
    end
    
    return H_ODT
end

function particles_in_ODT(results)
    indicies = []
    number = 0.0
   for i in 1:length(results.x_trajectories)
        if abs(results.x_trajectories[i][end]) < 0.05 && abs(results.z_trajectories[i][end]) < 0.05 
            push!(indicies, i)
            number += exp(-results.photons_scattered[i]/14000)
        end 
    end
    return number, indicies
end

function generate_sequence!(sequence, timing)
    t_last = 0.0
    i_last = 1
   for params in sequence
        params.t_start = t_last
        params.t_end = t_last + timing[i_last]
        t_last = t_last + timing[i_last]
        i_last += 1
    end
end


# function get_ODT_Hamiltonian_matrix(package, peak_intensity, pol, wavelength=1064e-9)
#     H_ODT = zeros(ComplexF64, n_states, n_states)
#     for i in 1:16
#         H_ODT[i,i] = -1.0
#     end
#     return H_ODT
# end
                                        
function bootstrap_iteration(results; n_values=nothing)
    if n_values == nothing
        n_values = length(results.times)
    end
    indicies = rand(1:length(results.times),n_values)
    results1 = deepcopy(results)
    results1.x_trajectories = Array{Vector{Float64}}(fill([],n_values))
    results1.y_trajectories = Array{Vector{Float64}}(fill([],n_values)) 
    results1.z_trajectories = Array{Vector{Float64}}(fill([],n_values))
    results1.x_velocities = Array{Vector{Float64}}(fill([],n_values))
    results1.y_velocities = Array{Vector{Float64}}(fill([],n_values))
    results1.z_velocities = Array{Vector{Float64}}(fill([],n_values))
    results1.times = Array{Vector{Float64}}(fill([],n_values))
    results1.photons_scattered = zeros(n_values)
    
    for i in 1:n_values
        i_bs = indicies[i]
        results1.x_trajectories[i] = results.x_trajectories[i_bs]
        results1.y_trajectories[i] = results.y_trajectories[i_bs]
        results1.z_trajectories[i] = results.z_trajectories[i_bs]
        results1.x_velocities[i] = results.x_velocities[i_bs]
        results1.y_velocities[i] = results.y_velocities[i_bs]
        results1.z_velocities[i] = results.z_velocities[i_bs]
        results1.times[i] = results.times[i_bs]
        results1.photons_scattered[i] = results.photons_scattered[i_bs]
    end
    return results1
end

function gaussian(x, p)
   A, σ, x0 = p[1], p[2], p[3]
    return A * exp.(-(x .- x0).^2 / (2*σ^2))
end

function bootstrap_size(results; iter = 10, n_values=nothing)
    sizes = []
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
        _,sx, sy, sz = plot_size(results1, "all")
        push!(sizes, (sx[end]*sy[end]*sz[end])^(1/3))
    end
        
    return mean(sizes), std(sizes)
end
    
function bootstrap_size_fit(results; iter = 10, n_values=nothing, t_end=5e-3, range=0.5, n_grids=40)
    sizes = []
    x_grid = collect(LinRange(-range, range, n_grids))
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
    
        id_survived = survived(t_end, results1.times)
    
        xs = [results1.x_trajectories[j][end] for j in id_survived]
        density_x = distribution_to_func(x_grid, xs)
        p0 = [10, 0.1, 0.0]
        fit_x = curve_fit(gaussian, x_grid, density_x, p0)
        sx = fit_x.param[2]
    
        ys = [results1.y_trajectories[j][end] for j in id_survived]
        density_y = distribution_to_func(x_grid, xs)
        p0 = [10, 0.1, 0.0]
        fit_y = curve_fit(gaussian, x_grid, density_y, p0)
        sy = fit_y.param[2]
    
        zs = [results1.z_trajectories[j][end] for j in id_survived]
        density_z = distribution_to_func(x_grid, zs)
        p0 = [10, 0.1, 0.0]
        fit_z = curve_fit(gaussian, x_grid, density_z, p0)
        sz = fit_z.param[2]

    
    
        push!(sizes, (sx[end]*sy[end]*sz[end])^(1/3))
    end
        
    return mean(sizes), std(sizes)
end

function distribution_to_func(x_grid, trajectories)
    func = x_grid .* 0.0
    dx = x_grid[2]-x_grid[1]
    for x in trajectories
       id_x = Int((x - x_grid[1]) ÷ dx)
        if 1<= id_x <= length(x_grid)
            func[id_x] += 1.0
        end
    end
    return func
end

    
function bootstrap_survival(results; iter=10, n_values = nothing)
    signals = []

    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
        n = survival_signal(5e-3, results1, photon_budget=12000)
        push!(signals, n)
    end
        
    return mean(signals), std(signals)
end
    
function bootstrap_temperature(results; iter=10, n_values=nothing)
    temperatures = []

    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
        Tx = plot_temperature(results1, "x")[end]
        Ty = plot_temperature(results1, "y")[end]
        Tz = plot_temperature(results1, "z")[end]
        push!(temperatures, (Tx[end]*Ty[end]*Tz[end])^(1/3))
    end
    
    return mean(temperatures), std(temperatures)
    
end
    
function bootstrap_temperature_fit(results; iter=10, n_values=nothing, range=0.5, n_grids=20,t_end=5e-3)
    temperatures = []
    x_grid = collect(LinRange(-range, range, n_grids))
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
    
        id_survived = survived(t_end, results1.times)
    
        xs = [results1.x_velocities[j][end] for j in id_survived]
        density_x = distribution_to_func(x_grid, xs)
        p0 = [10, 0.1, 0.0]
        fit_x = curve_fit(gaussian, x_grid, density_x, p0)
        k_B = 1.38e-23
        sx = m * fit_x.param[2]^2 /k_B *1e6
    
        ys = [results1.y_velocities[j][end] for j in id_survived]
        density_y = distribution_to_func(x_grid, xs)
        p0 = [10, 0.1, 0.0]
        fit_y = curve_fit(gaussian, x_grid, density_y, p0)
        sy = m * fit_y.param[2]^2 /k_B *1e6
    
        zs = [results1.z_velocities[j][end] for j in id_survived]
        density_z = distribution_to_func(x_grid, zs)
        p0 = [10, 0.1, 0.0]
        fit_z = curve_fit(gaussian, x_grid, density_z, p0)
        sz = m * fit_z.param[2]^2 /k_B *1e6

       

    
        push!(temperatures, (sx[end]*sy[end]*sz[end])^(1/3))
    end
        
    return mean(temperatures), std(temperatures)
end
   
using Optim

# Define the 2D Gaussian function
function gaussian2d(x, y, p)
    xc, yc, σx, σy, A = p
    return A * exp.(-((x .- xc) .^ 2 / (2 * σx^2) + (y .- yc) .^ 2 / (2 * σy^2)))
end


function least_sq_cost(params)
    predicted = [gaussian2d(xx, zz, params) for xx in x, zz in z]
    return sum((predicted - grids).^2)
end

using Optim
;

function get_trapped_indicies(param, results)
   t_max =  param.t_end*1e3
    trapped_indicies = Int[]
    for i in 1:length(results.times)
        if results.times[i][end] ≈ t_max
           push!(trapped_indicies, i) 
        end
    end
    return trapped_indicies
end;

function make_2D_image(results, x_min, x_max, n_pixels;t=0.0,photon_budget=Inf)
    """ grids = population """
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
             max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
   dt = plot_ts[2] - plot_ts[1]
    it = Int(t ÷ dt) + 1

    grids = zeros(n_pixels, n_pixels)
    x = LinRange(x_min, x_max, n_pixels)
    dx = x[2]-x[1]
    z = LinRange(x_min, x_max, n_pixels)
    for i in 1:length(results.times)
        if it > length(results.times[i])
            continue
        end
        x_end = results.x_trajectories[i][it]
        z_end = results.z_trajectories[i][it]
        n_photons = sum(results.A_populations[i][1:it]*dt*Γ/1000)
       if  (x_min < x_end < x_max) && (x_min < z_end < x_max)
           ix = Int((x_end - x_min) ÷ dx + 1)
            iz = Int((z_end - x_min) ÷ dx + 1)
            grids[ix, iz] += 1 * exp(-n_photons / photon_budget)
        end
    end
    return x, z, grids
end

function take_camera_image(results, x_min, x_max, n_pixels;t=0.0, t_img=0.0, const_scattering=false, noise=0.0,photon_budget=Inf)
    x, z, grids = make_2D_image(results, x_min, x_max, n_pixels;t=t, photon_budget=photon_budget)
    dt = 0.5
    if const_scattering
        grids = grids .* dt 
    else
        grids = grids .* scattering_rate_at_t(results, t+0.1) * dt 
    end
    if t_img <= 0.5
        return x, z, grids
    else
        t_curr = t + dt
        while t_curr < t + t_img
            ~,~,grids1 = make_2D_image(results, x_min, x_max, n_pixels;t=t_curr, photon_budget=photon_budget)
            if const_scattering
                grids = grids .+ grids1 * dt
            else
                grids = grids .+ grids1 * scattering_rate_at_t(results, t_curr) * dt
            end
            t_curr += dt
        end
        grids = grids .+ (rand(length(x), length(z)).- 0.5).*t_img * noise
        return x, z, grids # grids = brightness
    end
end

function goodness(t_end, results)
    
    times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
    
    # Sample number of photons before the molecule is lost to vibrational dark state
    n_molecules = length(times)
    
    _survived = survived(t_end, times)
    @printf("Survival: %i / %i", length(_survived), n_molecules)
    println()
    
    surviving_x_trajectories = x_trajectories[_survived]
    surviving_y_trajectories = y_trajectories[_survived]
    surviving_z_trajectories = z_trajectories[_survived]
    
    n = 0
    for survived_idx ∈ _survived
        idx_end = length(x_trajectories[survived_idx])
        r = distance(x_trajectories[survived_idx], y_trajectories[survived_idx], z_trajectories[survived_idx], idx_end)
        if r > 1.0
            n += 0.0
        elseif r > 0.5
            n += 0.1 * exp(-results.photons_scattered[survived_idx]/14000)
        elseif r > 0.3
            n += (0.6 - r)* exp(-results.photons_scattered[survived_idx]/14000)
        elseif r <= 0.3
            n += ((0.3 - r) * 3 + 0.3) * exp(-results.photons_scattered[survived_idx]/14000)
        end
    end
    return n/n_molecules
end
;
function survived(t_end, times)
    _survived = Int64[]
    for i ∈ eachindex(times)
        if abs(times[i][end] - t_end*1e3) <= 1.0
            push!(_survived, i)
        end
    end
    return _survived
end;
function cloud_size(trajectories, i)
    std(trajectory[i] for trajectory ∈ trajectories)
end

"""
    Evaluates the density given a set of trajectories.
"""
function density(t_end, results)
    
    times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
    
    # Sample number of photons before the molecule is lost to vibrational dark state
    n_molecules = length(times)
    
    _survived = survived(t_end, times)
    println(_survived)
    
    surviving_x_trajectories = x_trajectories[_survived]
    surviving_y_trajectories = y_trajectories[_survived]
    surviving_z_trajectories = z_trajectories[_survived]
    
    n = length(_survived)
    _density = Float64(n)
    if n > 1
        idx_end = length(surviving_x_trajectories[1])
        σ_x = cloud_size(surviving_x_trajectories, idx_end)
        σ_y = cloud_size(surviving_y_trajectories, idx_end)
        σ_z = cloud_size(surviving_z_trajectories, idx_end)
        _density /= σ_x * σ_y * σ_z
    end
    return _density
end

function distance(x_trajectory, y_trajectory, z_trajectory, idx)
    return sqrt(x_trajectory[idx]^2 + y_trajectory[idx]^2 + z_trajectory[idx]^2)
end


function survival_signal(t_end, results; photon_budget = 14000)
    
    times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
    
    # Sample number of photons before the molecule is lost to vibrational dark state
    n_molecules = length(times)
    
    _survived = survived(t_end, times)
#     @printf("Survival: %i / %i", length(_survived), n_molecules)
#     println()
    
    surviving_x_trajectories = x_trajectories[_survived]
    surviving_y_trajectories = y_trajectories[_survived]
    surviving_z_trajectories = z_trajectories[_survived]
    
    n = 0
    for survived_idx ∈ _survived
        idx_end = length(x_trajectories[survived_idx])
        n += exp(-results.photons_scattered[survived_idx]/photon_budget)
    end
    return n/n_molecules
end
;                




function set_H_zero!(H)
    @turbo for i in eachindex(H)
        H.re[i] = 0.0
        H.im[i] = 0.0
    end
    return nothing
end

struct Jump
    s ::Int64
    s′::Int64
    q::Int64
    r ::ComplexF64
end

function base_to_soa!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @inbounds for i in eachindex(ρ_soa)
        ρ_soa.re[i] = real(ρ[i])
        ρ_soa.im[i] = imag(ρ[i])
    end
    return nothing
end

function update_eiωt!(eiωt::StructArray{<:Complex}, ω::Array{<:Real}, τ::Real)
    @turbo for i ∈ eachindex(ω)
        eiωt.im[i], eiωt.re[i] = sincos( ω[i] * τ )
    end
    
    return nothing
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

function mul_by_im_minus!(C::StructArray{<:Complex})
    @turbo for i ∈ eachindex(C)
        a = C.re[i]
        C.re[i] = C.im[i]
        C.im[i] = -a
    end
end

function soa_to_base!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @inbounds for i in eachindex(ρ_soa)
        ρ[i] = ρ_soa.re[i] + im * ρ_soa.im[i]
    end
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
        diffusion_constant = sim_params.diffusion_constant
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
    
    
    
    rand1 = rand()
    diffusion_direction = 1.0
    if rand1 < 0.5
        diffusion_direction= -1
    end
    integrator.u[n_states + n_excited + 4] += p.diffusion_constant/p.mass * diffusion_direction
    
    rand1 = rand()
    diffusion_direction = 1.0
    if rand1 < 0.5
        diffusion_direction= -1
    end
    integrator.u[n_states + n_excited + 5] += p.diffusion_constant/p.mass * diffusion_direction
    
    rand1 = rand()
    diffusion_direction = 1.0
    if rand1 < 0.5
        diffusion_direction= -1
    end
    integrator.u[n_states + n_excited + 6] += p.diffusion_constant/p.mass * diffusion_direction
    
    
    
    
    p.time_to_decay = rand(p.decay_dist)
    return nothing
end
        

function make_problem_with_param_constant_diffusion(molecule_package, package_A, param; variable_budget=true, randomize_initial_vector = true, ψ0=nothing)       
    t_start = param.t_start
    t_end = param.t_end
    pol1_x, pol2_x, pol3_x, pol4_x = param.pol1_x, param.pol2_x, param.pol3_x, param.pol4_x

    s1, s2, s3, s4 = param.s1, param.s2, param.s3, param.s4
    s_ramp_time = param.s_ramp_time * Γ
    s_ramp_factor = param.s_ramp_to_factor
    
    Δ1, Δ2, Δ3, Δ4 = param.Δ1, param.Δ2, param.Δ3, param.Δ4
    B_gradient = param.B_gradient
    temp = param.temp
    diameter = param.diameter
    B_ramp_time = param.B_ramp_time * Γ
    displacement = param.displacement
    kick = param.kick
    
    imbalance = param.pol_imbalance 
    sx_imbalance, sy_imbalance, sz_imbalance = param.s_imbalance[1], param.s_imbalance[2], param.s_imbalance[3] 
    x_center_y, x_center_z, y_center_x, y_center_z, z_center_x, z_center_y = param.off_center .* k
    retro_loss = param.retro_loss
    
    #x_center_y *= rand()
    #x_center_z *= rand()
    #y_center_x *= rand()
    #y_center_z *= rand()
    #z_center_x *= rand()
    #z_center_y *= rand()
    
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
    
    ODT_intensity = param.ODT_intensity # 13.3*1e3/(50e-4)^2
    ODT_size = param.ODT_size # (30e-6, 2e-3, 30e-6)

    α = param.ODT_revolutions
    r_max = param.ODT_rmax * 1.1
    t_max = param.ODT_tmax * 1.1

    n = 1000
    arclengths = zeros(n)
    # spiral: r = τ; theta = 2πα * τ
    τs = LinRange(0,1.1,n)
    dτ = τs[2]-τs[1]
    length_curr = 0.0

    for i in 1:n
        r = r_max/1 * τs[i]
        θ = 2*π * α * τs[i]
        length_curr += sqrt((r_max/1)^2 +(2π*α*r)^2) * dτ
        arclengths[i] = length_curr
    end

    velocity = length_curr / t_max
    ts = arclengths / velocity
    
    
    photon_budget = 14000
    
    if variable_budget == true
        photon_budget = param.photon_budget
    end
    
    n_states = length(states)
    particle = OpticalBlochEquations.Particle()
    ψ₀ = zeros(ComplexF64, n_states)
    ψ₀[1] = 1.0
    H₀ = zeros(ComplexF64, n_states, n_states)
    
    extra_data = MutableNamedTuple(
        Zeeman_Hx=Zeeman_x_mat,
        Zeeman_Hy=Zeeman_y_mat,
        Zeeman_Hz=Zeeman_z_mat,
        H_ODT_static = StructArray(get_ODT_Hamiltonian_matrix(package, package_A, ODT_intensity, [0,1.0,0])),
        H_ODT = StructArray(get_ODT_Hamiltonian_matrix(package, package_A, ODT_intensity, [0,1.0,0])),
        ODT_position = [0.0, 0.0]
        
    )
    
    sim_params = MutableNamedTuple(
        B_gradient = param.B_gradient,
        B_offset = (param.Bx_offset,param.By_offset,param.Bz_offset),
        n_excited = n_excited,
        B_ramp_time = B_ramp_time,
        photon_budget = rand(Exponential(photon_budget)),
        ODT_size = ODT_size,
        dark_lifetime = param.dark_lifetime,
        FC_mainline = param.FC_mainline,

        ODT_rmax = param.ODT_rmax,
        ODT_tmax = param.ODT_tmax,
        ODT_revolutions = param.ODT_revolutions,
        ODT_motion_t_start = param.ODT_motion_t_start,
        ODT_motion_t_stop = param.ODT_motion_t_stop,
        interpolation_times = ts, interpolation_τs = τs,
        
        s_total = s1 + s2 + s3 + s4,
        diffusion_constant = param.diffusion_constant
        )

    t_span = (t_start, t_end) ./ (1/Γ);
    
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
    pol1_x = pol1_x.*sqrt(1 - imbalance) + flip(pol1_x).*sqrt(imbalance)*exp(im*2π*rand1)
    rand2 = rand()
    pol2_x = pol2_x.*sqrt(1 - imbalance) + flip(pol2_x).*sqrt(imbalance)*exp(im*2π*rand2)
    rand3 = rand()
    pol3_x = pol3_x.*sqrt(1 - imbalance)  + flip(pol3_x).*sqrt(imbalance)*exp(im*2π*rand3)
    rand4 = rand()
    pol4_x = pol4_x.*sqrt(1 - imbalance) + flip(pol4_x).*sqrt(imbalance)*exp(im*2π*rand4)
    
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
        
    p = schrodinger_stochastic_constant_diffusion(particle, states, lasers, d, ψ₀, m/(ħ*k^2/Γ), n_excited; sim_params=sim_params, extra_data=extra_data, λ=λ, Γ=Γ, update_H_and_∇H=update_H_and_∇H)

    prob = ODEProblem(ψ_stochastic_potential!, p.ψ, t_span, p)
    if randomize_initial_vector
        randomize_initial_vector!(prob.p, x_dist, y_dist, z_dist, vx_dist, vy_dist, vz_dist)
        random_initial_state!(prob.p)
     else
        prob.p.ψ .= ψ0
     end
            
    
    
    return prob
end
            
            
function simulate_particles_constant_diffusion(package, package_A, sequence_of_params; variable_budget=false)
    
    n_values = sequence_of_params[1].n_values
    n_threads=Threads.nthreads()
    batch_size = fld(n_values, n_threads)
    remainder = n_values - batch_size * n_threads
    prog_bar = Progress(n_values)

    n_states = length(package.states)
    n_excited = package.n_excited

    x_trajectories = Array{Vector{Float64}}(fill([],n_values))
    y_trajectories = Array{Vector{Float64}}(fill([],n_values)) 
    z_trajectories = Array{Vector{Float64}}(fill([],n_values))
    x_velocities = Array{Vector{Float64}}(fill([],n_values))
    y_velocities = Array{Vector{Float64}}(fill([],n_values))
    z_velocities = Array{Vector{Float64}}(fill([],n_values))
    A_populations = Array{Vector{Float64}}(fill([],n_values))
    times = Array{Vector{Float64}}(fill([],n_values))
    photons_scattered = zeros(n_values)
    sum_diffusion_x = zeros(n_values)
    sum_diffusion_y = zeros(n_values)                        
    sum_diffusion_z = zeros(n_values)
                            
    results = MutableNamedTuple(x_trajectories = x_trajectories, y_trajectories= y_trajectories, z_trajectories=z_trajectories,
                                x_velocities = x_velocities, y_velocities=y_velocities, z_velocities=z_velocities,
                                times=times, A_populations=A_populations,
                                n_values=n_values, photons_scattered=photons_scattered, n_states = n_states, n_excited=n_excited
                                )

    cb = ContinuousCallback(condition, SE_collapse_pol_constant_diffusion!, nothing, save_positions=(false,false))

    Threads.@threads for i ∈ 1:n_threads
    
        _batch_size = i <= remainder ? (batch_size + 1) : batch_size
        batch_start_idx = 1 + (i <= remainder ? (i - 1) : remainder) + batch_size * (i-1)

        for j ∈ batch_start_idx:(batch_start_idx + _batch_size - 1) 
            ψ0 = nothing
            for i_param in 1:length(sequence_of_params)
                randomize = false
                if i_param == 1
                     randomize = true
                end
                params = sequence_of_params[i_param]      
                prob_copy = make_problem_with_param_constant_diffusion(package, package_A, params, variable_budget=variable_budget, randomize_initial_vector = randomize, ψ0=ψ0)
                ψ0 = continue_particle_constant_diffusion!(j, results, prob_copy)    
                if abs(params.t_end - results.times[j][end]*1e-3) > 1e-3
                          break # this particle has already died
                end
            end

            next!(prog_bar)
        end
    end
    
#     results = MutableNamedTuple(x_trajectories = x_trajectories, y_trajectories= y_trajectories, z_trajectories=z_trajectories,
#                                 x_velocities = x_velocities, y_velocities=y_velocities, z_velocities=z_velocities,
#                                 times=times, A_populations=A_populations,
#                                 n_values=n_values, photons_scattered=photons_scattered)
    return results
end
                                
function continue_particle_constant_diffusion!(j, results, prob_copy)
     # j = particle id in results
    cb = ContinuousCallback(condition, SE_collapse_pol_constant_diffusion!, nothing, save_positions=(false,false))
                
    sol = DifferentialEquations.solve(prob_copy, alg=Tsit5(), reltol=1e-3, callback=cb, saveat=4000, maxiters=50000000)

    n_states = results.n_states
    n_excited = results.n_excited
    
    plot_us = sol.u
    plot_ts = sol.t
    results.x_trajectories[j] = [results.x_trajectories[j]; [real(u[n_states + n_excited + 1]) for u in plot_us]./k*1e3 ]
    results.y_trajectories[j] = [results.y_trajectories[j]; [real(u[n_states + n_excited + 2]) for u in plot_us]./k*1e3 ]
    results.z_trajectories[j] = [results.z_trajectories[j]; [real(u[n_states + n_excited + 3]) for u in plot_us]./k*1e3 ]
    results.x_velocities[j] = [results.x_velocities[j] ; [real(u[n_states + n_excited + 4]) for u in plot_us]./k*Γ ]
    results.y_velocities[j] = [results.y_velocities[j] ; [real(u[n_states + n_excited + 5]) for u in plot_us]./k*Γ ]
    results.z_velocities[j] = [results.z_velocities[j] ; [real(u[n_states + n_excited + 6]) for u in plot_us]./k*Γ ]

    results.times[j] = [results.times[j] ; plot_ts./Γ*1e3]

    results.A_populations[j] = [results.A_populations[j] ; [sum(real.(norm.(u[n_states - n_excited + 1 : n_states]).^2)) for u in plot_us] ]

    results.photons_scattered[j] += prob_copy.p.n_scatters 
                                                        
     return sol.u[end]       
end   
       
function merge_results(list_of_results)
    n_values = 0
    for results in list_of_results
        n_values += length(results.x_trajectories)
    end
    x_trajectories = Array{Vector{Float64}}(fill([],n_values))
    y_trajectories = Array{Vector{Float64}}(fill([],n_values)) 
    z_trajectories = Array{Vector{Float64}}(fill([],n_values))
    x_velocities = Array{Vector{Float64}}(fill([],n_values))
    y_velocities = Array{Vector{Float64}}(fill([],n_values))
    z_velocities = Array{Vector{Float64}}(fill([],n_values))
    A_populations = Array{Vector{Float64}}(fill([],n_values))
    times = Array{Vector{Float64}}(fill([],n_values))
    photons_scattered = zeros(n_values)
    
    n_current = 1
    for results in list_of_results
        n = length(results.x_trajectories)
        x_trajectories[n_current:n_current + n - 1] = results.x_trajectories
        y_trajectories[n_current:n_current + n - 1] = results.y_trajectories
        z_trajectories[n_current:n_current + n - 1] = results.z_trajectories
        x_velocities[n_current:n_current + n - 1] = results.x_velocities
        y_velocities[n_current:n_current + n - 1] = results.y_velocities
        z_velocities[n_current:n_current + n - 1] = results.z_velocities
        A_populations[n_current:n_current + n - 1] = results.A_populations
        times[n_current:n_current + n - 1] = results.times
        photons_scattered[n_current:n_current + n - 1] = results.photons_scattered
        n_current += n
    end
                            
    merged_results = MutableNamedTuple(x_trajectories = x_trajectories, y_trajectories= y_trajectories, z_trajectories=z_trajectories,
                                x_velocities = x_velocities, y_velocities=y_velocities, z_velocities=z_velocities,
                                times=times, A_populations=A_populations,
                                n_values=n_values, photons_scattered=photons_scattered,
                                n_states=list_of_results[1].n_states, n_excited=list_of_results[1].n_excited
                                )
    return merged_results
end

function merge_result_lists(list_of_lists)
    n_each = length(list_of_lists[1])
    merged_list = []
    for i in 1:n_each
        temp_list = []
        for list_of_results in list_of_lists
            push!(temp_list, list_of_results[i])
        end
        merged_results = merge_results(temp_list)
        push!(merged_list, merged_results)
    end
    return merged_list
end                                                    
                                                    
     
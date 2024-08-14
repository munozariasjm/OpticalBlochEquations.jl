
using
    QuantumStates,
    OpticalBlochEquations,
    UnitsToValue

import MutableNamedTuples: MutableNamedTuple
import StructArrays: StructArray, StructVector
import StaticArrays: @SVector, SVector
import LinearAlgebra: norm, ⋅, adjoint!, diag
import LoopVectorization: @turbo
using Parameters


Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1))/sqrt(2)
Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1))/sqrt(2)
Zeeman_z(state, state′) = Zeeman(state, state′, 0)


# function get_CaOH_package()
#     λ = 626e-9
#     Γ = 2π* 6.4e6
#     m = @with_unit 57 "u"
#     k = 2π / λ
#     _μB = (μ_B / h) * 1e-4
    
#     QN_bounds = (
#         S = 1/2, 
#         I = 1/2, 
#         Λ = 0, 
#         N = 0:3
#     )
#     X_state_basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

#     X_state_operator = :(
#         BX * Rotation + 
#         DX * RotationDistortion + 
#         γX * SpinRotation + 
#         bFX * Hyperfine_IS + 
#         cX * (Hyperfine_Dipolar/3)
#     )

#     X_state_parameters = QuantumStates.@params begin
#         BX = 10023.0841
#         DX = 1.154e-2
#         γX = 34.7593
#         bFX = 2.602
#         cX = 2.053
#     end

#     X_state_ham = Hamiltonian(basis=X_state_basis, operator=X_state_operator, parameters=X_state_parameters)

#     # Add Zeeman terms
   
#     X_state_ham = add_to_H(X_state_ham, :B_x, (gS * _μB * 1e-6) * Zeeman_x)
#     X_state_ham = add_to_H(X_state_ham, :B_y, (gS * _μB * 1e-6) * Zeeman_y)
#     X_state_ham = add_to_H(X_state_ham, :B_z, (gS * _μB * 1e-6) * Zeeman_z)
#     X_state_ham.parameters.B_x = 0.
#     X_state_ham.parameters.B_y = 0.
#     X_state_ham.parameters.B_z = 0.

#     evaluate!(X_state_ham)
#     QuantumStates.solve!(X_state_ham)
#     ;

#     QN_bounds = (
#         S = 1/2,
#         I = 1/2,
#         Λ = (-1,1),
#         J = 1/2:5/2
#     )
#     A_state_basis = enumerate_states(HundsCaseA_Rot, QN_bounds)

#     A_state_operator = :(
#         T_A * DiagonalOperator +
#         Be_A * Rotation + 
#         Aso_A * SpinOrbit +
#         q_A * ΛDoubling_q +
#         p_A * ΛDoubling_p2q + q_A * (2ΛDoubling_p2q)
#     )

#     # Spectroscopic constants for CaOH, A state
#     A_state_parameters = QuantumStates.@params begin
#         T_A = 15998.122 * 299792458 * 1e-4
#         Be_A = 0.3412200 * 299792458 * 1e-4
#         Aso_A = 66.8181 * 299792458 * 1e-4
#         p_A = -0.04287 * 299792458 * 1e-4
#         q_A = -0.3257e-3 * 299792458 * 1e-4
#     end

#     A_state_ham = Hamiltonian(basis=A_state_basis, operator=A_state_operator, parameters=A_state_parameters)
#     evaluate!(A_state_ham)
#     QuantumStates.solve!(A_state_ham)
#     ;

#     A_state_J12_pos_parity_states = A_state_ham.states[5:8]

#     QN_bounds = (
#         S = 1/2, 
#         I = 1/2, 
#         Λ = (-1,1), 
#         N = 0:3
#     )
#     A_state_caseB_basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

#     ground_states = X_state_ham.states[5:16]
#     excited_states = convert_basis(A_state_J12_pos_parity_states, A_state_caseB_basis)

#     states = [ground_states; excited_states]
#     n_excited = length(excited_states)

#     for state ∈ states
#         state.E *= 1e6
#     end
#     ;

#     d = zeros(ComplexF64, 16, 16, 3)
#     d_ge = zeros(ComplexF64, 12, 4, 3)

#     basis_tdms = get_tdms_two_bases(X_state_ham.basis, A_state_caseB_basis, TDM)
#     tdms_between_states!(d_ge, basis_tdms, ground_states, excited_states)
#     d[1:12, 13:16, :] .= d_ge
    
#     #     Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1))/sqrt(2)
#     #     Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1))/sqrt(2)
#     #     Zeeman_z(state, state′) = Zeeman(state, state′, 0)

#     Zeeman_x_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_x, ground_states, excited_states) .* (2π*gS*_μB/Γ))
#     Zeeman_y_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_y, ground_states, excited_states) .* (2π*gS*_μB/Γ))
#     Zeeman_z_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_z, ground_states, excited_states) .* (2π*gS*_μB/Γ))
                

#     package = MutableNamedTuple(states=states, n_excited=n_excited, d=d, Zeeman_x_mat=Zeeman_x_mat, Zeeman_y_mat=Zeeman_y_mat, Zeeman_z_mat=Zeeman_z_mat, Γ = Γ , k = k, m = m)
#     return package
# end;

function get_CaOH_package()
    λ = 626e-9
    Γ = 2π* 6.4e6
    m = @with_unit 57 "u"
    k = 2π / λ
    _μB = (μ_B / h) * 1e-4
    
    QN_bounds = (
        S = 1/2, 
        I = 1/2, 
        Λ = 0, 
        N = 0:3
    )
    X_state_basis = enumerate_states(HundsCaseB_LinearMolecule, QN_bounds)

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
   
    X_state_ham = add_to_H(X_state_ham, :B_x, (gS * _μB * 1e-6) * Zeeman_x)
    X_state_ham = add_to_H(X_state_ham, :B_y, (gS * _μB * 1e-6) * Zeeman_y)
    X_state_ham = add_to_H(X_state_ham, :B_z, (gS * _μB * 1e-6) * Zeeman_z)
    X_state_ham.parameters.B_x = 0.
    X_state_ham.parameters.B_y = 0.
    X_state_ham.parameters.B_z = 1e-5

    evaluate!(X_state_ham)
    QuantumStates.solve!(X_state_ham)
    ;

    QN_bounds = (
        S = 1/2,
        I = 1/2,
        Λ = (-1,1),
        J = 1/2:5/2
    )
    A_state_basis = enumerate_states(HundsCaseA_LinearMolecule, QN_bounds)

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
    A_state_caseB_basis = enumerate_states(HundsCaseB_LinearMolecule, QN_bounds)

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
    
    #     Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1))/sqrt(2)
    #     Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1))/sqrt(2)
    #     Zeeman_z(state, state′) = Zeeman(state, state′, 0)

    Zeeman_x_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_x, ground_states, excited_states) .* (2π*gS*_μB/Γ))
    Zeeman_y_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_y, ground_states, excited_states) .* (2π*gS*_μB/Γ))
    Zeeman_z_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_z, ground_states, excited_states) .* (2π*gS*_μB/Γ))
                

    package = MutableNamedTuple(states=states, n_excited=n_excited, d=d, Zeeman_x_mat=Zeeman_x_mat, Zeeman_y_mat=Zeeman_y_mat, Zeeman_z_mat=Zeeman_z_mat, Γ = Γ , k = k, m = m)
    return package
end;


function get_CaOH_package_12_A_states()
    λ = 626e-9
    Γ = 2π* 6.4e6
    m = @with_unit 57 "u"
    k = 2π / λ
    _μB = (μ_B / h) * 1e-4
    
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













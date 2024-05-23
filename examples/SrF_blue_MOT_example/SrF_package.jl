
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


function get_SrF_package()
    λ = 663.3e-9
    Γ = 2π * 6.6e6
    m = @with_unit 107 "u"
    k = 2π / λ
    μ_B = 9.2740100783e-24
    _μB = (μ_B / h) * 1e-4
    
     # X states
     HX_operator = :(
        BX * Rotation + 
        DX * RotationDistortion + 
        γX * SpinRotation + 
        bFX * Hyperfine_IS + 
        cX * (Hyperfine_Dipolar/3)
    )

    parameters = @params begin
        BX = 0.25359 * 299792458 * 1e-4
        DX = 2.49e-7 * 299792458 * 1e-4
        γX = 74.79485
        bFX = 97.0834 + 30.268/3
        cX = 30.268
    end

    QN_bounds = (S=1/2, I=1/2, Λ=0, N=0:3)
    basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

    SrF_000_N0to3_Hamiltonian = Hamiltonian(basis=basis, operator=HX_operator, parameters=parameters)

 

    full_evaluate!(SrF_000_N0to3_Hamiltonian)
    QuantumStates.solve!(SrF_000_N0to3_Hamiltonian)
    ;


    # A states
    H_operator = :(
        T_A * DiagonalOperator +
        Be_A * Rotation + 
        Aso_A * SpinOrbit + 
        q_A * ΛDoubling_q +
        p_A * ΛDoubling_p2q + q_A * (2ΛDoubling_p2q) + 
        temp * Hyperfine_IF
    ) # had to add a small hyperfine splitting so that when zeeman terms are added, mF remains a good quantum number 
    # (breaks degeneracy between hyperfine states so that the eigenstates of H found by solver are eigenstates of m)

    parameters = @params begin
        T_A = 15072.09 * 299792458 * 1e-4
        Be_A =  0.2536135 * 299792458 * 1e-4
        Aso_A = 281.46138 * 299792458 * 1e-4
        p_A = -0.133002 * 299792458 * 1e-4
        q_A = -0.3257e-3 * 299792458 * 1e-4
        temp = 0
    end

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), J=1/2:5/2)
    basis = enumerate_states(HundsCaseA_Rot, QN_bounds)

    # gL = 1
    # gL_prime = -0.083*3 #SrF
    # Zeeman_A(state, state1) = gL * _μB * Zeeman_L(state, state1) + gS * _μB * Zeeman_S(state, state1) + gL_prime * _μB * Zeeman_glprime(state, state1)
    SrF_A000_J12to52_Hamiltonian = Hamiltonian(basis=basis, operator=H_operator, parameters=parameters)
    # SrF_A000_J12to52_Hamiltonian = add_to_H(SrF_A000_J12to52_Hamiltonian, :B_z, Zeeman_A)
    # SrF_A000_J12to52_Hamiltonian.parameters.B_z = 1e-3

    evaluate!(SrF_A000_J12to52_Hamiltonian)
    QuantumStates.solve!(SrF_A000_J12to52_Hamiltonian)
    ;

    HA_J12_pos_parity_states = SrF_A000_J12to52_Hamiltonian.states[5:8]

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), N=0:2)
    basis_to_convert = enumerate_states(HundsCaseB_Rot, QN_bounds)

    states_A_J12_caseB = convert_basis(HA_J12_pos_parity_states, basis_to_convert)
    ;

    _, HX_N1_states = subspace(SrF_000_N0to3_Hamiltonian.states, (N=1,))
    states = [HX_N1_states; states_A_J12_caseB]
    
    for state ∈ states
        state.E *= 1e6
    end
    ;

    # Calculate transtion dipole moment between X and A states:
    d = zeros(ComplexF64, 16, 16, 3)
    d_ge = zeros(ComplexF64, 12, 4, 3)
    basis_tdms = get_tdms_two_bases(SrF_000_N0to3_Hamiltonian.basis, basis_to_convert, TDM)
    tdms_between_states!(d_ge, basis_tdms, HX_N1_states, states_A_J12_caseB)
    d[1:12, 13:16, :] .= d_ge
    ;

    Zeeman_x_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_x, HX_N1_states, states_A_J12_caseB) .* (2π*gS*_μB/Γ))
    Zeeman_y_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_y, HX_N1_states, states_A_J12_caseB) .* (2π*gS*_μB/Γ))
    Zeeman_z_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_z, HX_N1_states, states_A_J12_caseB) .* (2π*gS*_μB/Γ))
                

    package = MutableNamedTuple(states=states, n_excited=length(states_A_J12_caseB), d=d, Zeeman_x_mat=Zeeman_x_mat, Zeeman_y_mat=Zeeman_y_mat, Zeeman_z_mat=Zeeman_z_mat, Γ = Γ , k = k, m = m)
    return package
end;














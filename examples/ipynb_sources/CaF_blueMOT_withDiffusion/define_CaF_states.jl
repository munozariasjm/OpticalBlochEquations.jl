using QuantumStates

function define_CaOH_states()

    # Define the X state Hamiltonian for the N = 0...3 rotational states of CaF
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
        DX = 4.8078e-7
        γX = 39.65895
        bFX = 109.1840
        cX = 40.1182
    end

    X_state_ham = Hamiltonian(basis=X_state_basis, operator=X_state_operator, parameters=X_state_parameters)
    evaluate!(X_state_ham)
    QuantumStates.solve!(X_state_ham)

    # Define the A state Hamiltonian for the N = 0...3 rotational states of CaF
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

    # Spectroscopic constants for CaF, A state
    A_state_parameters = QuantumStates.@params begin
        T_A = 16526.750 * 299792458 * 1e-4
        Be_A = 0.348781 * 299792458 * 1e-4
        Aso_A = 71.429 * 299792458 * 1e-4
        p_A = -0.044517 * 299792458 * 1e-4
        q_A = -2.916e-4 * 299792458 * 1e-4
    end

    A_state_ham = Hamiltonian(basis=A_state_basis, operator=A_state_operator, parameters=A_state_parameters)
    evaluate!(A_state_ham)
    QuantumStates.solve!(A_state_ham)

    A_state_J12_pos_parity_states = [A_state_ham.states[5:8];A_state_ham.states[9:16]] # J=1/2 and J=3/2

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
    
    return ground_states, excited_states
    
end
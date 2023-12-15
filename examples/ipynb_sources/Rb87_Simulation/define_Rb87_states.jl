function define_Rb87_states()

    # Define the 5^2S_{1/2} state
    QN_bounds = (
        L = 0,
        S = 1/2,
        J = 1/2,
        I = 3/2
    )
    S12_basis = enumerate_states(AtomicState, QN_bounds)

    S12_operator = :(
        A_S12 * hyperfine_magnetic_dipole
    )

    S12_parameters = QuantumStates.@params begin
        A_S12 = 3.417341305452145e9 # magnetic dipole constant in Hz
    end

    S12_ham = Hamiltonian(basis=S12_basis, operator=S12_operator, parameters=S12_parameters)
    evaluate!(S12_ham)
    QuantumStates.solve!(S12_ham)

    # Define the 5^2P_{3/2} state
    QN_bounds = (
        L = 1,
        S = 1/2,
        J = 3/2,
        I = 3/2
    )
    P32_basis = enumerate_states(AtomicState, QN_bounds)

    P32_operator = :(
        T * identity +
        A_P32 * hyperfine_magnetic_dipole +
        B_P32 * hyperfine_electric_quadrupole
    )

    P32_parameters = QuantumStates.@params begin
        T = 384.2304844685e12 # energy of 5^2P_{3/2} relative to 5^2S_{1/2}
        A_P32 = 84.7185e6 # magnetic dipole constant in Hz
        B_P32 = 12.4965e6 # electric quadrupole constant in Hz
    end

    P32_ham = Hamiltonian(basis=P32_basis, operator=P32_operator, parameters=P32_parameters)
    evaluate!(P32_ham)
    QuantumStates.solve!(P32_ham)

    ground_states = S12_ham.states
    excited_states = P32_ham.states

    states = [ground_states; excited_states]
    n_excited = length(excited_states)
    
    return ground_states, excited_states

end
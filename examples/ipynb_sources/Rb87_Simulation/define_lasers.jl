function define_lasers(
        states
    )
    
    # function that determines the polarization as a function of time
    ϵ_func(ϵ) = t -> ϵ
    
    # function that determines the intensity as a function of time
    s_func(s) = (x,t) -> (t < 10e-3 / (1/Γ)) ? s : 0.0

    s = 1.0
    det = -2.0Γ
    
    # define laser1 parameters
    s1   = s
    det1 = det
    pol1 = σ⁺
    ω1   = 2π * (energy(states[end]) - energy(states[8])) + det1
    k̂1   = +x̂

    # define laser2 parameters
    s2   = s
    det2 = det
    pol2 = σ⁺
    ω2   = 2π * (energy(states[end]) - energy(states[8])) + det2
    k̂2   = -x̂
    
    # define laser3 parameters
    s3   = s
    det3 = det
    pol3 = σ⁺
    ω3   = 2π * (energy(states[end]) - energy(states[8])) + det3
    k̂3   = +ŷ

    # define laser4 parameters
    s4   = s
    det4 = det
    pol4 = σ⁺
    ω4   = 2π * (energy(states[end]) - energy(states[8])) + det4
    k̂4   = -ŷ
    
    # define laser5 parameters
    s5   = s
    det5 = det
    pol5 = σ⁺
    ω5   = 2π * (energy(states[end]) - energy(states[8])) + det5
    k̂5   = +ẑ

    # define laser6 parameters
    s6   = s
    det6 = det
    pol6 = σ⁺
    ω6   = 2π * (energy(states[end]) - energy(states[8])) + det6
    k̂6   = -ẑ
    
    # define lasers
    laser1 = Field(k̂1, ϵ_func(rotate_pol(pol1, k̂1)), ω1, s_func(s1))
    laser2 = Field(k̂2, ϵ_func(rotate_pol(pol2, k̂2)), ω2, s_func(s2))
    laser3 = Field(k̂3, ϵ_func(rotate_pol(pol3, k̂3)), ω3, s_func(s3))
    laser4 = Field(k̂4, ϵ_func(rotate_pol(pol4, k̂4)), ω4, s_func(s4))
    laser5 = Field(k̂5, ϵ_func(rotate_pol(pol5, k̂5)), ω5, s_func(s5))
    laser6 = Field(k̂6, ϵ_func(rotate_pol(pol6, k̂6)), ω6, s_func(s6))
    
    lasers = [laser1, laser2, laser3, laser4, laser5, laser6]
    # lasers = [laser5, laser6]
    
end

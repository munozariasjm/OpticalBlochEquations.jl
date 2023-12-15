"""
    Compute the Hamiltonian for the atom-tweezer interaction.
"""
function get_H_tweezer(states, ground_states, excited_states, d, all_states, peak_intensity, pol, wavelength=808e-9)
    Is = π*h*c*Γ/(3λ^3) # saturation intensity in W/m^2
    s = peak_intensity / Is
    f_ODT = c/wavelength
    
    states = [ground_states; excited_states]
    n_states = length(states)
    E0 = sqrt(s)/(2 * √2) # factor?
    H_tweezer = zeros(ComplexF64, n_states, n_states)
    
    d = tdms_between_states(all_states, all_states)
    fs = energy.(all_states)

    for q in 1:3
        for p in 1:3
            for i in 1:n_states
                for j in 1:n_states
                    for l in 1:length(all_states)
                        H_tweezer[i,j] -= 2π * Γ * (E0^2/4 * d[min(i,l),max(i,l),q] * pol[q] * d[min(j,l),max(j,l),p] * pol[p]) * (1/(fs[l]-fs[i] - f_ODT) + 1/(fs[l]-fs[i] + f_ODT))
                    end
                end
            end
        end
    end
    
    return StructArray(H_tweezer)
end

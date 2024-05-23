Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1))/sqrt(2)
Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1))/sqrt(2)
Zeeman_z(state, state′) = Zeeman(state, state′, 0)

Zeeman_x_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_x, X_states, A_states[1:4]) .* (2π*gS*_μB/Γ))
Zeeman_y_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_y, X_states, A_states[1:4]) .* (2π*gS*_μB/Γ))
Zeeman_z_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_z, X_states, A_states[1:4]) .* (2π*gS*_μB/Γ))

# # add excited state shifts
# # we'll use that the matrix elements are the same as those in the F=0,1 states in the ground state, just with a smaller g-factor (by a ratio 0.021 to 1/3, with the same sign)
# Zeeman_x_mat[13:16,13:16] = Zeeman_x_mat[1:4,1:4] .* (0.021 / (1/3))
# Zeeman_y_mat[13:16,13:16] = Zeeman_x_mat[1:4,1:4] .* (0.021 / (1/3))
# Zeeman_z_mat[13:16,13:16] = Zeeman_x_mat[1:4,1:4] .* (0.021 / (1/3))
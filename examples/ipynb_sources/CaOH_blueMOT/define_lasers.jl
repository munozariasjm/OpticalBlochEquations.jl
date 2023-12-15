function get_Δ_from_exp(voltage, aom_freq)
    # return Δ1, Δ2 in MHz
    Δ1 = 57 - 7.4*(5.5-voltage)
    Δ2 = Δ1 + 51.24 - aom_freq
    return Δ1, Δ2
end

function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function gaussian_intensity_along_axes(r, axes, centers)
    """1/e^2 width = 5mm Gaussian beam """
    d2 = (r[axes[1]] - centers[1])^2 + (r[axes[2]] - centers[2])^2   
    return exp(-2*d2/(5e-3/(1/k))^2)
end

# function define_lasers(states,
#         s1,
#         s2,
#         s3,
#         s_ramp_time,
#         s_ramp_to,
#         pol_imbalance,
#         s_imbalance,
#         retro_loss,
#         off_center,
#         pointing_error,
#         pol1_x,
#         pol2_x,
#         pol3_x,
#         voltage,
#         aom_freq,
#         sideband_freq
#     )

#     Δs = get_Δ_from_exp(voltage, aom_freq)
#     Δ1 = Δs[1] * 1e6 * 2π 
#     Δ2 = Δs[2] * 1e6 * 2π
    
#     ω1 = 2π * (energy(states[end]) - energy(states[1])) + Δ1
#     ω2 = 2π * (energy(states[end]) - energy(states[5])) + Δ2
#     ω3 = ω2 + 2π * 1e6 * sideband_freq
    
#     x_center_y = rand() * off_center[1] * k
#     x_center_z = rand() * off_center[2] * k
#     y_center_x = rand() * off_center[3] * k
#     y_center_z = rand() * off_center[4] * k
#     z_center_x = rand() * off_center[5] * k
#     z_center_y = rand() * off_center[6] * k
    
#     ϵ_(ϵ, f) = t -> ϵ
#     s_func(s) = (x,t) -> s
#     s_gaussian(s, axes, centers) = (r,t) -> s * gaussian_intensity_along_axes(r, axes, centers)
    
#     s_gaussian_ramp(s, factor, ramp_time, axes, centers) = (r,t) -> ((s*factor-s)/ramp_time * min(t, ramp_time) + s) * gaussian_intensity_along_axes(r, axes, centers)
    
#     rand1 = rand()
#     pol1_x = pol1_x.*sqrt(1 - rand1*pol_imbalance) + flip(pol1_x).*sqrt(rand1*pol_imbalance)
#     rand2 = rand()
#     pol2_x = pol2_x.*sqrt(1 - rand2*pol_imbalance) + flip(pol2_x).*sqrt(rand2*pol_imbalance)
#     rand3 = rand()
#     pol3_x = pol3_x.*sqrt(1 - rand3*pol_imbalance) + flip(pol3_x).*sqrt(rand3*pol_imbalance)
#     rand3 = rand()
    
#     sx_rand = 1/2-rand()
#     sy_rand = 1/2-rand()
#     sz_rand = 1/2-rand()
    
#     ϕs = [exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand())]
    
#     kx = x̂ + [0, pointing_error[1],pointing_error[2]]
#     kx = kx ./ sqrt(kx[1]^2 + kx[2]^2 + kx[3]^2)
#     ky = ŷ + [pointing_error[3],0.0,pointing_error[4]]
#     ky = ky ./ sqrt(ky[1]^2 + ky[2]^2 + ky[3]^2)
#     kz = ẑ + [pointing_error[5],pointing_error[6],0.0]
#     kz = kz / sqrt(kz[1]^2 + kz[2]^2 + kz[3]^2)
    
#     s1x = s1 * (1+s_imbalance[1]*sx_rand)
#     s1y = s1 * (1+s_imbalance[2]*sy_rand)
#     s1z = s1 * (1+s_imbalance[3]*sz_rand)
#     k̂ = kx; ϵ1 = ϕs[1]*rotate_pol(pol1_x, k̂); ϵ_func1 = ϵ_(ϵ1, 1); laser1 = Field(k̂, ϵ_func1, ω1,  s_gaussian_ramp(s1x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
#     k̂ = -kx; ϵ2 = ϕs[2]*rotate_pol(pol1_x, k̂); ϵ_func2 = ϵ_(ϵ2, 2); laser2 = Field(k̂, ϵ_func2, ω1, s_gaussian_ramp(s1x*(1-retro_loss), s_ramp_to, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
#     k̂ = ky; ϵ3 = ϕs[3]*rotate_pol(pol1_x, k̂); ϵ_func3 = ϵ_(ϵ3, 3); laser3 = Field(k̂, ϵ_func3, ω1,  s_gaussian_ramp(s1y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
#     k̂ = -ky; ϵ4 = ϕs[4]*rotate_pol(pol1_x, k̂); ϵ_func4 = ϵ_(ϵ4, 4); laser4 = Field(k̂, ϵ_func4, ω1,  s_gaussian_ramp(s1y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
#     k̂ = +kz; ϵ5 = ϕs[5]*rotate_pol(flip(pol1_x), k̂); ϵ_func5 = ϵ_(ϵ5, 5); laser5 = Field(k̂, ϵ_func5, ω1,  s_gaussian_ramp(s1z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
#     k̂ = -kz; ϵ6 = ϕs[6]*rotate_pol(flip(pol1_x), k̂); ϵ_func6 = ϵ_(ϵ6, 6); laser6 = Field(k̂, ϵ_func6, ω1, s_gaussian_ramp(s1z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

#     lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]

#     s2x = s2 * (1+s_imbalance[1]*sx_rand)
#     s2y = s2 * (1+s_imbalance[2]*sy_rand)
#     s2z = s2 * (1+s_imbalance[3]*sz_rand)
#     k̂ = +kx; ϵ7 = ϕs[1]*rotate_pol(pol2_x, k̂); ϵ_func7 = ϵ_(ϵ7, 1); laser7 = Field(k̂, ϵ_func7, ω2, s_gaussian_ramp(s2x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
#     k̂ = -kx; ϵ8 = ϕs[2]*rotate_pol(pol2_x, k̂); ϵ_func8 = ϵ_(ϵ8, 2); laser8 = Field(k̂, ϵ_func8, ω2, s_gaussian_ramp(s2x*(1-retro_loss), s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
#     k̂ = +ky; ϵ9 = ϕs[3]*rotate_pol(pol2_x, k̂); ϵ_func9 = ϵ_(ϵ9, 3); laser9 = Field(k̂, ϵ_func9, ω2, s_gaussian_ramp(s2y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
#     k̂ = -ky; ϵ10 = ϕs[4]*rotate_pol(pol2_x, k̂); ϵ_func10 = ϵ_(ϵ10, 4); laser10 = Field(k̂, ϵ_func10, ω2, s_gaussian_ramp(s2y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
#     k̂ = +kz; ϵ11 = ϕs[5]*rotate_pol(flip(pol2_x), k̂); ϵ_func11 = ϵ_(ϵ11, 5); laser11 = Field(k̂, ϵ_func11, ω2, s_gaussian_ramp(s2z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
#     k̂ = -kz; ϵ12 = ϕs[6]*rotate_pol(flip(pol2_x), k̂); ϵ_func12 = ϵ_(ϵ12, 6); laser12 = Field(k̂, ϵ_func12, ω2, s_gaussian_ramp(s2z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

#     lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]

#     s3x = s3 * (1+s_imbalance[1]*sx_rand)
#     s3y = s3 * (1+s_imbalance[2]*sy_rand)
#     s3z = s3 * (1+s_imbalance[3]*sz_rand)
#     k̂ = +kx; ϵ13 = ϕs[1]*rotate_pol(pol3_x, k̂); ϵ_func13 = ϵ_(ϵ13, 1); laser13 = Field(k̂, ϵ_func13, ω3, s_gaussian_ramp(s3x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
#     k̂ = -kx; ϵ14 = ϕs[2]*rotate_pol(pol3_x, k̂); ϵ_func14 = ϵ_(ϵ14, 2); laser14 = Field(k̂, ϵ_func14, ω3, s_gaussian_ramp(s3x*(1-retro_loss), s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
#     k̂ = +ky; ϵ15 = ϕs[3]*rotate_pol(pol3_x, k̂); ϵ_func15 = ϵ_(ϵ15, 3); laser15 = Field(k̂, ϵ_func15, ω3, s_gaussian_ramp(s3y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
#     k̂ = -ky; ϵ16 = ϕs[4]*rotate_pol(pol3_x, k̂); ϵ_func16 = ϵ_(ϵ16, 4); laser16 = Field(k̂, ϵ_func16, ω3, s_gaussian_ramp(s3y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
#     k̂ = +kz; ϵ17 = ϕs[5]*rotate_pol(flip(pol3_x), k̂); ϵ_func17 = ϵ_(ϵ17, 5); laser17 = Field(k̂, ϵ_func17, ω3, s_gaussian_ramp(s3z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
#     k̂ = -kz; ϵ18 = ϕs[6]*rotate_pol(flip(pol3_x), k̂); ϵ_func18 = ϵ_(ϵ18, 6); laser18 = Field(k̂, ϵ_func18, ω3, s_gaussian_ramp(s3z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

#     lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]
    
#     lasers = [lasers_1; lasers_2; lasers_3]

# end

function define_lasers(
        states,
        s1,
        s2,
        s3,
        s4,
        Δ1,
        Δ2,
        Δ3,
        Δ4,
        pol1_x,
        pol2_x,
        pol3_x,
        pol4_x,
        s_ramp_time,
        s_ramp_to,
        pol_imbalance,
        s_imbalance,
        retro_loss,
        off_center,
        pointing_error
    )
    
    ω1 = 2π * (energy(states[end]) - energy(states[1])) + Δ1 * 1e6 * 2π 
    ω2 = 2π * (energy(states[end]) - energy(states[8])) + Δ2 * 1e6 * 2π
    ω3 = 2π * (energy(states[end]) - energy(states[8])) + Δ3 * 1e6 * 2π
    ω4 = 2π * (energy(states[end]) - energy(states[8])) + Δ4 * 1e6 * 2π
    
    x_center_y = rand() * off_center[1] * k
    x_center_z = rand() * off_center[2] * k
    y_center_x = rand() * off_center[3] * k
    y_center_z = rand() * off_center[4] * k
    z_center_x = rand() * off_center[5] * k
    z_center_y = rand() * off_center[6] * k
    
    ϵ_(ϵ, f) = t -> ϵ
    s_func(s) = (x,t) -> s
    s_gaussian(s, axes, centers) = (r,t) -> s * gaussian_intensity_along_axes(r, axes, centers)
    
    s_gaussian_ramp(s, factor, ramp_time, axes, centers) = (r,t) -> ((s*factor-s)/ramp_time * min(t, ramp_time) + s) * gaussian_intensity_along_axes(r, axes, centers)
    
    rand1 = rand()
    pol1_x = pol1_x.*sqrt(1 - rand1*pol_imbalance) + flip(pol1_x).*sqrt(rand1*pol_imbalance)
    rand2 = rand()
    pol2_x = pol2_x.*sqrt(1 - rand2*pol_imbalance) + flip(pol2_x).*sqrt(rand2*pol_imbalance)
    rand3 = rand()
    pol3_x = pol3_x.*sqrt(1 - rand3*pol_imbalance) + flip(pol3_x).*sqrt(rand3*pol_imbalance)
    rand4 = rand()
    pol4_x = pol3_x.*sqrt(1 - rand4*pol_imbalance) + flip(pol4_x).*sqrt(rand4*pol_imbalance)
    rand4 = rand()
    
    sx_rand = 1/2-rand()
    sy_rand = 1/2-rand()
    sz_rand = 1/2-rand()
    
    ϕs = [exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand())]
    
    kx = x̂ + [0, pointing_error[1],pointing_error[2]]
    kx = kx ./ sqrt(kx[1]^2 + kx[2]^2 + kx[3]^2)
    ky = ŷ + [pointing_error[3],0.0,pointing_error[4]]
    ky = ky ./ sqrt(ky[1]^2 + ky[2]^2 + ky[3]^2)
    kz = ẑ + [pointing_error[5],pointing_error[6],0.0]
    kz = kz / sqrt(kz[1]^2 + kz[2]^2 + kz[3]^2)
    
    s1x = s1 * (1+s_imbalance[1]*sx_rand)
    s1y = s1 * (1+s_imbalance[2]*sy_rand)
    s1z = s1 * (1+s_imbalance[3]*sz_rand)
    k̂ = kx; ϵ1 = ϕs[1]*rotate_pol(pol1_x, k̂); ϵ_func1 = ϵ_(ϵ1, 1); laser1 = Field(k̂, ϵ_func1, ω1,  s_gaussian_ramp(s1x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ2 = ϕs[2]*rotate_pol(pol1_x, k̂); ϵ_func2 = ϵ_(ϵ2, 2); laser2 = Field(k̂, ϵ_func2, ω1, s_gaussian_ramp(s1x*(1-retro_loss), s_ramp_to, s_ramp_time,  (2,3), (x_center_y, x_center_z)))
    k̂ = ky; ϵ3 = ϕs[3]*rotate_pol(pol1_x, k̂); ϵ_func3 = ϵ_(ϵ3, 3); laser3 = Field(k̂, ϵ_func3, ω1,  s_gaussian_ramp(s1y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ4 = ϕs[4]*rotate_pol(pol1_x, k̂); ϵ_func4 = ϵ_(ϵ4, 4); laser4 = Field(k̂, ϵ_func4, ω1,  s_gaussian_ramp(s1y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ5 = ϕs[5]*rotate_pol(flip(pol1_x), k̂); ϵ_func5 = ϵ_(ϵ5, 5); laser5 = Field(k̂, ϵ_func5, ω1,  s_gaussian_ramp(s1z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ6 = ϕs[6]*rotate_pol(flip(pol1_x), k̂); ϵ_func6 = ϵ_(ϵ6, 6); laser6 = Field(k̂, ϵ_func6, ω1, s_gaussian_ramp(s1z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]

    s2x = s2 * (1+s_imbalance[1]*sx_rand)
    s2y = s2 * (1+s_imbalance[2]*sy_rand)
    s2z = s2 * (1+s_imbalance[3]*sz_rand)
    k̂ = +kx; ϵ7 = ϕs[1]*rotate_pol(pol2_x, k̂); ϵ_func7 = ϵ_(ϵ7, 1); laser7 = Field(k̂, ϵ_func7, ω2, s_gaussian_ramp(s2x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ8 = ϕs[2]*rotate_pol(pol2_x, k̂); ϵ_func8 = ϵ_(ϵ8, 2); laser8 = Field(k̂, ϵ_func8, ω2, s_gaussian_ramp(s2x*(1-retro_loss), s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = +ky; ϵ9 = ϕs[3]*rotate_pol(pol2_x, k̂); ϵ_func9 = ϵ_(ϵ9, 3); laser9 = Field(k̂, ϵ_func9, ω2, s_gaussian_ramp(s2y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ10 = ϕs[4]*rotate_pol(pol2_x, k̂); ϵ_func10 = ϵ_(ϵ10, 4); laser10 = Field(k̂, ϵ_func10, ω2, s_gaussian_ramp(s2y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ11 = ϕs[5]*rotate_pol(flip(pol2_x), k̂); ϵ_func11 = ϵ_(ϵ11, 5); laser11 = Field(k̂, ϵ_func11, ω2, s_gaussian_ramp(s2z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ12 = ϕs[6]*rotate_pol(flip(pol2_x), k̂); ϵ_func12 = ϵ_(ϵ12, 6); laser12 = Field(k̂, ϵ_func12, ω2, s_gaussian_ramp(s2z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]

    s3x = s3 * (1+s_imbalance[1]*sx_rand)
    s3y = s3 * (1+s_imbalance[2]*sy_rand)
    s3z = s3 * (1+s_imbalance[3]*sz_rand)
    k̂ = +kx; ϵ13 = ϕs[1]*rotate_pol(pol3_x, k̂); ϵ_func13 = ϵ_(ϵ13, 1); laser13 = Field(k̂, ϵ_func13, ω3, s_gaussian_ramp(s3x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ14 = ϕs[2]*rotate_pol(pol3_x, k̂); ϵ_func14 = ϵ_(ϵ14, 2); laser14 = Field(k̂, ϵ_func14, ω3, s_gaussian_ramp(s3x*(1-retro_loss), s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = +ky; ϵ15 = ϕs[3]*rotate_pol(pol3_x, k̂); ϵ_func15 = ϵ_(ϵ15, 3); laser15 = Field(k̂, ϵ_func15, ω3, s_gaussian_ramp(s3y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ16 = ϕs[4]*rotate_pol(pol3_x, k̂); ϵ_func16 = ϵ_(ϵ16, 4); laser16 = Field(k̂, ϵ_func16, ω3, s_gaussian_ramp(s3y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ17 = ϕs[5]*rotate_pol(flip(pol3_x), k̂); ϵ_func17 = ϵ_(ϵ17, 5); laser17 = Field(k̂, ϵ_func17, ω3, s_gaussian_ramp(s3z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ18 = ϕs[6]*rotate_pol(flip(pol3_x), k̂); ϵ_func18 = ϵ_(ϵ18, 6); laser18 = Field(k̂, ϵ_func18, ω3, s_gaussian_ramp(s3z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))

    lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]

    s4x = s4 * (1+s_imbalance[1]*sx_rand)
    s4y = s4 * (1+s_imbalance[2]*sy_rand)
    s4z = s4 * (1+s_imbalance[3]*sz_rand)
    k̂ = +kx; ϵ19 = ϕs[1]*rotate_pol(pol4_x, k̂); ϵ_func19 = ϵ_(ϵ19, 1); laser19 = Field(k̂, ϵ_func19, ω4, s_gaussian_ramp(s4x, s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = -kx; ϵ20 = ϕs[2]*rotate_pol(pol4_x, k̂); ϵ_func20 = ϵ_(ϵ20, 2); laser20 = Field(k̂, ϵ_func20, ω4, s_gaussian_ramp(s4x*(1-retro_loss), s_ramp_to, s_ramp_time, (2,3), (x_center_y, x_center_z)))
    k̂ = +ky; ϵ21 = ϕs[3]*rotate_pol(pol4_x, k̂); ϵ_func21 = ϵ_(ϵ21, 3); laser21 = Field(k̂, ϵ_func21, ω4, s_gaussian_ramp(s4y, s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = -ky; ϵ22 = ϕs[4]*rotate_pol(pol4_x, k̂); ϵ_func22 = ϵ_(ϵ22, 4); laser22 = Field(k̂, ϵ_func22, ω4, s_gaussian_ramp(s4y*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,3), (y_center_x, y_center_z)))
    k̂ = +kz; ϵ23 = ϕs[5]*rotate_pol(flip(pol4_x), k̂); ϵ_func23 = ϵ_(ϵ23, 5); laser23 = Field(k̂, ϵ_func23, ω4, s_gaussian_ramp(s4z, s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    k̂ = -kz; ϵ24 = ϕs[6]*rotate_pol(flip(pol4_x), k̂); ϵ_func24 = ϵ_(ϵ24, 6); laser24 = Field(k̂, ϵ_func24, ω4, s_gaussian_ramp(s4z*(1-retro_loss), s_ramp_to, s_ramp_time,  (1,2), (z_center_x, z_center_y)))
    
    lasers_4 = [laser19, laser20, laser21, laser22, laser23, laser24]
    
    lasers = [lasers_1; lasers_2; lasers_3; lasers_4]

end
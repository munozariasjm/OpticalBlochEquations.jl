function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function define_lasers(
        states,
        s1,
        s2,
        s3,
        Δ1,
        Δ2,
        Δ3,
        pol1_x,
        pol2_x,
        pol3_x
    )
    
    ω1 = 2π * (energy(states[end]) - energy(states[1])) + Δ1 * 1e6 * 2π 
    ω2 = 2π * (energy(states[end]) - energy(states[4])) + Δ2 * 1e6 * 2π
    ω3 = 2π * (energy(states[end]) - energy(states[4])) + Δ3 * 1e6 * 2π
    
    ϵ_(ϵ, f) = t -> ϵ
    s_func(s) = (x,t) -> s
    
    kx = x̂ 
    ky = ŷ
    kz = ẑ
    
    k̂ = +kx; ϵ1 = rotate_pol(pol1_x, k̂); ϵ_func1 = ϵ_(ϵ1, 1); laser1 = Field(k̂, ϵ_func1, ω1, s_func(s1))
    k̂ = -kx; ϵ2 = rotate_pol(pol1_x, k̂); ϵ_func2 = ϵ_(ϵ2, 2); laser2 = Field(k̂, ϵ_func2, ω1, s_func(s1))
    k̂ = +ky; ϵ3 = rotate_pol(pol1_x, k̂); ϵ_func3 = ϵ_(ϵ3, 3); laser3 = Field(k̂, ϵ_func3, ω1, s_func(s1))
    k̂ = -ky; ϵ4 = rotate_pol(pol1_x, k̂); ϵ_func4 = ϵ_(ϵ4, 4); laser4 = Field(k̂, ϵ_func4, ω1, s_func(s1))
    k̂ = +kz; ϵ5 = rotate_pol(flip(pol1_x), k̂); ϵ_func5 = ϵ_(ϵ5, 5); laser5 = Field(k̂, ϵ_func5, ω1, s_func(s1))
    k̂ = -kz; ϵ6 = rotate_pol(flip(pol1_x), k̂); ϵ_func6 = ϵ_(ϵ6, 6); laser6 = Field(k̂, ϵ_func6, ω1, s_func(s1))
    
    lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]

    k̂ = +kx; ϵ7  = rotate_pol(pol2_x, k̂); ϵ_func7 = ϵ_(ϵ7, 1); laser7 = Field(k̂, ϵ_func7, ω2, s_func(s2))
    k̂ = -kx; ϵ8  = rotate_pol(pol2_x, k̂); ϵ_func8 = ϵ_(ϵ8, 2); laser8 = Field(k̂, ϵ_func8, ω2, s_func(s2))
    k̂ = +ky; ϵ9  = rotate_pol(pol2_x, k̂); ϵ_func9 = ϵ_(ϵ9, 3); laser9 = Field(k̂, ϵ_func9, ω2, s_func(s2))
    k̂ = -ky; ϵ10 = rotate_pol(pol2_x, k̂); ϵ_func10 = ϵ_(ϵ10, 4); laser10 = Field(k̂, ϵ_func10, ω2, s_func(s2))
    k̂ = +kz; ϵ11 = rotate_pol(flip(pol2_x), k̂); ϵ_func11 = ϵ_(ϵ11, 5); laser11 = Field(k̂, ϵ_func11, ω2, s_func(s2))
    k̂ = -kz; ϵ12 = rotate_pol(flip(pol2_x), k̂); ϵ_func12 = ϵ_(ϵ12, 6); laser12 = Field(k̂, ϵ_func12, ω2, s_func(s2))
    
    lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]

    k̂ = +kx; ϵ13 = rotate_pol(pol3_x, k̂); ϵ_func13 = ϵ_(ϵ13, 1); laser13 = Field(k̂, ϵ_func13, ω3, s_func(s3))
    k̂ = -kx; ϵ14 = rotate_pol(pol3_x, k̂); ϵ_func14 = ϵ_(ϵ14, 2); laser14 = Field(k̂, ϵ_func14, ω3, s_func(s3))
    k̂ = +ky; ϵ15 = rotate_pol(pol3_x, k̂); ϵ_func15 = ϵ_(ϵ15, 3); laser15 = Field(k̂, ϵ_func15, ω3, s_func(s3))
    k̂ = -ky; ϵ16 = rotate_pol(pol3_x, k̂); ϵ_func16 = ϵ_(ϵ16, 4); laser16 = Field(k̂, ϵ_func16, ω3, s_func(s3))
    k̂ = +kz; ϵ17 = rotate_pol(flip(pol3_x), k̂); ϵ_func17 = ϵ_(ϵ17, 5); laser17 = Field(k̂, ϵ_func17, ω3, s_func(s3))
    k̂ = -kz; ϵ18 = rotate_pol(flip(pol3_x), k̂); ϵ_func18 = ϵ_(ϵ18, 6); laser18 = Field(k̂, ϵ_func18, ω3, s_func(s3))
    
    lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]

    k̂ = +kx; ϵ19 = rotate_pol([im/√2,0,im/√2], k̂); ϵ_func19 = ϵ_(ϵ19, 1); laser19 = Field(k̂, ϵ_func19, ω1, s_func(s1))
    k̂ = -kx; ϵ20 = rotate_pol([0,1,0], k̂); ϵ_func20 = ϵ_(ϵ20, 1); laser20 = Field(k̂, ϵ_func20, ω1, s_func(s1))
    k̂ = +kx; ϵ21 = rotate_pol([im/√2,0,im/√2], k̂); ϵ_func21 = ϵ_(ϵ21, 1); laser21 = Field(k̂, ϵ_func21, ω2, s_func(s2))
    k̂ = -kx; ϵ22 = rotate_pol([0,1,0], k̂); ϵ_func22 = ϵ_(ϵ22, 1); laser22 = Field(k̂, ϵ_func22, ω2, s_func(s2))
    
    lasers_linear = [laser19, laser20, laser21, laser22]
    
    # lasers = [lasers_1; lasers_2; lasers_3; lasers_linear]
    # lasers = [lasers_1; lasers_2; lasers_linear]
    
    lasers = [lasers_1; lasers_2; lasers_3]
    
end

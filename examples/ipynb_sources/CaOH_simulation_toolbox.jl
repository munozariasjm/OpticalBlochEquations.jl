using
    QuantumStates,
    OpticalBlochEquations,
    DifferentialEquations,
    UnitsToValue,
    Printf,
    MutableNamedTuples,
    Serialization,
    StaticArrays

import StructArrays: StructArray, StructVector

using Revise
    

function sample_direction(r=1.0)
    θ = 2π * rand()
    z = rand() * 2 - 1
    return SVector(r * sqrt(1 - z^2) * cos(θ), r * sqrt(1 - z^2) * sin(θ), r * z)
end
export sample_direction;


import ProgressMeter: Progress, next!

function force_scan_custom(prob::T1, scan_values::T2, prob_func!::F1, param_func::F2, output_func::F3, force_cb_func!::F4; n_threads=Threads.nthreads(), reshape=false) where {T1,T2,F1,F2,F3,F4}

    # total size of the scan
    n_values = 1
    n_params = 0
    for (var, values) in scan_values  # scan_values must be a dictionary.
        n_values = n_values * length(values)
        n_params = n_params + 1
    end
    
    # dimension of the outputs

 
    params = zeros(Float64, n_values, n_params)
    forces = zeros(Float64, n_values)
    populations = zeros(Float64, n_values, length(prob.p.states))
    batch_size = fld(n_values, n_threads)
    remainder = n_values - batch_size * n_threads
    prog_bar = Progress(n_values)

    Threads.@threads for i ∈ 1:n_threads
        prob_copy = deepcopy(prob)
        # Threads.@spawn begin
            # prob_func!(_prob, scan_values, i)
        force_cb = PeriodicCallback(force_cb_func!, prob_copy.p.period)
        if :callback ∈ keys(prob_copy.kwargs)
            cbs = prob_copy.kwargs[:callback]
            prob_copy = remake(prob_copy, callback=CallbackSet(cbs, force_cb))
        else
            prob_copy = remake(prob_copy, callback=force_cb)
        end
        _batch_size = i <= remainder ? (batch_size + 1) : batch_size - 1
        batch_start_idx = 1 + ((i <= remainder) ? i : remainder) + batch_size * (i-1)
        for j ∈ batch_start_idx:(batch_start_idx + _batch_size)

            
            prob_copy = prob_func!(prob_copy, scan_values, j)
            sol = solve(prob_copy, alg=DP5(),reltol=1e-3,abstol=1e-6)
            params[j,:] .= param_func(prob_copy, scan_values, j)
            forces[j] = output_func(prob_copy.p, sol)
            prob_copy.p.force_last_period = (0, 0, 0)

            populations[j,:] .= prob_copy.p.populations
            

            
            next!(prog_bar)
        end

    end
 
    return params, forces, populations

    
end
export force_scan_custom;

function round_freqs!(states, fields, freq_res)
    """
    Rounds frequencies of state energies and fields by a common denominator.
    
    freq_res::Float: all frequencies are rounded by this value (in units of Γ)
    """
    for i in eachindex(fields)
        fields.ω[i] = round_freq(fields.ω[i], freq_res)
    end
    for i in eachindex(states)
        states.E[i] = round_freq(states.E[i], freq_res)
    end
    return nothing
end
export round_freqs!;

function remake_obe_B(p, B)
    λ = p.extra_p.λ
    Γ = p.extra_p.Γ
 
    
    particle = Particle()
    
    
    freq_res = p.freq_res
    
    extra_p = deepcopy(p.extra_p)
    extra_p.Ham_X.parameters.B_z = B

    # re-diagonalize Hamiltonians
    evaluate!(extra_p.Ham_X)
    QuantumStates.solve!(extra_p.Ham_X)

    _, X_states = subspace(extra_p.Ham_X.states, (N=1,))
    for state in X_states
       state.E *= 1e6 
    end
    states = [X_states; extra_p.A_states]
    
    d = zeros(ComplexF64, 16, 16, 3)
    d_ge = zeros(ComplexF64, 12, 4, 3)
    tdms_between_states!(d_ge, extra_p.basis_tdms, X_states, extra_p.A_states)
    d[1:12, 13:16, :] .= d_ge
    
    ρ0 = zeros(ComplexF64, length(states), length(states)) 
    ρ0[2,2] = 1.0
    

    lasers = p.extra_p.lasers

    extra_p.X_states = X_states
    
    new_p = obe(ρ0, particle, states, lasers, d, d, true, true, λ, Γ, freq_res, extra_p)
    return new_p
end

# function remake_obe_B(p, B)
    # λ = p.extra_p.λ
    # Γ = p.extra_p.Γ
 
    
    # particle = Particle()
    
    
    # freq_res = p.freq_res
    
    # extra_p = deepcopy(p.extra_p)
    # extra_p.Ham_X.parameters.B_z = B
    # extra_p.Hax_A.parameters.B_z = B

    # # re-diagonalize Hamiltonians
    # evaluate!(extra_p.Ham_X)
    # QuantumStates.solve!(extra_p.Ham_X)

    # _, X_states = subspace(extra_p.Ham_X.states, (N=1,))
    # for state in X_states
    #    state.E *= 1e6 
    # end



    






    
    
    # d = zeros(ComplexF64, 16, 16, 3)
    # d_ge = zeros(ComplexF64, 12, 4, 3)
    # tdms_between_states!(d_ge, extra_p.basis_tdms, X_states, extra_p.A_states)
    # d[1:12, 13:16, :] .= d_ge
    
    # ρ0 = zeros(ComplexF64, length(states), length(states)) 
    # ρ0[2,2] = 1.0
    
    # states = [X_states; A_states]
    # lasers = p.extra_p.lasers

    # extra_p.X_states = X_states
    
    # new_p = obe(ρ0, particle, states, lasers, d, d, true, true, λ, Γ, freq_res, extra_p)
    # return new_p
# end
export remake_obe_B;

function remake_obe_laser_det(p, which_laser, det)
    states = [p.extra_p.X_states; p.extra_p.A_states]
    ρ0 = zeros(ComplexF64, length(states), length(states)) 
    ρ0[2,2] = 1.0

    particle = Particle()

    d = p.d

    λ = p.extra_p.λ
    Γ = p.extra_p.Γ

    freq_res = p.freq_res
    
    extra_p = deepcopy(p.extra_p)

    
    J12_energy = energy(states[1])
    J32_energy = energy(states[10])
    A_energy = energy(states[13])
   
    # Parameters for laser addressing J=1/2
    if which_laser == 1
        δJ12_red = det
    else
        δJ12_red = extra_p.δJ12_red
    end

    
    if which_laser == 2
        δJ12_blue = det * Γ
    else
        δJ12_blue = extra_p.δJ12_blue
    end


    if which_laser == 3
        δJ32_blue = det * Γ
    else
        δJ32_blue = extra_p.δJ32_blue
    end


    # Parameters for laser addressing J=1/2

    s_J12_red = extra_p.s_J12_red
    pol_J12_red = extra_p.pol_J12_red
    ω_J12_red = 2π * (A_energy - J12_energy) + δJ12_red

    
    s_J12_blue = extra_p.s_J12_blue
    pol_J12_blue = extra_p.pol_J12_blue
    ω_J12_blue = 2π * (A_energy - J12_energy) + δJ12_blue


    s_J32_blue = extra_p.s_J32_blue
    pol_J32_blue = extra_p.pol_J32_blue
    ω_J32_blue = 2π * (A_energy - J32_energy) + δJ32_blue

    # This function sets the polarization as a function of time; here it's just set to be constant


    ϵ1 = ϵ(rotate_pol(pol_J12_red, +x̂)); laser1 = Field(+x̂, ϵ1, ω_J12_red, s_J12_red)
    ϵ2 = ϵ(rotate_pol(pol_J12_red, -x̂)); laser2 = Field(-x̂, ϵ2, ω_J12_red, s_J12_red)
    ϵ3 = ϵ(rotate_pol(pol_J12_red, +ŷ)); laser3 = Field(+ŷ, ϵ3, ω_J12_red, s_J12_red)
    ϵ4 = ϵ(rotate_pol(pol_J12_red, -ŷ)); laser4 = Field(-ŷ, ϵ4, ω_J12_red, s_J12_red)
    ϵ5 = ϵ(rotate_pol(pol_J12_red, +ẑ)); laser5 = Field(+ẑ, ϵ5, ω_J12_red, s_J12_red)
    ϵ6 = ϵ(rotate_pol(pol_J12_red, -ẑ)); laser6 = Field(-ẑ, ϵ6, ω_J12_red, s_J12_red)
    lasers_J12_red = [laser1, laser2, laser3, laser4, laser5, laser6]

    ϵ7  = ϵ(rotate_pol(pol_J12_blue, +x̂)); laser7  = Field(+x̂, ϵ7,  ω_J12_blue, s_J12_blue)
    ϵ8  = ϵ(rotate_pol(pol_J12_blue, -x̂)); laser8  = Field(-x̂, ϵ8,  ω_J12_blue, s_J12_blue)
    ϵ9  = ϵ(rotate_pol(pol_J12_blue, +ŷ)); laser9  = Field(+ŷ, ϵ9,  ω_J12_blue, s_J12_blue)
    ϵ10 = ϵ(rotate_pol(pol_J12_blue, -ŷ)); laser10 = Field(-ŷ, ϵ10, ω_J12_blue, s_J12_blue)
    ϵ11 = ϵ(rotate_pol(pol_J12_blue, +ẑ)); laser11 = Field(+ẑ, ϵ11, ω_J12_blue, s_J12_blue)
    ϵ12 = ϵ(rotate_pol(pol_J12_blue, -ẑ)); laser12 = Field(-ẑ, ϵ12, ω_J12_blue, s_J12_blue)
    lasers_J12_blue = [laser7, laser8, laser9, laser10, laser11, laser12]

    ϵ13  = ϵ(rotate_pol(pol_J32_blue, +x̂)); laser13  = Field(+x̂, ϵ13,  ω_J32_blue, s_J32_blue)
    ϵ14 = ϵ(rotate_pol(pol_J32_blue, -x̂)); laser14 = Field(-x̂, ϵ14,  ω_J32_blue, s_J32_blue)
    ϵ15 = ϵ(rotate_pol(pol_J32_blue, +ŷ)); laser15 = Field(+ŷ, ϵ15,  ω_J32_blue, s_J32_blue)
    ϵ16 = ϵ(rotate_pol(pol_J32_blue, -ŷ)); laser16 = Field(-ŷ, ϵ16, ω_J32_blue, s_J32_blue)
    ϵ17 = ϵ(rotate_pol(pol_J32_blue, +ẑ)); laser17 = Field(+ẑ, ϵ17, ω_J32_blue, s_J32_blue)
    ϵ18 = ϵ(rotate_pol(pol_J32_blue, -ẑ)); laser18 = Field(-ẑ, ϵ18, ω_J32_blue, s_J32_blue)
    lasers_J32_blue = [laser13, laser14, laser15, laser16, laser17, laser18]

    lasers = [lasers_J12_red; lasers_J12_blue; lasers_J32_blue]
    ;
    extra_p.lasers = lasers
    extra_p.δJ12_red = δJ12_red
    extra_p.δJ12_blue = δJ12_blue
    extra_p.δJ32_blue = δJ32_blue
    

    new_p = obe(ρ0, particle, states, lasers, d, d, true, true, λ, Γ, freq_res, extra_p)

    return new_p

end
export remake_obe_laser_det;

function remake_obe_laser_sat(p, which_laser, s)
    states = [p.extra_p.X_states; p.extra_p.A_states]
    ρ0 = zeros(ComplexF64, length(states), length(states)) 
    ρ0[2,2] = 1.0

    particle = Particle()

    d = p.d

    λ = p.extra_p.λ
    Γ = p.extra_p.Γ

    freq_res = p.freq_res
    
    extra_p = deepcopy(p.extra_p)

    
    J12_energy = energy(states[1])
    J32_energy = energy(states[10])
    A_energy = energy(states[13])
   
    # Parameters for laser addressing J=1/2
    if which_laser == 1
        s_J12_red = s
    else
        s_J12_red = extra_p.s_J12_red
    end

    
    if which_laser == 2
        s_J12_blue = s
    else
        s_J12_blue = extra_p.s_J12_blue
    end


    if which_laser == 3
        s_J32_blue = s
    else
        s_J32_blue = extra_p.s_J32_blue
    end


    # Parameters for laser addressing J=1/2

    δJ12_red = p.extra_p.δJ12_red
    pol_J12_red = p.extra_p.pol_J12_red
    ω_J12_red = 2π * (A_energy - J12_energy) + δJ12_red

    
    δJ12_blue = p.extra_p.δJ12_blue
    pol_J12_blue = p.extra_p.pol_J12_blue
    ω_J12_blue = 2π * (A_energy - J12_energy) + δJ12_blue


    δJ32_blue = p.extra_p.δJ32_blue
    pol_J32_blue = p.extra_p.pol_J32_blue
    ω_J32_blue = 2π * (A_energy - J32_energy) + δJ32_blue

    # This function sets the polarization as a function of time; here it's just set to be constant


    ϵ1 = ϵ(rotate_pol(pol_J12_red, +x̂)); laser1 = Field(+x̂, ϵ1, ω_J12_red, s_J12_red)
    ϵ2 = ϵ(rotate_pol(pol_J12_red, -x̂)); laser2 = Field(-x̂, ϵ2, ω_J12_red, s_J12_red)
    ϵ3 = ϵ(rotate_pol(pol_J12_red, +ŷ)); laser3 = Field(+ŷ, ϵ3, ω_J12_red, s_J12_red)
    ϵ4 = ϵ(rotate_pol(pol_J12_red, -ŷ)); laser4 = Field(-ŷ, ϵ4, ω_J12_red, s_J12_red)
    ϵ5 = ϵ(rotate_pol(pol_J12_red, +ẑ)); laser5 = Field(+ẑ, ϵ5, ω_J12_red, s_J12_red)
    ϵ6 = ϵ(rotate_pol(pol_J12_red, -ẑ)); laser6 = Field(-ẑ, ϵ6, ω_J12_red, s_J12_red)
    lasers_J12_red = [laser1, laser2, laser3, laser4, laser5, laser6]

    ϵ7  = ϵ(rotate_pol(pol_J12_blue, +x̂)); laser7  = Field(+x̂, ϵ7,  ω_J12_blue, s_J12_blue)
    ϵ8  = ϵ(rotate_pol(pol_J12_blue, -x̂)); laser8  = Field(-x̂, ϵ8,  ω_J12_blue, s_J12_blue)
    ϵ9  = ϵ(rotate_pol(pol_J12_blue, +ŷ)); laser9  = Field(+ŷ, ϵ9,  ω_J12_blue, s_J12_blue)
    ϵ10 = ϵ(rotate_pol(pol_J12_blue, -ŷ)); laser10 = Field(-ŷ, ϵ10, ω_J12_blue, s_J12_blue)
    ϵ11 = ϵ(rotate_pol(pol_J12_blue, +ẑ)); laser11 = Field(+ẑ, ϵ11, ω_J12_blue, s_J12_blue)
    ϵ12 = ϵ(rotate_pol(pol_J12_blue, -ẑ)); laser12 = Field(-ẑ, ϵ12, ω_J12_blue, s_J12_blue)
    lasers_J12_blue = [laser7, laser8, laser9, laser10, laser11, laser12]

    ϵ13  = ϵ(rotate_pol(pol_J32_blue, +x̂)); laser13  = Field(+x̂, ϵ13,  ω_J32_blue, s_J32_blue)
    ϵ14 = ϵ(rotate_pol(pol_J32_blue, -x̂)); laser14 = Field(-x̂, ϵ14,  ω_J32_blue, s_J32_blue)
    ϵ15 = ϵ(rotate_pol(pol_J32_blue, +ŷ)); laser15 = Field(+ŷ, ϵ15,  ω_J32_blue, s_J32_blue)
    ϵ16 = ϵ(rotate_pol(pol_J32_blue, -ŷ)); laser16 = Field(-ŷ, ϵ16, ω_J32_blue, s_J32_blue)
    ϵ17 = ϵ(rotate_pol(pol_J32_blue, +ẑ)); laser17 = Field(+ẑ, ϵ17, ω_J32_blue, s_J32_blue)
    ϵ18 = ϵ(rotate_pol(pol_J32_blue, -ẑ)); laser18 = Field(-ẑ, ϵ18, ω_J32_blue, s_J32_blue)
    lasers_J32_blue = [laser13, laser14, laser15, laser16, laser17, laser18]

    lasers = [lasers_J12_red; lasers_J12_blue; lasers_J32_blue]
    ;
    extra_p.lasers = lasers
    extra_p.s_J12_red = s_J12_red
    extra_p.s_J12_blue = s_J12_blue
    extra_p.s_J32_blue = s_J32_blue
    

    new_p = obe(ρ0, particle, states, lasers, d, d, true, true, λ, Γ, freq_res, extra_p)

    return new_p

end
export remake_obe_laser_sat;

function remake_obe_freq_res(p, freq_res)
    λ = p.extra_p.λ
    Γ = p.extra_p.Γ
    d = p.d
 
    particle = Particle()
    
    extra_p = p.extra_p
    states = [p.extra_p.X_states; p.extra_p.A_states]

    ρ0 = zeros(ComplexF64, length(states), length(states)) 
    ρ0[2,2] = 1.0

    lasers = p.extra_p.lasers
    

    new_p = obe(ρ0, particle, states, lasers, d, d, true, true, λ, Γ, freq_res, extra_p)
    return new_p
end


function prob_func!(prob, scan_values, j)
    current_indicies = unpack_indicies(j, scan_values)

    new_p = prob.p

        
    if haskey(scan_values, "B")
        current_B = scan_values["B"][current_indicies["B"]]

        if haskey(new_p.extra_p.current_config, "B")
            if new_p.extra_p.current_config["B"] != current_indicies["B"]
                new_p = remake_obe_B(prob.p, current_B)
            end
        else
            new_p = remake_obe_B(prob.p, current_B)
        end
    end

    if haskey(scan_values, "δ1")
        current_δ1 = scan_values["δ1"][current_indicies["δ1"]]

        if haskey(new_p.extra_p.current_config, "δ1") # Don't remake obe if the parameter didn't change.
            if new_p.extra_p.current_config["δ1"] != current_indicies["δ1"]
                new_p = remake_obe_laser_det(new_p, 1, current_δ1)
            end
        else
            new_p = remake_obe_laser_det(new_p, 1, current_δ1)
        end
    end

    if haskey(scan_values, "δ2")
        current_δ2 = scan_values["δ2"][current_indicies["δ2"]]

        if haskey(new_p.extra_p.current_config, "δ2")
            if new_p.extra_p.current_config["δ2"] != current_indicies["δ2"]
                new_p = remake_obe_laser_det(new_p, 2, current_δ2)
            end
        else
            new_p = remake_obe_laser_det(new_p, 2, current_δ2)
        end
    
    end

    if haskey(scan_values, "δ3")
        current_δ3 = scan_values["δ3"][current_indicies["δ3"]]

        if haskey(new_p.extra_p.current_config, "δ2")
            if new_p.extra_p.current_config["δ2"] != current_indicies["δ2"]
                new_p = remake_obe_laser_det(new_p, 3, current_δ3)
            end
        else
            new_p = remake_obe_laser_det(new_p, 3, current_δ3)
        end
        
    end

    if haskey(scan_values, "δ23")
        current_δ23 = scan_values["δ23"][current_indicies["δ23"]]

        if haskey(new_p.extra_p.current_config, "δ23")
            if new_p.extra_p.current_config["δ23"] != current_indicies["δ23"]
                new_p = remake_obe_laser_det(new_p, 2, current_δ23)
                new_p = remake_obe_laser_det(new_p, 3, current_δ23)
            end
        else
            new_p = remake_obe_laser_det(new_p, 2, current_δ23)
            new_p = remake_obe_laser_det(new_p, 3, current_δ23)
        end     
    end

    if haskey(scan_values, "s1")
        current_s1 = scan_values["s1"][current_indicies["s1"]]

        if haskey(new_p.extra_p.current_config, "s1")
            if new_p.extra_p.current_config["s1"] != current_indicies["s1"]
                new_p = remake_obe_laser_sat(new_p, 1, current_s1)
            end
        else
            new_p = remake_obe_laser_sat(new_p, 1, current_s1)
        end
    end

    if haskey(scan_values, "s2")
        current_s2 = scan_values["s2"][current_indicies["s2"]]

        if haskey(new_p.extra_p.current_config, "s2")
            if new_p.extra_p.current_config["s2"] != current_indicies["s2"]
                new_p = remake_obe_laser_sat(new_p, 2, current_s2)
            end
        else
            new_p = remake_obe_laser_sat(new_p, 2, current_s2)
        end
    end

    if haskey(scan_values, "s3")
        current_s3 = scan_values["s3"][current_indicies["s3"]]

        if haskey(new_p.extra_p.current_config, "s3")
            if new_p.extra_p.current_config["s3"] != current_indicies["s3"]
                new_p = remake_obe_laser_sat(new_p, 3, current_s3)
            end
        else
            new_p = remake_obe_laser_sat(new_p, 3, current_s3)
        end
    end

    if haskey(scan_values, "s23")
        current_s23 = scan_values["s23"][current_indicies["s23"]]

        if haskey(new_p.extra_p.current_config, "s23")
            if new_p.extra_p.current_config["s23"] != current_indicies["s23"]
                new_p = remake_obe_laser_sat(new_p, 2, current_s23)
                new_p = remake_obe_laser_sat(new_p, 3, current_s23)
            end
        else
            new_p = remake_obe_laser_sat(new_p, 2, current_s23)
            new_p = remake_obe_laser_sat(new_p, 3, current_s23)
        end
    end


    # if haskey(scan_values, "v")
    #     current_v = scan_values["v"][current_indicies["v"]]
    #     if (abs(current_v) > 0.5 / (Γ/k)) && (new_p.freq_res < 1e-1)
    #         new_p = remake_obe_freq_res(new_p, 1e-1)
    #     elseif (abs(current_v) <= 0.5 / (Γ/k) )&& (new_p.freq_res > 1e-2)
    #         new_p = remake_obe_freq_res(new_p, 1e-2)
    #     end
    #     new_p.v .= (0, 0, current_v) 
    #     new_p.v .= round_vel(new_p.v, new_p.freq_res) 
    # end

    if haskey(scan_values, "v")
        current_v = scan_values["v"][current_indicies["v"]]
        new_p.v .= (0, 0, current_v) 
        new_p.v .= round_vel(new_p.v, new_p.freq_res) 
    end

    if haskey(scan_values, "r")
        current_r = scan_values["r"][current_indicies["r"]]
        new_p.r0 .= [0,0,current_r]
    else
        new_p.r0 .= rand(uniform_dist, 3)
    end

    new_p.extra_p.current_config = current_indicies
    prob = remake(prob; p=new_p, u0=new_p.ρ0_vec)
            
    return prob
end
export prob_func!;

function param_func(prob, scan_values, j)
    current_indicies = unpack_indicies(j, scan_values)
    current_value = []
    for (key, index) in current_indicies
         push!(current_value, scan_values[key][index])
    end
    
    return current_value
end
export param_func;

function output_func(p, sol)
    f = p.force_last_period
    f_proj = f[3]#(f ⋅ p.v) #/ norm(p.v)
    return f_proj
end
export output_func;


function unpack_indicies(i, scan_values)
    i = i-1
    current_indicies = Dict{String, Int32}()
    my_keys = collect(keys(scan_values))
    for key_i in length(my_keys):-1:1
        var = my_keys[key_i]
        values = scan_values[var]
        
        current_indicies[var] = mod(i,length(values)) + 1
        i = i ÷ length(values)
    end
    return current_indicies
end
export unpack_indicies;

function pack_indicies(indicies, scan_values)
    j = 0
    for (key, index) in indicies
        j = j * length(scan_values[key])
        j = j + index - 1
    end
    return j+1
end
export pack_indicies;

function get_output_slice_1d(fixed_indicies, scan_values, forces)
    """ Fix all indicies except one, and plot output vs. that index.
    
    fixed_indicies should be a dictionary with "param_name" => index, 
    and should have only 1 less parameter than scan_values.
    """

    out = Float64[]
    # find which key in scan_values is not fixed.
    key = ""
    for (var, values) in scan_values
        if !haskey(fixed_indicies, var)
           key = var
        end
    end
    
    if key == ""
        return 0
    end
    
    for slice_i in 1:1:length(scan_values[key])
        indicies = Dict(key => slice_i)
        for (var, values) in scan_values
            if var != key
                indicies[var]= fixed_indicies[var]
            end
        end
        j = pack_indicies(indicies, scan_values)
        push!(out, forces[j])
    end
    return out
end
export get_output_slice;

function get_output_slice(fixed_indicies, scan_values, forces)
    """ Fix some indicies.
    
    fixed_indicies should be a dictionary with "param_name" => index, 
    and should have only 1 or more less parameter than scan_values.

    The output will be nested vectors. Use something like
        using TensorCast
        @cast test_tensor[j,i] := test[i][j]
    to convert to matrix.
    """
    out = []
    # find which keys in scan_values is not fixed.
    scan_keys = []
    for (var, values) in scan_values
        if !haskey(fixed_indicies, var)
           push!(scan_keys, var)
        end
    end
    
    if length(scan_keys) == 1
        return get_output_slice_1d(fixed_indicies, scan_values, forces)
    end
    
    key = scan_keys[1]
    temp_dict = deepcopy(fixed_indicies)
    for i in 1:1:length(scan_values[key])
        temp_dict[key] = i
        slice = get_output_slice(temp_dict, scan_values, forces)
        push!(out, slice)
    end

    return out
end
export get_output_slice_2d

function average_1_param(scan_values, forces, param_to_avg)

    shortened_scan_values = Dict()
    for (key, value) in scan_values
        if key != param_to_avg
            shortened_scan_values[key] = value
        end
    end
    summed_output = zeros(Float64, length(forces)÷length(scan_values[param_to_avg]))
    
    for j in 1:1:length(forces)
       current_indicies = unpack_indicies(j, scan_values) 
        shortened_indicies = Dict()
        for (key, index) in current_indicies
            if key != param_to_avg
                shortened_indicies[key] = index
            end
        end
        jj = pack_indicies(shortened_indicies, shortened_scan_values)
        summed_output[jj] += forces[j]
    end
    
    return shortened_scan_values, summed_output ./ length(scan_values[param_to_avg])
end
export average_1_param;


function get_CaOH_Hamiltonian()
    # X states
    HX_operator = :(
        BX * Rotation + 
        DX * RotationDistortion + 
        γX * SpinRotation + 
        bFX * Hyperfine_IS + 
        cX * (Hyperfine_Dipolar/3)
    )

    parameters = @params begin
        BX = 0.33441 * 299792458 * 1e-4
        DX = 0.3869e-6 * 299792458 * 1e-4
        γX = 0.001134 * 299792458 * 1e-4
        bFX = 2.602
        cX = 2.053
    end

    QN_bounds = (S=1/2, I=1/2, Λ=0, N=0:3)
    basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

    CaOH_000_N0to3_Hamiltonian = Hamiltonian(basis=basis, operator=HX_operator, parameters=parameters)

    # Add Zeeman term
    _μB = (μ_B / h) * (1e-6 * 1e-4)
    Zeeman_z(state, state′) = Zeeman(state, state′, 0)
    CaOH_000_N0to3_Hamiltonian = add_to_H(CaOH_000_N0to3_Hamiltonian, :B_z, gS * _μB * Zeeman_z)
    CaOH_000_N0to3_Hamiltonian.parameters.B_z = 1e-6 #todo

    full_evaluate!(CaOH_000_N0to3_Hamiltonian)
    QuantumStates.solve!(CaOH_000_N0to3_Hamiltonian)
    ;


    # A states
    H_operator = :(
        T_A * DiagonalOperator +
        Be_A * Rotation + 
        Aso_A * SpinOrbit + 
        q_A * ΛDoubling_q +
        p_A * ΛDoubling_p2q + q_A * (2ΛDoubling_p2q)
    )

    parameters = @params begin
        T_A = 15998.122 * 299792458 * 1e-4
        Be_A = 0.3412200 * 299792458 * 1e-4
        Aso_A = 66.8181 * 299792458 * 1e-4
        p_A = -0.04287 * 299792458 * 1e-4
        q_A = -0.3257e-3 * 299792458 * 1e-4
    end

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), J=1/2:5/2)
    basis = enumerate_states(HundsCaseA_Rot, QN_bounds)

    CaOH_A000_J12to52_Hamiltonian = Hamiltonian(basis=basis, operator=H_operator, parameters=parameters)
    evaluate!(CaOH_A000_J12to52_Hamiltonian)
    QuantumStates.solve!(CaOH_A000_J12to52_Hamiltonian)
    ;

    HA_J12_pos_parity_states = CaOH_A000_J12to52_Hamiltonian.states[5:8]

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), N=0:2)
    basis_to_convert = enumerate_states(HundsCaseB_Rot, QN_bounds)

    states_A_J12_caseB = convert_basis(HA_J12_pos_parity_states, basis_to_convert)
    ;

    _, HX_N1_states = subspace(CaOH_000_N0to3_Hamiltonian.states, (N=1,))
    states = [HX_N1_states; states_A_J12_caseB]
    for state ∈ states
        state.E *= 1e6
    end
    ;

    # Calculate transtion dipole moment between X and A states:
    d = zeros(ComplexF64, 16, 16, 3)
    d_ge = zeros(ComplexF64, 12, 4, 3)
    basis_tdms = get_tdms_two_bases(CaOH_000_N0to3_Hamiltonian.basis, basis_to_convert, TDM)
    tdms_between_states!(d_ge, basis_tdms, HX_N1_states, states_A_J12_caseB)
    d[1:12, 13:16, :] .= d_ge
    ;

    return MutableNamedTuple(Ham_X=CaOH_000_N0to3_Hamiltonian, states=states, d=d, basis_tdms=basis_tdms)
end
export get_CaOH_Hamiltonian
;

function average_1_param_vector(scan_values, populations, param_to_avg)
    avg_output = zeros(Float64, (size(populations)[1]÷length(scan_values[param_to_avg]), size(populations)[2]))
    for col in 1:1:size(populations)[2]
        shortened_scan_values, avg_output[:,col] = average_1_param(scan_values, populations[:,col],param_to_avg)
    end
    return shortened_scan_values, avg_output
end
export average_1_param_vector
;


# gravity(r) = SVector(0.0, -9.81, 0.0)

function f(idx, r, v, p, time)
    B = r[3] * p.B_gradient
    a = p.acceleration_func(B * sign(v[3]), abs(v[3])) * sign(v[3])
    return SVector(0,0,a)
end 
export f;

function save(particles, p, s)
    for i in 1:size(particles, 1)
        idx = particles.idx[i]

        push!(s.trajectories[idx], particles.r[i])
        push!(s.velocities[idx], particles.v[i])
        B = particles.r[i][3] * p.B_gradient
        v = particles.v[i]
        push!(s.A_populations[idx], [p.A_population_func(B * sign(v[3]), abs(v[3]))])
    end
    return nothing
end
export save;

function update(particles, p, s, dt, time, idx)
    hbar = 1.05457182e-34 
    for i in 1:size(particles, 1)
        B = particles.r[i][3] * p.B_gradient
        v = particles.v[i]
        Pe = p.A_population_func(B * sign(v[3]), abs(v[3]))
        scattering_prob = dt * Γ * Pe
        if rand() < scattering_prob 
            v_rand = sample_direction(hbar * k / m)
            particles.v[i] += [0,0,v_rand[3]]
            # if rand() < 0.5
            #     particles.v[i] += sample_direction(hbar * k / m) # SVector(0,0,hbar * k / m)
            # else
            #     particles.v[i] -= SVector(0,0,hbar * k / m)
            # end
        end
    end
    return nothing
end
export update;


function init_1D_MOT_distribution(T, diameter)
    kB = BoltzmannConstant.val
    m = @with_unit 57 "u"
    σ = sqrt(kB * T / m)
    
    r = (Normal(0, diameter), Normal(0, diameter), Normal(0, diameter))
    v = (Normal(0, σ), Normal(0, σ), Normal(0, σ))
    a = (Normal(0, 0), Normal(0, 0), Normal(0, 0))
    return r, v, a
end
;


function propagate_particles!(r, v, a, alg, particles, f::F1, save::F2, discard::F3, save_every, delete_every, max_steps, update, p, s, dt, use_adaptive, dt_min, dt_max, abstol, randomize) where {F1, F2, F3}

    n = length(particles)
    n_threads = Threads.nthreads()
    chunk_size = ceil(Int64, n / n_threads)
    
    total_points = n_threads * max_steps
    prog_bar = Progress(total_points)
    
    Threads.@threads for i in 1:n_threads

        p_ = deepcopy(p)
        
        start_idx   = (i-1)*chunk_size+1
        end_idx     = min(i*chunk_size, n)
        chunk_idxs  = start_idx:end_idx
        actual_chunk_size = length(chunk_idxs)
        
        particles_chunk = particles[chunk_idxs]
        if randomize
            initialize_dists_particles!(r, v, a, start_idx, particles_chunk, dt, use_adaptive)
        end

        idx = 1
        save_idx = 1
        time = 0.0

        for step in 0:(max_steps - 1)

            update(particles_chunk, p_, s, dt, time, idx)

            if step % save_every == 0
                save(particles_chunk, p_, s)
            end

            if step % delete_every == 0
                discard_particles!(particles_chunk, discard)
            end


            if alg == "euler"
                dtstep_euler!(particles_chunk, f, abstol, p_, dt_min, dt_max, time)
            elseif alg == "rkf12"
                dtstep_eulerrich!(particles_chunk, f, abstol, p_, dt_min, dt_max, time)
            end

            idx += 1
            time += dt
            next!(prog_bar)

            if iszero(length(particles_chunk))
                break
            end
        end
    end

    return nothing
end
;

function make_grids(shortened_scan_values, avg_acceleration)
    dims = []
    for (key, value) in shortened_scan_values
        push!(dims, length(value))
    end
    println(reverse(collect(keys(shortened_scan_values))))
    dims = Tuple(reverse(dims))
    return reshape(avg_acceleration, dims);
end
export make_grids;


function get_YO_Hamiltonian()
    # X states
    HX_operator = :(
        BX * Rotation + 
        DX * RotationDistortion + 
        γX * SpinRotation + 
        bFX * Hyperfine_IS + 
        cX * (Hyperfine_Dipolar/3)
    )

    parameters = @params begin
        BX = 11.6336 * 1e3
        DX = 9.581 * 1e-3
        γX = -9.2254
        bFX = -762.976 + ( -28.236 )/3
        cX = -28.236 
    end


    QN_bounds = (S=1/2, I=1/2, Λ=0, N=0:3)
    basis = enumerate_states(HundsCaseB_Rot, QN_bounds)

    YO_000_N0to3_Hamiltonian = Hamiltonian(basis=basis, operator=HX_operator, parameters=parameters)

    # Add Zeeman term
    _μB = (μ_B / h) * (1e-6 * 1e-4)
    Zeeman_z(state, state′) = Zeeman(state, state′, 0)
    YO_000_N0to3_Hamiltonian = add_to_H(YO_000_N0to3_Hamiltonian, :B_z, gS * _μB * Zeeman_z)
    YO_000_N0to3_Hamiltonian.parameters.B_z = 1e-6

    full_evaluate!(YO_000_N0to3_Hamiltonian)
    QuantumStates.solve!(YO_000_N0to3_Hamiltonian)
    ;


    # A states
    H_operator = :(
        T_A * DiagonalOperator +
        Be_A * Rotation + 
        q_A * ΛDoubling_q +
        p_A * ΛDoubling_p2q + q_A * (2ΛDoubling_p2q)
    )

    parameters = @params begin
        T_A = 16742.2 * 299792458 * 1e-4
        Be_A = 0.3857 * 299792458 * 1e-4
        p_A = -0.150343 * 299792458 * 1e-4
        q_A = -0.1331e-3 * 299792458 * 1e-4
    end

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), J=1/2:5/2)
    basis = enumerate_states(HundsCaseA_Rot, QN_bounds)

    YO_A000_J12to52_Hamiltonian = Hamiltonian(basis=basis, operator=H_operator, parameters=parameters)
    evaluate!(YO_A000_J12to52_Hamiltonian)
    QuantumStates.solve!(YO_A000_J12to52_Hamiltonian)
    ;

    HA_J12_pos_parity_states = YO_A000_J12to52_Hamiltonian.states[5:8]

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), N=0:2)
    basis_to_convert = enumerate_states(HundsCaseB_Rot, QN_bounds)

    states_A_J12_caseB = convert_basis(HA_J12_pos_parity_states, basis_to_convert)
    ;

    _, HX_N1_states = subspace(YO_000_N0to3_Hamiltonian.states, (N=1,))
    states = [HX_N1_states; states_A_J12_caseB]
    for state ∈ states
        state.E *= 1e6
    end
    ;

    # Calculate transtion dipole moment between X and A states:
    d = zeros(ComplexF64, 16, 16, 3)
    d_ge = zeros(ComplexF64, 12, 4, 3)
    basis_tdms = get_tdms_two_bases(YO_000_N0to3_Hamiltonian.basis, basis_to_convert, TDM)
    tdms_between_states!(d_ge, basis_tdms, HX_N1_states, states_A_J12_caseB)
    d[1:12, 13:16, :] .= d_ge
    ;

    return MutableNamedTuple(Ham_X=YO_000_N0to3_Hamiltonian, states=states, d=d, basis_tdms=basis_tdms)
end
;

using ImageFiltering
function find_equilibrium_index(t, y; rel_tol=0.2)
    dt = t[2] - t[1]
    dydt = (y[2:end] - y[1:end-1])./dt
    abs_tol = rel_tol * abs(y[end]-y[1])/(t[end]-t[1])
    dydt_smooth = imfilter(dydt, Kernel.gaussian((100,)))
    index = findfirst(x->abs(x)<abs_tol, dydt_smooth)
    if index == nothing
        return length(t)
    else
        return index
    end
end;



function discard_particles!(particles, discard)
    @inbounds for i in 1:size(particles, 1)
        particles.dead[i] = discard(particles.r[i], particles.v[i])
    end
    StructArrays.foreachfield(x -> deleteat!(x, particles.dead), particles)
    return nothing
end

function discard_out_of_bound(r, v)
    if abs(r[3]) > abs( maximum(scan_values["B"]) / B_gradient )
        return true
    elseif abs(v[3]) > abs( maximum(scan_values["v"])* (Γ / k))
        return true
    else
        return false
    end
end
;


function get_SrF_Hamiltonian()
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

    # Add Zeeman term
    _μB = (μ_B / h) * (1e-6 * 1e-4)
    Zeeman_z(state, state′) = Zeeman(state, state′, 0)
    SrF_000_N0to3_Hamiltonian = add_to_H(SrF_000_N0to3_Hamiltonian, :B_z, gS * _μB * Zeeman_z)
    SrF_000_N0to3_Hamiltonian.parameters.B_z = 1e-3 #todo

    full_evaluate!(SrF_000_N0to3_Hamiltonian)
    QuantumStates.solve!(SrF_000_N0to3_Hamiltonian)
    ;


    # A states
    H_operator = :(
        T_A * DiagonalOperator +
        Be_A * Rotation + 
        Aso_A * SpinOrbit + 
        q_A * ΛDoubling_q +
        p_A * ΛDoubling_p2q + q_A * (2ΛDoubling_p2q) 
    ) # had to add a small hyperfine splitting so that when zeeman terms are added, mF remains a good quantum number 
    # (breaks degeneracy between hyperfine states so that the eigenstates of H found by solver are eigenstates of m)

    parameters = @params begin
        T_A = 15072.09 * 299792458 * 1e-4
        Be_A =  0.2536135 * 299792458 * 1e-4
        Aso_A = 281.46138 * 299792458 * 1e-4
        p_A = -0.133002 * 299792458 * 1e-4
        q_A = -0.3257e-3 * 299792458 * 1e-4
    end

    QN_bounds = (S=1/2, I=1/2, Λ=(-1,1), J=1/2:5/2)
    basis = enumerate_states(HundsCaseA_Rot, QN_bounds)

    gL = 1
    gL_prime = -0.083*3 #SrF
    Zeeman_A(state, state1) = gL * _μB * Zeeman_L(state, state1) + gS * _μB * Zeeman_S(state, state1) + gL_prime * _μB * Zeeman_glprime(state, state1)
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

    return MutableNamedTuple(Ham_X=SrF_000_N0to3_Hamiltonian, Ham_A = SrF_A000_J12to52_Hamiltonian, states=states, d=d, basis_tdms=basis_tdms)
end
;
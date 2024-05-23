    using OpticalBlochEquations

function generate_sequence!(sequence, timing) 
    t_last = 0.0
    i_last = 1
   for params in sequence
        params.t_start = t_last
        params.t_end = t_last + timing[i_last]
        t_last = t_last + timing[i_last]
        i_last += 1
    end
end


function get_average_diffusion(results)
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end 
    trapped = []       
    out_x= [] 
    out_y = []
    out_z = []                                                                
    for i in 1:length(results.times)
         if length(results.times[i]) == max_t_id   
             push!(trapped, i)
            push!(out_x, results.sum_diffusion_x[i] / results.photons_scattered[i])
            push!(out_y, results.sum_diffusion_y[i] / results.photons_scattered[i])
            push!(out_z, results.sum_diffusion_z[i] / results.photons_scattered[i])
         end                                                                                   
    end 
#    plot(trapped, out_x, linewidth=2,title="average diffusion (<p^2>-<p>^2)", xlabel="trapped particle index", size=(500,300),dpi=300, label="x")
#    plot!(trapped, out_y, linewidth=2,label="y") 
#    plot!(trapped, out_z, linewidth=2,label="z")                                                                             
   return (mean(out_x) + mean(out_y) + mean(out_z) ) / 3                                                                                                                                                                         
end 
    

function set_H_zero!(H)
    @turbo for i in eachindex(H)
        H.re[i] = 0.0
        H.im[i] = 0.0
    end
    return nothing
end

# struct Jump
#     s ::Int64
#     s′::Int64
#     q::Int64
#     r ::ComplexF64
# end

# function base_to_soa!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
#     @inbounds for i in eachindex(ρ_soa)
#         ρ_soa.re[i] = real(ρ[i])
#         ρ_soa.im[i] = imag(ρ[i])
#     end
#     return nothing
# end

# function update_eiωt!(eiωt::StructArray{<:Complex}, ω::Array{<:Real}, τ::Real)
#     @turbo for i ∈ eachindex(ω)
#         eiωt.im[i], eiωt.re[i] = sincos( ω[i] * τ )
#     end
    
#     return nothing
# end



# function mul_by_im_minus!(C::StructArray{<:Complex})
#     @turbo for i ∈ eachindex(C)
#         a = C.re[i]
#         C.re[i] = C.im[i]
#         C.im[i] = -a
#     end
# end

# function soa_to_base!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
#     @inbounds for i in eachindex(ρ_soa)
#         ρ[i] = ρ_soa.re[i] + im * ρ_soa.im[i]
#     end
#     return nothing
# end





        

            
            
                    

function state_overlap(state1, state2)
    overlap = 0.0 *im
    @turbo for i ∈ eachindex(state1)
        state1_re = state1.re[i]
        state1_im = state1.im[i]
        state2_re = state2.re[i]
        state2_im = state2.im[i]
        overlap += state1_re*state2_re + state1_im*state2_im + im*(state1_re*state2_im - state2_re*state1_im)
    end
    return overlap
end

function operator_matrix_expectation_complex(O, state)
    O_re = 0.0
    O_im = 0.0
    @turbo for i ∈ eachindex(state)
        re_i = state.re[i]
        im_i = state.im[i]
        for j ∈ eachindex(state)
            re_j = state.re[j]
            im_j = state.im[j]
            cicj_re = re_i * re_j + im_i * im_j # real part of ci* * cj
            cicj_im = re_i * im_j - im_i * re_j
            O_re += O.re[i,j] * cicj_re - O.im[i,j] * cicj_im
            O_im += O.re[i,j] * cicj_im + O.im[i,j] * cicj_re
        end
    end
    return (O_re, O_im)
end


                                                                                
function find_diffusion_constant(params; run_time=1e-3, n_particles=20, ramp_time=1e-6, temp=1e-4, diameter=80e-6)
    params_copy = deepcopy(params)
    params_copy.n_values = n_particles
    params_copy.B_ramp_time = ramp_time
    params_copy.s_ramp_time = ramp_time
    params_copy.temp = temp
    params_copy.diameter = diameter
    sequence = [params_copy]
    durations = [run_time]
    generate_sequence!(sequence, durations)
    
    results = simulate_particles_diffusion(package, package_A, sequence)
    diffusion_constant = get_average_diffusion(results)
    return diffusion_constant
end; 
      
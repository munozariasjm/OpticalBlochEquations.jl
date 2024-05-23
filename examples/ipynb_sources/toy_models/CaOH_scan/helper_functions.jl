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
   return (sqrt(mean(out_x)) + sqrt(mean(out_y)) + sqrt(mean(out_z)) )/ 3                                                                                                                                                                         
end 
    
function get_diffusion(results)
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
                                                                           
   return out_x, out_y, out_z                                                                           
end
        
    

function set_H_zero!(H)
    @turbo for i in eachindex(H)
        H.re[i] = 0.0
        H.im[i] = 0.0
    end
    return nothing
end

struct Jump
    s ::Int64
    s′::Int64
    q::Int64
    r ::ComplexF64
end

function base_to_soa!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @inbounds for i in eachindex(ρ_soa)
        ρ_soa.re[i] = real(ρ[i])
        ρ_soa.im[i] = imag(ρ[i])
    end
    return nothing
end

function update_eiωt!(eiωt::StructArray{<:Complex}, ω::Array{<:Real}, τ::Real)
    @turbo for i ∈ eachindex(ω)
        eiωt.im[i], eiωt.re[i] = sincos( ω[i] * τ )
    end
    
    return nothing
end



function mul_by_im_minus!(C::StructArray{<:Complex})
    @turbo for i ∈ eachindex(C)
        a = C.re[i]
        C.re[i] = C.im[i]
        C.im[i] = -a
    end
end

function soa_to_base!(ρ::Array{<:Complex}, ρ_soa::StructArray{<:Complex})
    @inbounds for i in eachindex(ρ_soa)
        ρ[i] = ρ_soa.re[i] + im * ρ_soa.im[i]
    end
    return nothing
end



function condition(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[p.n_states+i])
    end
    
    _condition = integrated_excited_pop - p.time_to_decay
    
    r = 0.0
    for i ∈ 1:3
        r += norm(u[p.n_states + p.n_excited + i])^2
    end
    r = sqrt(r)
    if r >= 3e-3*k # terminate if the particle is more than 3 mm from the centre
       terminate!(integrator) 
    elseif integrator.p.n_scatters > integrator.p.sim_params.photon_budget # also terminate if too many photons have been scattered
        terminate!(integrator)
    end
    return _condition
end

function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function get_ODT_Hamiltonian_matrix(package, package_A, peak_intensity, pol, wavelength=1064e-9)
    """ Return the ODT Hamiltonian in the basis given by "package"."""
    h = 6.62607015e-34    
    c = 2.99792458e8 
    Γ = package.Γ
    λ = (1/package.k)/(2π)                                   
    Is = π*h*c*Γ/(3λ^3) / 10 # saturation intensity in mW/cm^2
    s = peak_intensity / Is
    f_ODT = c/wavelength 
    
    n_states = length(package.states)
    E0 = sqrt(s)/(2 * √2) # factor?
    H_ODT = zeros(ComplexF64, n_states, n_states)
    
    d = package_A.d
#     fs = energy.(package_A.states)

    
    for q in 1:3
        for p in 1:3
            
            for i in 1:n_states
                for j in 1:n_states
                    for l in 1:length(package_A.states)
                    H_ODT[i,j] -= 2π * Γ * (E0^2/4 * d[min(i,l),max(i,l),q] * pol[q] * d[min(j,l),max(j,l),p] * pol[p]) * 
                                    (1/(package_A.states[l].E-package_A.states[i].E - f_ODT) + 1/(package_A.states[l].E-package_A.states[i].E + f_ODT))
                    end
                end
            end
            
        end
    end
    
    return H_ODT
end

function update_ODT_center!(p1, t1)::Nothing
    Γ = p1.Γ
    p = p1.sim_params
    if t1 >= p.ODT_motion_t_start*Γ
        t = min(t1/Γ - p.ODT_motion_t_start,  p.ODT_motion_t_stop - p.ODT_motion_t_start )
        iτ = searchsortedfirst(p.interpolation_times,t)
        τ = p.interpolation_τs[iτ] + (p.interpolation_τs[iτ+1]-p.interpolation_τs[iτ])/(p.interpolation_times[iτ+1]-p.interpolation_times[iτ])*(t-p.interpolation_times[iτ])
        # τ = func_t_to_τ(t)
        p1.extra_data.ODT_position[1] = p.ODT_rmax * τ * cos(2*π*τ* p.ODT_revolutions)
        p1.extra_data.ODT_position[2] = p.ODT_rmax * τ * sin(2*π*τ* p.ODT_revolutions)
    end
    return nothing
end
export update_ODT_center!;

function gaussian_intensity_along_axes(r, axes, centers,k)
    """1/e^2 width = 5mm Gaussian beam """
    d2 = (r[axes[1]] - centers[1])^2 + (r[axes[2]] - centers[2])^2   
    return exp(-2*d2/(5e-3/(1/k))^2)
end






function randomize_initial_vector!(p, r_dist, v_dist)
    k = 2π/p.λ
    Γ = 2π/p.Γ
    n_excited = p.sim_params.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = rand(r_dist)*k
    p.ψ[n_states + n_excited + 2] = rand(r_dist)*k
    p.ψ[n_states + n_excited + 3] = rand(r_dist)*k
    p.ψ[n_states + n_excited + 4] = rand(v_dist)*k/Γ
    p.ψ[n_states + n_excited + 5] = rand(v_dist)*k/Γ
    p.ψ[n_states + n_excited + 6] = rand(v_dist)*k/Γ
end

function init_MOT_distribution(T, diameter,displacement,kick,m)
    kB = 1.381e-23
#     m = @with_unit 57 "u"
    σ = sqrt(kB * T / m)
    
    r = Normal(displacement, diameter)
    v = Normal(kick, σ)
    return r, v
end
export init_MOT_distribution


function randomize_initial_vector!(p, x_dist, y_dist, z_dist, vx_dist, vy_dist, vz_dist)
    k = 2π / p.λ
    Γ = p.Γ
    n_excited = p.sim_params.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = rand(x_dist)*k
    p.ψ[n_states + n_excited + 2] = rand(y_dist)*k
    p.ψ[n_states + n_excited + 3] = rand(z_dist)*k
    p.ψ[n_states + n_excited + 4] = rand(vx_dist)*k/Γ
    p.ψ[n_states + n_excited + 5] = rand(vy_dist)*k/Γ
    p.ψ[n_states + n_excited + 6] = rand(vz_dist)*k/Γ
end
export randomize_initial_vector!

function fixed_initial_vector!(p, x, y, z, vx, vy, vz)
     n_excited = p.sim_params.n_excited
    n_states = length(p.states)
    p.ψ[n_states + n_excited + 1] = x*k
    p.ψ[n_states + n_excited + 2] = y*k
    p.ψ[n_states + n_excited + 3] = z*k
    p.ψ[n_states + n_excited + 4] = vx*k/Γ
    p.ψ[n_states + n_excited + 5] = vy*k/Γ
    p.ψ[n_states + n_excited + 6] = vz*k/Γ
end

function random_initial_state!(p)
    n_excited = p.sim_params.n_excited
    n_states = length(p.states)
   rn = rand() * (n_states - n_excited)
    i = Int(floor(rn))+1
    p.ψ[1:n_states].=0.0
    p.ψ[i] = 1.0
end
export random_initial_state!

            
            
                    

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
    return results, diffusion_constant
end; 
      
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
    return exp(-2*d2/(6e-3/(1/k))^2)
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
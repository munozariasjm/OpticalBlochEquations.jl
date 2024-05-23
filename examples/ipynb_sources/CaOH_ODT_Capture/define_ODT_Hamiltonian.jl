α = sim_params.ODT_revs # number of revoltions
r_max = sim_params.ODT_rmax #m
t_max = sim_params.ODT_motion_t_stop #s

n = 10000
arclengths = zeros(n)
# spiral: r = τ; theta = 2πα * τ
τs = LinRange(0,1,n)
dτ = τs[2]-τs[1]
length_curr = 0.0

for i in 1:n
    local r = r_max/1 * τs[i]
    θ = 2*π * α * τs[i]
    global length_curr += sqrt((r_max/1)^2 +(2*π*α*r)^2) * dτ
    arclengths[i] = length_curr
end

velocity = length_curr / t_max
as = arclengths ./ velocity

ODT_τs = τs
ODT_as = as

func_t_to_τ(t, τs, as) = τs[searchsortedfirst(as, t)]

@inline function update_ODT_center_spiral!(p, extra_data, t1)
    if t1 >= p.ODT_motion_t_start*Γ
        t = min(t1 - p.ODT_motion_t_start*Γ, p.ODT_motion_t_stop*Γ - p.ODT_motion_t_start*Γ)
        τ = func_t_to_τ(t/Γ, extra_data.ODT_τs, extra_data.ODT_as)
        p.ODT_position[1] = p.ODT_rmax * τ * cos(2π*τ*p.ODT_revs)
        p.ODT_position[2] = p.ODT_rmax * τ * sin(2π*τ*p.ODT_revs)
    end
    return nothing
end

function update_ODT_center_circle!(p, extra_data, t1)
    # if t1 >= p.ODT_motion_t_start*Γ
        # t = min(t1 - p.ODT_motion_t_start*Γ, p.ODT_motion_t_stop*Γ - p.ODT_motion_t_start*Γ)
    τ = t1 * (1/Γ)
    p.ODT_position[1] = p.ODT_rmax * cos(2π*τ*p.ODT_revs/p.ODT_motion_t_stop)
    p.ODT_position[2] = p.ODT_rmax * sin(2π*τ*p.ODT_revs/p.ODT_motion_t_stop)
    return nothing
end

function ODT_radius(p, t)
    p.ODT_rad * t / (p.τ_ODT / (1/Γ))
end

function ODT_center(p, t)
    rad = ODT_radius(p, t) / (1/k)
    freq = p.ODT_freq / Γ
    ODT_radius(p, t) * cos(2π * freq * t), ODT_radius(p, t) * sin(2π * freq * t) # trace out a circle
end

"""
    Compute the Hamiltonian for the molecule-ODT interaction.
"""
function get_H_ODT(states, X_states, A_states, peak_intensity, pol, wavelength=1064e-9)
    Is = π*h*c*Γ/(3λ^3) / 10 # saturation intensity in mW/cm^2
    s = peak_intensity / Is
    f_ODT = c/wavelength
    
    n_states = length(states)
    E0 = sqrt(s)/(2 * √2) # factor?
    H_ODT = zeros(ComplexF64, n_states, n_states)
    
    all_states = [X_states; A_states]
    d = tdms_between_states(all_states, all_states)
    fs = energy.(all_states)

    for q in 1:3
        for p in 1:3
            for i in 1:n_states
                for j in 1:n_states
                    for l in 1:length(all_states)
                        H_ODT[i,j] -= 2π * Γ * (E0^2/4 * d[min(i,l),max(i,l),q] * pol[q] * d[min(j,l),max(j,l),p] * pol[p]) * (1/(fs[l]-fs[i] - f_ODT) + 1/(fs[l]-fs[i] + f_ODT))
                    end
                end
            end
        end
    end
    
    return H_ODT
end
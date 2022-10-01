evaluate_field(field::Laser, k, r, ω, t) = (field.ϵ + im * field.ϵ_im) * cis(k * r - ω * t)
export evaluate_field

function update_fields!(fields::StructVector{Field{T}}, r, t) where T
    """
    Fields must be specified as one of the following types:
    """
    for i in eachindex(fields)
        fields.E[i] .= fields.f[i](fields.ω[i], r, t)
    end
    return nothing
end
export update_fields!

# function ρ_and_force!(du, u, p, τ)

#     p.particle.r = p.particle.v .* τ

#     mat_to_vec_minus1!(u, p.ρ)
#     base_to_soa!(p.ρ, p.ρ_soa)
#     #p.ρ_soa .= ρ

#     # Update the Hamiltonian according to the new time τ
#     update_H!(τ, p.particle.r, p.lasers, p.H, p.conj_mat, p.d, p.d_nnz)

#     # Apply a transformation to go to the Heisenberg picture
#     update_eiωt!(p.eiωt, p.ω, τ)
#     Heisenberg!(p.ρ_soa, p.eiωt)

#     # Compute coherent evolution terms
#     # im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.A12, p.B12, p.T1, p.T2, p.HJ, p.tmp1, p.tmp2)
#     im_commutator!(p.dρ_soa, p.H, p.ρ_soa, p.tmp1, p.tmp2, p.HJ)

#     # Add the terms ∑ᵢ Jᵢ ρ Jᵢ†
#     # We assume jumps take the form Jᵢ = sqrt(Γ)|g⟩⟨e| such that JᵢρJᵢ† = Γ^2|g⟩⟨g|ρₑₑ
#     @inbounds for i in eachindex(p.Js)
#         J = p.Js[i]
#         p.dρ_soa.re[J.s′, J.s′] += J.r^2 * p.ρ_soa.re[J.s, J.s]
#     end

#     # The left-hand side also needs to be transformed into the Heisenberg picture
#     # To do this, we require the transpose of the `ω` matrix
#     # Heisenberg!(p.dρ_soa, p.ω_trans, τ)
#     Heisenberg!(p.dρ_soa, p.eiωt, -1)
#     soa_to_base!(p.dρ, p.dρ_soa)

#     mat_to_vec!(p.dρ, du)
#     du[end] = derivative_force(p.ρ, p, τ)
#     # u[end] = force(p.ρ, p, τ)

#     return nothing
# end
# export ρ_and_force!

function derivative_force(p, ρ, τ)

    @unpack ρ_soa, lasers, Γ, d, d_nnz = p

    r = p.particle.v .* τ
    update_lasers!(r, lasers, τ)

    F = SVector(0, 0, 0)

    for q in 1:3

        ampl = SVector(0, 0, 0)
        @inbounds for i in 1:length(lasers)
            s = lasers.s[i]
            k = lasers.k[i]
            ω = lasers.ω[i]
            x = h * Γ * s / (4π * √2)
            ampl += k * x * (im * lasers.f_re_q[i][q] - lasers.f_im_q[i][q])
        end

        d_q = @view d[:,:,q]
        d_nnz_q = d_nnz[q]
        @inbounds for i ∈ d_nnz_q
            F += ampl * d_q[i] * ρ_soa[i] + conj(ampl * d_q[i] * ρ_soa[i])
        end
    end

    return real(F[1])
end
export derivative_force

@with_kw struct Field{T<:Function}
    f::T                                                # function for the field
    ω::Float64                                          # angular frequency of field
    s::Float64                                          # saturation parameter
    ϵ::SVector{3, Float64}                              # polarization vector
    k::SVector{3, Float64} = zeros(Float64, 3)          # k-vector
    E::MVector{3, ComplexF64} = zeros(ComplexF64, 3)    # the actual field components
end
export Field

function calculate_force_from_period_callable(p, sol; times=nothing)
    """
    Integrates the force resulting from `sol` over a time period designated by `period`.
    """
    F = 0.0
    for t ∈ times
        F += force(p, sol(t), t)
    end
    return F / length(times)
end
export calculate_force_from_period_callable

# function average_forces(iterator_info, forces, param_to_average, f_value, f_average)
#     params, iterator = iterator_info
#     typeof_param = typeof(f_value(first(iterator)))
#     forces_dict = Dict{typeof_param, Tuple{Int64, Float64}}()
#     for (i, param) ∈ enumerate(iterator)
#         force = forces[i]
#         param_value = f_value(param)
#         if haskey(forces_dict, param_value)
#             occurrences, current_force = forces_dict[param_value]
#             forces_dict[param_value] = (occurrences + 1, current_force + f_average(force, param))
#         else
#             forces_dict[param_value] = (1, f_average(force, param))
#         end
#     end
#     forces_dict = sort(forces_dict)
#     unique_values = collect(keys(forces_dict))
#     forces_with_occurrences = collect(values(forces_dict))
#     averaged_forces = [x[2] / x[1] for x in forces_with_occurrences]
    
#     sorted_idxs = sortperm(unique_values)
#     return unique_values[sorted_idxs], averaged_forces[sorted_idxs]
# end

function force_scan(params, scan_params, iterate_func; nthreads=Threads.nthreads())

    # Make an iterator from `scan_params`
    iterator_product = iterate_func(values(scan_params)...)
    n_chunks = cld(length(iterator_product), nthreads)
    iterator = Iterators.partition(iterator_product, n_chunks)
    iterated_values = keys(scan_params)
    n_distinct = sum(length.(iterator))

    params_chunks = [deepcopy(params) for _ ∈ 1:nthreads]
    
    prog_bar = Progress(n_distinct)

    tasks = Vector{Task}(undef, nthreads)
    forces = SVector{3, Float64}[] #Vector{SVector{3, Float64}}(undef, n_distinct)

    @sync for (i, scan_params_chunk) ∈ enumerate(iterator)

        tasks[i] = Threads.@spawn begin
            iterated_values = keys(scan_params)

            params_chunk = params_chunks[i]
            forces_chunk = SVector{3, Float64}[]

            t_end = 250
            tspan = (0.0, t_end)
            times = range(t_end - params_chunk.period, t_end, step=params_chunk.period * 2e-3)

            ρ0 = zeros(ComplexF64, (length(params_chunk.states), length(params_chunk.states)))
            ρ0[1,1] = 1.0
            prob = ODEProblem(ρ!, ρ0, tspan, params_chunk)#, callback=AutoAbstol(false, init_curmax=0.0))

            for j ∈ eachindex(scan_params_chunk)
                for (k, iterated_value) ∈ enumerate(iterated_values)
                    setproperty!(params_chunk, iterated_value, scan_params_chunk[j][k])
                end
                round_params(params_chunk) # round params to `freq_res` accuracy just in case they were updated
                sol = solve(prob, alg=DP5(), saveat=times, abstol=1e-5) #abstol=1e-6, reltol=1e-7, dense=false, saveat=times)
                push!(forces_chunk, calculate_force_from_period(params_chunk, sol))
                next!(prog_bar)
            end
            forces_chunk
        end
    end
    forces = vcat(fetch.(tasks)...)
    return (iterated_values, iterator_product), forces

    ### Alternatively, use `@threads` to 
    # Threads.@threads for scan_param ∈ collect(iterator_product)

    #     params_chunk = deepcopy(params)
    #     forces_chunk = SVector{3, Float64}[]

    #     t_end = 500
    #     tspan = (0.0, t_end)
    #     times = range(t_end - params_chunk.period, t_end, 10000)

    #     ρ0 = zeros(ComplexF64, (length(params_chunk.states), length(params_chunk.states)))
    #     ρ0[1,1] = 1.0
    #     prob = ODEProblem(ρ!, ρ0, tspan, params_chunk)#, callback=AutoAbstol(false, init_curmax=0.0))

    #     # params_chunk.B = scan_param[1]

    #     for (k, iterated_value) ∈ enumerate(iterated_values)
    #         setproperty!(params_chunk, iterated_value, scan_param[k])
    #     end
    #     sol = solve(prob, alg=DP5(), saveat=times) #abstol=1e-6, reltol=1e-7, dense=false, saveat=times)
    #     push!(forces_chunk, calculate_force_from_period(params_chunk, sol))
    #     # next!(prog_bar)
    #     # forces_chunk
    #     # forces = [forces; forces_chunk]
    # end
end
export force_scan

"""
    param_scan(evaluate_func, params, scan_params; nthreads)

    Evaluates an `ODEProblem` by looping over both the set of params [`outer_scan_params`; `inner_scan_params`].

    update_func(params, outer_scan_param) --> updates the parameters `p` of the passed `prob`
    evaluate_func() --> 
"""
function param_scan(update_func::F1, evaluate_func::F2, prob::ODEProblem, outer_scan_params, inner_scan_params; nthreads=Threads.nthreads()) where {F1,F2}

    # Make an iterator from `inner_scan_params`, break them into chunks to prepare for threading
    iterator_product = Iterators.product(values(inner_scan_params)...)
    n_chunks = cld(length(iterator_product), nthreads)
    iterator = Iterators.partition(iterator_product, n_chunks)
    iterated_values = keys(inner_scan_params)
    n_distinct = sum(length.(iterator))

    params_chunks = [deepcopy(params) for _ ∈ 1:nthreads]
    
    prog_bar = Progress(n_distinct)

    tasks = Vector{Task}(undef, nthreads)
    return_values = []

    for outer_scan_param ∈ outer_scan_params
        update_func(prob, outer_scan_param)

        @sync for (i, scan_params_chunk) ∈ enumerate(iterator)

            tasks[i] = Threads.@spawn begin
                iterated_values = keys(scan_params)

                params_chunk = params_chunks[i]
                forces_chunk = SVector{3, Float64}[]

                t_end = 300
                tspan = (0.0, t_end)
                times = range(t_end - params_chunk.period, t_end, 1000)

                ρ0 = zeros(ComplexF64, (length(params_chunk.states), length(params_chunk.states)))
                ρ0[1,1] = 1.0
                prob = ODEProblem(ρ!, ρ0, tspan, params_chunk)

                for j ∈ eachindex(scan_params_chunk)
                    for (k, iterated_value) ∈ enumerate(iterated_values)
                        setproperty!(params_chunk, iterated_value, scan_params_chunk[j][k])
                    end
                    round_params(params_chunk) # round params to `freq_res` accuracy just in case they were updated
                    sol = solve(prob, alg=DP5(), abstol=1e-5)
                    push!(forces_chunk, evaluate_func(params_chunk, sol))
                    next!(prog_bar)
                end
                forces_chunk
            end
        end
    end
    forces = vcat(fetch.(tasks)...)
    return (iterated_values, iterator_product), forces

end
export force_scan

# Define callbacks to calculate force and terminate integration once it has converged
# condition(u, t, integrator) = false #integrator.p.force
# affect!(integrator) = terminate!(integrator)
# cb = DiscreteCallback(condition, affect!)

function force_callback!(integrator)
    # if integrator.t > 10integrator.p.period
    p = integrator.p
    force = force_noupdate(p) / p.n_force_values
    modded_idx = mod1(p.force_idx, p.n_force_values)
    p.forces[modded_idx] += force
    for i ∈ 1:10
        modded_idx1 = mod1(p.force_idx - i * 100, p.n_force_values)
        modded_idx2 = mod1(modded_idx1 + 1, p.n_force_values)
        p.force_chunks[i] += p.forces[modded_idx1]
        p.force_chunks[i] -= p.forces[modded_idx2]
    end
    p.force_idx += 1
    # end
    return nothing
end
;
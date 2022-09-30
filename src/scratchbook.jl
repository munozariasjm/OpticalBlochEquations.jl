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
import DifferentialEquations: ODEProblem, solve
import ProgressMeter: Progress
import Parameters: @with_kw

function force(p, ρ, τ)
    F = SVector(0.0, 0.0, 0.0)

    @unpack fields, Γ, d, d_nnz = p
    r = p.r0 + p.v .* τ
    update_fields!(fields, r, τ)
    base_to_soa!(ρ, p.ρ_soa)
    update_eiωt!(p.eiωt, p.ω, τ)
    Heisenberg!(p.ρ_soa, p.eiωt)

    for i ∈ eachindex(fields)

        s = fields.s[i]
        x = sqrt(s) / (2 * √2)
        k = fields.k[i]

        @inbounds for q ∈ 1:3
            # With SI units, should be x = -h * Γ * sqrt(s) / (2 * √2 * p.λ)
            ampl = k .* (x * im * fields.E[i][q])
            # Note h * Γ / λ = 2π ħ * Γ / Λ = ħ * k * Γ, so this has units units of ħ k Γ
            d_q = @view d[:,:,q]
            d_nnz_q = d_nnz[q]
            @inbounds for j ∈ d_nnz_q
                F -= ampl * d_q[j] * conj(p.ρ_soa[j]) #+ conj(ampl * d_q[j] * ρ[j])
            end
        end
    end
    F += conj(F)
    return real.(F)
end
export force

function calculate_force_from_period(p, sol; force_idxs=nothing)
    """
    Integrates the force resulting from `sol` over a time period designated by `period`.
    """
    F = SVector(0.0, 0.0, 0.0)
    if isnothing(force_idxs)
        for i ∈ eachindex(sol.t)
            F += force(p, sol.u[i], sol.t[i])
        end
        return F ./ length(sol.t)
    else
        for i ∈ eachindex(force_idxs)
            F += force(p, sol.u[i], sol.t[force_idxs[i]])
        end
        return F ./ length(force_idxs)
    end
end
export calculate_force_from_period

function find_idx_for_time(time_to_find, times, backwards)
    """
    Search backwards in the array.
    """
    if backwards
        times = reverse(times)
    end
    start_time = times[1]
    found_idx = 0
    for (i, time) in enumerate(times)
        if abs(start_time - time) > time_to_find
            found_idx = i
            break
        end
    end
    if backwards
        found_idx = length(times) + 1 - found_idx
    end
    
    return found_idx
end
export find_idx_for_time

function force_scan(prob, scan_values::T, prob_func!::F1, param_func::F2, output_func::F3; n_threads=Threads.nthreads()) where {T,F1,F2,F3}

    n_values = length(first(scan_values))
    batch_size = fld(n_values, n_threads)
    remainder = n_values - batch_size * n_threads
    params = zeros(Float64, n_values)
    forces = zeros(Float64, n_values)

    prog_bar = Progress(n_values)

    @sync for i ∈ 1:n_threads
        _prob = deepcopy(prob)
        Threads.@spawn begin
            prob_func!(_prob.p, scan_values, i)
            _batch_size = i <= remainder ? (batch_size + 1) : batch_size - 1
            batch_start_idx = 1 + ((i <= remainder) ? i : remainder) + batch_size * (i-1)
            for j ∈ batch_start_idx:(batch_start_idx + _batch_size)
                prob_func!(_prob.p, scan_values, j)
                sol = solve(_prob, alg=DP5(), abstol=1e-4)
                params[j] = param_func(_prob.p, scan_values, j)
                forces[j] = output_func(_prob.p, sol)
                next!(prog_bar)
            end
        end
    end
    return params, forces
end
export force_scan

idx_finder = x->findall.(.==(unique(x)), Ref(x) )
function average_forces(params, forces)
    unique_params = unique(params)
    params_idxs = idx_finder(params)
    average_forces = zeros(length(unique_params))
    for (i, (param, idxs)) ∈ enumerate(zip(unique_params, params_idxs))
        param_forces = forces[idxs]
        average_forces[i] = mean(param_forces)
    end
    return unique_params, average_forces
end
export average_forces
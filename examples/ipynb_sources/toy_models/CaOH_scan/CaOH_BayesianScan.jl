using Revise

using
    QuantumStates,
    OpticalBlochEquations,
    DifferentialEquations,
    UnitsToValue,
    LinearAlgebra,
    Printf,
    Plots,
    Random,
    StatsBase
;

using Distributions

import MutableNamedTuples: MutableNamedTuple
import StructArrays: StructArray, StructVector
import StaticArrays: @SVector, SVector
import LinearAlgebra: norm, ⋅, adjoint!, diag
import LoopVectorization: @turbo
using BenchmarkTools
using Parameters
using LsqFit

import ProgressMeter: Progress, next!

const λ = 626e-9
const Γ = 2π* 6.4e6
const m = @with_unit 57 "u"
const k = 2π / λ
_μB = (μ_B / h) * 1e-4;

include("CaOH_scan_helper_repump_Christian.jl")
working_dir = "C:\\Google Drive\\github\\OpticalBlochEquations\\examples\\ipynb_sources\\toy_models\\CaOH_scan"

function survived(t_end, times, trajectories)
    _survived = Int64[]
    for i ∈ eachindex(trajectories)
        if times[i][end] ≈ t_end*1e3
            push!(_survived, i)
        end
    end
    return _survived
end

function cloud_size_std(trajectories, i)
    std(trajectory[i] for trajectory ∈ trajectories if length(trajectory) >= i)
end

"""
    Evaluates the density given a set of trajectories.
"""
function density(t_end, results)
    
    times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
    
    # Sample number of photons before the molecule is lost to vibrational dark state
    n_molecules = length(times)
    
    _survived = survived(t_end, times, x_trajectories)
    println(_survived)
    
    surviving_x_trajectories = x_trajectories[_survived]
    surviving_y_trajectories = y_trajectories[_survived]
    surviving_z_trajectories = z_trajectories[_survived]
    
    n = length(_survived)
    _density = Float64(n)
    if n > 1
        idx_end = length(surviving_x_trajectories[1])
        σ_x = cloud_size(surviving_x_trajectories, idx_end)
        σ_y = cloud_size(surviving_y_trajectories, idx_end)
        σ_z = cloud_size(surviving_z_trajectories, idx_end)
        _density /= σ_x * σ_y * σ_z
    end
    return _density
end

function distance(x_trajectory, y_trajectory, z_trajectory, idx)
    return sqrt(x_trajectory[idx]^2 + y_trajectory[idx]^2 + z_trajectory[idx]^2)
end

function exponential(x, p)
    σ, A = p
   return A * exp.(-x.^2/(2*σ^2))
end

function exp_fit(results, _survived)
    
    xs = [distance(results.x_trajectories[survived_idx], results.y_trajectories[survived_idx], results.z_trajectories[survived_idx], idx_end) for survived_idx ∈ _survived]

    hist_data = fit(Histogram, xs, 0:0.1:5)
    hist_data.isdensity = true
    v = collect(hist_data.edges[1])
    dv = v[2]-v[1]
    v = v[1:end-1] .+ dv/2
    fv = hist_data.weights ./ (sum(hist_data.weights) * dv)
    
    v_fit = curve_fit(exponential, v, fv, [0.3, 3.])
    σ, A = v_fit.param

    return σ, σ_error
end

goodness_val(r) = r > 0.1 ? 1/r^3 : 1/0.1^3
"""
    Evaluates how many particles are within a 0.3 mm radius.
"""
function goodness(t_end, results)
    
    times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
    
    # Sample number of photons before the molecule is lost to vibrational dark state
    n_molecules = length(times)
    
    _survived = survived(t_end, times, x_trajectories)
    @printf("Survival: %i / %i", length(_survived), n_molecules)
    println()

    n = 0.0
    # for i ∈ eachindex(x_trajectories)
    #     r = sqrt(x_trajectories[i][end]^2 + y_trajectories[i][end]^2 + z_trajectories[i][end]^2)
    #     if i ∈ _survived
    #         n += goodness_val(r)
    #     end
    # end

    σ, σ_error = exp_fit(results, _survived)
    if σ_error < σ
        n = length(_survived) * 1/σ^3
    end

    # try
    #     σ, σ_error = exp_fit(results, _survived)
    #     if σ_error < σ
    #         n = length(_survived) * 1/σ^3
    #     end
    # catch
    #     n = 0.0
    # end

    return n/n_molecules
end

using LsqFit

function Gaussian(x, p)
    σ, x0, A = p
   return A * exp.(-(x.-x0).^2/(2*σ^2)) 
end

function cloud_size(i, trajectories)
    
    xs = [trajectory[i] for trajectory ∈ trajectories]

    hist_data = fit(Histogram, xs, nbins=5)
    hist_data.isdensity = true
    v = collect(hist_data.edges[1])
    dv = v[2]-v[1]
    v = v[1:end-1] .+ dv/2
    fv = hist_data.weights ./ (sum(hist_data.weights) * dv)

    v_fit = curve_fit(Gaussian, v, fv, [0.5, 0., 10])
    σ, x0, A = v_fit.param

    return σ
end

package = get_CaOH_package()
n_states = length(package.states)
n_excited = package.n_excited

"""
    Function to optimize (density).
"""
function f(x)
    s1 = x[1]
    s3 = x[2]
    s4 = x[3]
    Δ1 = x[4]*Γ
    Δ3 = Δ1
    Δ4 = x[5]*Γ
    # ramp_time = 10e-3
    # ramp_time = 20e-3
    B_gradient = -x[6]
    repump_rate = x[7]
    ramp_time = x[8] * 1e-3

    temp = @with_unit 0.050 "mK"
    diameter = @with_unit 2.0 "mm"
    displacement = [0.0, 0.0, 0.0]
    kick = [0,0, 0.0, 0.0]
    
    params = MutableNamedTuple(t_end = 40e-3, n_values = 20,
                           pol1_x=σ⁻, pol2_x=σ⁻, pol3_x=σ⁺, pol4_x=σ⁻, 
                           s1=s1, s2=0., s3=s3, s4=s4,
                           Δ1=Δ1, Δ2=0., Δ3=Δ3, Δ4=Δ4,
                           B_gradient = B_gradient,

                           temp=temp, diameter=diameter, 
                           displacement=displacement, kick=kick,
                           ramp_time = ramp_time,
                           photon_budget=15000, dark_lifetime=1/(repump_rate * 0.1e6), FC_mainline=0.95,

                           pol_imbalance=0.02,
                           s_imbalance = (0.1, 0.1, 0.1), retro_loss=0.0,

                           off_center=[2,2,2,2,2,2].*1e-3,
                           pointing_error =[0., 0., 0, 0, 0, 0]
    )
    
    results = simulate_particles_repump(package, params)
    
    @printf("s1 = %.2f; s3 = %.2f; s4 = %.2f; Δ1 = %.2f Γ; Δ3 = Δ1; Δ4 = %.2f Γ; B = %.2f; repump_rate = %.2f; ramp_time = %.2f", x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8])
    println()
    
    _goodness = goodness(params.t_end, results)
    @printf("goodness = %.3f", _goodness)
    println()
    
    return _goodness
end

using BayesianOptimization, GaussianProcesses

model = ElasticGPE(8,                            # 6 input dimensions
                   mean = MeanConst(1.),         
                   kernel = SEArd([1., 1, 1, 1, 1, 1, 1, 1], 5.),
                   logNoise = 0.,
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples
set_priors!(model.mean, [Normal(0., 0.01)])

# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every x steps
modeloptimizer = MAPGPOptimizer(
    every = 1, 
    noisebounds = [-3, 3], # bounds of the logNoise
    kernbounds = [[-3, -3, -3, -3, -3, -3, -3, -3, -5], [3, 3, 3, 3, 3, 3, 3, 3, 5]],  # bounds of the parameters GaussianProcesses.get_param_names(model.kernel)
    maxeval = 1000
)

opt = BOpt(f,
           model,
           UpperConfidenceBound(),             # type of acquisition
           modeloptimizer,
           initializer_iterations=300,
           [0.0, 0.0, 0.0, +0.0, -10.0, 0, 0, 0],        # lowerbounds
           [7.0, 7.0, 10., +7.0, -0.0, 50, 3, 40],        # upperbounds         
           repetitions = 1,                          # evaluate the function for each input x times
           maxiterations=300,                       # evaluate at x input positions
           sense = Max,                              # maximize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 10,       # run the NLopt method from x random initial conditions each time
                                 maxtime = 5.0,      # run the NLopt method for at most 1.0 second each time
                                 maxeval = 5000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
           verbosity = BayesianOptimization.Progress)

result = boptimize!(opt)
    
while true
    maxiterations!(opt, 50)
    result = boptimize!(opt)
    serialize("optimized_3freqs_with_imperfections_fullscan_density.jl", opt)
    println("===== Autosaved =====")
end

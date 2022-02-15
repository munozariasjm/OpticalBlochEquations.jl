using Revise

using OpticalBlochEquations

using Plots
using SparseArrays
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using PhysicalConstants.CODATA2018
import DifferentialEquations: ODEProblem, solve, DP5

const h     = PlanckConstant.val
const ħ     = h / 2π
const μB    = BohrMagneton.val
const c     = SpeedOfLightInVacuum.val
const ϵ0    = VacuumElectricPermittivity.val

# Define constants for the problem
ΩP0 = @with_unit 2π * 10 "MHz"
tP  = @with_unit 0.0 "μs"
τP  = @with_unit 30 "μs"
ω0P = @with_unit -10.5 "μs"
ΩS0 = @with_unit 2π * 10 "MHz" # maximum Rabi frequency
tS  = @with_unit 0.0 "μs"  # Stokes pulse offset time
τS  = @with_unit 30 "μs"  # Stokes pulse width
ω0S = @with_unit 0.0 "μs"    # Stokes initial frequency
α   = @with_unit 40 "THz"   # pump sweep rate
β   = @with_unit 40 "THz"   # Stokes sweep rate
Δ0  = @with_unit 1.5 "GHz"

# Generate states
m1 = Manifold(F=0, ω=0, μ=0)
m2 = Manifold(F=0, ω=0, μ=0)
m3 = Manifold(F=0, ω=0, μ=0)
states = [m1.states..., m2.states..., m3.states...]

@inline function Ω(t, Ω0, offset, τ_width)
    return Ω0 * exp(-(t - offset)^2 / (2τ_width^2))
end

@inline function ωt(t, start, rate, offset)
    return start + rate * (t-offset)
end

using Parameters
function f(H, t, p)
    @unpack ΩS0, tS, τS, ΩP0, tP, τP, α, β, ω0S, ω0P = p
    H.re[1,2] = H.re[2,1] = Ω(t, ΩP0, tP, τP)
    H.re[2,3] = H.re[3,2] = Ω(t, ΩS0, tS, τS)
    H.re[2,2] = ωt(t, 0, α, ω0S)
    H.re[3,3] = ωt(t, 0, β, ω0P)
end

p = @params ΩS0, ΩP0, tS, tP, τS, τP, α, β, ω0S, ω0P
d = zeros(3, 3, 3)
(dψ, ψ, p_) = schrödinger(states, lasers, d, f, p)
@btime ψ!(dψ, ψ, p_, 1.0)

tmax = 15.0e-6
tspan = (-tmax, tmax); tstops = -tmax:tmax/4000:tmax
prob = ODEProblem(ψ!, ψ, tspan, p_)
@time sol = solve(prob, alg=DP5(), saveat=tstops, abstol=1.0e-8, reltol=1.0e-6)

ρ₁₁ = [norm(x[1])^2 for x in sol.u]
ρ₂₂ = [norm(x[2])^2 for x in sol.u]
ρ₃₃ = [norm(x[3])^2 for x in sol.u]
plot(sol.t, ρ₁₁)
plot!(sol.t, ρ₂₂)
plot!(sol.t, ρ₃₃, 0.0:0.2:1.0)

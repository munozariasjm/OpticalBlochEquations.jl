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

const λ     = @with_unit 626 "nm"
const Γ     = @with_unit 2π * 8.3 "MHz"
const M     = @with_unit 50 "u"
const E₀    = c / λ

# Generate states
m1 = Manifold(F=0, ω=0, μ=0)
m2 = Manifold(F=0, ω=E₀/Γ, μ=0, Γ=Γ)
states = [m1.states..., m2.states...]

δ = -4Γ

# Generate lasers
x̂ = [1, 0, 0]
ŷ = [0, 1, 0]
ẑ = [0, 0, 1]
l1 = define_laser(x̂, ẑ, (E₀ + δ)/Γ, 20)
# l2 = define_laser(-x̂, ŷ, (E₀ + δ)/Γ, 20)

lasers = [l1]

d = zeros(2, 2, 3)
d[2,1,2] = d[1,2,2] = 1

(dρ, ρ₀, p) = obe(states, lasers, d)
ρ!(dρ, ρ₀, p, 1.0)

@btime ρ!(dρ, ρ₀, p, 1.0)

tspan = (0., 12.56); tstops = 0.0:0.02512:12.56
prob = ODEProblem(ρ!, ρ₀, tspan, p)
@time sol = solve(prob, alg=DP5(), save_everystep=true, tstops=tstops)

ρ₁₁ = [real(x[1,1]) for x in sol.u]
ρ₂₂ = [real(x[2,2]) for x in sol.u]
plot(sol.t / 2π, ρ₁₁)
plot!(sol.t / 2π, ρ₂₂, yticks=0.0:0.2:1.0)

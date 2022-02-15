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
m1 = manifold(F=2, ω=0, μ=0)
m2 = manifold(F=3, ω=E₀/Γ, μ=0, Γ=Γ)
states = [m1.states..., m2.states...]

δ = -2.73Γ

# Generate lasers
l1 = define_laser([1,0,0], [0,0,1], (E₀ + δ)/Γ, 4*1.352464)

lasers = [l1]

d = zeros(12, 12, 3)
d[1, 8, 1] = 0.25819889
d[2, 9, 1] = 0.4472136
d[3, 10, 1] = 0.63245553
d[4, 11, 1] = 0.81649658
d[5, 12, 1] = 1

d[1, 7, 2] = -0.57735027
d[2, 8, 2] = -0.73029674
d[3, 9, 2] = -0.77459667
d[4, 10, 2] = -0.73029674
d[5, 11, 2] = -0.57735027

d[1, 6, 3] = 1
d[2, 7, 3] = 0.81649658
d[3, 8, 3] = 0.63245553
d[4, 9, 3] = 0.4472136
d[5, 10, 3]= 0.25819889

for q in 1:3, i in 1:5, j in 6:12
    d[j,i,q] = d[i,j,q]
end

(dρ, ρ, p) = obe(states, lasers, d)
@btime ρ!(dρ, ρ, p, 1.0)

tspan = (0., 600); tstops = 0:1:600
prob = ODEProblem(ρ!, ρ, tspan, p)
@time sol = solve(prob, alg=DP5(), save_everystep=true)

ρ₁₁ = [real(x[1,1]) for x in sol.u]
ρ₂₂ = [real(x[2,2]) for x in sol.u]
ρ₃₃ = [real(x[3,3]) for x in sol.u]
ρ₄₄ = [real(x[4,4]) for x in sol.u]
ρ₅₅ = [real(x[5,5]) for x in sol.u]
plot(sol.t, ρ₁₁)
plot!(sol.t, ρ₂₂)
plot!(sol.t, ρ₃₃)
plot!(sol.t, ρ₄₄)
plot!(sol.t, ρ₅₅, yticks=0.0:0.2:1.0)

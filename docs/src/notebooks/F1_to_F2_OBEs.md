```@meta
EditURL = "<unknown>/F1_to_F2_OBEs.ipynb"
```

````@example F1_to_F2_OBEs
{
  "cells": [
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "using Revise"
      ],
      "id": "10a3a475",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "using OpticalBlochEquations"
      ],
      "id": "17425b0f",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "using Plots\n",
        "using BenchmarkTools\n",
        "using DifferentialEquations\n",
        "using LinearAlgebra"
      ],
      "id": "a0d26321",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "const λ = @with_unit 606 \"nm\"\n",
        "const Γ = @with_unit 2π * 8 \"MHz\"\n",
        "const M = @with_unit 50 \"u\"\n",
        "const E₀ = c / λ\n",
        "const k = 2π / λ\n",
        ";"
      ],
      "id": "42099b46",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Wigner D-matrix to rotate polarization vector\n",
        "D(cosβ, sinβ, α, γ) = [\n",
        "    (1/2)*(1 + cosβ)*exp(-im*(α + γ)) -(1/√2)*sinβ*exp(-im*α) (1/2)*(1 - cosβ)*exp(-im*(α - γ));\n",
        "    (1/√2)*sinβ*exp(-im*γ) cosβ -(1/√2)*sinβ*exp(im*γ);\n",
        "    (1/2)*(1 - cosβ)*exp(im*(α - γ)) (1/√2)*sinβ*exp(im*α) (1/2)*(1 + cosβ)*exp(im*(α + γ))\n",
        "];\n",
        "\n",
        "function rotate_pol(pol, k)\n",
        "    # Rotates polarization `pol` onto the quantization axis `k`\n",
        "    cosβ = k[3]\n",
        "    sinβ = sqrt(1 - cosβ^2)\n",
        "    α = 0.0\n",
        "    if abs(cosβ) < 1\n",
        "        γ = atan(k[2], k[1])\n",
        "    else\n",
        "        γ = 0.0\n",
        "    end\n",
        "    return inv(D(cosβ, sinβ, α, γ)) * pol\n",
        "end;"
      ],
      "id": "80329749",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "function find_idx_for_time(time_to_find, times, backwards)\n",
        "    if backwards\n",
        "        times = reverse(times)\n",
        "    end\n",
        "    start_time = times[1]\n",
        "    found_idx = 0\n",
        "    for (i, time) in enumerate(times)\n",
        "        if abs(start_time - time) > time_to_find\n",
        "            found_idx = i\n",
        "            break\n",
        "        end\n",
        "    end\n",
        "    if backwards\n",
        "        found_idx = length(times) + 1 - found_idx\n",
        "    end\n",
        "    \n",
        "    return found_idx\n",
        "end;"
      ],
      "id": "9196cb13",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "function calculate_force_from_period(p, sol, force_idxs)\n",
        "    \"\"\"\n",
        "    Integrates the force resulting from `sol` over a time period designated by `period`.\n",
        "    \"\"\"\n",
        "    force = 0.0\n",
        "    for i in force_idxs\n",
        "        force += OpticalBlochEquations.force(p, sol.u[i], sol.t[i])\n",
        "    end\n",
        "    t = sol.t[force_idxs[end]] - sol.t[force_idxs[1]]\n",
        "    return force / (length(force_idxs))\n",
        "end;"
      ],
      "id": "7e6026eb",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Generate states\n",
        "m1 = manifold(F=1, ω=0, μ=0)\n",
        "m2 = manifold(F=2, ω=E₀, μ=0, Γ=Γ)\n",
        "states = [m1.states..., m2.states...]\n",
        "\n",
        "δ = -2.5Γ\n",
        "s = 1.0\n",
        "\n",
        "# Generate lasers\n",
        "x = [1., 0, 0]\n",
        "y = [0., 1, 0]\n",
        "z = [0., 0, 1]\n",
        "\n",
        "σ_m = [1., 0., 0.]\n",
        "σ_p = [0., 0., 1.]\n",
        "\n",
        "l1 = Laser(-x, rotate_pol(σ_m, -x), E₀ + δ, s)\n",
        "l2 = Laser( x, rotate_pol(σ_p,  x), E₀ + δ, s)\n",
        "\n",
        "l3 = Laser(-y, rotate_pol(σ_m, -y), E₀ + δ, s)\n",
        "l4 = Laser( y, rotate_pol(σ_p,  y), E₀ + δ, s)\n",
        "\n",
        "l5 = Laser(-z, rotate_pol(σ_m, -z), E₀ + δ, s)\n",
        "l6 = Laser( z, rotate_pol(σ_p,  z), E₀ + δ, s)\n",
        "\n",
        "lasers = [l1, l2, l3, l4, l5, l6]\n",
        "\n",
        "d = zeros(8, 8, 3)\n",
        "d_1 = [\n",
        "    0.0 0.0 0.40824829 0.0 0.0;\n",
        "    0.0 0.0 0.0 0.70710678 0.0;\n",
        "    0.0 0.0 0.0 0.0 1.0\n",
        "]\n",
        "d_2 = [\n",
        "    0.0 -0.70710678 0.0 0.0 0.0;\n",
        "    0.0 0.0 -0.81649658 0.0 0.0;\n",
        "    0.0 0.0 0.0 -0.70710678 0.0\n",
        "]\n",
        "d_3 = [\n",
        "    1.0 0.0 0.0 0.0 0.0;\n",
        "    0.0 0.70710678 0.0 0.0 0.0;\n",
        "    0.0 0.0 0.40824829 0.0 0.0\n",
        "]\n",
        "\n",
        "d[1:3,4:8,1] = d_1\n",
        "d[4:8,1:3,1] = d_1'\n",
        "d[1:3,4:8,2] = d_2\n",
        "d[4:8,1:3,2] = d_2'\n",
        "d[1:3,4:8,3] = d_3\n",
        "d[4:8,1:3,3] = d_3'\n",
        ";"
      ],
      "id": "2040bf21",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "using StaticArrays\n",
        "\n",
        "ρ0 = zeros(ComplexF64, length(states), length(states))\n",
        "ρ0[1,1] = 1.0\n",
        "\n",
        "# Frequencies are rounded to a multiple of `freq_res`, and are measured in units of Γ\n",
        "freq_res = 1e-3\n",
        "\n",
        "(dρ, ρ, p) = obe(states, lasers, d, ρ0, freq_res=freq_res)\n",
        "ρ!(dρ, ρ, p, 0.0)\n",
        "\n",
        "ω_min = freq_res\n",
        "period = 2π / ω_min\n",
        "display(period)"
      ],
      "id": "25599d32",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "@btime ρ!($dρ, $ρ, $p, $1.0)"
      ],
      "id": "7e0ff6b6",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "t_end = 2π * 2000\n",
        "tspan = (0., t_end)\n",
        "times = range(tspan[1], tspan[2], 200000)\n",
        "\n",
        "p.particle.r0 = SVector(0, 0, 0) / (1 / k)\n",
        "p.particle.v = SVector(0.0, 0.0, 0.1)\n",
        "p.particle.v = round_vel(p.particle.v, λ, Γ, freq_res) / (Γ / k)\n",
        "println(p.particle.v)\n",
        "\n",
        "prob = ODEProblem(ρ!, ρ0, tspan, p, callback=AutoAbstol(false, init_curmax=0.0))\n",
        "\n",
        "# @time sol = solve(prob, alg=DP5(), abstol=1e-6, reltol=1e-3, dense=false, saveat=times)\n",
        "@time sol = solve(prob, alg=DP5(), abstol=1e-6, reltol=1e-7, dense=false, saveat=times)\n",
        ";"
      ],
      "id": "dc71a19e",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "plot(sol.t[1:100:end], [real(u[1,1]) for u in sol.u[1:100:end]], size=(1200, 400))\n",
        "plot!(sol.t[1:100:end], [real(u[2,2]) for u in sol.u[1:100:end]], legend=false)"
      ],
      "id": "757224eb",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "offset = 100\n",
        "period_idx = find_idx_for_time(period, sol.t, true)\n",
        "force_idxs = (period_idx - offset):(length(times) - offset)\n",
        "force = calculate_force_from_period(p, sol, force_idxs)\n",
        "\n",
        "println(\"Excited population: \", real(sum(diag(mean(sol.u[force_idxs]))[4:end])))\n",
        "print(\"Force: \", (Γ / (2π / λ)) * (ħ * k * Γ * force / M))"
      ],
      "id": "c27c91f3",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "vs = collect(0.0:0.50:5.00)\n",
        "forces = zeros(length(vs))\n",
        "exc_pop = zeros(length(vs))\n",
        "\n",
        "t_end = 2π * 2000\n",
        "tspan = (0., t_end)\n",
        "times = range(tspan[1], tspan[2], 200000)\n",
        "\n",
        "@time begin\n",
        "    Threads.@threads for i in 1:length(vs)\n",
        "        \n",
        "        v = vs[i]\n",
        "        \n",
        "        previous_force = 0.0\n",
        "        force = 1.0\n",
        "        error = 1.0\n",
        "        \n",
        "        p_ = deepcopy(p)\n",
        "        \n",
        "        p_.particle.r0 = SVector(0.0, 0.0, 0.0) / (1 / k)\n",
        "        p_.particle.v = SVector(0.0, 2.0, v)\n",
        "        p_.particle.v = round_vel(p_.particle.v, λ, Γ, freq_res) / (Γ / k)\n",
        "        \n",
        "        vs[i] = sqrt(sum(p_.particle.v.^2))\n",
        "        \n",
        "        ρ0 = zeros(ComplexF64,(8, 8))\n",
        "        ρ0[1,1] = 1.0\n",
        "\n",
        "        prob = ODEProblem(ρ!, ρ0, tspan, p_)\n",
        "        sol = solve(prob, alg=DP5(), abstol=1e-6, reltol=1e-7, saveat=times)\n",
        "\n",
        "        offset = 0\n",
        "        period_idx = find_idx_for_time(period, sol.t, true)\n",
        "        force_idxs = (period_idx - offset):(length(times) - offset)\n",
        "        \n",
        "        previous_force = force\n",
        "        force = calculate_force_from_period(p_, sol, force_idxs)\n",
        "\n",
        "        forces[i] = force\n",
        "        exc_pop[i] = real(sum(diag(mean(sol.u[force_idxs]))[13:end]))\n",
        "            \n",
        "    end\n",
        "end\n",
        "forces_ħkΓ = (Γ / k) * (forces) * 1e3\n",
        ";"
      ],
      "id": "16e16fa1",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "plot(vs, real.(forces), size=(600,400))"
      ],
      "id": "08013182",
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Julia 1.7.1",
      "language": "julia",
      "name": "julia-1.7"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


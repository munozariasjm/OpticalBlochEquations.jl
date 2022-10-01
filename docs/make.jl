using Documenter
using OpticalBlochEquations

makedocs(
    sitename = "OpticalBlochEquations",
    format = Documenter.HTML(),
    modules = [OpticalBlochEquations]
)

deploydocs(
    repo = "github.com/christian-hh/OpticalBlochEquations.jl.git"
)

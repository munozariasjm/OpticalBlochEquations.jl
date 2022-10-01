using Documenter
using OpticalBlochEquations

makedocs(
    sitename = "OpticalBlochEquations",
    format = Documenter.HTML(),
    modules = [OpticalBlochEquations]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

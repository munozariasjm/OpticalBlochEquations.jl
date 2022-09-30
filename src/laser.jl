@with_kw struct Laser
    k::SVector{3, Float64}         # k-vector
    ϵ_re::SVector{3, Float64}      # real part of polarization
    ϵ_im::SVector{3, Float64}      # imaginary part of polarization
    ϵ_cart::SVector{3, ComplexF64} # polarization in cartesian coordinates 
    ω::Float64                     # frequency
    s::Float64                     # saturation parameter
    kr::Float64                    # value of k ⋅ r, defaults to 0
    E::MVector{3, ComplexF64}      # f = exp(i(kr - ωt))
    re::Float64
    im::Float64
    Laser(k, ϵ, ω, s) = new(k, real.(ϵ), imag.(ϵ), sph2cart * ϵ, ω, s, 0.0, MVector(0.0, 0.0, 0.0), 1.0, 0.0)
end
export Laser
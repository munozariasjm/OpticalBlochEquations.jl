function make_couplings(H)
    ds_state1 = Int64[]
    ds_state2 = Int64[]
    ds = Float64[]
    for i ∈ axes(H, 1)
        for j ∈ i:size(H, 2)
            if norm(H[i,j]) > 1e-10
                push!(ds_state1, j)
                push!(ds_state2, i)
                push!(ds, H[i,j])
            end
        end
    end
    return (ds_state1, ds_state2, ds)
end

function make_parameters_fast(
    particle, states, fields, d, ψ₀, mass, n_excited;
    sim_params=nothing, extra_data=nothing, λ=1.0, Γ=2π, H_extra=nothing, H_func=nothing, ω_offset=0.0)

    ##
    ds_state1_ODT, ds_state2_ODT, ds_ODT = make_couplings(H_extra)
    ##

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    type_to_use = Float64
    
    ψ = rand(Complex{type_to_use}, n_states)
    dψ = rand(Complex{type_to_use}, n_states)
    
    ψ_mutable = MVector{16, Complex{type_to_use}}(ψ)
    dψ_mutable = MVector{16, Complex{type_to_use}}(dψ)
    
    # Static arrays
    ks = rand(type_to_use, n_fields, 3)
    for i ∈ 1:n_fields
        ks[i,:] .= fields.k[i]
    end
    ks_static = SMatrix{n_fields,3}(ks)

    ωs = zeros(type_to_use, n_fields)
    for i ∈ 1:n_fields
        ωs[i] = fields.ω[i]
    end
    ωs_static = SVector{n_fields, type_to_use}(ωs)
    
    ϵs = rand(Complex{type_to_use}, n_fields, 3)
    for i ∈ 1:n_fields
        fields.ϵ_val[i] = fields.ϵ[i](0.0) # initialize polarization value
        ϵs[i,:] .= fields.ϵ_val[i]
    end
    ϵs_static = SMatrix{n_fields,3}(ϵs)

    for i ∈ eachindex(fields)
        fields.s[i] = fields.s_func[i](particle.r, 0.0)
    end
    sat_params = fields.s
    sat_params_static = SVector{n_fields, type_to_use}(sat_params)

    # Mutable arrays
    krs = zeros(type_to_use, n_fields)
    krs_mutable = MVector{n_fields, type_to_use}(krs)
    
    ims = rand(type_to_use, n_fields)
    res = rand(type_to_use, n_fields)
    ims_mutable = MVector{n_fields, type_to_use}(ims)
    res_mutable = MVector{n_fields, type_to_use}(res)
    
    exp_vals = rand(type_to_use, n_fields)
    exp_vals_mutable = MVector{n_fields, type_to_use}(exp_vals)
    exp_vals_static = SVector{n_fields, type_to_use}(exp_vals)
    
    Es = rand(Complex{type_to_use}, n_fields, 3)
    Es_mutable = MMatrix{n_fields,3}(Es)
    
    # define transition dipole moments
    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]

    Js_summed = zeros(Float64, length(n_excited))
    for i ∈ eachindex(Js_summed)
        for J ∈ Js
            if J.s == (n_states - n_excited + i)
                Js_summed[i] += real(J.r)
            end
        end
    end

    # 
    ds1 = SVector{10, type_to_use}(ds[1])
    ds2 = SVector{10, type_to_use}(ds[2])
    ds3 = SVector{10, type_to_use}(ds[3])
    
    ds = [ds1; ds2; ds3]
    dEs = MVector{length(ds), Complex{type_to_use}}(zeros(Complex{type_to_use}, length(ds)))
    dEs_soa=StructArray(dEs)
    
    ds_added=SVector{length(ds_ODT), Complex{type_to_use}}(ds_ODT)
    ds_state1_added=SVector{length(ds_state1_ODT), Int64}(ds_state1_ODT)
    ds_state2_added=SVector{length(ds_state2_ODT), Int64}(ds_state2_ODT)
    dEs_soa_added=StructArray(ds_added)
    
    H = deepcopy(dEs)

    ds_state1 = SVector{length(ds), Int64}([ds_state1[1]; ds_state1[2]; ds_state1[3]])
    ds_state2 = SVector{length(ds), Int64}([ds_state2[1]; ds_state2[2]; ds_state2[3]]) 

    ωs_states = states.E
    ωs_states_static = SVector{16, type_to_use}(ωs_states)
    
    #
    ψ_repeats = rand(Complex{type_to_use}, length(ds))
    ψ_repeats = MVector{length(ds), Complex{type_to_use}}(ψ_repeats)
    ψ_repeats_soa = StructArray(ψ_repeats)
    
    dψ_repeats = rand(Complex{type_to_use}, length(ds))
    dψ_repeats = MVector{length(ds), Complex{type_to_use}}(dψ_repeats)
    dψ_repeats_soa = StructArray(dψ_repeats)
    
    ψ_soa = StructArray(ψ_mutable)
    
    f = MVector{3, Complex{type_to_use}}(zeros(Complex{type_to_use}, 3))
    
    E = MVector{3, Complex{type_to_use}}(zeros(Complex{type_to_use}, 3))

    decay_dist = Exponential(1)

    # set position
    u₀ = zeros(ComplexF64, length(ψ₀)+n_excited+6)
    for i ∈ 1:n_states
        u₀[i] = ψ₀[i]
    end
    for i in 1:3
        u₀[n_states+n_excited+i] = particle.r[i]
        u₀[n_states+n_excited+3+i] = particle.v[i]
    end
    u₀ = MVector{26, ComplexF64}(u₀)

    return MutableNamedTuple(
        
        n_scatters=0,
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),

        krs=krs_mutable,
        ks=ks_static,
        ωs=ωs_static, 
        exp_vals=exp_vals_mutable,
        Es=Es_mutable,
        ϵs=ϵs_static,
        ims=ims_mutable,
        res=res_mutable,
        
        # total electric fields
        E_soa=StructArray(E),
        E_k=[SVector{3, Complex{type_to_use}}(zeros(Complex{type_to_use}, 3)) for _ ∈ 1:3], # should this be a tuple?
        E_k_soa=StructArray(MMatrix{3, 3, Complex{type_to_use}}(zeros(Complex{type_to_use}, 3, 3))),
        sat_params=sat_params_static,
        
        # dipole matrix couplings
        d=d,
        ds=ds,
        ds1=ds1,
        ds2=ds2,
        ds3=ds3,
        ds_state1=ds_state1,
        ds_state2=ds_state2,
        ψ_repeats1_soa=deepcopy(ψ_repeats_soa),
        ψ_repeats2_soa=deepcopy(ψ_repeats_soa),
        dψ_repeats1_soa=deepcopy(dψ_repeats_soa),
        dψ_repeats2_soa=deepcopy(dψ_repeats_soa),
        # d_exp=SVector{3, Complex{type_to_use}}(zeros(Complex{type_to_use}, 3)),
        d_exp=MVector{3, Complex{type_to_use}}(zeros(Complex{type_to_use}, 3)),
        # d_exp=MVector{3, type_to_use}(zeros(type_to_use, 3)),
        
        # extra terms
        ds_added=ds_added,
        ds_state1_added=ds_state1_added,
        ds_state2_added=ds_state2_added,
        E_ODT=sim_params.ODT_intensity,
        
        # Hamiltonian terms
        dEs_soa=dEs_soa,
        dEs_soa_added=dEs_soa_added,
        
        # states information
        ωs_states=ωs_states_static,
        eiωt=StructArray(MVector{length(states), Complex{type_to_use}}(zeros(Complex{type_to_use}, length(states)))),
        
        #
        ψ_soa=ψ_soa,
        dψ_soa=deepcopy(ψ_soa),
        n_states=length(ψ_soa),
        n_excited=n_excited,
        n_ground=n_states - n_excited,
        
        sim_params=sim_params,
        extra_data=extra_data,
        
        k=k,
        Γ=Γ,
        
        f=f,

        mass=mass,

        H_func=H_func,

        Js=Js,

        u₀=u₀,

        ω_offset=ω_offset
    )
end
export make_parameters_fast

function ψ_stochastic_fast!(du, u, p, t)
    
    r = SVector(real(u[p.n_states + p.n_excited + 1]), real(u[p.n_states + p.n_excited + 2]), real(u[p.n_states + p.n_excited + 3]))

    # normalize u
    normalize_u!(u, p.n_states)

    # transfer data from u to ψ
    u_to_ψ!(u, p.ψ_soa, p.n_states)
    
    update_eiωt!(p.eiωt, p.ωs_states, t)
    # update_eiωt!(p.eiωt, p.ωs_states, p.ω_offset, t)
    
    Heisenberg_turbo_state!(p.ψ_soa, p.eiωt, -1)

    ψ_ordering_soa_both!(p.ψ_repeats1_soa, p.ψ_repeats2_soa, p.ψ_soa, p.ds_state1, p.ds_state2)
    
    # update E fields
    update_fields_fast_composed!(p.krs, p.ks, r, p.ωs, t, p.exp_vals, p.Es, p.ϵs, p.ims, p.res)
    update_E_and_E_k!(p.E_soa, p.E_k, p.Es, p.ks, p.sat_params)
    
    # perform H = -dE
    update_dE!(p.dEs_soa, p.ds, p.E_soa)
    
    # perform -iHψ, and put directly into p.dψ_soa
    dψ_Hψ_simple!(p.dψ_soa, p.ψ_soa, p.dEs_soa, p.ds_state1, p.ds_state2)
    
    # add decays
    update_decay!(p.dψ_soa, p.ψ_soa, p.n_states, p.n_excited)

    # custom code
    scalar, ∇H = p.H_func(p, r, t)
    dψ_Hψ_added!(p.dψ_soa, p.ψ_soa, p.dEs_soa_added, p.ds_state1_added, p.ds_state2_added, scalar)
    
    Heisenberg_turbo_state!(p.dψ_soa, p.eiωt)

    # transfer data from dψ to du
    ψ_to_u!(p.dψ_soa, du, p.n_states)
    
    # calculate force, f = -im * k * ⟨H⟩
    d_expectation!(p.ψ_repeats1_soa, p.ψ_repeats2_soa, p.ds, p.d_exp)
    update_force!(p.f, p.d_exp, p.E_k)

    # add to force
    f_added_expectation = scalar * operator_expectation_state(p.dEs_soa_added, p.ψ_soa, p.ds_state1_added, p.ds_state2_added)
    f_added = ∇H .* (-f_added_expectation)

    # print(f_added_expectation)
    update_position_and_force!(du, u, p.f, f_added, p.mass, p.n_states, p.n_excited)
    
    @inbounds @fastmath for i ∈ 1:p.n_excited
        _ψ = p.ψ_soa[p.n_states - p.n_excited + i]
        du[p.n_states + i] = _ψ.re^2 + _ψ.im^2
    end

    return nothing
end
export ψ_stochastic_fast!

### FUNCTION FOR DECAY ###
function update_decay!(dψ_soa, ψ_soa, n_states, n_excited)
    for i ∈ (n_states-n_excited+1):n_states
        dψ_soa[i] -= ψ_soa[i]/2
    end
    return nothing
end

### FUNCTIONS FOR STATES ###
@inline function update_eiωt!(eiωt, ωs, ω_offset, t)
    @turbo for i ∈ eachindex(ωs)
        eiωt.im[i], eiωt.re[i] = sincos(ωs[i] * t + ω_offset * t) # did adding the offset make this slower? need to check
    end
    return nothing
end

@inline function update_eiωt!(eiωt, ωs, t)
    @turbo for i ∈ eachindex(ωs)
        eiωt.im[i], eiωt.re[i] = sincos(ωs[i] * t)
    end
    return nothing
end

function normalize_ψ!(ψ, n_states)
    ψ_norm = zero(Float64)
    @turbo for i ∈ 1:n_states
        ψ_norm += ψ.re[i]^2 + ψ.im[i]^2
    end
    ψ_norm = sqrt(ψ_norm)
    @turbo for i ∈ 1:n_states
        ψ.re[i] /= ψ_norm
        ψ.im[i] /= ψ_norm
    end                           
    return nothing
end

@inline function normalize_u!(u, n_states)
    # slower than it needs to be probably
    u_norm = zero(Float64)
    @inbounds @fastmath for i ∈ 1:n_states
        u_norm += real(u[i])^2 + imag(u[i])^2
    end
    u_norm = sqrt(u_norm)
    @inbounds @fastmath for i ∈ 1:n_states
        u[i] /= u_norm
    end                           
    return nothing
end

### POSITION AND FORCE FUNCTIONS ###
function update_position_and_force!(du, u, f, f_added, mass, n_states, n_excited)
    @inbounds @fastmath for i ∈ 1:3
        du[n_states + n_excited + i] = u[n_states + n_excited + i + 3] # update d(position)/dt
        du[n_states + n_excited + 3 + i] = (f[i] + f_added[i]) / mass # update d(velocity)/dt
    end
    return nothing
end

@inline function update_force!(f, d_exp, E_k)
    a = -im * (d_exp[1] * E_k[1][1] + d_exp[2] * E_k[1][2] + d_exp[3] * E_k[1][3])
    b = -im * (d_exp[1] * E_k[2][1] + d_exp[2] * E_k[2][2] + d_exp[3] * E_k[2][3])
    c = -im * (d_exp[1] * E_k[3][1] + d_exp[2] * E_k[3][2] + d_exp[3] * E_k[3][3])
    f[1] = a + conj(a)
    f[2] = b + conj(b)
    f[3] = c + conj(c)
    return nothing
end

### CONVERSION FROM ψ_REPEATS TO/FROM ψ ###
@inline function u_to_ψ!(u, ψ, n_states) # can u be made soa? this would help slightly here, but not sure if compatible with the differential equation solver...
    @inbounds @fastmath for i ∈ 1:n_states
        ψ.re[i] = real(u[i])
        ψ.im[i] = imag(u[i])
    end
    return nothing
end

@inline function ψ_to_u!(ψ, u, n_states) # can u be made soa? this would help slightly here, but not sure if compatible with the differential equation solver...
    @inbounds @fastmath for i ∈ 1:n_states
        u[i] = ψ.re[i] + im * ψ.im[i]
    end
    return nothing
end

@inline function ψ_ordering_soa!(ψ_soa, ψ_repeats_soa, ds_state)
    # there must be a better way to do this resorting...
    # maybe associate each state with a list of transitions, so that each easier to sum over them?
    # this will probably work for this function but not the reverse mapping
    @turbo for i ∈ eachindex(ψ_soa)
        ψ_soa.re[i] = zero(Float64)
        ψ_soa.im[i] = zero(Float64)
    end
    @inbounds for i ∈ eachindex(ψ_repeats_soa)
        idx = ds_state[i]
        ψ_soa.re[idx] += ψ_repeats_soa.re[i]
        ψ_soa.im[idx] += ψ_repeats_soa.im[i]
    end
    return nothing
end

@inline function ψ_ordering_soa_both!(ψ_repeats1_soa, ψ_repeats2_soa, ψ_soa, ds_state1, ds_state2)
    @inbounds for i ∈ eachindex(ψ_repeats1_soa, ψ_repeats2_soa)
        idx1 = ds_state1[i]
        ψ_repeats1_soa.re[i] = ψ_soa.re[idx1]
        ψ_repeats1_soa.im[i] = ψ_soa.im[idx1]
        idx2 = ds_state2[i]
        ψ_repeats2_soa.re[i] = ψ_soa.re[idx2]
        ψ_repeats2_soa.im[i] = ψ_soa.im[idx2]
    end
    return nothing
end

@inline function ψ_ordering_soa_reverse!(ψ_soa, ψ_repeats1_soa, ψ_repeats2_soa, ds_state1, ds_state2) 
    # there must be a better way to do this resorting...
    # maybe associate each state with a list of transitions, so that each easier to sum over them?
    # this will probably work for this function but not the reverse mapping
    @turbo for i ∈ eachindex(ψ_soa)
        ψ_soa.re[i] = zero(Float64)
        ψ_soa.im[i] = zero(Float64)
    end
    @inbounds @fastmath for i ∈ eachindex(ψ_repeats1_soa)
        idx1 = ds_state1[i]
        ψ_soa.re[idx1] += ψ_repeats1_soa.re[i]
        ψ_soa.im[idx1] += ψ_repeats1_soa.im[i]
        idx2 = ds_state2[i]
        ψ_soa.re[idx2] += ψ_repeats2_soa.re[i]
        ψ_soa.im[idx2] += ψ_repeats2_soa.im[i]
    end
    return nothing
end

@inline function ψ_ordering_soa_both_dE!(ψ_repeats1_soa, ψ_repeats2_soa, ψ_soa, ds_state1, ds_state2, E_soa)
    E_re = E_soa.re[1]
    E_im = E_soa.im[1]
    @inbounds @fastmath for i ∈ 1:10
        idx1 = ds_state1[i]
        ψ_soa1_re = ψ_soa.re[idx1]
        ψ_soa1_im = ψ_soa.im[idx1]
        ψ_repeats1_soa.re[i] = ψ_soa1_re * E_re - ψ_soa1_im * E_im
        ψ_repeats1_soa.im[i] = ψ_soa1_im * E_re + ψ_soa1_re * E_im
        idx2 = ds_state2[i]
        ψ_soa2_re = ψ_soa.re[idx2]
        ψ_soa2_im = ψ_soa.im[idx2]        
        ψ_repeats2_soa.re[i] = ψ_soa2_re * E_re - ψ_soa2_im * E_im
        ψ_repeats2_soa.im[i] = ψ_soa2_im * E_re + ψ_soa2_re * E_im
    end
    E_re = E_soa.re[2]
    E_im = E_soa.im[2]
    @inbounds for i ∈ 11:20
        idx1 = ds_state1[i]
        ψ_soa1_re = ψ_soa.re[idx1]
        ψ_soa1_im = ψ_soa.im[idx1]
        ψ_repeats1_soa.re[i] = ψ_soa1_re * E_re - ψ_soa1_im * E_im
        ψ_repeats1_soa.im[i] = ψ_soa1_im * E_re + ψ_soa1_re * E_im
        idx2 = ds_state2[i]
        ψ_soa2_re = ψ_soa.re[idx2]
        ψ_soa2_im = ψ_soa.im[idx2]        
        ψ_repeats2_soa.re[i] = ψ_soa2_re * E_re - ψ_soa2_im * E_im
        ψ_repeats2_soa.im[i] = ψ_soa2_im * E_re + ψ_soa2_re * E_im
    end
    # @inbounds for i ∈ 21:30
    #     idx1 = ds_state1[i]
    #     ψ_repeats1_soa.re[i] = ψ_soa.re[idx1] * E_soa[3]
    #     ψ_repeats1_soa.im[i] = ψ_soa.im[idx1] * E_soa[3]
    #     idx2 = ds_state2[i]
    #     ψ_repeats2_soa.re[i] = ψ_soa.re[idx2] * E_soa[3]
    #     ψ_repeats2_soa.im[i] = ψ_soa.im[idx2] * E_soa[3]
    # end
    return nothing
end

### SCHROEDINGER EVOLUTION ###
function dψ_Hψ_simple!(dψ_soa, ψ_soa, dEs_soa, ds_state1, ds_state2) # includes a factor i
    # should be able to make this faster, possibly by transforming and using the faster matrix multiplication...
    @turbo for i ∈ eachindex(ψ_soa)
        dψ_soa.re[i] = zero(Float64)
        dψ_soa.im[i] = zero(Float64)
    end    
    @inbounds @fastmath for i ∈ eachindex(dEs_soa)
        idx1 = ds_state1[i]
        idx2 = ds_state2[i]
        dψ_soa[idx1] += conj(dEs_soa[i]) * (im * ψ_soa[idx2])
        dψ_soa[idx2] += dEs_soa[i] * (im * ψ_soa[idx1])
    end
    return nothing
end

function dψ_Hψ_added!(dψ_soa, ψ_soa, dEs_soa, ds_state1, ds_state2, scalar) # includes a factor i
    # should be able to make this faster, possibly by transforming and using the faster matrix multiplication...   
    @inbounds @fastmath for i ∈ eachindex(dEs_soa)
        idx1 = ds_state1[i]
        idx2 = ds_state2[i]
        dψ_soa[idx1] -= conj(dEs_soa[i]) * (im * ψ_soa[idx2]) * scalar
        if idx2 != idx1
            dψ_soa[idx2] -= dEs_soa[i] * (im * ψ_soa[idx1]) * scalar
        end
    end
    return nothing
end

function dψ_Hψ_soa!(dψ_repeats_soa, dEs_soa, ψ_repeats_soa) # includes a factor -i
    @turbo for i ∈ eachindex(dψ_repeats_soa)
        # dψ_copy_soa.re[i] = dEs_soa.re[i] * ψ_copy_soa.re[i] - dEs_soa.im[i] * ψ_copy_soa.im[i]
        # dψ_copy_soa.im[i] = dEs_soa.im[i] * ψ_copy_soa.re[i] + dEs_soa.re[i] * ψ_copy_soa.im[i]
        dψ_repeats_soa.im[i] = -(dEs_soa.re[i] * ψ_repeats_soa.re[i] - dEs_soa.im[i] * ψ_repeats_soa.im[i]) # negative sign
        dψ_repeats_soa.re[i] = dEs_soa.im[i] * ψ_repeats_soa.re[i] + dEs_soa.re[i] * ψ_repeats_soa.im[i]
    end
    return nothing
end

function dψ_Hψ_soa!(dψ_repeats1_soa, dψ_repeats2_soa, dEs_soa, ψ_repeats1_soa, ψ_repeats2_soa) # includes a factor -i
    @turbo for i ∈ eachindex(dψ_repeats1_soa, dψ_repeats2_soa)
        dψ_repeats1_soa.im[i] = -(dEs_soa.re[i] * ψ_repeats1_soa.re[i] - dEs_soa.im[i] * ψ_repeats1_soa.im[i])
        dψ_repeats1_soa.re[i] = dEs_soa.im[i] * ψ_repeats1_soa.re[i] + dEs_soa.re[i] * ψ_repeats1_soa.im[i]
        dψ_repeats2_soa.im[i] = dEs_soa.re[i] * ψ_repeats2_soa.re[i] - dEs_soa.im[i] * ψ_repeats2_soa.im[i]
        dψ_repeats2_soa.re[i] = -(dEs_soa.im[i] * ψ_repeats2_soa.re[i] + dEs_soa.re[i] * ψ_repeats2_soa.im[i])
    end
    return nothing
end

### UPDATE HAMILTONIAN ###
function update_dE!(dEs, ds, E) # includes a factor -1 (since H = -dE)
    @inbounds @fastmath for i ∈ 1:10
        dEs[i] = ds[i] * E[1]
    end
    @inbounds @fastmath for i ∈ 11:20
        dEs[i] = ds[i] * E[2]
    end
    @inbounds @fastmath for i ∈ 21:30
        dEs[i] = ds[i] * E[3]
    end
    return nothing
end

# function update_dE!(dEs, ds, E) # includes a factor -1 (since H = -dE)
#     @inbounds @fastmath for j ∈ 1:3
#         E_re = E.re[j]
#         E_im = E.im[j]
#         for i ∈ 1:10
#             dEs.re[i,j] = ds[i,j] * E_re
#             dEs.im[i,j] = ds[i,j] * E_im
#         end
#     end
#     return nothing
# end

# function update_dE!(dEs, ds, E) # includes a factor -1 (since H = -dE)
#     E_re = E.re[1]
#     E_im = E.im[1]
#     @inbounds @fastmath for i ∈ 1:10
#         dEs.re[i] = ds[i] * E_re[1]
#         dEs.im[i] = ds[i] * E_im[1]
#     end
#     E_re = E.re[2]
#     E_im = E.im[2]
#     @inbounds @fastmath for i ∈ 11:20
#         dEs.re[i] = ds[i] * E_re[2]
#         dEs.im[i] = ds[i] * E_im[2]
#     end
#     E_re = E.re[3]
#     E_im = E.im[3]
#     @inbounds @fastmath for i ∈ 21:30
#         dEs.re[i] = ds[i] * E_re[3]
#         dEs.im[i] = ds[i] * E_im[3]
#     end
#     return nothing
# end

### UPDATE EXPECTATION OF d ###
function d_expectation!(ψ_repeats1_soa, ψ_repeats2_soa, ds, d_exp)
    
    d_exp1_re = zero(eltype(ds))
    d_exp1_im = zero(eltype(ds))
    @turbo for i ∈ 1:10
        ψ1_adj_re = ψ_repeats1_soa.re[i]
        ψ1_adj_im = ψ_repeats1_soa.im[i]
        ψ2_re = ψ_repeats2_soa.re[i]
        ψ2_im = -ψ_repeats2_soa.im[i]
        d = ds[i]
        a_re = ψ1_adj_re * ψ2_re - ψ1_adj_im * ψ2_im
        d_exp1_re += d * a_re
        a_im = ψ1_adj_re * ψ2_im + ψ1_adj_im * ψ2_re
        d_exp1_im += d * a_im
    end
    d_exp[1] = d_exp1_re + im * d_exp1_im
    
    d_exp2_re = zero(eltype(ds))
    d_exp2_im = zero(eltype(ds))
    @turbo for i ∈ 11:20
        ψ1_adj_re = ψ_repeats1_soa.re[i]
        ψ1_adj_im = ψ_repeats1_soa.im[i]
        ψ2_re = ψ_repeats2_soa.re[i]
        ψ2_im = -ψ_repeats2_soa.im[i]
        d = ds[i]
        a_re = ψ1_adj_re * ψ2_re - ψ1_adj_im * ψ2_im
        d_exp2_re += d * a_re
        a_im = ψ1_adj_re * ψ2_im + ψ1_adj_im * ψ2_re
        d_exp2_im += d * a_im
    end
    d_exp[2] = d_exp2_re + im * d_exp2_im
    
    d_exp3_re = zero(eltype(ds))
    d_exp3_im = zero(eltype(ds))
    @turbo for i ∈ 21:30
        ψ1_adj_re = ψ_repeats1_soa.re[i]
        ψ1_adj_im = ψ_repeats1_soa.im[i]
        ψ2_re = ψ_repeats2_soa.re[i]
        ψ2_im = -ψ_repeats2_soa.im[i]
        d = ds[i]
        a_re = ψ1_adj_re * ψ2_re - ψ1_adj_im * ψ2_im
        d_exp3_re += d * a_re
        a_im = ψ1_adj_re * ψ2_im + ψ1_adj_im * ψ2_re
        d_exp3_im += d * a_im
    end
    d_exp[3] = d_exp3_re + im * d_exp3_im
    
    return nothing
end

@inline function d_expectation(ψ_repeats1_soa, ψ_repeats2_soa, ds, d_exp, q)
    dq_exp = zero(eltype(ds))
    @inbounds @fastmath for i ∈ eachindex(ds)
        ψ1_adj_re = ψ_repeats1_soa.re[i]
        ψ1_adj_im = ψ_repeats1_soa.im[i]
        ψ2_re = ψ_repeats2_soa.re[i]
        ψ2_im = -ψ_repeats2_soa.im[i]
        d = ds[i]
        a_re = ψ1_adj_re * ψ2_re - ψ1_adj_im * ψ2_im
        dq_exp += d * a_re
    end
    d_exp[q] = dq_exp
    return nothing
end

function operator_expectation(ψ_repeats1_soa, ψ_repeats2_soa, H)
    H_exp_re = zero(Float64)
    H_exp_im = zero(Float64)
    @turbo for i ∈ eachindex(ψ_repeats1_soa, ψ_repeats2_soa)
        ψ1_adj_re = ψ_repeats1_soa.re[i]
        ψ1_adj_im = ψ_repeats1_soa.im[i]
        ψ2_re = ψ_repeats2_soa.re[i]
        ψ2_im = -ψ_repeats2_soa.im[i]

        a_re = ψ1_adj_re * ψ2_re - ψ1_adj_im * ψ2_im
        a_im = ψ1_adj_re * ψ2_im + ψ1_adj_im * ψ2_re
        
        H_re = H.re[i]
        # H_im = H.im[i]
        H_exp_re += H_re * a_re #- H_im * a_im
        H_exp_im += H_re * a_im #+ H_im * a_re
    end
    return H_exp_re + im * H_exp_im
end

function operator_expectation_state(o, ψ_soa, ds_state1, ds_state2)
    o_exp_re = zero(Float64)
    o_exp_im = zero(Float64)
    @inbounds @fastmath for i ∈ eachindex(o)
        idx1 = ds_state1[i]
        idx2 = ds_state2[i]
        o_exp_re += o.re[i] * (ψ_soa.re[idx1] * ψ_soa.re[idx2] + ψ_soa.im[idx1] * ψ_soa.im[idx2])
        if idx1 != idx2
            o_exp_re += o.re[i] * (ψ_soa.re[idx2] * ψ_soa.re[idx1] + ψ_soa.im[idx2] * ψ_soa.im[idx1])
        end
    end
    return o_exp_re
end
export operator_expectation_state

### UPDATE FIELDS ###
@inline function update_fields_fast1!(krs, ks, r, ωs, t, exp_vals)
    @inbounds @fastmath for i ∈ eachindex(krs)
        krs_i = zero(eltype(krs))
        for j ∈ 1:3
            krs_i += ks[i,j] * r[j]
        end
        krs[i] = krs_i
        exp_vals[i] = -krs_i + ωs[i] * t
    end
    return nothing
end

@inline function update_fields_fast2!(res, ims, exp_vals)
    @turbo for i ∈ eachindex(exp_vals)
        ims[i], res[i] = sincos(exp_vals[i])
    end
    return nothing
end

@inline function update_fields_fast3!(Es, ϵs, res, ims)
    @inbounds @fastmath for i ∈ axes(Es,1)
        prefactor = res[i] + im * ims[i]
        for j ∈ 1:3
            Es[i,j] = prefactor * conj(ϵs[i,j])
        end
    end
    return nothing
end

@inline function update_fields_fast_composed!(krs, ks, r, ωs, t, exp_vals, Es, ϵs, ims, res)
    
    update_fields_fast1!(krs, ks, r, ωs, t, exp_vals)
    update_fields_fast2!(res, ims, exp_vals)
    update_fields_fast3!(Es, ϵs, res, ims)
    
    return nothing
end

@inline function update_E_and_E_k!(E, E_k, Es, ks, sat_params) # multiplication of the two matrices Es and ks might be faster
    # this can definitely be faster
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] = (zero(eltype(E_k[i])), zero(eltype(E_k[i])), zero(eltype(E_k[i])))
        E[i] = 0.0
    end
    @inbounds @fastmath for i ∈ axes(Es, 1)
        E_i = (sqrt(sat_params[i]) / (2√2)) * Es[i,:] # can make the prefactor constants
        k_i = ks[i,:]
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
        E .+= E_i
    end
    return nothing
end

# what are some other ideas?
# directly using sparse arrays
# using all staticarrays, and maybe making some things matrices rather than vectors
# make no reference to indices at all

# Design 1: one long vector with all ds
# Design 2: three vectors, one for each of d_1, d_2, d_3 (q = -1, 0, +1)

# H: a vector with the same length as nonzero elements in d
# dE: a vector with the values d_iq*E_q

# function ψ_stochastic_fast!(du, u, p, t)
    
#     # transfer data from u to ψ
#     u_to_ψ!(u, p.ψ_soa, p.n_states)
    
#     normalize_ψ!(p.ψ_soa)
    
#     update_eiωt!(p.eiωt, p.ωs_states, t)
    
#     Heisenberg_turbo_state!(p.ψ_soa, p.eiωt, -1)

#     ψ_ordering_soa_both!(p.ψ_repeats1_soa, p.ψ_repeats2_soa, p.ψ_soa, p.ds_state1, p.ds_state2)
    
#     # update E fields
#     update_fields_fast_composed!(p.krs, p.ks, p.r, p.ωs, t, p.exp_vals, p.Es, p.ϵs, p.ims, p.res)
#     update_E_and_E_k!(p.E_soa, p.E_k, p.Es, p.ks, p.sat_params)
    
#     # perform H = -dE
#     update_dE!(p.dEs_soa, p.ds, p.E_soa)
    
#     # perform -iHψ, and put directly into p.dψ_soa
#     dψ_Hψ_simple!(p.dψ_soa, p.ψ_soa, p.dEs_soa, p.ds_state1, p.ds_state2)
    
#     # custom code
#     r = p.r
#     update_ODT_center_spiral!(p.sim_params, p.extra_data, t)
#     ODT_x = p.sim_params.ODT_position[1] / (1 / p.k)
#     ODT_z = p.sim_params.ODT_position[2] / (1 / p.k)
#     ODT_y = 0e-3 / (1 / p.k)
#     ODT_size = p.sim_params.ODT_size .* p.k
#     scalar_ODT = exp(-2(r[1]-ODT_x)^2/ODT_size[1]^2) * exp(-2(r[2]-ODT_y)^2/ODT_size[2]^2) * exp(-2(r[3]-ODT_z)^2/ODT_size[3]^2)
    
#     dψ_Hψ_added!(p.dψ_soa, p.ψ_soa, p.dEs_soa_added, p.ds_state1_added, p.ds_state2_added, scalar_ODT)
    
#     # perform -iHψ
#     # dψ_Hψ_soa!(p.dψ_repeats_soa, p.dEs_soa, p.ψ_repeats2_soa)
#     # dψ_Hψ_soa!(p.dψ_repeats1_soa, p.dψ_repeats2_soa, p.dEs_soa, p.ψ_repeats2_soa, p.ψ_repeats1_soa)
    
#     # transform back to dψ
#     # ψ_ordering_soa!(p.dψ_soa, p.dψ_repeats_soa, p.ds_state1)
#     # ψ_ordering_soa_reverse!(p.dψ_soa, p.dψ_repeats1_soa, p.dψ_repeats2_soa, p.ds_state1, p.ds_state2)
    
#     # this can be done all at once probably
    
#     Heisenberg_turbo_state!(p.dψ_soa, p.eiωt)
    
#     # transfer data from dψ to du
#     ψ_to_u!(p.dψ_soa, du, p.n_states)
    
#     # calculate force, f = -im * k * ⟨H⟩
#     d_expectation!(p.ψ_repeats1_soa, p.ψ_repeats2_soa, p.ds, p.d_exp)
#     update_force!(p.f, p.d_exp, p.E_k)
    
#     update_position_and_force!(du, u, f, p.n_states, p.n_excited)
    
#     return nothing
# end
# ;
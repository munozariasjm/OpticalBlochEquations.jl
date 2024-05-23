using Serialization
using Printf,Plots




function log_test_info(saving_dir, test_i, params)
    # make a new folder under saving_dir for this test
    folder = @sprintf("test%d", test_i)
    while isdir(joinpath(saving_dir, folder))
        test_i += 1
        folder = @sprintf("test%d", test_i)
    end
    folder_dir = joinpath(saving_dir, folder)
    mkdir(folder_dir)
    
    # save current parameters to .jl file
    serialize(joinpath(folder_dir, "params.jl"), params)
    
    write_test_info(saving_dir, test_i, params)
    
    return test_i
end

function log_test_info_with_molecule_package(saving_dir, test_i, params, package)
    # make a new folder under saving_dir for this test
    folder = @sprintf("test%d", test_i)
    while isdir(joinpath(saving_dir, folder))
        test_i += 1
        folder = @sprintf("test%d", test_i)
    end
    folder_dir = joinpath(saving_dir, folder)
    mkdir(folder_dir)

    # save current parameters to .jl file
    serialize(joinpath(folder_dir, "params.jl"), params)

    serialize(joinpath(folder_dir, "package.jl"), package)

    write_test_info(saving_dir, test_i, params)

    return test_i
end



function load_test_params(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    if isdir(folder_dir) == false
        @printf("%s is not found.", folder_dir)
        println()
       return nothing 
    end
    params = deserialize(joinpath(folder_dir, "params.jl"))
    return params
end

function save_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    serialize(joinpath(folder_dir, "results.jl"), results)
end


function pol2str(pol)
    if pol == σ⁺
        return "+"
    elseif pol == σ⁻
        return "-"
    end
end

function display_test_info(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    params = deserialize(joinpath(folder_dir, "params.jl"))
    header = "-"^50
    println(header)
    @printf("test %d information", test_i)
    println()
    println(header)
    @printf("propagation time = %.2f ms", params.t_end*1e3)
    println()
    @printf("particle number = %d", params.n_values)
    println()
    println(header)
    
    @printf("Laser parameters:")
    println()
    @printf("Polarizations (+x beam): %s, %s, %s, %s", 
            pol2str(params.pol1_x), pol2str(params.pol2_x), pol2str(params.pol3_x), pol2str(params.pol4_x))
    println()
    @printf("Polarization imbalance: %.3f", params.pol_imbalance)
    println()
    @printf("Detunings (MHz): %.2f, %.2f, %.2f, %.2f", params.Δ1/(2π)/1e6, params.Δ2/(2π)/1e6, params.Δ3/(2π)/1e6, params.Δ4/(2π)/1e6)
    println()
    @printf("Saturations: %.2f, %.2f, %.2f, %.2f", params.s1, params.s2, params.s3, params.s4)
    println()
    @printf("Power imbalance (x,y,z): %.3f, %.3f, %.3f", params.s_imbalance[1], params.s_imbalance[2], params.s_imbalance[3])
    println()
    println(header)
    
    @printf("max B field gradient: (%.2f, %.2f, %.2f) G/cm", -params.B_gradient/2, params.B_gradient/2, -params.B_gradient)
    println()
    @printf("B field ramp time: %.1f ms", params.B_ramp_time*1e3)
    println()
    println(header)
    
    println("Initial state:")
    @printf("Cloud radius = %.2f mm", params.diameter*1e3)
    println()
    @printf("Cloud temperature = %.2f mK", params.temp*1e3)
    println()
    @printf("Displacement from centre = (%.2f, %.2f, %.2f) mm", params.displacement[1],params.displacement[2],params.displacement[3])
    println()
    @printf("Centre of mass velocity = (%.2f, %.2f, %.2f) m/s", params.kick[1], params.kick[2], params.kick[3])
    println()
    println(header)
end
;


function write_summarized_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    
    open(joinpath(folder_dir, "results.txt"), "w") do file 
        header = ("-"^50)*"\n"
        write(file, header)
        
        write(file, @sprintf("test %d results \n", test_i))

        write(file, header)
        
        ~, x = plot_size(results, "x")
        ~, y = plot_size(results, "y")
        ~, z = plot_size(results, "z")
        write(file, @sprintf("Final cloud size: (%.2f, %.2f, %.2f) mm \n", x[end], y[end], z[end]))
        
        ~, Tx = plot_temperature(results, "x")
        ~, Ty = plot_temperature(results, "y")
        ~, Tz = plot_temperature(results, "z")
        write(file, @sprintf("Final temperature: (%.2f, %.2f, %.2f) μK \n", Tx[end], Ty[end], Tz[end]))
        
        n_photon = plot_photons_scattered(results)
        write(file, @sprintf("Average photons scattered: %.0f \n", n_photon))
        rate = plot_scattering_rate(results)
        write(file, @sprintf("Average scattering rate: %.3f MHz \n", rate))
        
        write(file, header)
    end
end


function save_results(saving_dir, test_i, results)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    serialize(joinpath(folder_dir, "results.jl"), results)
    # write_summarized_results(saving_dir, test_i, results)
end

function load_results(saving_dir, test_i)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    if isdir(folder_dir)==false
        @printf("%s is not found.", folder_dir)
        println()
       return nothing 
    end
    results = deserialize(joinpath(folder_dir, "results.jl"))
    return results
end

function summarize_results(results)
    header = "-"^50
    println(header)
#     @printf("Molecules trapped: %i out of %i", length(results.trapped_indicies), results.n_values)
#     println()
    
    ~,x = plot_size(results, "x")
    ~,y = plot_size(results, "y")
    ~,z = plot_size(results, "z")
    @printf("Final cloud size: (%.2f, %.2f, %.2f) mm", x[end], y[end], z[end])
    println()
    
    ~,Tx = plot_temperature(results, "x")
    ~,Ty = plot_temperature(results, "y")
    ~,Tz = plot_temperature(results, "z")
    @printf("Final temperature: (%.2f, %.2f, %.2f) μK", Tx[end], Ty[end], Tz[end])
    println()
    
    n_photon = plot_photons_scattered(results)
    @printf("Average photons scattered: %i", n_photon)
    println()
    rate = plot_scattering_rate(results)
    @printf("Average scattering rate: %.3f MHz", rate)
    println()
    
    println(header)
end


function summarize_results(saving_dir, test_i)
    header = "-"^50
    println(header)
    @printf("test %d results", test_i)
    println()
   results = load_results(saving_dir, test_i)
    summarize_results(results)
end

function make_scan_folder(lists, working_dir, scan_i, comments)
    folder = @sprintf("scan%d", scan_i)
    while isdir(joinpath(working_dir, folder))
        scan_i += 1
        folder = @sprintf("scan%d", scan_i)
    end
    folder_dir = joinpath(working_dir, folder)
    mkdir(folder_dir)
    
    serialize(joinpath(folder_dir, "lists.jl"), lists)
    
    open(joinpath(folder_dir, "comments.txt"), "w") do file
        write(file, comments)
    end;
    return folder_dir
end;


function write_test_info(saving_dir, test_i, params)
    folder = @sprintf("test%d", test_i)
    folder_dir = joinpath(saving_dir, folder)
    
    open(joinpath(folder_dir, "info.txt"), "w") do file  
        header = ("-"^50)*"\n"
        write(file, header)

        write(file, @sprintf("test %d information \n", test_i))

        write(file, header)

        write(file, @sprintf("propagation time = %.2f ms \n", params.t_end*1e3))

        write(file, @sprintf("particle number = %d \n", params.n_values))

        write(file, header)
        
        write(file, "Laser parameters:\n")
        
        write(file, @sprintf("Polarizations (+x beam): %s, %s, %s, %s \n", 
            pol2str(params.pol1_x), pol2str(params.pol2_x), pol2str(params.pol3_x), pol2str(params.pol4_x)))
            
        write(file, @sprintf("Polarization imbalance: %.3f \n", params.pol_imbalance))
    
        write(file, @sprintf("Detunings (MHz): %.2f, %.2f, %.2f, %.2f \n", params.Δ1/(2π)/1e6, params.Δ2/(2π)/1e6, params.Δ3/(2π)/1e6, params.Δ4/(2π)/1e6))

        write(file, @sprintf("Saturations: %.2f, %.2f, %.2f, %.2f \n", params.s1, params.s2, params.s3, params.s4))
        
        write(file, @sprintf("Power imbalance (x, y, z): %.3f, %.3f, %.3f \n", params.s_imbalance[1], params.s_imbalance[2], params.s_imbalance[3]))

        write(file, header)
        
        write(file,  @sprintf("max B field gradient: (%.2f, %.2f, %.2f) G/cm \n", -params.B_gradient/2, params.B_gradient/2, -params.B_gradient))
    
        write(file, @sprintf("B field ramp time: %.1f ms \n", params.B_ramp_time*1e3))

        write(file, header)
        
        write(file, "Initial state: \n")
    
        write(file, @sprintf("Cloud radius = %.2f mm \n", params.diameter*1e3))
        
        write(file, @sprintf("Cloud temperature = %.2f mK \n", params.temp*1e3))
    
        write(file, @sprintf("Displacement from centre = (%.2f, %.2f, %.2f) mm \n", params.displacement[1]*1e3,params.displacement[2]*1e3,params.displacement[3]*1e3))
 
        write(file, @sprintf("Centre of mass velocity = (%.2f, %.2f, %.2f) m/s \n", params.kick[1], params.kick[2], params.kick[3]))
    
        write(file, header)
    end;
end
;


function get_points_from_results(results, it)
    points = []
     for i in 1:length(results.times)
         if it < length(results.times[i])
             push!(points, (results.x_trajectories[i][it], results.y_trajectories[i][it],results.z_trajectories[i][it]))
         end
     end
     return points
 end;
 
 
 
 function plot_survival(params, results; keep=false, label="")
     max_t_id = 1
      plot_ts = Float64[]
     for i in 1:length(results.times)
          if length(results.times[i]) > max_t_id
               max_t_id = length(results.times[i])  
              plot_ts = results.times[i]
          end                                                                                                          
      end
 
     dt = plot_ts[2] - plot_ts[1]
     time_plot = plot_ts
 
     survival = []
 
     for i in 1:Int(params.t_end*1e3 ÷ dt)
         points = get_points_from_results(results, i)
         num = 0
         for p in points
            if p[1]^2 + p[2]^2 + p[3]^2 < (2)^2
                 num += 1
             end
         end
         push!(survival, length(points))
     end
 
     time_plot = LinRange(0, Int(params.t_end*1e3 ÷ dt)*dt, Int(params.t_end*1e3 ÷ dt))
     if keep
         plot!(time_plot,survival, linewidth=3, ylim=(0,survival[1]+5), label=label)
     else
         plot(time_plot,survival, linewidth=3,ylim=(0,survival[1]+5), label=label)
     end
     plot!(title="Survived molecules", xlabel="time (ms)", size = (400,300), legend = true, dpi=300)
     ;
     return time_plot, survival
 end;
 
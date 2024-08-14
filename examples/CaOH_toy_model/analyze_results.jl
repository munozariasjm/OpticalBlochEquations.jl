
using Printf, Plots

using Distributions


using LsqFit
using Optim



function plot_all_trajectories(results, direction)
    if direction == "x"
        plot(legend=false, title="x position", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.x_trajectories[j])
        end
    elseif direction == "y"
        plot(legend=false, title="y position", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.y_trajectories[j])
        end
    elseif direction == "z"
        plot(legend=false, title="z position", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.z_trajectories[j])
        end
    elseif direction == "all"
        plot(legend=false, title="distance from centre", xlabel="time (ms)",ylabel="position (mm)")
        for j in 1:length(results.times)
           plot!(results.times[j], sqrt.(results.z_trajectories[j].^2 + results.x_trajectories[j].^2 + results.y_trajectories[j].^2))
        end
    end    
end

function plot_all_velocities(results, direction)
    if direction == "x"
        plot(legend=false, title="x velcocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.x_velocities[j])
        end
    elseif direction == "y"
        plot(legend=false, title="y velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.y_velocities[j])
        end
    elseif direction == "z"
        plot(legend=false, title="z velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], results.z_velocities[j])
        end
    elseif direction == "all"
        plot(legend=false, title="speed", xlabel="time (ms)",ylabel="velocity (m/s)")
        for j in 1:length(results.times)
           plot!(results.times[j], sqrt.(results.z_velocities[j].^2 + results.x_velocities[j].^2 + results.y_velocities[j].^2))
        end
    end    
end
function plot_survival_scattering_rate(param, results)
    trapped = get_trapped_indicies(param, results)
    plot(legend=false, xlabel="time (ms)", ylabel="MHz")
   for p in trapped
      plot!(results.times[p], results.A_populations[p] .* Γ * 1e-6)
   end
   plot!(title="scattering rate")

   out = []
   for i in 1:length(results.times[trapped[1]])
      n_total = 0
       n_num = 0.01
       for p in trapped
           if length(results.times[p]) >= i
               n_total += results.A_populations[p][i]
               n_num += 1
           end
       end
       push!(out, n_total/n_num)
   end
   plot!(results.times[trapped[1]], out.*Γ*1e-6, linewidth=3, color=:red)
   return out.*Γ*1e-6   
end

                                                                                           
function plot_survival_velocities(results, direction)
    max_t_id = 1
    plot_ts = Float64[]
   for i in 1:length(results.times)
        if length(results.times[i]) > max_t_id
             max_t_id = length(results.times[i])  
            plot_ts = results.times[i]
        end                                                                                                          
    end                                                                                             
   if direction == "x"
       plot(legend=false, title="x velcocity", xlabel="time (ms)",ylabel="velocity (m/s)")
       for j in 1:length(results.times)
          if length(results.times[j]) < max_t_id 
               continue
          end                                                                                                      
          plot!(results.times[j], results.x_velocities[j])
       end
   elseif direction == "y"
       plot(legend=false, title="y velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
       for j in 1:length(results.times)
          if length(results.times[j]) < max_t_id 
               continue
          end                                                                                                             
              plot!(results.times[j], results.y_velocities[j])
       end
   elseif direction == "z"
       plot(legend=false, title="z velocity", xlabel="time (ms)",ylabel="velocity (m/s)")
       for j in 1:length(results.times)
          if length(results.times[j]) < max_t_id 
               continue
          end                                                                                                                      
          plot!(results.times[j], results.z_velocities[j])
       end
   elseif direction == "all"
       plot(legend=false, title="speed", xlabel="time (ms)",ylabel="velocity (m/s)")
       for j in 1:length(results.times)
          if length(results.times[j] )< max_t_id
               continue
          end                                                                                                                              
          plot!(results.times[j], sqrt.(results.z_velocities[j].^2 + results.x_velocities[j].^2 + results.y_velocities[j].^2))
       end
   end    
end                                                                                            
                                                                   

function plot_scattering_rate(results)
   plot(legend=false, xlabel="time (ms)", ylabel="MHz")
   for p in 1:length(results.times)
      plot!(results.times[p], results.A_populations[p] .* Γ * 1e-6)
   end
   plot!(title="scattering rate")

   max_t_id = 1
    plot_ts = Float64[]
   for i in 1:length(results.times)
        if length(results.times[i]) > max_t_id
             max_t_id = length(results.times[i])  
            plot_ts = results.times[i]
        end                                                                                                          
    end

   out = []
   for i in 1:max_t_id
      n_total = 0
       n_num = 0.01
       for p in 1:length(results.times)
           if length(results.times[p]) >= i
               n_total += results.A_populations[p][i]
               n_num += 1
           end
       end
       push!(out, n_total/n_num)
   end
   plot!(plot_ts, out.*Γ*1e-6, linewidth=3, color=:red)
   return mean(out.*Γ*1e-6)                                                                    
end

function scattering_rate_at_t(results, t)
    max_t_id = 1
    plot_ts = Float64[]
   for i in 1:length(results.times)
        if length(results.times[i]) > max_t_id
             max_t_id = length(results.times[i])  
            plot_ts = results.times[i]
        end                                                                                                          
    end                                                                                               
    dt = plot_ts[2]-plot_ts[1]
    i = Int(t ÷ dt) + 1
                                                                                                           
     n_total = 0
     n_num = 0.01
     for p in 1:length(results.times)
         if length(results.times[p]) >= i
             n_total += results.A_populations[p][i]
             n_num += 1
         end
     end                                                                                                       
     return n_total/n_num *Γ*1e-6                                                                                                    
end   


function plot_photons_scattered(results)
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end 
    trapped = []       
    out= []                                                                                
    for i in 1:length(results.times)
         if length(results.times[i]) == max_t_id   
             push!(trapped, i)     
               push!(out, results.photons_scattered[i])                                                                             
         end                                                                                   
    end 
   scatter(trapped, out, linewidth=2,title="photon scattered for survived molecules",xlabel="trapped particle index", size=(500,300),dpi=300, legend=false)
   return mean(out)                                                                                                                                                                         
end


                                                                          
function plot_size(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
                                                                                                                         
    if direction == "x"
         plot(legend=false, title="x_size", xlabel="time (ms)",ylabel="size (mm)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_trajectories)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.x_trajectories[j][i])
                    end    
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, σs)                                                                                                                           
          return plot_ts, σs
     elseif direction == "y"
         plot(legend=false, title="y_size", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            y_at_t = Float64[]
            for j in 1:length(results.y_trajectories)
                if length(results.y_trajectories[j]) >= i    
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0
                        push!(y_at_t, results.y_trajectories[j][i])     
                    end
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(y_at_t))
         end
                                                                                                                                                     plot!(plot_ts, σs)
                                                                                                                                                     return plot_ts, σs
     elseif direction == "z"
         plot(legend=false, title="z_size", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            z_at_t = Float64[]
            for j in 1:length(results.z_trajectories)
                if length(results.z_trajectories[j]) >= i  
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0
                        push!(z_at_t, results.z_trajectories[j][i]) 
                    end
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(z_at_t))
         end
         plot!(plot_ts, σs)
         return plot_ts, σs
     elseif direction == "all"
         
         plot_ts, σx = plot_size(results, "x")
         ~,  σy = plot_size(results, "y")
         ~,  σz = plot_size(results, "z")
         plot(legend=true, title="cloud size", xlabel="time (ms)",ylabel="size (mm)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz
     end  
     
 end
                                                                                                                                                
                                                                                                                                            function plot_centre(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
                                                                                                                         
    if direction == "x"
         plot(legend=false, title="x centre", xlabel="time (ms)",ylabel="size (mm)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_trajectories)
                if length(results.x_trajectories[j]) >= i                                                                                                          push!(x_at_t, results.x_trajectories[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(x_at_t))
         end
          plot!(plot_ts, σs)                                                                                                                           
          return plot_ts, σs
     elseif direction == "y"
         plot(legend=false, title="y centre", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            y_at_t = Float64[]
            for j in 1:length(results.y_trajectories)
                if length(results.y_trajectories[j]) >= i                                                                                                          push!(y_at_t, results.y_trajectories[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(y_at_t))
         end
                                                                                                                                                     plot!(plot_ts, σs)
                                                                                                                                                     return plot_ts, σs
     elseif direction == "z"
         plot(legend=false, title="z centre", xlabel="time (ms)",ylabel="size (mm)")
         for i in 1:max_t_id
            z_at_t = Float64[]
            for j in 1:length(results.z_trajectories)
                if length(results.z_trajectories[j]) >= i                                                                                                          push!(z_at_t, results.z_trajectories[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(z_at_t))
         end
         plot!(plot_ts, σs)
         return plot_ts, σs
     elseif direction == "all"
         
         plot_ts, σx = plot_centre(results, "x")
         ~,  σy = plot_centre(results, "y")
         ~,  σz = plot_centre(results, "z")
         plot(legend=true, title="cloud centre", xlabel="time (ms)",ylabel="size (mm)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz
     end  
     
 end
                                    
                                    
                                                                                                                                           function plot_centre_velocity(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
                                                                                                                         
    if direction == "x"
         plot(legend=false, title="x centre", xlabel="time (ms)",ylabel="velocity (m/s)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_trajectories)
                if length(results.x_trajectories[j]) >= i                                                                                                          push!(x_at_t, results.x_velocities[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(x_at_t))
         end
          plot!(plot_ts, σs)                                                                                                                           
          return plot_ts, σs
     elseif direction == "y"
         plot(legend=false, title="y centre", xlabel="time (ms)",ylabel="velocity (m/s)")
         for i in 1:max_t_id
            y_at_t = Float64[]
            for j in 1:length(results.y_trajectories)
                if length(results.y_trajectories[j]) >= i                                                                                                          push!(y_at_t, results.y_velocities[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(y_at_t))
         end
                                                                                                                                                     plot!(plot_ts, σs)
                                                                                                                                                     return plot_ts, σs
     elseif direction == "z"
         plot(legend=false, title="z centre", xlabel="time (ms)",ylabel="velocity (m/s)")
         for i in 1:max_t_id
            z_at_t = Float64[]
            for j in 1:length(results.z_trajectories)
                if length(results.z_trajectories[j]) >= i                                                                                                          push!(z_at_t, results.z_velocities[j][i])           
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, mean(z_at_t))
         end
         plot!(plot_ts, σs)
         return plot_ts, σs
     elseif direction == "all"
         
         plot_ts, σx = plot_centre(results, "x")
         ~,  σy = plot_centre(results, "y")
         ~,  σz = plot_centre(results, "z")
         plot(legend=true, title="cloud centre", xlabel="time (ms)",ylabel="size (mm)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz
     end  
     
 end

 
 function plot_temperature(results, direction)
                                                                                         
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
              max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
                                                                                                                         
    
    σs = Float64[]                                                                                                                     
    k_B = 1.381e-23                                                                                                                     
    if direction == "x"
         plot(legend=false, title="x temperature", xlabel="time (ms)",ylabel="temperature (μK)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.x_velocities)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.x_velocities[j][i])
                    end    
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, m*σs.^2/k_B*1e6)                                                                                                                        
          return plot_ts, m*σs.^2/k_B*1e6
     elseif direction == "y"
         plot(legend=false, title="y temperature", xlabel="time (ms)",ylabel="temperature (μK)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.y_velocities)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.y_velocities[j][i])
                    end    
               end                                                                                                                      
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, m*σs.^2/k_B*1e6)                                                                                                                        
          return plot_ts, m*σs.^2/k_B*1e6
     elseif direction == "z"
         plot(legend=false, title="z temperature", xlabel="time (ms)",ylabel="temperature (μK)")
 
         for i in 1:max_t_id
            x_at_t = Float64[]
            for j in 1:length(results.z_velocities)
                if length(results.x_trajectories[j]) >= i 
                    r = sqrt(results.x_trajectories[j][i]^2 + results.y_trajectories[j][i]^2 + results.z_trajectories[j][i]^2)
                    if r < 1.0 
                        push!(x_at_t, results.z_velocities[j][i])
                    end    
               end                                                                                                                    
            end                                                                                                                         
            push!(σs, std(x_at_t))
         end
          plot!(plot_ts, m*σs.^2/k_B*1e6)                                                                                                                        
          return plot_ts, m*σs.^2/k_B*1e6
     elseif direction == "all"
         plot_ts, σx = plot_temperature(results, "x")
         ~,  σy = plot_temperature(results, "y")
         ~,  σz = plot_temperature(results, "z")
         plot(legend=true, title="Temperature", xlabel="time (ms)",ylabel="T (uK)")
         plot!(plot_ts, σx, label="x")
         plot!(plot_ts, σy, label="y")
         plot!(plot_ts, σz, label="z")
         return plot_ts, σx, σy, σz                                              
         
      end 
 end


 
function make_2D_image(results, x_min, x_max, n_pixels;t=0.0,photon_budget=Inf)
    """ grids = population """
    max_t_id = 1
     plot_ts = Float64[]
    for i in 1:length(results.times)
         if length(results.times[i]) > max_t_id
             max_t_id = length(results.times[i])  
             plot_ts = results.times[i]
         end                                                                                                          
     end
   dt = plot_ts[2] - plot_ts[1]
    it = Int(t ÷ dt) + 1

    grids = zeros(n_pixels, n_pixels)
    x = LinRange(x_min, x_max, n_pixels)
    dx = x[2]-x[1]
    z = LinRange(x_min, x_max, n_pixels)
    for i in 1:length(results.times)
        if it > length(results.times[i])
            continue
        end
        x_end = results.x_trajectories[i][it]
        z_end = results.z_trajectories[i][it]
        n_photons = sum(results.A_populations[i][1:it]*dt*Γ/1000)
       if  (x_min < x_end < x_max) && (x_min < z_end < x_max)
           ix = Int((x_end - x_min) ÷ dx + 1)
            iz = Int((z_end - x_min) ÷ dx + 1)
            grids[ix, iz] += 1 * exp(-n_photons / photon_budget)
        end
    end
    return x, z, grids
end

function take_camera_image(results, x_min, x_max, n_pixels;t=0.0, t_img=0.0, const_scattering=false, noise=0.0,photon_budget=Inf)
    x, z, grids = make_2D_image(results, x_min, x_max, n_pixels;t=t, photon_budget=photon_budget)
    dt = 0.5
    if const_scattering
        grids = grids .* dt 
    else
        grids = grids .* scattering_rate_at_t(results, t+0.1) * dt 
    end
    if t_img <= 0.5
        return x, z, grids
    else
        t_curr = t + dt
        while t_curr < t + t_img
            ~,~,grids1 = make_2D_image(results, x_min, x_max, n_pixels;t=t_curr, photon_budget=photon_budget)
            if const_scattering
                grids = grids .+ grids1 * dt
            else
                grids = grids .+ grids1 * scattering_rate_at_t(results, t_curr) * dt
            end
            t_curr += dt
        end
        grids = grids .+ (rand(length(x), length(z)).- 0.5).*t_img * noise
        return x, z, grids # grids = brightness
    end
end




function get_trapped_indicies(param, results)
    t_max =  param.t_end*1e3
     trapped_indicies = Int[]
     for i in 1:length(results.times)
         if results.times[i][end] ≈ t_max
            push!(trapped_indicies, i) 
         end
     end
     return trapped_indicies
 end;
 
 
 function goodness(t_end, results)
     
     times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
     
     # Sample number of photons before the molecule is lost to vibrational dark state
     n_molecules = length(times)
     
     _survived = survived(t_end, times)
     @printf("Survival: %i / %i", length(_survived), n_molecules)
     println()
     
     surviving_x_trajectories = x_trajectories[_survived]
     surviving_y_trajectories = y_trajectories[_survived]
     surviving_z_trajectories = z_trajectories[_survived]
     
     n = 0
     for survived_idx ∈ _survived
         idx_end = length(x_trajectories[survived_idx])
         r = distance(x_trajectories[survived_idx], y_trajectories[survived_idx], z_trajectories[survived_idx], idx_end)
         if r > 1.0
             n += 0.0
         elseif r > 0.5
             n += 0.1 * exp(-results.photons_scattered[survived_idx]/14000)
         elseif r > 0.3
             n += (0.6 - r)* exp(-results.photons_scattered[survived_idx]/14000)
         elseif r <= 0.3
             n += ((0.3 - r) * 3 + 0.3) * exp(-results.photons_scattered[survived_idx]/14000)
         end
     end
     return n/n_molecules
 end
 ;
 function survived(t_end, times)
     _survived = Int64[]
     for i ∈ eachindex(times)
         if abs(times[i][end] - t_end*1e3) <= 1.0
             push!(_survived, i)
         end
     end
     return _survived
 end;
                                                
                                                
function survived_t(results, t; rmax=10.0)
    i = 1
    while i <= length(results.times)
         if length(results.times[i]) > 1
              break
         end 
         i += 1                                               
    end                                                    
    dt = results.times[i][2]-results.times[i][1]
    t_j = Int(floor(t ÷ dt))
    ids = Int64[]
    for i in 1:length(results.times)
        if length(results.times[i]) >= t_j
            if (results.x_trajectories[i][t_j])^2 + (results.y_trajectories[i][t_j])^2 + (results.y_trajectories[i][t_j])^2 <= rmax^2                         
                push!(ids, i)
            end                                                    
        end
    end
    return ids
end;

 
function survival_signal(t, results; photon_budget = 14000,rmax=10.0)
    
    times, x_trajectories, y_trajectories, z_trajectories = results.times, results.x_trajectories, results.y_trajectories, results.z_trajectories
    
    # Sample number of photons before the molecule is lost to vibrational dark state
    n_molecules = length(times)
    
    _survived = survived_t(results, t, rmax=rmax)
    
    surviving_x_trajectories = x_trajectories[_survived]
    surviving_y_trajectories = y_trajectories[_survived]
    surviving_z_trajectories = z_trajectories[_survived]
    
    n = 0
    for survived_idx ∈ _survived
        idx_end = length(x_trajectories[survived_idx])
        rate =  results.photons_scattered[survived_idx]/results.times[survived_idx][end]                                              
        n += exp(-rate * t/photon_budget)
    end
    return n/n_molecules
end
;          





       
function merge_results(list_of_results)
    n_values = 0
    for results in list_of_results
        n_values += length(results.x_trajectories)
    end
    x_trajectories = Array{Vector{Float64}}(fill([],n_values))
    y_trajectories = Array{Vector{Float64}}(fill([],n_values)) 
    z_trajectories = Array{Vector{Float64}}(fill([],n_values))
    x_velocities = Array{Vector{Float64}}(fill([],n_values))
    y_velocities = Array{Vector{Float64}}(fill([],n_values))
    z_velocities = Array{Vector{Float64}}(fill([],n_values))
    A_populations = Array{Vector{Float64}}(fill([],n_values))
    times = Array{Vector{Float64}}(fill([],n_values))
    photons_scattered = zeros(n_values)
    
    n_current = 1
    for results in list_of_results
        n = length(results.x_trajectories)
        x_trajectories[n_current:n_current + n - 1] = results.x_trajectories
        y_trajectories[n_current:n_current + n - 1] = results.y_trajectories
        z_trajectories[n_current:n_current + n - 1] = results.z_trajectories
        x_velocities[n_current:n_current + n - 1] = results.x_velocities
        y_velocities[n_current:n_current + n - 1] = results.y_velocities
        z_velocities[n_current:n_current + n - 1] = results.z_velocities
        A_populations[n_current:n_current + n - 1] = results.A_populations
        times[n_current:n_current + n - 1] = results.times
        photons_scattered[n_current:n_current + n - 1] = results.photons_scattered
        n_current += n
    end
                            
    merged_results = MutableNamedTuple(x_trajectories = x_trajectories, y_trajectories= y_trajectories, z_trajectories=z_trajectories,
                                x_velocities = x_velocities, y_velocities=y_velocities, z_velocities=z_velocities,
                                times=times, A_populations=A_populations,
                                n_values=n_values, photons_scattered=photons_scattered,
                                n_states=list_of_results[1].n_states, n_excited=list_of_results[1].n_excited
                                )
    return merged_results
end

function merge_result_lists(list_of_lists)
    n_each = length(list_of_lists[1])
    merged_list = []
    for i in 1:n_each
        temp_list = []
        for list_of_results in list_of_lists
            push!(temp_list, list_of_results[i])
        end
        merged_results = merge_results(temp_list)
        push!(merged_list, merged_results)
    end
    return merged_list
end                           



                                        
function bootstrap_iteration(results; n_values=nothing)
    if n_values == nothing
        n_values = length(results.times)
    end
    indicies = rand(1:length(results.times),n_values)
    results1 = deepcopy(results)
    results1.x_trajectories = Array{Vector{Float64}}(fill([],n_values))
    results1.y_trajectories = Array{Vector{Float64}}(fill([],n_values)) 
    results1.z_trajectories = Array{Vector{Float64}}(fill([],n_values))
    results1.x_velocities = Array{Vector{Float64}}(fill([],n_values))
    results1.y_velocities = Array{Vector{Float64}}(fill([],n_values))
    results1.z_velocities = Array{Vector{Float64}}(fill([],n_values))
    results1.times = Array{Vector{Float64}}(fill([],n_values))
    results1.photons_scattered = zeros(n_values)
    
    for i in 1:n_values
        i_bs = indicies[i]
        results1.x_trajectories[i] = results.x_trajectories[i_bs]
        results1.y_trajectories[i] = results.y_trajectories[i_bs]
        results1.z_trajectories[i] = results.z_trajectories[i_bs]
        results1.x_velocities[i] = results.x_velocities[i_bs]
        results1.y_velocities[i] = results.y_velocities[i_bs]
        results1.z_velocities[i] = results.z_velocities[i_bs]
        results1.times[i] = results.times[i_bs]
        results1.photons_scattered[i] = results.photons_scattered[i_bs]
    end
    return results1
end

function gaussian(x, p)
   A, σ, x0 = p[1], p[2], p[3]
    return A * exp.(-(x .- x0).^2 / (2*σ^2))
end

function bootstrap_size(results, t; iter = 10, n_values=nothing)
    sizes = []
    dt = results.times[1][2]-results.times[1][1]
    t_j = Int(floor(t ÷ dt))                                                
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
        _,sx, sy, sz = plot_size(results1, "all")
        push!(sizes, (sx[t_j]*sy[t_j]*sz[t_j])^(1/3))
    end
        
    return mean(sizes), std(sizes)
end
    
function bootstrap_size_fit(results, t; iter = 10, n_values=nothing, range=0.5, n_grids=40, rmax=10.0)
    sizes = []
    x_grid = collect(LinRange(-range, range, n_grids))
    dt = results.times[1][2]-results.times[1][1]
    t_j = Int(floor(t ÷ dt))
                                                            
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
    
        id_survived = survived_t(results1, t,rmax=rmax)
    
        xs = [results1.x_trajectories[j][t_j] for j in id_survived]
        density_x = distribution_to_func(x_grid, xs)
        p0 = [10, 0.1, 0.0]
        fit_x = curve_fit(gaussian, x_grid, density_x, p0)
        sx = fit_x.param[2]
    
        ys = [results1.y_trajectories[j][t_j] for j in id_survived]
        density_y = distribution_to_func(x_grid, ys)
        p0 = [10, 0.1, 0.0]
        fit_y = curve_fit(gaussian, x_grid, density_y, p0)
        sy = fit_y.param[2]
    
        zs = [results1.z_trajectories[j][t_j] for j in id_survived]
        density_z = distribution_to_func(x_grid, zs)
        p0 = [10, 0.1, 0.0]
        fit_z = curve_fit(gaussian, x_grid, density_z, p0)
        sz = fit_z.param[2]

    
    
        push!(sizes, (sx[end]*sy[end]*sz[end])^(1/3))
    end
        
    return mean(sizes), std(sizes)
end

function distribution_to_func(x_grid, trajectories)
    func = x_grid .* 0.0
    dx = x_grid[2]-x_grid[1]
    for x in trajectories
       id_x = Int((x - x_grid[1]) ÷ dx)
        if 1<= id_x <= length(x_grid)
            func[id_x] += 1.0
        end
    end
    return func
end

    
function bootstrap_survival(results, t; iter=10, n_values = nothing, rmax=10.0)
    signals = []

    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
        n = survival_signal(t, results1, photon_budget=14000, rmax=rmax)
        push!(signals, n)
    end
        
    return mean(signals), std(signals)
end
    
function bootstrap_temperature(results, t; iter=10, n_values=nothing)
    temperatures = []
    dt = results.times[1][2]-results.times[1][1]
    t_j = Int(floor(t ÷ dt))                                                     
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
        Tx = plot_temperature(results1, "x")[end]
        Ty = plot_temperature(results1, "y")[end]
        Tz = plot_temperature(results1, "z")[end]
        push!(temperatures, (Tx[t_j]*Ty[t_j]*Tz[t_j])^(1/3))
    end
    
    return mean(temperatures), std(temperatures)
    
end
    
function bootstrap_temperature_fit(results, t; iter=10, n_values=nothing, range=0.5, n_grids=20,rmax=10.0)
    temperatures = []
    dt = results.times[1][2]-results.times[1][1]
    t_j = Int(floor(t ÷ dt))                                                            
    x_grid = collect(LinRange(-range, range, n_grids))
    for i in 1:iter
        results1 = bootstrap_iteration(results, n_values=n_values)
    
        id_survived = survived_t(results1, t,rmax=rmax)
    
        xs = [results1.x_velocities[j][t_j] for j in id_survived]
        density_x = distribution_to_func(x_grid, xs)
        p0 = [10, 0.1, 0.0]
        fit_x = curve_fit(gaussian, x_grid, density_x, p0)
        k_B = 1.38e-23
        sx = m * fit_x.param[2]^2 /k_B *1e6
    
        ys = [results1.y_velocities[j][t_j] for j in id_survived]
        density_y = distribution_to_func(x_grid, ys)
        p0 = [10, 0.1, 0.0]
        fit_y = curve_fit(gaussian, x_grid, density_y, p0)
        sy = m * fit_y.param[2]^2 /k_B *1e6
    
        zs = [results1.z_velocities[j][t_j] for j in id_survived]
        density_z = distribution_to_func(x_grid, zs)
        p0 = [10, 0.1, 0.0]
        fit_z = curve_fit(gaussian, x_grid, density_z, p0)
        sz = m * fit_z.param[2]^2 /k_B *1e6

       

    
        push!(temperatures, (sx[end]*sy[end]*sz[end])^(1/3))
    end
        
    return mean(temperatures), std(temperatures)
end
   


# Define the 2D Gaussian function
function gaussian2d(x, y, p)
    xc, yc, σx, σy, A = p
    return A * exp.(-((x .- xc) .^ 2 / (2 * σx^2) + (y .- yc) .^ 2 / (2 * σy^2)))
end


function least_sq_cost(params)
    predicted = [gaussian2d(xx, zz, params) for xx in x, zz in z]
    return sum((predicted - grids).^2)
end

function distance(x_trajectory, y_trajectory, z_trajectory, idx)
    return sqrt(x_trajectory[idx]^2 + y_trajectory[idx]^2 + z_trajectory[idx]^2)
end

;

using CairoMakie 
using JLD2
using DifferentialEquations

myfont = "TeX Computer Modern"

function mu_f(s, μ_max, Kₛ, Iₛ)
  s==0 ? 0 : μ_max * s / (Kₛ + s + Iₛ*s*s)
end

####################################
########## MONOD ###################
####################################

μlist = [1.5 10.0;
        3.0 3.0;
        0.75  0.75;
        1.0 6.0;
        2.0 0.0;
        6.00 2.0;
        1.0 1.0]

Ex = [0.1/0.29]
Ey = [100-0.1/0.29]./10

npaths = size(μlist,1)

rootpath = pwd()

#output1 = load_object(rootpath .* "/simulations/data/1A-continuous/500h_001h_ODE_7runs")
#output2 = load_object(rootpath .* "/simulations/data/1A-continuous/500h_001h_SDE_7runs")

#ODE_memory, Yode_dict = output1
#SDE_memory, Ysde_dict = output2

x_t = load_object(rootpath .* "/simulations/data/1A-continuous/500h_001h_ODE_7runs_nolog_x_t.hdf5")
ODE_memory = load_object(rootpath .* "/simulations/data/1A-continuous/500h_001h_ODE_7runs_nolog_x.hdf5")
SDE_memory = load_object(rootpath .* "/simulations/data/1A-continuous/500h_001h_SDE_7runs_nolog_x.hdf5")

SP = [Ex;Ey]

fig = CairoMakie.Figure(size = (300, 300), 
  fontsize = 16)
ax1 = fig[1, 1] = CairoMakie.Axis(fig, xlabel = "", ylabel = "", title = "", 
        aspect = 1, backgroundcolor = :white,
        xlabelfont = myfont,
        ylabelfont = myfont,
#        xticklabelfont = myfont,
 #       yticklabelfont = myfont,
        xgridvisible = false,
        ygridvisible = false,)

odeSolD001(x,y) = Point(-10*mu_f(x, .3, 10.0, 0.0) * y + 0.01 * (100 - x), mu_f(x, 0.3, 10.0, 0.0)*y-0.01y) 

CairoMakie.streamplot!(ax1, odeSolD001, 0..12, 0..12, 
  colormap = :grayC, #CairoMakie.Reverse(:grayC),
  arrow_size = 0.0,
  grid=false,
  density=0.75,
  ticks = false,
  #alpha=0.25,
  linewidth=0.0001
)

fig

for i in 1:npaths
  if i==npaths
    CairoMakie.lines!(ax1,ODE_memory[i, 1:50001, 2], ODE_memory[i, 1:50001, 1], 
    color=(:black, 1.0), #color=(RGB(167/255, 103/255, 130/255), 0.75), 
    grid=false,
    ticks = false,
    linewidth=1.0)
  else
    CairoMakie.lines!(ax1,ODE_memory[i, 1:50001, 2], ODE_memory[i, 1:50001, 1], 
    color=(:black, 1.0), 
    linewidth=1.0)
  end
end


CairoMakie.scatter!(ax1, SP[1,:],SP[2,:], color=:deepskyblue, markersize=12)
CairoMakie.limits!(ax1, 0, 12, 0, 12)
ax1.xticks = 0:3:12
ax1.yticks = 0:3:12
fig

rootpath = "/u/29/magalha1/unix/Pictures/"

CairoMakie.save(rootpath .* "ODE_Monod.pdf", fig)
CairoMakie.save(rootpath .* "ODE_Monod.png", fig)


fig = CairoMakie.Figure(size = (300, 300), 
  fontsize = 16)
ax1 = fig[1, 1] = CairoMakie.Axis(fig, xlabel = "", ylabel = "", title = "", 
          aspect = 1, backgroundcolor = :white,
#          xlabelfont = myfont,
#         ylabelfont = myfont,
#          xticklabelfont = myfont,
#          yticklabelfont = myfont,
          ygridvisible = false,
          xgridvisible = false)

for i in 1:npaths
  if i==npaths
    CairoMakie.lines!(ax1,ODE_memory[i, 1:50001, 2], ODE_memory[i, 1:50001, 1], 
    color=(RGB(167/255, 103/255, 130/255), 0.75), 
    grid=false,
    ticks = false,
    linewidth=1.5, linestyle=:dot)
  else
    CairoMakie.lines!(ax1,ODE_memory[i, 1:50001, 2], ODE_memory[i, 1:50001, 1], 
    color=(:black, 1.0), 
    linewidth=1.5, linestyle=:dot)
  end
end


for i in 1:npaths
  if i==npaths
    CairoMakie.lines!(ax1,SDE_memory[i, 1:50001, 2], SDE_memory[i, 1:50001, 1], 
    color=(RGB(167/255, 103/255, 130/255), 1.0), 
    grid=false,
    ticks = false,
    linewidth=1.0)
  else
    CairoMakie.lines!(ax1,SDE_memory[i, 1:50001, 2], SDE_memory[i, 1:50001, 1], 
    color=(:black, 1.0), 
    linewidth=1.0)
  end
end

CairoMakie.scatter!(ax1, SP[1,:],SP[2,:], color=:deepskyblue, markersize=12)
CairoMakie.limits!(ax1, 0, 12, 0, 12)
ax1.xticks = 0:3:12
ax1.yticks = 0:3:12

fig
CairoMakie.save(rootpath .* "SDE_Monod.pdf", fig)
CairoMakie.save(rootpath .* "SDE_Monod.png", fig)


####################################
########## HALDANE #################
####################################
rootpath = pwd()

x_t = load_object(rootpath .* "/simulations/data/2A-continuous/25h_0005h_ODE_7runs_nolog_x_t.hdf5")
ODE_memory = load_object(rootpath .* "/simulations/data/2A-continuous/25h_0005h_ODE_7runs_nolog_x.hdf5")
SDE_memory = load_object(rootpath .* "/simulations/data/2A-continuous/25h_0005h_SDE_10runs_nolog_x.hdf5")

#=
output1 = load_object(rootpath .* "/simulations/data/2A-continuous/25h_0005h_ODE_10runs")
output2 = load_object(rootpath .* "/simulations/data/2A-continuous//25h_0005h_SDE_10runs")
ODE_memory, Yode_dict = output1
SDE_memory, Ysde_dict = output2
=#
Ex = [2.0 0.0876483]
Ey = [0.0 1.9123517]
Esx = [1.14092]
Esy = [0.85908]

SP = [Ex;Ey]
NONSP = [Esx;Esy]

fig = CairoMakie.Figure(size = (300, 300), 
  fontsize = 16)
ax1 = fig[1, 1] = CairoMakie.Axis(fig, xlabel = "", ylabel = "", title = "", 
#    xlabelfont = myfont,
#    ylabelfont = myfont,
#    xticklabelfont = myfont,
#    yticklabelfont = myfont,
    ygridvisible = false,
    xgridvisible = false,
    aspect = 1, backgroundcolor = :white)

odeSolD07(x,y) = Point(-mu_f(x, 5.0, 0.5, 5.0) * y + 0.7 * (2 - x), mu_f(x, 5.0, 0.5, 5)*y-0.7y) 

CairoMakie.streamplot!(ax1, odeSolD07, 0:1.0:4, 0:1.0:4, 
  colormap = :grayC, #CairoMakie.Reverse(:grayC),
  arrow_size = 0.001,
  grid=false,
  density=0.75,
  ticks = false,
  linewidth=0.0001
)

for i in 1:10
  if i==10
    CairoMakie.lines!(ax1,ODE_memory[i, 1:5001, 2], ODE_memory[i, 1:5001, 1], 
    color=(:black, 1.0),#RGB(167/255, 103/255, 130/255),
    arrow_size = 1/3, 
    grid=false,
    ticks = false,
    linewidth=1.0)
  else
    CairoMakie.lines!(ax1,ODE_memory[i, 1:5001, 2], ODE_memory[i, 1:5001, 1], 
    color=:black,
    arrow_size = 1/3, 
    grid=false,
    ticks = false,
    linewidth=1.0)
  end
end

CairoMakie.scatter!(ax1, SP[1,:],SP[2,:], color=:deepskyblue, markersize=12)
CairoMakie.scatter!(ax1, NONSP[1,:],NONSP[2,:], color=:red, markersize=12)
CairoMakie.limits!(ax1, 0, 4, 0, 4)

fig

rootpath = "/u/29/magalha1/unix/Pictures/"

CairoMakie.save(rootpath .* "ODE_Haldane.pdf", fig)
CairoMakie.save(rootpath .* "ODE_Haldane.png", fig)

fig = CairoMakie.Figure(size = (300, 300), 
  fontsize = 16)
ax1 = fig[1, 1] = CairoMakie.Axis(fig, xlabel = "", ylabel = "", title = "", 
#    xlabelfont = myfont,
#    ylabelfont = myfont,
#    xticklabelfont = myfont,
#    yticklabelfont = myfont,
    ygridvisible = false,
    xgridvisible = false,
    aspect = 1, backgroundcolor = :white)


for i in 1:10
  if i==10
    CairoMakie.lines!(ax1,ODE_memory[i, 1:5001, 2], ODE_memory[i, 1:5001, 1], 
    color=RGB(167/255, 103/255, 130/255),
    arrow_size = 1/3, 
    grid=false,
    ticks = false,
    linewidth=1.5, linestyle=:dot)
  else
    CairoMakie.lines!(ax1,ODE_memory[i, 1:5001, 2], ODE_memory[i, 1:5001, 1], 
    color=(:black, 1.0),
    arrow_size = 1/3, 
    grid=false,
    ticks = false,
    linewidth=1.5,linestyle=:dot)
  end
end


for i in 1:10
  if i==10
    CairoMakie.lines!(ax1,SDE_memory[i, 1:5001, 2], SDE_memory[i, 1:5001, 1], 
    color=RGB(167/255, 103/255, 130/255), 
    linetype=:steppre, 
    arrow_size = 1/3, 
    grid=false,
    ticks = false,
    linewidth=1.0)
  else
    CairoMakie.lines!(ax1,SDE_memory[i, 1:5001, 2], SDE_memory[i, 1:5001, 1], 
    color=(:black, 1.0), 
    linetype=:steppre,
    arrow_size = 1/3, 
    grid=false,
    linewidth=1.0)
  end
end

CairoMakie.scatter!(ax1, SP[1,:],SP[2,:], color=:deepskyblue, markersize=12)
CairoMakie.scatter!(ax1, NONSP[1,:],NONSP[2,:], color=:red, markersize=12)
CairoMakie.limits!(ax1, 0, 4, 0, 4)

fig

rootpath = "/u/29/magalha1/unix/Pictures/"

CairoMakie.save(rootpath .* "SDE_Haldane.pdf", fig)
CairoMakie.save(rootpath .* "SDE_Haldane.png", fig)


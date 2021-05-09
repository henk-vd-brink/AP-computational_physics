import matplotlib.pyplot as plt
import anim
import functions
import plot

plt.close("all")

#Physical paramters
kb = 1.38E-23
T0 = 119.8          

#Initialize the simulation here!
n = 8                   #FCC unit cells per dimension (2k for k = 1, 2, 3, ...)
rho = 0.88              #initialize density (N/V) here!
Tint = T0               #initalize temperature here!

#System parameters
Nt = 200                #amount of timesteps
h = 0.5*10**(-2)        #magnitude of timesteps
 
#System dimensions
N = functions.N(n)      #amount of particles based on # unit cells
L = (N/rho)**(1/3)      #width of box in sigma

   
#Define initial FCC lattice and velocities
pos = functions.FCC(Nt, n, L)
vel = functions.v_int(N, Tint, T0)

#Solve the system
pos, T, P, Energies, vels = functions.systemsolver("Leapfrog", Nt, h, L, pos, vel, Tint, T0)

#Animate the positions
animation = anim.make_3d_animation(L, pos[0:1], delay=10*h/1E-2, initial_view=(20, -70),rotate_on_play=0)
plot.veldist(vel,T,T0)
plot.EnergyTP(Energies, T, T0, vels, P, N)
plot.diff(pos)
plot.pathplot(pos)
functions.rdf(pos,L)
functions.report(T, P, Energies, N)
import numpy as np
import matplotlib.pyplot as plt

def rmin(x,y,z,L):
    """
    Minimal distance calculation function.
    For a given set of x and y and z coordinates, calculates the shortest distance
    between two particles or their respective images is.
    
    Returns distances delta x, delta y, delta z and combined distance delta r.
    """
    threshold = 1E-5*L       #minimal particle distance to avoid approaching
                             #singularity in force    
    deltax = (x.reshape(len(x),1)-x.T + L/2) % L - L/2
    deltay = (y.reshape(len(y),1)-y.T + L/2) % L - L/2
    deltaz = (z.reshape(len(z),1)-z.T + L/2) % L - L/2
    deltar = np.sqrt(deltax**2 + deltay**2 + deltaz**2)
    deltar[deltar < threshold] = threshold
    return deltax, deltay, deltaz, deltar

def LJforce(x,y,z,L):
    """
    Lennard-Jones force calculation. Derivative of the Lennard-Jones force
    for x, y and z coordinates. Uses the rmin function to find distances to
    other particles.
    
    Returns arrays of indivual forces of each particle on every other particle.
    """
    dx, dy, dz, dr = rmin(x,y,z,L) 
    dr8 = dr**(-8)
    F = 48*dr8**2*dr**2 - 24*dr8
    Fx = F*dx
    Fy = F*dy
    Fz = F*dz
    return Fx, Fy, Fz

def Energy(x,y,z,v,L):
    """
    Energy function. Calculates the kinetic and potential energy per
    particle using Ekin = 1/2*m*v^2 and Epot = 4*(1/r^-12 - 1/r^-6). 
    Input is x, y, z and velocity array(s).
    """
    Ekin = 0.5*np.nansum(v[:,0]**2+v[:,1]**2+v[:,2]**2)
    dx, dy, dz, dr = rmin(x,y,z,L) 
    dr6 = dr**(-6)
    Epot = 2*(dr6**2 - dr6)
    netEpot = np.nansum(Epot[Epot < 1E40])
    return Ekin, netEpot

def rdf(pos, L):
    """
    Radial distribution functions. Calculates RDF function for given input of
    positions, box dimensions.
    """
    plt.figure()
    
    N = pos.shape[1]
    V = L**3
    rho = N/V
    
    bins = N
    rmax = L/2
    deltar = rmax/bins
    
    for time in [0,-1]:
        _, _, _, dr = rmin(pos[time][:,0],pos[time][:,1],pos[time][:,2],L)
        g = np.zeros(bins)
        r = np.linspace(0,rmax,bins)
        for i in range(1,bins):
            for n in range(len(dr)):
                count = np.sum((dr[n,:] > i*(deltar)) & (dr[n,:] < (i+1)*deltar))
                g[i] += count / (4*np.pi*((i+1)**2*deltar**3))
        g *= 2/(n+1)/(N-1)/rho
#        if time == 0:
#            plt.stem(r[g > 0.1*max(g)], g[g > 0.1*max(g)]/max(g))
#        else:
        if time == 0:
            string = "Initial position/ideal FCC"
        else:
            string = "Final timestep"
        plt.plot(r, g/max(g), label = string)
    plt.xlabel('Radius [\u03C3]')
    plt.ylabel('g(r)/max(g(r))')
    plt.legend()
    plt.show()
    
def pressure(pos,L,Temp,T0,N):
    """
    Pressure function. Takes positions, amount of particles, box dimensions, 
    temperature and initialized temperature as input. Calculates the pressure 
    of the system using virial theorem. Returns pressure value.
    """
    dx,dy,dz,dr = rmin(pos[:][:,0],pos[:][:,1],pos[:][:,2],L)
    fx,fy,fz = LJforce(pos[:][:,0],pos[:][:,1],pos[:][:,2],L)
    Pressure = 1 + (T0/Temp)/6/N*np.sum(np.sum(dx*fx+dy*fy+dz*fz,axis=1),axis=0)
    return Pressure

def FCC(Nt, n, L):
    """
    Sets initial FCC positions and creates position matrix.
    """
    k = 0
    pos = np.zeros(shape=(Nt, N(n), 3), dtype=float)
    #Creating cubic lattice paramters
    dH = L/n
    #define position array
    for z in range(0,n):
        if z % 2 == 0:
            for x in range(0,n):
                if x % 2 == 0:
                    for y in range(0,n,2):
                        pos[0][k,:] = x, y, z
                        k += 1
                else:
                    for y in range(1,n,2):
                        pos[0][k,:] = x, y, z
                        k += 1
        else:
            for x in range(0,n):
                if x % 2 == 0:
                    for y in range(1,n,2):
                        pos[0][k,:] = x, y, z
                        k += 1
                else:
                    for y in range(0,n,2):
                        pos[0][k,:] = x, y, z
                        k += 1
    pos[0] = pos[0]*dH + dH/2
    return pos

def systemsolver(algorithm, Nt, h, L, pos, vel, Tint, T0, lthreshold = False):
    """
    Calculates the time evolution of the system using the selected algorithm
    using all relevent paramters.
    """
    Np = pos.shape[1]
    
    if lthreshold == False:
        lthreshold = Nt
    ForceM = np.zeros(shape=(Nt, Np, 3), dtype=float).T
    Energies = np.zeros(shape=(Nt,2),dtype=float)
    
    T = np.zeros((Nt-1,1))
    P = np.zeros((Nt-1,1))
    lamb = np.zeros((Nt,1))    
    vels = np.zeros((Nt,Np,3))
    if algorithm == "Verlet":
        """
        Verlet algorithm
        """
        ForceM_old = np.array([0, 0, 0])
        for i in range(0,Nt-1):
            pos[i+1] = (pos[i] + h*vel +(h**2/2)*ForceM_old) % L
             
            F  = LJforce(pos[i+1][:,0],pos[i+1][:,1],pos[i+1][:,2],L)
            ForceM = np.array([np.sum(F[0],axis = 1),np.sum(F[1],axis = 1),np.sum(F[2],axis = 1)]).T
            
            Energies[i] = Energy(pos[i][:,0], pos[i][:,1], pos[i][:,2], vel, L)
            
            if i < lthreshold:
                lamb[i+1] = np.sqrt((Np-1)*3/2*Tint/T0/Energies[i,0])
            else:
                lamb[i+1] = 1
            vel *= lamb[i+1]
            vel += (h/2)*(ForceM+ForceM_old)
        
            ForceM_old = ForceM
            T[i] = Energies[i,0]/(Np-1)*T0/3*2
            P[i] = pressure(pos[i],L,T[i],T0,Np)

    elif algorithm == "Leapfrog":
        """
        Leapfrog algorithm
        """
        for i in range(0,Nt-1):
            F  = LJforce(pos[i][:,0],pos[i][:,1],pos[i][:,2],L)
            ForceM = np.array([np.sum(F[0],axis = 1),np.sum(F[1],axis = 1),np.sum(F[2],axis = 1)]).T
            vel += h*ForceM
            pos[i+1] = (pos[i] + h*vel) % L
            
            Energies[i+1] = Energy(pos[i+1][:,0], pos[i+1][:,1], pos[i+1][:,2], vel, L)
            if i < lthreshold:
                lamb[i+1] = np.sqrt((Np-1)*3/2*Tint/T0/Energies[i+1,0])
            else:
                lamb[i+1] = 1
            lamb[i+1] = np.sqrt((Np-1)*3/2*Tint/T0/Energies[i+1,0])
            vel *= lamb[i+1]
            T[i] = Energies[i,0]/(Np-1)*T0/3*2
            P[i] = pressure(pos[i],L,T[i],T0,Np)
            status = str(i)+'/'+str(Nt)
            print("Status: ", status)
            
            vels[i] = vel
    else:
        print("Incorrect algorithm input")
    


    return pos, T, P, Energies, vels
    
def N(n):
    "Calculates amount of particles for given amount of FCC cells."
    N = int(n**3/2)
    return N


def Terror(vel, T0):
    """
    Calculation of the error in temperature.
    """
    T = vel.shape[0]
    N = vel.shape[1]
    
    rd = np.random.randint(0, N, N**2).reshape(N, N)
    stds = np.zeros([T-1,1])
    means = np.zeros([T-1,1])
    
    for i in range(0,T-1):
        velxyz = vel[i]
        vel2 = np.sum(velxyz**2, axis=1)
        vsum = np.sum(vel2[rd], axis = 1)
        tmean = vsum/(N-1)*T0/3
        stds[i] = np.std(tmean)
        means[i] = np.mean(tmean)
    return stds,means

def v_int(Np, Tint, T0):
    """
    Sets initial velocities based on normal distribution.
    """
    vel = np.zeros(shape=(Np, 3), dtype=float)
    vel = np.random.normal(0, np.sqrt(3*(Tint/T0)),3*Np).reshape(Np,3)
    vel -= np.mean(vel, axis=0)
    return vel

def report(T, P, E, N):
    Nhalf = int(N/2)
    Tmean = np.mean(T[Nhalf:])
    Pmean = np.mean(P[Nhalf:])
    Epotmean = np.mean(E[Nhalf:,1])
    
    print("Mean temperature: %.2f" % Tmean)
    print("Mean pressure: %.2f" % Pmean)
    print("Mean potential energy: %.2f" % (Epotmean/N))
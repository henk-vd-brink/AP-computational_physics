import numpy as np
import matplotlib.pyplot as plt
import functions

def EnergyTP(E, T, T0, vel, P, N):
    """
    
    """
    
    Ekin = E[5:-1,0]/N
    Epot = E[5:-1,1]/N
    Esum = np.sum(E[5:-1,:],axis=1)/N
    Tsig,b  = functions.Terror(vel, T0)
    
    f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex = True)
    ax1.plot(Ekin,label = 'Kinetic energy')
    ax1.plot(Epot,label = 'Potential energy')
    ax1.plot(Esum,label = 'Sum')
    ax1.set_ylabel('Energy [\u03B5]')
    ax1.set_ylim([1.5*min(Epot),-2.5*min(Epot)])
    ax1.legend(loc = 9,ncol=3)
    
    ax2.fill_between(np.arange(0,len(T)-6), (T+Tsig)[5:-1, 0], (T-Tsig)[5:-1, 0], color = 'r', alpha = 0.5)
    ax2.plot(T[5:-1], label = 'Temperature [K]')
    ax2.set_ylim([np.mean(T)*0.9,np.mean(T)*1.1])
    ax2.set_ylabel('Temperature [K]')
    
    ax3.plot(P[5:-1], label = 'Pressure [\u03B2/\u03C1]')
    ax3.set_ylim([0,2*P[-1]])
    ax3.set_ylabel('Pressure [\u03B2/\u03C1]')
    ax3.set_xlabel('Timestep [h]')
    plt.show()

def pathplot(pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    for i in range(0,pos.shape[1],20):
        ax.scatter(pos[::10,i][:,0], pos[::10,i][:,1], pos[::10,i][:,2])
    plt.show()  
    
def veldist(vel,T,T0):
    plt.figure()
    speed = np.sqrt(np.sum(vel**2,axis = 1))
    velx = np.linspace(0,10,100)
    speedf = np.pi*4*(T0/2/np.pi/T[-1])**(3/2)*velx**2*np.exp(-velx**2*T0/(2*T[-1]))
    plt.hist(speed,bins = 41,normed = True, label = "Velocity histogram")  
    plt.plot(velx,speedf,label = "Maxwell-Boltzmann PDF")
    plt.xlabel("Speed")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    
def diff(pos):
    MSD = np.mean(abs(pos[:,:,0]-pos[0,:,0])**2,axis=1)
    time = np.arange(1, len(MSD)+1)
    D = MSD/time/6
    plt.figure()
    plt.plot(D)
    plt.ylabel('Diffusivity [\u03C3^2/h]')
    plt.xlabel('Timestep [h]')
    plt.show()

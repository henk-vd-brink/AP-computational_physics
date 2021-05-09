import numpy as np
import matplotlib.pyplot as plt
from parameters import Nt, W, L, a, thetadev, p2, h, p
from functions import forces, neighbours, torques, circles

pos = np.zeros((Nt, W*L, 2))

i = 0
for x in np.arange(L):
    for y in np.arange(W):
        pos[0, i, :] = a*[x, y]
        i += 1
ori = (np.ones(W*L)*(np.random.rand(W*L)*2 - 1)*thetadev)
velx = np.cos(ori)
vely = np.sin(ori)

Rmean = a
Rdev = a/10

radii = np.random.normal(a, a/10, size = ori.shape)

def closeangle(pos, ori, i):
    x = pos[i, :, 0]
    y = pos[i, :, 1]
    dx = x.T-x.reshape(x.shape[0], 1)
    dy = y.T-y.reshape(y.shape[0], 1)
    dr2 = dx**2 + dy**2
    dr2mask = (dr2>0) & (dr2<p2)
    orirep = np.tile(ori, (ori.shape[0],1))
    valid = np.ma.array(orirep, mask=~dr2mask)
    newangle = valid.mean(axis = 1).data*(np.random.rand(1, ori.shape[0])*0.1+0.95)[0]
    return newangle

plt.close("all")
fig, ax  = plt.subplots()
ax.set_aspect('equal')

def solver(pos, ori):
    for i in range(1, Nt):
        bisec, thetaout, dx, dy, dr2 = neighbours(pos, ori, i-1)
        
        T = torques(ori, dr2, bisec, thetaout)
        F = forces(thetaout, ori, dr2, dx, dy, radii)
        
        ori += h**2/2*T
        velx = np.cos(ori)/10
        vely = np.sin(ori)/10
        
        pos[i] = (pos[i-1] + h*np.array([velx,vely]).T +(h**2/2)*F.T)
        
        
        ax.cla()
        ax.set_aspect('equal')        
#        pos[i] = pos[i-1] + np.array([velx, vely]).T
       
        color = np.ones(100)*0.1
        color[thetaout > np.pi] = 0.2
        
        circles(pos[i,:, 0], pos[i, :, 1], radii, c = color, alpha = 1)
        circles(pos[i,:, 0], pos[i, :, 1], np.ones(radii.shape)*p, c = color, alpha = 0.2)

#        plt.scatter(pos[i,:, 0], pos[i, :, 1])
        plt.quiver(pos[i, :, 0], pos[i ,: ,1], velx, vely)
        print(i)
        plt.xlim((-20, 20))
        plt.ylim((-20, 20))
        plt.show()
        plt.pause(0.001)

solver(pos, ori)
i=Nt-1

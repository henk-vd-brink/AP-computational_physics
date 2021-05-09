import numpy as np
#Physical constants
kbt = 1e+2
epsilon = 0.25
sigma = 0.8

#setting amount of different angles
n = 6
theta = 2*np.pi/n*np.arange(n)

#Max chain length
N = 360

#iterations
k = 100
import numpy as np
from parameters import epsilon, sigma, kbt, theta, k, N
from scipy import stats
import matplotlib.pyplot as plt

def ljpot(roptions, index, xy):
    """
    Inputs:
        roptions - potential positions for next bead      
        index - bead number in the chain
        xy - coordinates of all beads        

    Lennard Jones potential function. Calculates total energy of given 
    positions.
    """
    Vjtot = np.zeros(len(roptions))
    for j in range(len(roptions)):
        deltaxyj = xy[0:index] - roptions[j]
        r2 = np.sum(deltaxyj**2, axis = 1)
        Vj = 4*epsilon*((sigma**2/r2)**6 - (sigma**2/r2)**3)
        Vjtot[j] = np.nansum(Vj)
    return Vjtot

def mcweight(ej):
    """
    Inputs:
        ej - Lennard Jones potential energies of positions j
        
    Attributes Monte Carlo weights to positions. Rescales energies if exponent
    gets too large. Rolls a random number and assigns it into one of the bins
    which correspond to the Boltzmann factor of the energies ej.
    """
    if max(ej/kbt > 100):
        ej *= 10**(-np.log10(max(ej/kbt)/100))
    wj = np.exp(-ej/kbt)
    w = np.sum(wj)
    if w == 0:
        return False
    else:
        randnum = w*np.random.rand()
        j = next(x[0] for x in enumerate(np.cumsum(wj)) if x[1] > randnum)
        return j

def neighbourhood(index, xy, roptions):
    """
    Inputs:
        index - bead number in the chain
        xy - coordinates of all beads
        roptions - potential positions for next bead
        
    For the last bead in the chain it is checked which other beads lie within
    or on a circle of radius sqrt(1.25). For each of these points, the checker 
    function is applied to check if the lines between adjacent points will 
    intersect the lines of the new possible positions. If this is the case, 
    these positions are removed from the set of valid potential points. 
    """
    
    xydis = xy[:index]-xy[index] 
    xy2 = np.sum(xydis**2, axis = 1)
    close = [j for j in range(len(xy2)) if xy2[j] <= 1.25]
    close.remove(index-1)   #remove the second to last point,
                            #since you cannot intersect this line
    valid = list(range(len(roptions)))
    for i in close:
        for j in range(len(roptions)):
            a = xy[i]
            b1 = xy[i+1]
            b2 = xy[i-1]
            c = xy[index]
            d = roptions[j]
            if checker(a,b1,c,d):
                if j in valid:
                    valid.remove(j)
            elif checker(a,b2,c,d) and i > 0:
                if j in valid:
                    valid.remove(j)
    return valid

def direction(a, b, c):
    """
    Inputs:
        - points a, b, c
        
    Checks what the orienation of 3 given points is.
    Output correspond to colinear, anti-clockwise or clockwise.
    """
    val = (b[1]-a[1])*(c[0]-b[0])-(b[0]-a[0])*(c[1]-b[1])
    if val == 0:
        return 1    #a, b, c are colinear
    elif val < 0:
        return 2    #a, b, c are anti-clockwise
    else:
        return 3    #a, b, c are clockwise
    
def checker(a, b, c, d):
    """
    Inputs:
        - points a, b, c d
        
    Checks if the linesegments between a and b, and c and d intersect. Calls
    the direction function to check for orientation. Operates on the assumption
    that starting/end points of the line segments are never exactly on the
    other line segment, which 
    """
    dir1 = direction(a, b, c)
    dir2 = direction(a, b, d)
    dir3 = direction(c, d, a)
    dir4 = direction(c, d, b)
    
    if dir1 is not dir2 and dir3 is not dir4:
        return True
    else:
        return False



def chainmaker(chains = None, k = None, flipping = True):
    """
    Inputs:
        - numpy array for storing chain coordinates
        - chain iteration k index
        - boolean for flipping
    
    Function used to produce k chains with a maximum length N. Calculates the
    position to place the next bead based on weights determined by Lennard-Jones
    potential and Monte Carlo methods. Flips the chain and starts building at 
    the start again if one side gets stuck. 
    """
    print("making chain %d" % k)
    xy = np.zeros((N, 2))
    xy[0] = [0, 0]
    xy[1] = [1, 0]
    
    i = 2
    flipped = False
    while i < len(xy):
        theta0 = 2*np.pi*np.random.rand()
        r0 = xy[i-1]
        roptions = r0 + np.transpose(np.array([np.sin(theta+theta0),
                                               np.cos(theta+theta0)]))
    
        if i > 2:
            valid = neighbourhood(i-1, xy, roptions)
            roptions = roptions[valid]
        if len(roptions) == 0:
            if flipping == True:
                if flipped == True:
                    xy=xy[:i]
                    break
                flipped = True
                xy[0:i] = np.flip(xy[0:i])
                continue
            else:
                xy=xy[:i]
                break
        ej = ljpot(roptions, i, xy)
        j = mcweight(ej)
    
        if type(j) == bool:
            xy=xy[:i]
            break
        xy[i] = roptions[j]
        i += 1
    beads = len(xy)
    length = np.sqrt(np.sum((xy[-1]-xy[0])**2))
        
    if chains is not None:
        chains[k, :len(xy), :] = xy
    return beads, length

def data(storechains = True, flipping = True):
    """
    Inputs:
        - Boolean for storing chain coordinate data or not
        - Boolean for flipping the chain if it gets stuck on one end or not
    
    Stores chain data generated by chainmaker function in lists and numpy
    array of chain coordinates. 
    """
    if storechains == True:
        global chains
        chains = np.zeros((k, N, 2))
        beads, lengths = zip(*[chainmaker(chains, i, flipping) for i in range(k)])
        return list(beads), list(lengths), chains
    else:
        beads, lengths = zip(*[chainmaker(flipping) for i in range(k)])
        return list(beads), list(lengths)
    
def e2edis(chains):
    """
    Inputs:
        - A set of polymer chains with size N
    
    For a given set of chains, this function calculates the end to end length
    of the subchains within a chain from point 0 to j. From this, the mean
    length of chains with any number of beads up to the maximum amount is
    calculated.
    """
    chains = np.where(chains[:,2:]!=0, chains[:,2:], np.nan)
    chains0 = np.repeat(chains[:,0,:], chains.shape[1],
                        axis=0).reshape(chains.shape)
    r2 = np.sum((chains - chains0)**2, axis=2)
    r2std = np.nanstd(r2, axis=0)/np.sqrt(np.sum(~np.isnan(r2), axis=0))
    r2std = r2std[~np.isnan(r2std)]
    
    r2mean = np.nanmean(r2, axis=0)
    r2mean = r2mean[~np.isnan(r2mean)]
    return r2mean, r2std, r2

def linefit(beads, r2mean):
    """
    Inputs:
        - Array with bead numbers
        - Array with mean lengths squared
        
    Calculates parameter a for the function y = a*(N-1)^1.5. Takes inputs of 
    number of beads N and corresponding end to end lengths y.
    """
    a = np.sum(r2mean*(beads-1)**1.5, )/np.sum((beads-1)**3, dtype=float)
    return a
    
def weibfit(beads):
    """
    Inputs:
        - Number of beads in each chain
    
    Fits a Weibull PDF to a given set of total beads in each chain.
    """
    beads = np.array(beads)
    a,b,c,d=stats.exponweib.fit(beads, 1, 1, loc=0)
    x = np.arange(int(max(1.5*beads)))
    exp = int(np.log10(kbt))
    plt.plot(x, stats.exponweib.pdf(x, a, b, scale = d),
             label = 'kbT = 10^%d: a = %.2f, c = %.2f' % (exp, a, b))

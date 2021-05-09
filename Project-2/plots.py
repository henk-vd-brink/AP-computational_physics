import numpy as np
import matplotlib.pyplot as plt
import functions
from parameters import kbt

def lengthplot(chains):
    """
    Inputs:
        - Array containing coordinates of generated chains
        
    Calls function to calculate the end to end distance and standard deviation
    for each amount of beads. Plots mean distance vs. bead number with
    corresponding standard deviation.
    """
    r2mean, r2std, r2 = functions.e2edis(chains)
    N = np.arange(r2mean.shape[0]) + 1
    a = functions.linefit(N, r2mean)
    y = a*(N-1)**1.5

    exp = int(np.log10(kbt))
    plt.figure()
    plt.title("kbT = 10^%d" % exp)
    plt.loglog(N, y, c = 'r', linestyle = '--', label = 'a = %.2f' % a)
    plt.xlim((2, 2*max(N)))
    plt.ylim((1, 2*max(r2mean)))
    plt.errorbar(N, r2mean, r2std,
                 linestyle = '', marker = 'o', markersize = 1, capsize = 2)
    plt.ylabel("<R²>")
    plt.xlabel("Polymer beads")
    plt.legend()
    plt.show()

    string = "fit1e"+str(exp)+"K.pdf"
    plt.savefig(string)


def chainplot(beads, chains, interval = np.arange(4)):
    """
    Inputs:
        - Amount of beads in each chain
        - Array with chain coordinates
        - Array with chain numbers to plot
    """
    for i in interval:
        plt.figure()
        plt.title("Chain %d" % (i+1))
        plt.plot(chains[i, :beads[i], 0], chains[i, :beads[i], 1])
        plt.scatter(chains[i, :beads[i], 0], chains[i, :beads[i], 1],color = 'k')
        txtstr = [str(i+1) for i in range(beads[i])]
        for j in range(beads[i]):
            plt.text(chains[i, j, 0], chains[i, j, 1], txtstr[j])
def beadhist(beads):
    """
    Inputs:
        - Array with number of beads per chain
    
    Generates a plot showing the frequency of chains of different lengths
    occuring.
    """
    
    exp = int(np.log10(kbt))
    plt.figure()
    plt.title("kbT = 10^%d" % exp)
    plt.hist(beads, bins = int(max(beads)/10), normed = True)
    plt.xlabel("Number of beads")
    plt.ylabel("Probability")
    functions.weibfit(beads)
    plt.legend()
    plt.show()
    string = "hist1e"+str(exp)+"K.pdf"
    plt.savefig(string)

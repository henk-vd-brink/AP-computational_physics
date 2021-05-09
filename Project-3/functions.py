import numpy as np
import matplotlib.pyplot as plt
from parameters import k, p2
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def forces(thetaout, ori, dr2, dx, dy, radii):
    
    dr = np.sqrt(dr2)
    mask = (dr2 < p2) & (dr2 > 0)
    Rsum = radii+radii.reshape(radii.shape[0],1)
    Frepx = k*(1-Rsum/abs(dr)*dx)
    Frepx[Frepx > 0] = 0
    if np.sum(Frepx > 0) > 0:
        print("overlap")
    Frepy = k*(1-Rsum/abs(dr)*dy)
    Frepy[Frepy > 0] = 0
    
    Frepxsum = np.nansum(np.ma.array(Frepx, mask = ~mask), axis=1).data
    Frepysum = np.nansum(np.ma.array(Frepy, mask = ~mask), axis=1).data

#    Fself = 
    Fin = np.ones(ori.shape)
    Fin[thetaout < np.pi] = 0
    Fxb = np.cos(ori)*Fin*(thetaout-np.pi)
    Fyb = np.sin(ori)*Fin*(thetaout-np.pi)
    
    Fx = Frepxsum + Fxb
    Fy = Frepysum + Fyb
    F = np.array([Fx, Fy])
    return F
    
    
def torques(ori, dr2, bisec, thetaout):
    Tbound = np.zeros(ori.shape)
    deltat = bisec-ori
    Tbound[thetaout > np.pi] = deltat[thetaout > np.pi]
    
    Tnoise = np.zeros(ori.shape[0])
#    Tnoise = -1 + 2*np.random.rand(ori.shape[0])
    
    Talign = np.zeros(ori.shape)
    mask = (dr2 < p2) & (dr2 > 0) 
    dpsi = ((ori.T - ori.reshape(ori.shape[0], 1))+np.pi) % 2*np.pi - np.pi
    
    dpsisum = np.nansum(np.ma.array(dpsi, mask = ~mask), axis = 1)
    Talign = dpsisum
    
    Tsum = Tbound + Tnoise + Talign
    Tsum = Tbound + Tnoise

    return Tsum

def neighbours(pos, ori, i):
    x = pos[i, :, 0]
    y = pos[i, :, 1]
    
    dx = x.T - x.reshape(x.shape[0], 1)  
    dy = y.T - y.reshape(y.shape[0], 1) 
    dr2 = dx**2 + dy**2
    dr2mask = (dr2>0) & (dr2<p2)
    thetas = np.ma.array(np.arctan(dy/dx), mask=~dr2mask)
    thetas[dx < 0] += np.pi
    thetain = np.array(np.max(thetas, axis = 1) - np.min(thetas, axis = 1))
    bisec = np.array(np.max(thetas, axis = 1) + np.min(thetas, axis = 1))/2
    thetaout = 2*np.pi-thetain
    
    return bisec, thetaout, dx, dy, dr2

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
   
    

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection
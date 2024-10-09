import numpy as np
def wind(kv, sig_v, sig_vpi, T, n = None, nsims = None, BMwind = None, BMprice = None):
    '''
    Numerical approximation of a realization of a solution to wind stochastic differential equation.
    
    Inputs
        kv: parameter of SDE
        sig_v: Standard deviation of wind
        sig_vpi: Covaraince of wind and price
        T: final time
        n: number of time steps
        BMwind: Pre-generated standard brownian motions, matrix with shape (nsims, n)

    Output: Matrix conatining values of wind speed over time with shape (nsims, n + 1)
    '''
    if type(BMwind) == type(None): #Checks to see if function is using pregenerated Brownian motion
        n = int(n)
        h = T/n
        BMwind = np.random.normal(scale = np.sqrt(h), size = nsims*n).reshape(nsims, n)
        BMprice = np.random.normal(scale = np.sqrt(h), size = nsims*n).reshape(nsims, n) #generates brownian motions
    else:
        n = np.size(BMwind[0,:])
        h = T/n
    sigmav = lambda s: 8*(1 + 0.375*np.sin(np.pi/(12*h)*(s + 2/h))) #functions used in SDE
    dsigmav = lambda s: 8*0.375*np.cos(np.pi/(12*h)*(s + 2/h))*(np.pi/(12*h))
    s = 0 #initial time
    v = np.zeros(shape = (nsims, (n + 1))) 
    v[:,0] = np.random.normal(sigmav(0), 0.226, nsims) #initial wind speed
    for i in range(0, n): #Numerically solves SDE for each simulation
        v[:,i + 1] = v[:,i] + kv*(sigmav(s) + 1/kv*dsigmav(s) - v[:,i])*h + \
            sig_v*v[:,i]*BMwind[:,i] + sig_vpi*v[:,i]*BMprice[:,i]
        s = s + h
    return v
import numpy as np
def wind(kv, sig_v, sig_vpi, T, n = None, BM = None):
    '''
    Numerical approximation of a realization of a solution to wind stochastic differential equation.
    
    Inputs
        kv: parameter
        sig_v: Standard deviation of wind
        sig_vpi: Covaraince of wind and price
        T: Time
        n: numer of steps
        BM: Pre-generated standard brownian motions

    Output: Array conatining values of wind speed over time
    '''
    if type(BM) == type(None): #Checks to see if function is using pregenerated Brownian motion
        n = int(n)
        cov = np.reshape(np.array([1, 0, 0, 1]), (2,2))
        BM = np.random.multivariate_normal([0,0], cov, size=n) #generates brownian motions
    else:
        n = np.size(BM[:,0])
    h = T/n
    sigmav = lambda s: 8*(1 + 0.375*np.sin(np.pi/(12*h)*(s + 2/h))) #functions used in SDE
    dsigmav = lambda s: 8*0.375*np.cos(np.pi/(12*h)*(s + 2/h))*(np.pi/(12*h))
    s = 0
    v = np.zeros(n + 1)
    v[0] = np.random.normal(sigmav(0), 0.226)
    for i in range(0 + 1, n + 1): #Numerically solves SDE
        v[i] = v[i - 1] + kv*(sigmav(s) + 1/kv*dsigmav(s) - v[i - 1])*h + np.sqrt(h)*sig_v*v[i - 1]*BM[i - 1, 0] + np.sqrt(h)*sig_vpi*v[i - 1]*BM[i - 1, 1]
        s = s + h
    return v
import numpy as np

T = 24
h = 1
N = int(T / h)
k_pi = 0.04 / h
sigma_pi_v = 0.01
sigma_pi_pi = 0.075

theta_v_bar = 8
alpha_v = 0.375
gamma = np.pi / (12 * h)
psi_v = 2 / h

def price_sde(T, N, k_pi, sigma_pi_v, sigma_pi_pi, simulation_times=None, wind_BM = None, price_BM = None):

    '''
    Simulation of the price process

    Inputs: 
    T: total time
    N: number of steps
    k_pi: mean reversion rate for price
    sig_pi_v: correlation of electricity price and wind speed
    sig_pi_pi: volatility of the price process
    simulation_times: how many times to simulate the process, default None
    wind_BM: standard brownian motion for the wind power, default None
    price_BM: standard brownian motion for the spot price, default None

    Output:
    A matrix containing the price of electricity at each time point
    '''
    h = T / N
    
    if price_BM is None or wind_BM is None:
        if price_BM is None: 
            price_BM = np.random.normal(0, np.sqrt(h), size = (simulation_times, N))
        
        if wind_BM is None:
            wind_BM = np.random.normal(0, np.sqrt(h), size = (simulation_times, N))
    else:
        h = T / np.size(price_BM[0, :])
    
    theta_v = lambda s: theta_v_bar * (1 + alpha_v * np.sin(gamma*(s + psi_v)))
    dtheta_v = lambda s: theta_v_bar * alpha_v * np.cos(gamma * (s + psi_v)) * gamma

    s = 0 
    price = np.zeros((simulation_times, N + 1))
    price[:, 0] = np.random.normal(theta_v(0), 0.075, simulation_times)
    
    for i in range(0, N):
        price[:, i+1] = price[:, i] + k_pi * (theta_v(s) + 1/k_pi * dtheta_v(s) - price[:, i]) * h \
            + np.sqrt(h) * sigma_pi_v * price[:, i] * wind_BM[:, i] \
            + np.sqrt(h) * sigma_pi_pi * price[:, i] * price_BM[:, i]
        s += h
    return price

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

def price_sde(T, h, k_pi, sigma_pi_v, sigma_pi_pi):

    '''
    Simulation of the price process

    Inputs: 
    T: total time
    h: time step size
    k_pi: mean reversion rate for price
    sig_pi_v: correlation of electricity price and wind speed
    sig_pi_pi: volatility of the price process
    wind_BM: standard brownian motion for the wind power
    price_BM: standard brownian motion for the spot price

    Output:
    A matrix contains the price of electricity at each time point
    '''

    N = int(T/h)
    cov = np.reshape(np.array([1, 0, 0, 1]), (2,2))
    brownianmotion = np.random.multivariate_normal([0,0], cov, size=N)

    theta_v = lambda s: theta_v_bar * (1 + alpha_v * np.sin(gamma*(s + psi_v)))
    dtheta_v = lambda s: theta_v_bar * alpha_v * np.cos(gamma * (s + psi_v)) * gamma

    s = 0 
    price = np.zeros(N + 1)
    price[0] = np.random.normal(theta_v(0), 0.075)
    
    for i in range(1, N + 1):
        price[i] = price[i - 1] + k_pi * (theta_v(s) + 1/k_pi * dtheta_v(s) - price[i - 1]) * h \
            + np.sqrt(h) * sigma_pi_v * price[i - 1] * brownianmotion[i-1, 0] \
            + np.sqrt(h) * sigma_pi_pi * price[i-1] * brownianmotion[i-1, 1]
        s += h
    return price


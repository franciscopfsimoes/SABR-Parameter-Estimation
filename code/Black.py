#Useful Python packages
import math
import numpy as np
from scipy.stats import norm

def D1(S, K, T, r, vol): #Returns value of function d1 in Black-Scholes' pricing formula

    return (1/(vol * np.sqrt(T))) * (np.log(S/K) + (r + (1 / 2) * (vol ** 2)) * T)


def D2(d1, T, vol): #Returns value of function d2 in Black-Scholes' pricing formula

    return  d1 - vol*math.sqrt(T)


def Vega(S, K, T, r, vol, iscall): #Returns the Black-Scholes' greek Vega

    d1 = D1(S, K, T, r, vol)

    if iscall == 2:

        vega = 2 * S * norm.cdf(d1, 0.0, 1.0) * np.sqrt(T)


    else:

        vega = S * norm.cdf(d1, 0.0, 1.0) * np.sqrt(T)

    return vega



def Price(S, K, T, vol, r, iscall): #Returns Black-Scholes' contract price

    d1 = D1(S, K, T, r, vol)

    d2 = D2(d1, T, vol)

    if iscall == 2: #Straddle price

        value = norm.cdf(d1) * S - math.exp(-r * T) * norm.cdf(d2) * K + math.exp(-r * T) * norm.cdf(-d2) * K - norm.cdf(-d1) * S

    elif iscall == 1: #Call price

        value = norm.cdf(d1) * S - math.exp(-r*T) * norm.cdf(d2) * K

    else: #Put price

        value = math.exp(-r * T) * norm.cdf(-d2) * K - norm.cdf(-d1) * S

    return value

def Vol(S, K, T, p, r, iscall): #Returns Black-Scholes' implied volatility using Newton-Raphson's method

#initial conditions of Newton-Raphson's method

    count = 0 #cycle counter

    maxiter= 1000 #maximum number of cycles

    epsilon = 1 #initial value for the error epsilon

    tol = 1e-3 #break condition for the size of epsilon

    vol = 0.5 #initial value for the volatility

    while epsilon > tol: #Newton-Raphson's method

        count += 1 #increase counter

        if count >= maxiter: #break condition for the maximum number of cycles
            break;

        origVol = vol #memory of the cycle's initial value of the volatility

        price = Price(S, K, T, vol, r, iscall) - p  # Difference between Black-Scholes's price of the contract considering volatility v and the market price of the contract

        vega = Vega(S, K, T, r, vol, iscall)  # Black-Scholes' greek Vega (\partial price / \partial v)

        vol = vol - price / vega #Newton-Raphson's step

        epsilon = math.fabs( (vol - origVol) / origVol) #Newton-Raphson's epsilon error after step

    return vol



"""testados contra pricer Black"""

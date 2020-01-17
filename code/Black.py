import math
from scipy.stats import norm

def D1(f, K, T, vol):

    return (1/(vol * math.sqrt(T))) * (math.log(f/K) + (1 / 2) * (vol ** 2) * T)


def D2(d1, T, vol):
    return  d1 - vol*math.sqrt(T)



def blackPrice(f, K, T, vol, r):

    d1 = D1(f, K, T, vol)

    d2 = D2(d1, T, vol)

    value = math.exp(-r*T) * (norm.cdf(d1) * f - norm.cdf(d2) * K)

    return value
"""testado contra pricer Black"""

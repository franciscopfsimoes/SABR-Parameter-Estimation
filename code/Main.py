from Black import blackPrice
import numpy as np
import math
from SABR import impVol
import matplotlib.pyplot as plt

'''
par = input("Do you have initial parameters alpha, beta, rho? (Y or N?)")

if par == 'Y':

    print('Enter the parameters: ')

    alpha = float(input("alpha :"))
    beta = float(input("beta :"))
    rho = float(input("rho :"))

else:

    alpha = 1
    beta = 1
    rho = 1

print('Enter the market data and specifics of your request: ')

f = float(input("Foward price of the underlying :"))
Vv = float(input("Volatility of the volatility :"))
K = float(input("Strike price :"))
T = float(input("Time to maturity :"))
r = float(input("Risk free interest rate :"))
'''

alpha = 0.5
beta = 1
rho = 0.5
f = 5
Vv = 0.5
T = 5
r = 0.05
a = 1
b= 1
'''K = 3

v = impVol(alpha, beta, rho, Vv, K, f, T)'''
K = np.arange(0.1, 20, 0.1)
vol = []

for k in K:
    v = impVol(alpha, beta, rho, Vv, k, f, T)
    vol.append(v)
    
plt.plot(K/f, vol)
plt.show()

'''volB = impVol(alpha, beta, rho, Vv, K, f, T)
p = blackPrice(f, K, T, volB, r)'''

'''print('The price of the call is: ', p)'''

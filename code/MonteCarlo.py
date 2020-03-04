#Useful Python packages
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

#Pre-processing/misc algorithms
import ExcelDate as date
import DataPreProcessing as dt

#Model specific algorithms
import Black
import SABR
import Estimating
import MonteCarloSparseIntraday as Generate

# Initial conditions
r = 0.05 #interest free rate (set to LIBROR rate at the time of the training data)

Generate.Simulation(1)

#Preprocessing data
arr = dt.preprocess("DayData.txt")

spot = np.asarray(list(map(float, arr[:,0]))) #Spot price of the underlying
strike = np.asarray(list(map(float, arr[:, 1]))) #Strike price of the option
duration = np.asarray(list(map(float, arr[:,2]))) #Duration of the contract (in Excel1900)
premium = np.asarray(list(map(float, arr[:, 3]))) #Market price of the contract
iscall = np.asarray(list(map(int, arr[:, 4]))) #type of product
vol = np.asarray(list(map(float, arr[:, 5]))) #Intrinsic Volatility Array

#Auxilary Arrays
S0K = [] #S0/K Array
fprice = []  # foward price


for i in range(len(spot)):

    # S0/K of the ith contract
    s = spot[i]/strike[i]
    S0K.append(s)

    # foward price of the stock of the ith contract
    f = spot[i] * math.exp(r*duration[i])

    fprice.append(f)


beta = 1

optarv = Estimating.ARV(beta, strike, fprice, duration, vol)

sabrvol = []
K = []

alphaopt = optarv[0]

rhoopt = optarv[1]

Vvopt = optarv[2]

for i in range(len(spot)):

    vi = SABR.impVol(alphaopt, beta, rhoopt, Vvopt, strike[i], fprice[i], duration[i])

    pi = Black.Price(spot[i], strike[i], duration[i], vi, r, iscall[i])

    err = math.sqrt((vi - vol[i])**2)/vol[i] * 100

    print("So your SABR volatility for the number ", i, " option is ", vi, ", while the actual volatility is ", vol[i], ", the error is ", round(err, 1), "%")

    if iscall[i] == 0:

        plt.plot(strike[i], vol[i], 'ro')


    elif iscall[i] == 1:

        plt.plot(strike[i], vol[i], 'bs')

    else:

        plt.plot(strike[i], vol[i], 'g^')

for k in np.arange(100, 3000, 1):
    vi = SABR.impVol(alphaopt, beta, rhoopt, Vvopt, k, fprice[0], duration[0])
    sabrvol.append(vi)
    K.append(k)

plt.plot(K, sabrvol, label='SABR')

plt.legend()
plt.ylabel('Intrinsic Volatility')
plt.xlabel('Strike Price')
plt.show()
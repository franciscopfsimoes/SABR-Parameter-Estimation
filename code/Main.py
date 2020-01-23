#Useful Python packages
import matplotlib.pyplot as plt
import numpy as np
import math

#Pre-processing/misc algorithms
import ExcelDate as date
import DataPreProcessing as dt

#Model specific algorithms
import Black
import SABR

# Initial conditions
r = 0.019 #interest free rate (set to LIBROR rate at the time of the training data)
iscall = 0 #type of options used for training (0 = put; 1 = call)

#Preprocessing data
arr = dt.preprocess("Data.txt")

spot = list(map(int, arr[:,0])) #Spot price of the underlying
matu = list(map(int, arr[:,1])) #Maturity date of the contract (in Excel1900)
strike = list(map(int, arr[:, 2])) #Strike price of the option
initime = list(map(int, arr[:, 5])) #Inital date of the contract (in Excel1900)
premium = list(map(float, arr[:, 8])) #Market price of the contract

#Auxilary Arrays
vol = [] #Volatility Array
S0K = [] #S0/K Array

for i in range(len(spot)):

    duration = (matu[i] - initime[i])/365 #Duration of the ith contract (in years)

    # Black's Implied Volatility of the ith contract
    v = Black.Vol(spot[i], strike[i], duration, premium[i], r, iscall)
    vol.append(v)

    # S0/K of the ith contract
    x = spot[i]/strike[i]
    S0K.append(x)

plt.plot(S0K, vol, 'ro')
plt.show()
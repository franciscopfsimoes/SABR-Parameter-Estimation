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

# Initial conditions
r = 0.019 #interest free rate (set to LIBROR rate at the time of the training data)
iscall = 0 #type of options used for training (0 = put; 1 = call)

#Preprocessing data
arr = dt.preprocess("Data.txt")
a = np.asarray(arr)

type = list(map(str, arr[:,0])) #type of product
spot = np.true_divide(np.asarray(list(map(int, arr[:,1]))),1000) #Spot price of the underlying
matu = np.asarray(list(map(int, arr[:,2]))) #Maturity date of the contract (in Excel1900)
strike = np.true_divide(np.asarray(list(map(int, arr[:, 3]))),1000) #Strike price of the option
initime = np.asarray(list(map(int, arr[:, 6]))) #Inital date of the contract (in Excel1900)
premium = np.true_divide(np.asarray(list(map(float, arr[:, 9]))),1000) #Market price of the contract

#Auxilary Arrays
vol = [] #BlackVolatility Array
S0K = [] #S0/K Array
fprice = []  # foward price
duration = [] #duration of the contract
iscall = [] #type of options: put(0), call(1), straddle(2)

lvNtm = []  # log Near-The-Money volatility
lfNtm = []  # log Near-The-Money foward price

for i in range(len(spot)):

    if type[i] == "European Put":
        iscall.append(0)

    elif type[i] == "European Call":
        iscall.append(1)

    elif type[i] == "European Straddle":
        iscall.append(2)

    d = (matu[i] - initime[i])/365 #Duration of the ith contract (in years)
    duration.append(d)


    # Black's Implied Volatility of the ith contract
    v = Black.Vol(spot[i], strike[i], duration[i], premium[i], r, iscall[i])
    vol.append(v)

    # S0/K of the ith contract
    s = spot[i]/strike[i]
    S0K.append(s)

    # foward price of the stock of the ith contract
    f = spot[i] * math.exp(r*duration[i])

    fprice.append(f)

'''
########################################Estimating beta########################################
    if s < 1.1: #Our Near-The-Money condition (should be a bit tighter)

        lV = math.log(v)

        lvNtm.append(lV)

        lF = math.log(f)

        lfNtm.append(lF)

#prepare log(f) and log(vol) Near-the-money arrays for linear regression
x = np.array(lfNtm).reshape((-1, 1))
y = np.array(lvNtm)

model = LinearRegression().fit(x, y) #Linear regression


beta = model.coef_ + 1 #beta as the slope of the linear regression model
'''

beta = 0.5 #until I get better data

######################################Estimating alpha, rho and Vv############################

alphaub = 30 #alpha upper bound
alphalb = 0.1 #alpha lower bound
alphastep = 1  #alpha increment step

rhoub = 0.9 #rho upper bound
rholb = -0.9 #rho lower bound
rhostep = 0.1 #rho increment step

Vvub = 50 #Vv upper bound
Vvlb = 0 #Vv lower bound
Vvstep = 1 #Vv incrment step


i = 0
opt = float("inf")

alphaopt=0
rhoopt=0
Vvopt =0

for alpha in np.arange(alphalb, alphaub, alphastep):

    for rho in np.arange(rholb, rhoub, rhostep):

        for Vv in np.arange(Vvlb, Vvub, Vvstep):

            sum = 0

            for i in range(len(vol)):

                est = SABR.impVol(alpha, beta, rho, Vv, strike[i], fprice[i], duration[i])

                dif = (est - vol[i])**2

                sum = sum + dif

            if sum <= opt:

                opt = sum

                alphaopt = alpha

                rhoopt = rho

                Vvopt = Vv


sabrvol = []
K = []

print("your optimal parameters are alpha:", alphaopt, " rho:", rhoopt, " Vv:", Vvopt)

for i in range(len(spot)):

    vi = SABR.impVol(alphaopt, beta, rhoopt, Vvopt, strike[i], fprice[i], duration[i])

    pi = Black.Price(spot[i], strike[i], duration[i], vi, r, iscall[i])

    err = math.sqrt((pi - premium[i])**2)/premium[i] * 100

    print("So your price for the number ", i, " option is ", pi, ", while the actual price is ", premium[i], ", the error is ", round(err, 1), "%")

    if iscall[i] == 0:

        plt.plot(strike[i], vol[i], 'ro')


    elif iscall[i] == 1:

        plt.plot(strike[i], vol[i], 'bs')

    else:

        plt.plot(strike[i], vol[i], 'g^')

for k in np.arange(1, 50, 0.1):
    vi = SABR.impVol(alphaopt, beta, rhoopt, Vvopt, k, fprice[i], duration[i])
    sabrvol.append(vi)
    K.append(k)

plt.plot(K, sabrvol, label='SABR')

plt.legend()
plt.ylabel('Implied Volatility')
plt.xlabel('Strike Price')
plt.show()
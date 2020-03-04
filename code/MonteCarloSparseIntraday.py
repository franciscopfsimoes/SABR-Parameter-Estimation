#Useful Python packages
import numpy as np
import math
import matplotlib.pyplot as plt
import random

import Black

def Simulation(num):

    T = 90 / 365


    alpha, beta, rho, Vv = 0.5, 1, -0.4, 20

    i = 0

    ti = 0

    f = open("DayData.txt", "w+")

    while i < num:

        Day(alpha, beta, rho, Vv, T, f)

        i = i + 1

        ti = ti + 0.00273790926 #increment one day

        if ti >= T:

            print('The number of days excedes the strike')

            break

    f.close()



def Day(alpha, beta, rho, Vv, T, f):

    dt = 1.90132588 * 10 ** (-5)  # 10 min in years

    S0 = 1000

    vol = []
    vol.append(alpha)
    spot = []
    spot.append(S0)
    price = []

    mean, stddev = 0, dt  # mean and standard deviation

    mu = 0.05

    p = 1

    z, e = 0, 0

    f.write("Spot Price\tStrike Price\tDuration\tPremium\tIsCall\tImplied Volatility\n")

    for it in range(0, 45, 1):

        #########################Simulate Stock Price and Volatility#####################

        dz = float(np.random.normal(mean, stddev, 1))

        z = z + dz

        dv = vol[it] * Vv * z

        v = vol[it] + dv

        vol.append(v)

        e = rho * z + math.sqrt(1 - rho**2) * float(np.random.normal(mean, stddev, 1))

        dS = mu * dt * spot[it] + math.exp((-mu*T)*(1-beta))* v * spot[it]**(beta) * e

        S = spot[it] + dS

        spot.append(S)

        #plt.plot(t, S, 'ro')

        ###########################Simulate Option Quote #########################

        if random.random() < p:

            K = float(np.random.normal(mean, 0.3, 1)) * S + S

            if random.random() < 0.5: # call

                iscall = 1


            else:

                iscall = 0

            premium = Black.Price(S, K, T, v, mu, iscall)
            price.append(premium)

            f.write("%f\t%f\t%f\t%f\t%d\t%f\n" % (S, K, T, premium, iscall,v))


'''
    plt.legend()
    plt.ylabel('Spot Price')
    plt.xlabel('Time')
    plt.show()
'''

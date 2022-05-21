#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy
import scipy.stats
from methods import black, sabr, estimating
from classes.Derivative import *
def foward(S, mu, T):
    f = float(S) * math.exp(mu * T)
    return f


def corrNum(rho):
    z1 = random.gauss(0, 1)
    z2 = rho * z1 + math.pow((1.0 - math.pow(rho, 2.0)), 0.5) * random.gauss(0, 1)
    return (z1, z2)


def sabrpathSim(num_steps, T, f0, alpha, beta, rho, Vv):
    dt = float(T) / float(num_steps)
    sqrtdt = float(math.sqrt(dt))
    f = []
    vol = []
    f.append(f0), vol.append(alpha)
    ft = f0
    alphat = alpha
    step = 1
    while step < num_steps:
        z = corrNum(rho)
        dWf = float(z[0]) * sqrtdt
        dWa = float(z[1]) * sqrtdt
        f_b = math.pow(abs(ft), beta)
        ft = ft + alphat * f_b * dWf
        alphat = alphat + alphat * Vv * dWa
        f.append(ft)
        vol.append(alphat)
        step += 1
    return (f, vol)  # returns paths as lists


def sabrfowardSim(num_steps, T, f0, alpha, beta, rho, Vv):
    dt = float(T) / float(num_steps)
    sqrtdt = float(math.sqrt(dt))
    ft = f0
    alphat = alpha
    step = 0
    while step < num_steps:
        z = corrNum(rho)
        ft = ft + alphat * math.pow(abs(ft), beta) * float(z[0]) * sqrtdt
        alphat = alphat + alphat * Vv * float(z[1]) * sqrtdt
        step += 1
    return ft


def pathPlot(num_steps, path):
    t = 0
    while t < num_steps:
        ax1 = plt.subplot(211)
        plt.plot(t, path[0][t], "bs")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(t, path[1][t], "bs")
        plt.setp(ax2.get_xticklabels(), fontsize=(12))
        t += 1
    plt.show()


def randStrike(f0):
    dK = float(np.random.normal(0, 0.30, 1)) * f0
    K = f0 + dK
    return K


def intervalStrike(f0, num_quotes):
    dK = float(2.0 * f0 / (num_quotes - 1))
    K = []
    i = 0
    while i < num_quotes:
        K.append(0.5 * f0 + i * dK)
        i += 1
    return K


def randputOrCall():
    if random.random() < 0.5:  # call
        iscall = 1
    else:
        iscall = 0
    return iscall


def dynamicQuotes(T, f0, alpha, beta, rho, Vv, num_quotes, time):
    num_steps = 200
    interval = time / num_quotes
    f = []
    vol = []
    duration = []
    strike = []
    type = []
    i = 0
    while i < num_quotes:
        path = sabrpathSim(num_steps, interval, f0, alpha, beta, rho, Vv)
        f0 = path[0][num_steps - 1]
        alpha = path[1][num_steps - 1]
        f.append(f0)
        vol.append(alpha)
        duration.append(T - (i + 1) * interval)
        strike.append(randStrike(f0))
        type.append(randputOrCall())
        i += 1
    return f, vol, duration, strike, type


def instaTestQuotes(derivative, alpha, beta, rho, Vv, num_quotes):
    f = []
    vol = []
    duration = []
    strike = []
    type = []
    k = intervalStrike(derivative.f0, num_quotes)
    i = 0
    while i < num_quotes:
        f.append(derivative.f0)
        vol.append(alpha)
        duration.append(derivative.T)
        strike.append(k[i])
        type.append(0)
        f.append(derivative.f0)
        vol.append(alpha)
        duration.append(derivative.T)
        strike.append(k[i])
        type.append(1)
        i += 1
    return f, vol, duration, strike, type


def valueAtMaturity(f, K, type):
    if type == 0:
        payoff = max(K - f, 0)

    else:
        payoff = max(f - K, 0)

    return payoff


def confidenceInterval(list, confidence):
    n = len(list)
    m = np.mean(list)
    std_err = scipy.stats.sem(list)
    h = std_err * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    start = m - h
    end = m + h
    return start, end


def expectedValuation(f0, alpha, duration, strike, type, beta, rho, Vv, num_simulations):
    i = 0
    num_steps = 100
    p = []
    while i < num_simulations:
        f = sabrfowardSim(num_steps, duration, f0, alpha, beta, rho, Vv)
        payoff = valueAtMaturity(f, strike, type)
        p.append(payoff)
        i += 1
    P = list(p)
    price = np.average(P)
    return price


def getPrice(quote, beta, rho, Vv, num_simulations):
    f0, vol, duration, strike, type = quote[0], quote[1], quote[2], quote[3], quote[4]
    price = []
    i = 0
    while i < len(f0):
        p = expectedValuation(
            f0[i],
            vol[i],
            duration[i],
            strike[i],
            type[i],
            beta,
            rho,
            Vv,
            num_simulations,
        )
        price.append(p)
        i += 1
    return price


def getPriceSimultaneousQuotes(quote, beta, rho, Vv, num_simulations):
    f0, vol, duration, strike, type = (
        quote[0][0],
        quote[1][0],
        quote[2][0],
        quote[3],
        quote[4],
    )
    sum = [0] * len(strike)
    num_steps = 100
    i = 0
    while i < num_simulations:
        f = sabrfowardSim(num_steps, duration, f0, vol, beta, rho, Vv)
        for j in np.arange(len(strike)):
            sum[j] = sum[j] + valueAtMaturity(f, strike[j], type[j])
        i += 1
    price = [s / num_simulations for s in sum]
    return price


def volInterval(price, quote):
    lb = price[1]
    ub = price[2]
    low = getVolatility(lb, quote)
    high = getVolatility(ub, quote)
    return low, high


def getVolatility(price, quote):
    V = []
    f0, vol, duration, strike, type = quote[0], quote[1], quote[2], quote[3], quote[4]
    i = 0
    while i < len(f0):
        v = black.Vol(f0[i], strike[i], duration[i], price[i], 0, type[i])
        V.append(v)
        i += 1
    return V


def getParameters(beta, quote, vol):
    f0, duration, strike = quote[0], quote[2], quote[3]
    optarv = estimating.ARV(beta, strike, f0, duration, vol)
    return optarv


def NormalizeStrike(quote):
    strike = quote[3]
    f0 = quote[0]
    Kf0 = []
    for i in np.arange(len(quote[0])):
        Kf0.append(float(strike[i] / f0[i]))
    return Kf0


def plotTheoreticalsabrVolSmile(alpha, beta, rho, Vv, f0, T):
    sabrvol = []
    K = []
    lb = round(0.1 * f0)
    ub = round(2.5 * f0)
    for k in np.arange(lb, ub, 1):
        vi = sabr.impVol(alpha, beta, rho, Vv, k, f0, T)
        sabrvol.append(vi)
        K.append(float(k / f0))
    plt.plot(K, sabrvol, "--", label="theoretical sabr")
    axes = plt.gca()
    axes.set_ylim([0, 2])


def plotQuotes(quote, vol):
    strike, type = quote[3], quote[4]
    Kf0 = NormalizeStrike(quote)
    for i in np.arange(len(vol)):
        if type[i] == 1:
            plt.plot(Kf0[i], vol[i], ms=4, c="r", marker="o")

        if type[i] == 0:
            plt.plot(Kf0[i], vol[i], ms=4, c="b", marker="^")


def plotFittedsabrVolSmile(alpha, beta, rho, Vv, f0, T):
    sabrvol = []
    K = []
    lb = round(0.1 * f0)
    ub = round(2.5 * f0)
    for k in np.arange(lb, ub, 1):
        vi = sabr.impVol(alpha, beta, rho, Vv, k, f0, T)
        sabrvol.append(vi)
        K.append(float(k / f0))

    plt.plot(K, sabrvol, label="fitted sabr")


def examplesabrVolSmile(alpha, beta, rho, Vv, f0, T):
    sabrvol = []
    K = []
    lb = 0.03
    ub = 0.15
    for k in np.linspace(lb, ub, 10000):
        vi = sabr.impVol(alpha, beta, rho, Vv, k, f0, T)
        sabrvol.append(vi)
        K.append(k)

    plt.plot(K, sabrvol, label="fitted sabr")


def MeanResidualsBS(vol, alpha):
    sum = 0
    for v in vol:
        sum = sum + abs(v - alpha)
    mean = sum / len(vol)
    return mean

    #############################MAIN FUNCTIONS###############################


def ExamplePath(num_steps, T, f0, alpha, beta, rho, Vv):
    path = sabrpathSim(num_steps, T, f0, alpha, beta, rho, Vv)
    pathPlot(num_steps, path)


def DynamicSimulation(T, f0, alpha, beta, rho, Vv, num_quotes, time, num_simulations):
    quote = dynamicQuotes(
        T, f0, alpha, beta, rho, Vv, num_quotes, time
    )  # f0, vol, duration, strike, type = quote[0], quote[1], quote[2], quote[3], quote[4]
    price = getPrice(quote, beta, rho, Vv, num_simulations)
    premium = price
    vol = getVolatility(premium, quote)
    plotQuotes(quote, vol)
    plotTheoreticalsabrVolSmile(alpha, beta, rho, Vv, f0, T)
    ARV = getParameters(beta, quote, vol)
    plotFittedsabrVolSmile(ARV[0], beta, ARV[1], ARV[2], f0, T)


def TestSimulation(derivative, alpha, beta, rho, Vv, num_quotes, num_simulations):

    quote = instaTestQuotes(
        derivative, alpha, beta, rho, Vv, num_quotes
    )  # f0, vol, duration, strike, type = quote[0], quote[1], quote[2], quote[3], quote[4]
    price = getPriceSimultaneousQuotes(quote, beta, rho, Vv, num_simulations)
    premium = price
    vol = getVolatility(premium, quote)
    plotQuotes(quote, vol)
    plotTheoreticalsabrVolSmile(alpha, beta, rho, Vv, derivative.f0, derivative.T)

    if Vv == 0:
        print(MeanResidualsBS(vol, alpha))

    # print("Fitting sabr...")
    # ARV = getParameters(beta, quote, vol);
    # plotFittedsabrVolSmile(ARV[0], beta, ARV[1], ARV[2], f0, T)

@hydra.main(config_path="conf", config_name="config.yaml")
def run(cfg):

    derivative = Derivative(cfg.parameters.T, cfg.parameters.f0)
    alpha = cfg.parameters.alpha 
    beta = cfg.parameters.beta 
    rho = cfg.parameters.rho 
    Vv = cfg.parameters.Vv 

    num_steps = cfg.montecarlo.num_steps 
    num_quotes = cfg.montecarlo.num_quotes 
    time_step = cfg.montecarlo.time_step
    num_simulations = cfg.montecarlo.num_simulations

    #ExamplePath(num_steps, T, f0, alpha, beta, rho, Vv)

    #DynamicSimulation(T, f0, alpha, beta, rho, Vv, num_quotes, time_step, num_simulations)

    TestSimulation(derivative, alpha, beta, rho, Vv, num_quotes, num_simulations)


    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    run()

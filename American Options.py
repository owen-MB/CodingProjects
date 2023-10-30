# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:01:49 2023

@author: owen_
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import time

def priceAmerican(T, steps, realizations, opt, brownian, regr, start = 1, degree = 4, r = 0, sigma = 1, strike = 1):
    dt = T / steps
    
    #Simulation of Paths of Geometric Brownian Motion
    #x = [dt*i for i in range(steps+1)]
    if brownian == "motion":
        s = np.random.normal(0, sigma*np.sqrt(dt), size = (realizations, steps+1))
        d = [(r - 0.5*sigma*sigma)*dt for i in range(steps+1)]
        s = s + d
        s = np.cumsum(s, axis = 1)
        s = np.exp(s)
        s = s * start
    if brownian == "bridge":
        x = (r - 0.5*sigma*sigma)*T + sigma*np.sqrt(T)*np.random.normal(0, 1, size = realizations)
        s = start*np.exp(x)
    #End of Path Simulation
    
    
    #Set-Up, initializing the Price vector
    if opt == "call" and brownian == "motion":
        P = np.copy( s[:, steps] ) - strike
        EBy = np.copy(s[:, steps])
    if opt == "put" and brownian == "motion":
        P = strike - np.copy( s[:, steps] )
        EBy = np.copy(s[:, steps])
    if opt == "call" and brownian == "bridge":
        P = np.copy( s ) - strike
        EBy = np.copy(s)
    if opt == "put" and brownian == "bridge":
        P = strike - np.copy( s )
        EBy = np.copy(s)
        
    NITM = P < 0 #Finding indices that are not-in-the-money
    P[NITM] = 0 #Setting the price to 0 when not-in-the-money
    C = np.copy(P) #Auxiliary array that saves the estimated values of continuation
    g = np.copy(P) #Auxiliary array that saves the estimated values of immediate exercise
    
    
    #Exercise Boundary Declarations
    EBx = np.full(realizations, T, dtype = "float") #The array that keeps the x-coordinate of each point in the Exercise Boundary
    
    for i in range(steps-1, -1, -1):
        if brownian == "bridge":
            z = np.random.normal(0, 1, size = realizations)
            x = x*(i/(i+1)) + (sigma*np.sqrt( i*dt/(i+1) ) )*z
            s = start*np.exp(x)
        
        
        if opt == "call" and brownian == "motion":
            ITM = s[:, i] > strike #Finding indices that are in-the-money
            a = s[ITM, i] #Taking just the paths that are in-the-money
        if opt == "put" and brownian == "motion":
            ITM = s[:, i] < strike #Finding indices that are in-the-money
            a = s[ITM, i] #Taking just the paths that are in-the-money
        if opt == "call" and brownian == "bridge":
            ITM = s > strike #Finding indices that are in-the-money
            a = s[ITM] #Taking just the paths that are in-the-money
        if opt == "put" and brownian == "bridge":
            ITM = s < strike #Finding indices that are in-the-money
            a = s[ITM] #Taking just the paths that are in-the-money
                
        b = np.exp(-r * dt) * P[ITM] #Taking the prices corresponding to the in-the-money paths
        
        if a.any():
            if regr == "Laguerre":
                beta = np.polynomial.laguerre.lagfit(a, b, degree, rcond=None, full=False, w=None) #Laguerre Polynomial Regression
                C[ITM] = np.polynomial.laguerre.lagval(a, beta) #Estimation of Value of Continuation
            if regr == "Isotonic":
                iso_reg = IsotonicRegression().fit(a, b) #Isotone Regression
                C[ITM] = iso_reg.predict(a) #Estimation of Value of Continuation        
            if opt == "put":
                g[ITM] = strike - a #Estimation of Value of Immediate Exercise
            if opt == "call":
                g[ITM] = a - strike
        
        auxInd = np.logical_and( ITM, C < g )
        P[ auxInd ] = g[auxInd]
        P[ np.logical_not(auxInd) ] = P[ np.logical_not(auxInd) ] * np.exp(-r * dt)
        
        #Exercise Boundary Actualization
        EBx[auxInd] = np.full(auxInd.sum(), dt*i)
        if brownian == "motion":
            EBy[auxInd] = s[auxInd, i]
        if brownian == "bridge":
            EBy[auxInd] = s[auxInd]
        #Exercise Boundary
    
    """Plotting Specifications
    
    plt.scatter(EBx, EBy, marker = "|", color = "red")
    plt.scatter([0], start, marker = "H",  color = "blue", label = "Starting Price")
    plt.scatter([T], strike, marker = "h",  color = "green", label = "Strike Price")
    plt.xlim(-0.01, T + 0.01)
    if opt == "call":
        plt.ylim(85, 140 )
    if opt == "put":
        plt.ylim(70, max(start, strike) + 10 )
    plt.legend(loc = "upper left")
    plt.title(f'Ex. Boundary for {opt} using Brownian {brownian} and {regr} Regression')
    plt.xlabel('Time to Expiry')
    plt.ylabel('Price of the Underlying')
    plt.show()
    """
    
    return np.average(P*np.exp(-r * dt))


"""
for S0 in [90, 100, 110]:
    for DEG in [1, 2, 3, 4, 5]:
        for OPT in {"call", "put"}:
            for BB in {"motion", "bridge"}:
                for RR in {"Laguerre"}:
                    print(f"Method of Brownian {BB}, {RR} Regression")
                    print(f"Starting Price of the Stock: {S0}")
                    t0 = time.time()
                    print("Predicted Price of the {OPT}: ", priceAmerican(1, 100, 100000, opt = OPT,
                                                        brownian = BB, regr = RR, start = S0, degree = DEG, r = 0.03, sigma = 0.15, strike = 100))
                    t1 = time.time()
                    print("Time Taken: ", t1 - t0)
                    print()
                    print()
                    
"""
TRU = {"put":{90: 10.728, 100: 4.821, 110: 1.8281}, "call":{90: 14.059, 100: 7.485, 110: 14.702}}



for S0 in [90, 100, 110]:
    for OPT in {"call", "put"}:
        for BB in {"motion", "bridge"}:
            for RR in {"Laguerre"}:
                for DEG in [4]:
                    relerrsd = []
                    for M in range(6, 7):
                        print(f"Method of Brownian {BB}, {RR} Regression of degree {DEG} using {10**M} realizations")
                        print(f"Starting Price of the Stock: {S0}")
                        t0 = time.time()
                        AUX = priceAmerican(1, 100, 10**M, opt = OPT,
                                                            brownian = BB, regr = RR, start = S0, degree = DEG, r = 0.03, sigma = 0.15, strike = 100)
                        t1 = time.time()
                        print(f"Predicted Price of the {OPT}: ", AUX)
                        RELERR = np.absolute(AUX - TRU[OPT][S0])/TRU[OPT][S0]
                        relerrsd.append(RELERR)
                        print(f"Relative Error: {RELERR}")
                        print("Time Taken: ", t1 - t0)
                        print()
                        print()
                    
                    """plt.plot(range(2, 7), relerrsd, label = f"Degree {DEG}")
                    plt.legend(loc = "upper right")
                    plt.title('Relative Errors with Laguerre Regression')
                    plt.xlabel('Time Steps in Log-Scale')
                    plt.ylabel('Relative Error')"""
                    
                        
                        
                        
                        
for S0 in [90, 100, 110]:
    for OPT in {"call", "put"}:
        for BB in {"motion", "bridge"}:
            for RR in {"Isotonic"}:
                for M in range(6, 7):
                    print(f"Method of Brownian {BB}, {RR} Regression using {10**M} realizations")
                    print(f"Starting Price of the Stock: {S0}")
                    t0 = time.time()
                    AUX = priceAmerican(1, 100, 10**M, opt = OPT,
                                        brownian = BB, regr = RR, start = S0, degree = DEG, r = 0.03, sigma = 0.15, strike = 100)
                    t1 = time.time()
                    print(f"Predicted Price of the {OPT}: ", AUX)
                    RELERR = np.absolute(AUX - TRU[OPT][S0])/TRU[OPT][S0]
                    print(f"Relative Error: {RELERR}")
                    print("Time Taken: ", t1 - t0)
                    print()
                    print()
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:04:31 2023

@author: owen_
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import time

def priceAmericanPutLSLaguerre(T, steps, realizations, start = 1, degree = 4, r = 0, sigma = 1, strike = 1):
    dt = T / steps
    
    #Simulation of Paths of Geometric Brownian Motion
    #x = [dt*i for i in range(steps+1)]

    s = np.random.normal(0, sigma*np.sqrt(dt), size = (realizations, steps+1))
    d = [(r - 0.5*sigma*sigma)*dt for i in range(steps+1)]
    s = s + d
    y = np.cumsum(s, axis = 1)
    y = np.exp(y)
    y = y * start
    #End of Path Simulation
    
    
    #Set-Up, initializing the Price vector
    P = strike - np.copy( y[:, steps] )
    NITM = P < 0 #Finding indices that are not-in-the-money
    P[NITM] = 0 #Setting the price to 0 when not-in-the-money
    C = np.copy(P) #Auxiliary array that saves the estimated values of continuation
    g = np.copy(P) #Auxiliary array that saves the estimated values of immediate exercise
    
    
    #Exercise Boundary Declarations
    EBx = np.full(realizations, T, dtype = "float") #The array that keeps the x-coordinate of each point in the Exercise Boundary
    EBy = np.copy(y[:, steps]) #The array that keeps the y-coordinate of each point in the Exercise Boundary
    
    for i in range(steps-1, -1, -1):
        ITM = y[:, i] < strike #Finding indices that are in-the-money
        
        a = y[ITM, i] #Taking just the paths that are in-the-money
        
        b = np.exp(-r * dt) * P[ITM] #Taking the prices corresponding to the in-the-money paths
        
        if a.any():
            beta = np.polynomial.laguerre.lagfit(a, b, degree, rcond=None, full=False, w=None) #Laguerre Polynomial Regression
            C[ITM] = np.polynomial.laguerre.lagval(a, beta) #Estimation of Value of Continuation
            g[ITM] = strike - a #Estimation of Value of Immediate Exercise
        
        auxInd = np.logical_and( ITM, C < g )
        P[ auxInd ] = g[auxInd]
        P[ np.logical_not(auxInd) ] = P[ np.logical_not(auxInd) ] * np.exp(-r * dt)
        
        #Exercise Boundary Actualization
        EBx[auxInd] = np.full(auxInd.sum(), dt*i)
        EBy[auxInd] = y[auxInd, i]
        #Exercise Boundary
    
    """Plotting Specifications """
    plt.scatter(EBx, EBy, marker = "1", color = "red")
    plt.scatter([0], start, marker = "H",  color = "blue", label = "Starting Price")
    plt.scatter([T], strike, marker = "h",  color = "green", label = "Strike Price")
    plt.xlim(-0.01, T + 0.01)
    plt.ylim(70, max(start, strike) + 10 )
    plt.legend(loc = "upper left")
    plt.title('Exercise Boundary using Longstaff-Schwartz and Laguerre Regression')
    plt.xlabel('Time to Expiry')
    plt.ylabel('Price of the Underlying')
    plt.show()
    """ """
    
    return np.average(P*np.exp(-r * dt))


def priceAmericanPutLSIsotone(T, steps, realizations, start = 1, r = 0, sigma = 1, strike = 1):
    dt = T / steps
    
    #Simulation of Paths of Geometric Brownian Motion
    #x = [dt*i for i in range(steps+1)]

    s = np.random.normal(0, sigma*np.sqrt(dt), size = (realizations, steps+1))
    d = [(r - 0.5*sigma*sigma)*dt for i in range(steps+1)]
    s = s + d
    y = np.cumsum(s, axis = 1)
    y = np.exp(y)
    y = y * start
    #End of Path Simulation
    
    
    
    #Set-Up, initializing the Price vector
    P = strike - np.copy( y[:, steps] )
    NITM = P < 0 #Finding indices that are not-in-the-money
    P[NITM] = 0 #Setting the price to 0 when not-in-the-money
    C = np.copy(P) #Auxiliary array that saves the estimated values of continuation
    g = np.copy(P) #Auxiliary array that saves the estimated values of immediate exercise
    
    
    #Exercise Boundary Declarations
    EBx = np.full(realizations, T, dtype = "float") #The array that keeps the x-coordinate of each point in the Exercise Boundary
    EBy = np.copy(y[:, steps]) #The array that keeps the y-coordinate of each point in the Exercise Boundary
    
    for i in range(steps-1, -1, -1):
        ITM = y[:, i] < strike #Finding indices that are in-the-money
        
        a = y[ITM, i] #Taking just the paths that are in-the-money
        b = np.exp(-r * dt) * P[ITM] #Taking the prices corresponding to the in-the-money paths
        
        if a.any():
            iso_reg = IsotonicRegression().fit(a, b) #Isotone Regression
            C[ITM] = iso_reg.transform(a) #Estimation of Value of Continuation
            g[ITM] = strike - a #Estimation of Value of Immediate Exercise
        
        auxInd = np.logical_and( ITM, C < g )
        P[ auxInd ] = g[auxInd]
        P[ np.logical_not(auxInd) ] = P[ np.logical_not(auxInd) ] * np.exp(-r * dt)
        
        #Exercise Boundary Actualization
        EBx[auxInd] = np.full(auxInd.sum(), dt*i)
        EBy[auxInd] = y[auxInd, i]
        #Exercise Boundary
    
    """Plotting Specifications """
    plt.scatter(EBx, EBy, marker = "1", color = "red")
    plt.scatter([0], start, marker = "H",  color = "blue", label = "Starting Price")
    plt.scatter([T], strike, marker = "h",  color = "green", label = "Strike Price")
    plt.xlim(-0.01, T + 0.01)
    plt.ylim(70, max(start, strike) + 10 )
    plt.legend(loc = "upper left")
    plt.title('Exercise Boundary using Longstaff-Schwartz and Isotonic Regression')
    plt.xlabel('Time to Expiry')
    plt.ylabel('Price of the Underlying')
    plt.show()
    """ """
    
    return np.average(P*np.exp(-r * dt))

def priceAmericanPutGustafssonLaguerre(T, steps, realizations, start = 1, degree = 4, r = 0, sigma = 1, strike = 1):
    dt = T / steps
    
    #Simulation of the End-Points of Geometric Brownian Motion
    x = (r - 0.5*sigma*sigma)*T + sigma*np.sqrt(T)*np.random.normal(0, 1, size = realizations)
    s = start*np.exp(x)
    
    
    #Exercise Boundary Declarations
    EBx = np.full(realizations, T, dtype = "float") #The array that keeps the x-coordinate of each point in the Exercise Boundary
    EBy = np.copy(s) #The array that keeps the y-coordinate of each point in the Exercise Boundary
    
    
    #Set-Up, initializing the Price vector
    P = strike - np.copy(s)
    NITM = P < 0 #Finding indices that are not-in-the-money
    P[NITM] = 0 #Setting the price to 0 when not-in-the-money
    C = np.copy(P) #Auxiliary array that saves the estimated values of continuation
    g = np.copy(P) #Auxiliary array that saves the estimated values of immediate exercise
    
    #Dynamic Programming
    for i in range(steps-1, 0, -1):
        #Generate GBM at time t_i
        z = np.random.normal(0, 1, size = realizations)
        x = x*(i/(i+1)) + (sigma*np.sqrt( i*dt/(i+1) ) )*z
        s = start*np.exp(x)
        
        #Check for in-the-moneyness and do the regression
                
        ITM = s < strike #Finding indices that are in-the-money
        
        a = s[ITM] #Taking just the paths that are in-the-money
        b = np.exp(-r * dt) * P[ITM] #Taking the prices corresponding to the in-the-money paths
        
        if a.any():
            beta = np.polynomial.laguerre.lagfit(a, b, degree, rcond=None, full=False, w=None) #Laguerre Polynomial Regression        
            C[ITM] = np.polynomial.laguerre.lagval(a, beta) #Estimation of Value of Continuation
            g[ITM] = strike - a #Estimation of Value of Immediate Exercise
        
        auxInd = np.logical_and( ITM, C < g )
        P[ auxInd ] = g[auxInd]
        P[ np.logical_not(auxInd) ] = P[ np.logical_not(auxInd) ] * np.exp(-r * dt)
        
        #Exercise Boundary Actualization
        EBx[auxInd] = np.full(auxInd.sum(), dt*i)
        EBy[auxInd] = s[auxInd]        
        #Exercise Boundary
    
    """Plotting Specifications """
    plt.scatter(EBx, EBy, marker = "1", color = "red")
    plt.scatter([0], start, marker = "H",  color = "blue", label = "Starting Price")
    plt.scatter([T], strike, marker = "h",  color = "green", label = "Strike Price")
    plt.xlim(-0.01, T + 0.01)
    plt.ylim(70, max(start, strike) + 10 )
    plt.legend(loc = "upper left")
    plt.title('Exercise Boundary using Gustafsson and Laguerre Regression')
    plt.xlabel('Time to Expiry')
    plt.ylabel('Price of the Underlying')
    plt.show()
    """ """
    
    return np.average(P*np.exp(-r * dt))


def priceAmericanPutGustafssonIsotone(T, steps, realizations, start = 1, r = 0, sigma = 1, strike = 1):
    dt = T / steps
    
    #Simulation of the End-Points of Geometric Brownian Motion
    x = (r - 0.5*sigma*sigma)*T + sigma*np.sqrt(T)*np.random.normal(0, 1, size = realizations)
    s = start*np.exp(x)
    
    #Exercise Boundary Declarations
    EBx = np.full(realizations, T, dtype = "float") #The array that keeps the x-coordinate of each point in the Exercise Boundary
    EBy = np.copy(s) #The array that keeps the y-coordinate of each point in the Exercise Boundary
    
    #Set-Up, initializing the Price vector
    P = strike - np.copy(s)
    NITM = P <= 0 #Finding indices that are not-in-the-money
    P[NITM] = 0 #Setting the price to 0 when not-in-the-money
    C = np.copy(P) #Auxiliary array that saves the estimated values of continuation
    g = np.copy(P) #Auxiliary array that saves the estimated values of immediate exercise
    
    #Dynamic Programming
    for i in range(steps-1, 0, -1):
        #Generate GBM at time t_i
        z = np.random.normal(0, 1, size = realizations)
        x = x*(i/(i+1)) + (sigma*np.sqrt( i*dt/(i+1) ) )*z
        s = start*np.exp(x)
        
        #Check for in-the-moneyness and do the regression
                
        ITM = s <= strike #Finding indices that are in-the-money
        
        a = s[ITM] #Taking just the paths that are in-the-money
        b = np.exp(-r * dt) * P[ITM] #Taking the prices corresponding to the in-the-money paths
        
        if a.any():
            iso_reg = IsotonicRegression().fit(a, b) #Isotone Regression
            C[ITM] = iso_reg.transform(a) #Estimation of Value of Continuation        
            g[ITM] = strike - a #Estimation of Value of Immediate Exercise
        
        auxInd = np.logical_and( ITM, C <= g )
        P[ auxInd ] = g[auxInd]
        P[ np.logical_not(auxInd) ] = P[ np.logical_not(auxInd) ] * np.exp(-r * dt)
        
        #Exercise Boundary Actualization
        EBx[auxInd] = np.full(auxInd.sum(), dt*i)
        EBy[auxInd] = s[auxInd]        
        #Exercise Boundary
    
    """Plotting Specifications """
    plt.scatter(EBx, EBy, marker = "1", color = "red")
    plt.scatter([0], start, marker = "H",  color = "blue", label = "Starting Price")
    plt.scatter([T], strike, marker = "h",  color = "green", label = "Strike Price")
    plt.xlim(-0.01, T + 0.01)
    plt.ylim(70, max(start, strike) + 10 )
    plt.legend(loc = "upper left")
    plt.title('Exercise Boundary using Gustafsson and Isotonic Regression')
    plt.xlabel('Time to Expiry')
    plt.ylabel('Price of the Underlying')
    plt.show()
    """ """
    
    return np.average(P*np.exp(-r * dt))


for S_0 in [90, 100, 110]:
    print("LS Method, Laguerre Regression")
    print(f"Starting Price of the Stock: {S_0}")
    t0 = time.time()
    print("Predicted Price of the Put: ", priceAmericanPutLSLaguerre(1, 100, 100000, start = S_0, degree = 4, r = 0.03, sigma = 0.15, strike = 100))
    t1 = time.time()
    print("Time Taken: ", t1 - t0)
    print()
    print()
    
    """
    
    print("LS Method, Isotonic Regression")
    print(f"Starting Price of the Stock: {S_0}")
    t0 = time.time()
    print("Predicted Price of the Put: ", priceAmericanPutLSIsotone(1, 100, 1000000, start = S_0, r = 0.03, sigma = 0.15, strike = 100))
    t1 = time.time()
    print("Time Taken: ", t1 - t0)
    print()
    print()    
    
    
    print("Gustafsson Method, Laguerre Regression")
    print(f"Starting Price of the Stock: {S_0}")
    t0 = time.time()
    print("Predicted Price of the Put: ", priceAmericanPutGustafssonLaguerre(1, 100, 1000000, start = S_0, degree = 4, r = 0.03, sigma = 0.15, strike = 100))
    t1 = time.time()
    print("Time Taken: ", t1 - t0)
    print()
    print()
    
    print("Gustafsson Method, Isotonic Regression")
    print(f"Starting Price of the Stock: {S_0}")
    t0 = time.time()
    print("Predicted Price of the Put: ", priceAmericanPutGustafssonIsotone(1, 100, 1000000, start = S_0, r = 0.03, sigma = 0.15, strike = 100))
    t1 = time.time()
    print("Time Taken: ", t1 - t0)
    print()
    print()
    
    """
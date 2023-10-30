# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:07:35 2023

@author: owen_
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_BM(steps, T, mu = 0, sigma = 1, start = 0):
    dt = T/steps
    x = [dt*i for i in range(steps+1)]
    
    s = np.random.normal(0, sigma*np.sqrt(dt), steps+1)
    d = [mu*dt for i in range(steps+1)]
    s = s + d
    s[0] = start
    y = np.cumsum(s)
    
    return x, y

def simulate_GBM(steps, T, r = 0, sigma = 1, start = 1):
    dt = T/steps
    x = [dt*i for i in range(steps+1)]
    
    s = np.random.normal(0, sigma*np.sqrt(dt), steps+1)
    d = [(r - 0.5*sigma*sigma)*dt for i in range(steps+1)]
    s = s + d
    s[0] = 0
    y = np.cumsum(s)
    y = np.exp(y)
    y = y * start
    
    return x, y


def simulate_BB(steps, T, mu = 0, sigma = 1, start = 0, end = 1):
    dt = T/steps
    x = [dt*i for i in range(steps+1)]
    y = np.copy(x)
    y[steps] = mu*T + sigma*np.sqrt(T)*np.random.normal(0, 1)
    
    for i in range(steps-1, -1, -1):
        z = np.random.normal(0, sigma * np.sqrt(i*dt/(i+1) ) )
        y[i] = y[i+1]*(i/(i+1)) + z
    
    y = y + start
            
    return x, y

def simulate_GBB(steps, T, r = 0, sigma = 1, start = 1):
    dt = T/steps
    x = [dt*i for i in range(steps+1)]
    y = np.copy(x)
    
    
    y[steps] = (r - 0.5*sigma*sigma)*T + sigma*np.random.normal(0, 1)
    
    for i in range(steps-1, -1, -1):
        z = np.random.normal(0, 1)
        y[i] = y[i+1]*(i/(i+1)) + (sigma*np.sqrt( i*dt/(i+1) ) )*z
    
    y = start * np.exp(y)
    
    return x, y





"""
a, b = simulate_BB(1000, 1, mu = 2, sigma = 3, start = 0)
plt.plot(a, b, label = "Brownian Bridge")
a, b = simulate_BM(1000, 1, mu = 2, sigma = 3, start = 0)
plt.plot(a, b, label = "Brownian Motion")
plt.xlim(-0.01, 1.01)
plt.legend(loc = "upper left")
plt.title("Brownian Motion vs Brownian Bridge")
plt.xlabel('Time t')
plt.ylabel('Value of W(t)')
plt.show()
"""
for i in range(50):
    a, b = simulate_BM(1000, 2, mu = 0, sigma = 1, start = 0)
    plt.plot(a, b)
plt.title("Realizations of Brownian Motion")
plt.xlabel('Time t')
plt.ylabel('Value of W(t)')
plt.show()

for i in range(50):
    a, b = simulate_BB(1000, 2, mu = 0, sigma = 1, start = 0)
    plt.plot(a, b)
plt.title("Realizations of Brownian Bridge")
plt.xlabel('Time t')
plt.ylabel('Value of W(t)')
plt.show()
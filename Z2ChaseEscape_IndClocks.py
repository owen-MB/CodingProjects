# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:49:55 2023

@author: owen_
"""

import numpy as np

import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation

#Constants
n = 131
eRate = 0.8

GRID = []
turns = 0

#Auxiliary Functions
def goodPrint(matrix):
    for j in matrix:
        print(j)
    print()
    
def nbors(v):
    ans = []
    i = v[0]
    j = v[1]
    ans.append( [(i+1) % n, j % n] )
    ans.append( [(i-1) % n, j % n] )
    ans.append( [i % n, (j+1) % n] )
    ans.append( [i % n, (j-1) % n] )
    return ans


#Main Code

#Define the grid to be displayed
grid = [ [0] * n for i in range(n) ]

grid[n//2][n//2] = 1


potB = [ ( np.random.exponential(1), [n//2, n//2] ) ]
potR = []
for k in nbors( [n//2, n//2] ):
    potR.append( ( np.random.exponential(1/eRate), k, [n//2, n//2] ) )


#Number of red vertices
R = 1

#Last movement
current = [n//2, n//2]

#Simulation loop
while R>0 and current[0] > 0 and current[0] < n-1 and current[1] > 0 and current[1] < n-1:
    red = 0
    turns += 1
    
    minB = min(potB, key = lambda t: t[0])
    minR = min(potR, key = lambda t: t[0])
    
    t = min(minR[0], minB[0])
    
    if t == minR[0]:
        red = 1
    
    if red:
        current = minR[1]
        grid[ current[0] ][ current[1] ] = 1
        R += 1
        
        potR[:] = [x for x in potR if x[1] != current]
        
        for k in nbors(current):
            if grid[ k[0] ][ k[1] ] == 0:
                potR.append( (t + np.random.exponential(1/eRate), k, current) )
            elif grid[ k[0] ][ k[1] ] == 2:
                potB.append( (t + np.random.exponential(1), current  ) )
    else:
        current = minB[1]
        grid[ current[0] ][ current[1] ] = 2
        R -= 1
        
        potB[:] = [x for x in potB if x[1] != current]
        
        potR[:] = [x for x in potR if not( (x[0] > minB[0]) and (x[2] == current) ) ]
        
        
        for k in nbors(current):
            if grid[ k[0] ][ k[1] ] == 1:
                potB.append( (minB[0] + np.random.exponential(1), k) )

    #GRID.append(grid)
    if turns % 40 == 0:    
        plt.imshow(grid, cmap = "inferno")
        plt.show()
    




"""
Figure = plt.figure()
gridders = plt.plot([])

gridders= GRID[ 3 ]
goodPrint(GRID[ 3 ])
goodPrint(GRID[ 2 ])
plt.show()




def AnimationFunction(frame):
    gridders = GRID[frame]

anim_created = FuncAnimation( Figure, AnimationFunction, frames = turns, interval = 25 )
plt.show()
plt.close()
"""
    
    
    
    
    
    
    
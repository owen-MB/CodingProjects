# CodingProjects
The file "Z2ChaseEscape_IndClocks" displays the simulation of the Chase-Escape model (a variation of first-passage percolation) on the graph of Z^2.
A brief description of the process at hand is the following. "Prey" nodes are trying to escape from "predator" nodes. Predator nodes can only occupy nodes previously occupied by prey nodes.
Prey nodes can only move to contiguous "empty" nodes. Each edge of the graph has two random clocks with an Exponential distribution; one of them has rate Lambda, and the other one has rate 1. Once an empty node becomes occupies with a prey node, all the edges
connecting to the neighboring nodes start their random clocks with rate Lambda. Once any clock ticks, the prey node multiplies to any of these. Similarly, the predator nodes are chasing the prey nodes, but their Exponential clock
has parameter 1 instead. The most important question is whether Prey nodes survive (meaning, stay alive for infinite time) given a value of Lambda.

The file "American Options" prices American calls and American puts using the Longstaff-Schwartz method. Both Browmian motions and Brownian bridges can be toggled on/off when simulating the movement of the underlying stock.
Two types of regression, Laguerre polynomials and Isotonic, are used to estimate the conditional expectation that represents the "Value of continuing to hold onto the option" needed at each time step.

import numpy as np


# Test Equation: dx/dt = p*x
# Analytical solution: x = x0*exp(p*t)
# for some initial condition x(0) = x0


def Model(t,x,p):
    return p*x    

def Jacobian(t,x,p):
    return p

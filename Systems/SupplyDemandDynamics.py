import numpy as np
from numpy import log


# The system of differential equations.
# We here have a 5 state ODE system.
# x: the number of interested buyers of an assert
# y: the number of interested sellers of an assert
# v: the number of people not owning the assert and not interested in buying
# z: the number of people owning the assert but not interested in selling
# s: the price of the assert

# Maybe M = log(x) could be interesting.

def Model(t,x,p): # 
    
    # Unpack parameter vector
    a,b,c,k,A,Q,M = p
    
    # Unpack state vector
    x,y,v,z,s = x
    
    # We define the Equation: xdot = f(x,t) = (f1, f2, f3, f4, f5)
    f1 =   a*(M-s)*v/M    - c*s*(M-s)*x*y
    f2 =   b*s*z  - c*s*(M-s)*x*y
    f3 = - a*(M-s)*v/M    + c*s*(M-s)*x*y + k*z + A*v*(Q-v)
    f4 = - b*s*z  + c*s*(M-s)*x*y - k*z
    f5 = (x-y)*s
    
    # The ODE system
    xdot = np.array([f1,f2,f3,f4,f5])
    
    return xdot
    







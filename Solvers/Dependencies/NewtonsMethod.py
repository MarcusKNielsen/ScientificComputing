
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve

def NewtonsMethod(Func, Jacobian, x0, TOL, MaxIter, args=()):
    
    x = x0                      # Initial guess
    res = TOL + 1               # Initial residual
    k = 0                       # While-loop counter


    while (k < MaxIter and res > TOL):
        
        k += 1
        f = Func(x,*args)
        Jf = Jacobian(x,*args)
        h = solve(Jf,-f)
        x = x + h
        res = norm(f - np.dot(Jf,h))

    if res < TOL:
        return x
    else:
        print("Failed")
        return np.repeat(np.nan,len(x))







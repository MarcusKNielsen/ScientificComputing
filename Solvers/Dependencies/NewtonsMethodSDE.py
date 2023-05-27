from numpy import size, identity
from numpy.linalg import norm
from scipy.linalg import solve

def NewtonsMethodSDE(Drift, JacobianDrift, xk, tk, dt, xinit, c, TOL, MaxIter, p):
    
    tnxt = tk + dt              # New time step
    x = xinit                   # Initial guess
    k = 0                       # While-loop counter
    
    I = identity(size(x))
    
    R  = x - dt*Drift( tnxt, x, p) - xk - c
    
    while (k < MaxIter and norm(R) > TOL):
        
        k += 1
        JR = I - dt*JacobianDrift(tnxt,x, p)
        dx  = solve(JR,-R)
        x  = x + dx
        R  = x - dt*Drift(tnxt,x, p) - xk - c

    return x




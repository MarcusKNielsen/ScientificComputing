from numpy import size, identity
from numpy.linalg import norm
from scipy.linalg import solve

def NewtonsMethodODE_Modified(Model, Jacobian, tnxt, dt, xinit, c, TOL, MaxIter, p):
    
    #tnxt = tk + dt             # New time step
    x = xinit                   # Initial guess
    k = 0                       # While-loop counter
    
    I = identity(size(x))
    
    R  = x - dt*Model( tnxt, x, p) - c
    Nfev = 1
    
    while (k < MaxIter and norm(R) > TOL):
        
        k  += 1
        JR  = I - dt*Jacobian(tnxt,x, p)
        dx  = solve(JR,-R)
        x   = x + dx
        R   = x - dt*Model(tnxt,x, p) - c
        
        Nfev += 2

    return x, Nfev

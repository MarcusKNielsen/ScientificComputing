from numpy import size, identity
from numpy.linalg import norm
from scipy.linalg import solve

def NewtonsMethodODE(Model, Jacobian, xk, tk, dt, xinit, TOL, MaxIter, args=()):
    
    tnxt = tk + dt              # New time step
    x = xinit                   # Initial guess
    k = 0                       # While-loop counter
    
    I = identity(size(x))
    Nfev = 1
    R  = x - dt*Model(tnxt,x, args) - xk
    
    while (k < MaxIter and norm(R) > TOL):
        
        k += 1
        JR = I - dt*Jacobian(tnxt,x, args)
        dx  = solve(JR,-R)
        x  = x + dx
        R  = x - dt*Model(tnxt,x, args) - xk
        Nfev += 2

    return x, Nfev




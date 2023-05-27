from numpy import size, identity
from numpy.linalg import norm
from scipy.linalg import lu, solve_triangular

def NewtonsMethodODE_Modified(Model, Jacobian, tnxt, dt, xinit, c, TOL, MaxIter, p):
    
    x = xinit                   # Initial guess
    k = 0                       # While-loop counter
    
    I = identity(size(x))
    
    Rnow  = x - dt*Model( tnxt, x, p) - c
    a = 1
    
    Nfev = 1
    Njac = 0
    Nlu  = 0
    
    while (k < MaxIter and norm(Rnow) > TOL):
        
        # Update loop counter
        k  += 1
        
        if a >= 1:
            
            # Jacobian evaluation
            JR  = I - dt*Jacobian(tnxt,x, p)
            Njac += 1
            
            # Compute LUP decomposition
            P,L,U = lu(JR)
            Nlu += 1
            
        # Solve system of equations
        b = P @ (-Rnow)
        y = solve_triangular(L,b,lower=True)
        dx = solve_triangular(U,y,lower=False)
        
        # Update x and R
        x    = x + dx
        Rold = Rnow
        Rnow = x - dt*Model(tnxt,x, p) - c
        a = norm(Rnow)/norm(Rold)
        Nfev += 1

    return x, Nfev, Njac, Nlu


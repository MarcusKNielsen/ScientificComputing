#from NewtonsMethodODE_Modified2 import NewtonsMethodODE_Modified as NewtonsMethod
from InexactNewtonsMethod import NewtonsMethodODE_Modified as NewtonsMethod
from numpy import zeros,size


def Solve(Model, Jacobian, tspan, x0, Nsteps, args):
    
    N  = Nsteps             # Number of steps in the solution
    t0 = tspan[0]           # Initial time
    tN = tspan[1]           # Final time
    dt = (tN - t0)/N        # Step size
    
    
    # Allocation of space for solution
    nx = size(x0)           # Number of states
    X = zeros([N+1,nx])       # States as a function of time
    T = zeros(N+1)            # Time
    
    
    # Setup initial conditions in solution vector
    X[0,:] = x0
    T[0]   = t0
    
    # Parameter for Newtons Method
    TOL = 10**(-8)          # Tolerance
    MaxIter = 100           # Maximum number of iterations

    # Counters
    Nfev    = 0
    Njac    = 0
    Nlu     = 0
    
    for i in range(N):
        # Prerequested for explicit Euler step
        x = X[i,:]
        t = T[i]
        
        # Explicit Euler as initial guess
        xinit = x + dt*Model(t, x, *args)
        tnxt  = t + dt
        Nfev += 1
        
        # Preform Newton's method for ODE
        xnxt, nfev, njac, nlu = NewtonsMethod(Model, Jacobian, tnxt, dt, xinit, x, TOL, MaxIter, *args)
        Nfev += nfev
        Njac += njac
        Nlu += nlu
        
        # Save the results
        X[i+1,:] = xnxt
        T[i+1]   = tnxt
    
    return T, X, Nfev, Njac, Nlu



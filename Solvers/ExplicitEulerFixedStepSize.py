from numpy import zeros,size


def Solve(Model, tspan, x0, Nsteps, args):
    
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
    
    # Function evaluation counter
    Nfev = 0
    
    for i in range(N):
        
        # Prerequested for explicit Euler step
        x = X[i,:]
        t = T[i]
        Nfev += 1
        f = Model(t, x, *args)
        
        # Calculate explicit Euler step: X(t+dt) = dt*f*X(t)
        X[i+1,:] = dt*f + X[i,:]
        T[i+1]   = t + dt
        
    return T,X,Nfev


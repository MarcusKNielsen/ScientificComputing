from numpy import zeros,size


def RK4(Model, xn, tn, h, args):
    
    hm = 0.5 * h # h mid step
    
    X1 = xn
    k1 = Model(tn,X1,args)
    
    X2 = xn + hm * k1
    k2 = Model(tn+hm,X2,args)
    
    X3 = xn + hm * k2
    k3 = Model(tn+hm,X3,args)
    
    X4 = xn + h * k3
    k4 = Model(tn+h,X4,args)   
    
    xnxt = xn + (h/6) * ( k1+2*k2+2*k3+k4 )
    
    return xnxt
    

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
    
    for n in range(N):
        
        # Prerequested for explicit Euler step
        xn = X[n,:]
        tn = T[n]
        
        # Calculate explicit Euler step: X(t+dt) = dt*f*X(t)
        X[n+1,:] = RK4(Model, xn, tn, dt, *args)
        T[n+1]   = tn + dt
        
        Nfev += 4
        
    return T,X, Nfev


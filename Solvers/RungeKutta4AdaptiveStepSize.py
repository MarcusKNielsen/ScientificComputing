from numpy import array, row_stack
import numpy as np


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


def Solve(Model, tspan, x0, h0, abstol, reltol, args):
    
    t = tspan[0]            # Initial time
    tf = tspan[1]           # Final time
    dt = h0                 # Initial time step
    x = x0                  # initial state
    
    # Error Controller Parameters
    epstol = 0.8            # Target
    facmin = 0              # Minimum allowed step factor
    facmax = np.inf         # Maximum allowed step factor
    
    
    # Allocation of space for solution
    X = array([x])          # States as a function of time
    T = array([t])          # Time
    R = array([0])
    H = array([dt])
    
    # Function evaluation,accept and reject step counter
    Nfev = 0
    Naccept = 0
    Nreject = 0
    
    while t < tf:
        
        if t+dt > tf:
            dt = tf - t


        AcceptStep = False
        while not AcceptStep:
            
            # Calculate the next x using a full dt step.
            xnxt = RK4(Model, x, t, dt, *args)          # Single full step calculation
            Nfev += 4
        
            # Calculate the next x using two half steps.
            dth = 0.5*dt                                # dt half
            x1nxt = RK4(Model, x, t, dth, *args)        # First half step calculation
            Nfev += 4
            th = t + dth                                # t plus a half step
            x2nxt = RK4(Model, x1nxt, th, dth, *args)   # Second half step calculation
            Nfev += 4
        
            # Error calculate
            e = x2nxt - xnxt
            
            r = np.max(np.abs(e) / np.maximum(abstol, np.abs(x2nxt) * reltol))
        
            AcceptStep = (r <= 1)
            if AcceptStep:
                t = t + dt
                x = x2nxt
                
                X = row_stack((X, x))
                T = np.append(T,t)
                Naccept += 1
                
                R = np.append(R,r)
                H = np.append(H,dt)
            
            elif AcceptStep == False:
                Nreject += 1
                
            # Step size controller
            dt = max(facmin,min((epstol/r)**(1/5),facmax))*dt
    
    return T,X,Nfev,R,H,Naccept,Nreject


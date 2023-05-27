from numpy import zeros, array, row_stack
#from NewtonsMethodODE_Modified2 import NewtonsMethodODE_Modified as NewtonsMethod
from InexactNewtonsMethod import NewtonsMethodODE_Modified as NewtonsMethod
import numpy as np


def ESDIRK23(Model, Jacobian, xn, tn, h, k3, args):
    
    # First we define all coefficients in the Butcher Tableau.
    # Coefficients which are zero are skipped.
    
    gamma = (2 - np.sqrt(2))/2
    
    # We define the time coefficients
    c2 = 2*gamma
    c3 = 1
    
    # The A matrix coefficients:
    a21 = gamma
    a22 = gamma
    
    a31 = (1-gamma)/2
    a32 = (1-gamma)/2
    a33 = gamma
    
    # The b vector (It is not used since it is equal to the 3 row in a)
    # b1 =  (1-gamma)/2
    # b2 =  (1-gamma)/2
    # b3 =  gamma
    
    # The d vector
    d1 = (1-6*gamma**2)/(12*gamma)
    d2 = (6*gamma*(1-2*gamma)*(1-gamma)-1)/(12*gamma*(1-2*gamma))
    d3 = (6*gamma*(1-gamma)-1)/(3*(1-2*gamma))

    # We define the 3 time stages:
    T2 = tn + c2 * h
    T3 = tn + c3 * h

    # We calculate the 3 intermediate stages:
    
    # new k1 is equal to k3 from last step
    k1 = k3
    
    ### For Newtons method we need:
    MaxIter=1000
    TOL=10**(-6)
    xinit = xn + c2*h*k1
    
    Nfev = 0
    Njac = 0
    Nlu  = 0
    
    
    dt = a22*h
    c  = xn + a21*h*k1
    
    # Find X2 using Newtons Method
    X2, nfev, njac, nlu = NewtonsMethod(Model, Jacobian, T2, dt, xinit, c, TOL, MaxIter, args)
    k2 = Model(T2,X2,args)
    Nfev += nfev
    Njac += njac
    Nlu += nlu
    

    dt = a33*h
    c  = xn + a31*h*k1 + a32*h*k2
    
    X3, nfev, njac, nlu = NewtonsMethod(Model, Jacobian, T3, dt, xinit, c, TOL, MaxIter, args)
    k3 = Model(T3,X3,args)
    Nfev += nfev
    Njac += njac
    Nlu += nlu
    
    # Calculate next step (xnxt) and next error (enxt)
    
    xnxt = X3
    
    enxt = h * ( d1 * k1 + d2 * k2 + d3 * k3)
    
    return xnxt, enxt, k3, Nfev, Njac, Nlu



def Solve(Model, Jacobian, tspan, x0, h0, abstol, reltol, args):

    t0 = tspan[0]           # Initial time
    tf = tspan[1]           # Final time
    dt = h0                 # Initial time step
    x = x0                  # initial state

    # Error Controller Parameters
    epstol = 0.5            # Target
    facmin = 0.0            # Minimum allowed step factor
    facmax = np.inf         # Maximum allowed step factor

    # Allocation of space for solution
    X = array([x])          # States as a function of time
    T = array([t0])         # Time
    E = zeros([len(x0)])    # Error array
    H = array([dt])         # Step array
    R = array([0])
    H = array([dt])
    
    # Function evaluation,accept and reject step counter
    Nfev    = 0
    Njac    = 0
    Nlu     = 0
    Naccept = 0
    Nreject = 0
    
    
    t = t0
    
    # We need to initiate k3 for t=t0
    k3 = Model( t0, x0, *args)
    Nfev += 1
    
    while t < tf:
        
        if t+dt > tf:
            dt = tf - t

        AcceptStep = False
        while not AcceptStep:
            
            xnxt, enxt, k3, nfev, njac, nlu = ESDIRK23(Model, Jacobian, x, t, dt, k3, *args)
            Nfev += nfev
            Nfev += nfev
            Njac += njac
            Nlu += nlu
            
            
            # We define r for the step size controller
            r = np.max(np.abs(enxt) / np.maximum(abstol, np.abs(xnxt) * reltol))
            
        
            AcceptStep = (r <= 1)
            if AcceptStep:
                
                t = t + dt
                x = xnxt
                
                X = row_stack((X, x))
                T = np.append(T,t)
                E = row_stack((E, enxt))
                R = np.append(R,r)
                H = np.append(H,dt)
                Naccept += 1
            
            elif AcceptStep == False:
                Nreject += 1
                
            # Step size controller
            dt = max(facmin,min((epstol/r)**(1/3),facmax))*dt
    
    return T,X,Nfev,Njac,Nlu,R,H,Naccept,Nreject




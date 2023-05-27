
import time
start_time = time.time()


from TestEquation import Model, Jacobian

p = -1                   # Parameters
x0 = 10                  # Initial condition
tspan = [0,10]          # time interval


from ExplicitEulerFixedStepSize import Solve

N = 100000

T,X = Solve(Model,tspan,x0,N,args=(p,))


#%% Plot of the solution

import matplotlib.pyplot as plt
import numpy as np

# Unpack state
X = X[0,:]

# Plot of the Solution
plt.figure(1)
plt.plot(T,X,label="Numerical")
plt.xlabel('Time')
plt.ylabel('Value')
plt.title("Solution to Test Equation")
plt.legend()

print("--- %s seconds ---" % (time.time() - start_time))


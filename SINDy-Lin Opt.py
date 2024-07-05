# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:20:27 2024

@author: Jin
"""



import pysindy as ps
from decimal import *


#from pysindy.utils import linear_damped_SHO
#from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
#from pysindy.utils import hopf
#from pysindy.utils import lorenz

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
#from scipy.io import loadmat


def SINDy_model(Threshold,initCondition=[2,0,1]):
    #return the model
   
    # Integrator keywords for solve_ivp
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'
    integrator_keywords['atol'] = 1e-12

    dt = .01
    t_train = np.arange(0, 50, dt)
    t_train_span = (t_train[0], t_train[-1])
   
    #initial conditions
    x0_train = initCondition
    #Training Data generation
    x_train = solve_ivp(linear_3D, t_train_span,x0_train, t_eval=t_train, **integrator_keywords).y.T
    poly_order = 5
   
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=Threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order)
        )
    # Fit the model
    model.fit(x_train, t=dt)
   
    return model
# 

def SINDyThresholdOptimization(model):
   
    a11 = model.coefficients()[0,1]
    a12 = model.coefficients()[1,1]
    a13 = model.coefficients()[2,1]
    a21 = model.coefficients()[0,2]                       
    a22 = model.coefficients()[1,2]
    a23 = model.coefficients()[2,2]               # -0.1    -2     0 
    a31 = model.coefficients()[0,3]               #  2      -0.1   0
    a32 = model.coefficients()[1,3]               #  0       0     -0.3
    a33 = model.coefficients()[2,3]
    
    modelerrorLinear = abs(a11-(-0.1))+abs(a12-(-2))+abs(a21-(2))+abs(a22-(-0.1))+abs(a33-(-0.3))
    # NonLinear Error is just absolute value of all the coefficent Minus the Linear Error
    
    modelerrorNonLinear = sum(sum(abs(model.coefficients()))) - (abs(a11) + abs(a12) + abs(a13) + abs(a21) + abs(a22) + abs(a23) + abs(a31) + abs(a32) + abs(a33))
    
    #modelerrorNonLinear = sum(sum(abs(model.coefficients()))) - modelerrorLinear
    #modelerrorNonLinear = sum(sum(abs(model.coefficients()))) - (abs(-0.01) + abs(-2) + abs(2) + abs(-0.1) + abs(-0.3))
    #errormatrix = np.array(model.coefficients())   
    #errormatrix[0:3,1:4]=0
    #modelerrorNonLinear = np.sum(np.square(errormatrix))
    #print('Current Linear error: ' + str(modelerrorLinear))
    #print('Current Non-Linear error: ' + str(modelerrorNonLinear))
    
    return modelerrorLinear, modelerrorNonLinear

ndecimals = 5
  
initSearchLineSpace = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
FixedLineSpace = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

errorsLinear = []
errorsNonLinear = []
xaxisforplot = []

for currentdigit in range(ndecimals):  # why set the range in "range(ndecimals)"
    errors = []
    # Loop to find the initial array
    for value in initSearchLineSpace:
        currenterror = []
        TestThreshold = value
        xaxisforplot.append(value)
        currentmodel = SINDy_model(TestThreshold, [2,0,1])
        
        currenterror = SINDyThresholdOptimization(currentmodel)
        #Optimaize for Non Linear Error
       # errors.append(currenterror[0])
       # errors.append(currenterror[0])                   # Minimize the Linear Error
        #errors.append(currenterror[1])                   # Minimize the NonLinear Error
        errors.append(currenterror[0]+currenterror[1])    # Minimize the Total Error
        
        errorsLinear.append(currenterror[0])
        errorsNonLinear.append(currenterror[1])
       
    
    guess = initSearchLineSpace[errors.index(min(errors))]
    #if currenterror[0] == 0 :
    #    print("The error is so small, which is: " +str(currenterror[0]) + ". We stop this at " + str(int(currentdigit+1)) + " digit")
    #   break
    print('for the '+ str(int(currentdigit+1)) +'th digit the best is ' + str(guess))
    tempSpace = []
  
    if errors.index(min(errors)) == 0:
        tempSpace.append(float( Decimal( str(guess) ) - Decimal("0.1")**Decimal(str(int(currentdigit+2))) + Decimal("0.1")**Decimal(str(int(currentdigit+3)))   ))
        for eachvalue in FixedLineSpace[1:]:
            tempSpace.append(float(Decimal(str(guess)) - Decimal("0.1")**Decimal(str(int(currentdigit+2))) + Decimal(str(eachvalue))*Decimal("0.1")**Decimal(str(int(currentdigit+1)))))
        initSearchLineSpace = tempSpace
        print('Next Search Space is:')
        print(initSearchLineSpace)
        #initSearchLineSpace = initSearchLineSpace * 0.1  # (10^-1)^5 decimals then 10^-2 *10^-5 = 10^-7 so should be 7 digits but 19 digits for 0.0013...03 in output
    else:
        tempSpace.append(float(Decimal(str(guess)) + Decimal("0.1")**Decimal(str(int(currentdigit+3)))) )
        for eachvalue in FixedLineSpace[1:]:
            tempSpace.append(float( Decimal(str(guess)) + Decimal(str(eachvalue))*Decimal("0.1")**Decimal(str(int(currentdigit+1)))))
        initSearchLineSpace = tempSpace
        print('Next Search Space is:')
        print(initSearchLineSpace)
        #initSearchLineSpace = guess + initSearchLineSpace * 0.1

print('For initial condition: ' + str('[2,0,1]') + ' for ' + str(ndecimals) + ' decimals places')
print('We found optimal threshold as: ' + str(guess))
print('Printing the model: ')

bestmodel = SINDy_model(guess,initCondition=[2,0,1])
Bestmodelerror = SINDyThresholdOptimization(bestmodel)
bestmodel.print()

print('We collected all the Linear and Non-Linear error, and then we plot!')

xaxis = np.linspace(0,len(errorsLinear),len(errorsLinear))
fig, ax1 = plt.subplots(figsize=(8,8))
ax2 = ax1.twinx()

ax1.plot(xaxis,errorsLinear,"-r")
# Linear error in red color
ax2.plot(xaxis,errorsNonLinear,"-g")
# NonLinear error in Green color

ax1.set_xticks(xaxis)
ax1.set_xticklabels(xaxisforplot, rotation=45 , ha='right')

ax1.set_xlabel('Threshold')
ax1.set_ylabel('Linear Error (red)')
ax2.set_ylabel('Non-Linear Error (green)')

plt.show()

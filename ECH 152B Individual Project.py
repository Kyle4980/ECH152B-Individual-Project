# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:39:09 2024

@author: Kyle Le
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

# constants, units are J/mol when applicable
T = 298.15
R = 8.314

GCO2 = -394359
GH2 = 0
GCH4 = -50460
GH2O = -228572
GO2 = 0
GCO = -137169

HCO2 = -393509
HH2 = 0
HCH4 = -74520
HH2O = -241818
HO2 = 0
HCO = -110525

# Cp constants from Van Ness
AO2 = 3.639
BO2 = 0.506
CO2 = 0
DO2 = -0.227

ACH4 = 1.702
BCH4 = 9.081
CCH4 = -2.164
DCH4 = 0

ACO2 = 5.457
BCO2 = 1.045
CCO2 = 0
DCO2 = -1.157

AH2 = 3.249
BH2 = 0.422
CH2 = 0
DH2 = 0.083

ACO = 3.376
BCO = 0.557
CCO = 0
DCO = -0.031

AH2O = 3.470
BH2O = 1.450
CH2O = 0
DH2O = 0.121


# Sabatier Reaction CO2 + 4H2 <--> CH4 + 2H2O
deltaG1_at_298 = np.round((1*GCH4 + 2*GH2O)-(1*GCO2 + 4*GH2),2)
K1_at_298 = np.exp(-deltaG1_at_298/(R*T))
deltaH0a = (1*HCH4 + 2*HH2O)-(1*HCO2 + 4*HH2)

#K calculation as usual assuming constant Cp
def calcK1(x):
    T0 = 298.15
    return K1_at_298*np.exp(-(deltaH0a/R)*(1/x - 1/T0))

# Incorporate Cp dependence on temperature to find a more accurate K. Refrenced Van Ness book.
def calcK1a(x):
    T0 = 298.15 # Kelvin
    T = x
    deltaA = (AH2O+ACH4)-(ACO2+AH2)
    deltaB = ((BH2O+BCH4)-(BCO2+BH2))*1e-3
    deltaC = ((CH2O+CCH4)-(CCO2+CH2))*1e-6
    deltaD = ((DH2O+DCH4)-(DCO2+DH2))*1e5

    K0 = np.exp(-deltaG1_at_298/(R*T0))
    K1 = np.exp((deltaH0a/(R*T0))*(1-T0/T))
    K2 = np.exp(deltaA*(np.log(T/T0)-(T-T0)/T)+0.5*deltaB*(((T-T0)**2)/T)+1/6*deltaC*(((T-T0)**2)
                *(T+2*T0))/T + 0.5*deltaD*((T-T0)**2)/(T**2*T0**2))
    Knew = K0*K1*K2
    return Knew

# This is an alternate way to calculate K from cp according to Van Ness, but 
# it gives a different K value, so I opted for the method above
def calcK1b(x):
    T0 = 298.15
    T = x
    deltaA = (AH2O+ACH4)-(ACO2+AH2)
    deltaB = ((BH2O+BCH4)-(BCO2+BH2))*1e-3
    deltaC = ((CH2O+CCH4)-(CCO2+CH2))*1e-6
    deltaD = ((DH2O+DCH4)-(DCO2+DH2))*1e5
    cp = deltaA * deltaB*T * deltaC*T**2 + deltaD*T**-2
    answer = np.exp(-((deltaG1_at_298-deltaH0a)/(R*T0)+(deltaH0a/(R*T))+cp/(T*R)*(T-T0)-(cp/R)*(-1/T**2)*(T-T0)))
    return answer

# Use fsolve to find the extent of reaction. 35 bar is the pressure of the system
# This was set up by hand with an ICE table, but cannot be solved by hand.
def epssolver(eps,K):
    x = 8-2*eps
    return ((1+eps)/x * ((2+2*eps)/x)**2) / (((1-eps)/x) * ((4-4*eps)/x)**4) - 35*K

x = np.linspace(298,400,102)
y = calcK1a(x)
z = np.zeros_like(y)
guess = np.linspace(0.975,0.924,len(x)) #create a loop for epsilon according to Aspen senstivitiy data
for i in range(len(y)):
    z[i] = so.fsolve(epssolver,guess[i],args = y[i])

#compositions
y_ch4 = (1+z)/(8-2*z)
y_h2o = (2+2*z)/(8-2*z)
y_co2 = (1-z)/(8-2*z)
y_h2 = (4-4*z)/(8-2*z)

plt.figure(figsize=(6,5))
plt.plot(x,y,'-',color='blue')
plt.xlabel(r'Temperature, K',fontsize=14)
plt.ylabel('K',fontsize=14)

ax2 = plt.gca().twinx()
ax2.plot(x,z,'-',color='r',label='epsilon')
ax2.plot(x,y_ch4,'-',color='g',label='CH4')
ax2.plot(x,y_h2o,'-',color='m',label='H2O')
ax2.plot(x,y_co2,'-',color='k',label='CO2')
ax2.plot(x,y_h2,'-',color='y',label='H2')

ymin, ymax = ax2.get_ylim()
ax2.set_ylim((ymin,ymax))
ax2.set_ylabel(r'$\epsilon_e, y_{x}$',fontsize=14)
plt.subplots_adjust(bottom=0.2) 
plt.legend(loc="lower center", bbox_to_anchor=(0.5,-0.25), ncol=5)
plt.title('Sabatier Reaction, P = 35 bar')
plt.show()



#RWGS reaction CO2 + H2 <--> CO + H2O
deltaG2_at_298 = np.round((1*GH2O + 1*GCO)-(1*GH2 + 1*GCO2),2)
K2_at_298 = np.exp(-deltaG2_at_298/(R*T))
deltaH0b = (1*HH2O + 1*HCO)-(1*HH2 + 1*HCO2)

def calcK2(x):
    T0 = 298.15
    return K2_at_298*np.exp(-(deltaH0b/R)*(1/x - 1/T0))

def calcK2a(x):
    T0 = 298.15 # Kelvin
    T = x
    deltaA = (ACO+AH2O)-(ACO2+AH2)
    deltaB = ((BCO+BH2O)-(BCO2+BH2))*1e-3
    deltaC = ((CCO+CH2O)-(CCO2+CH2))*1e-6
    deltaD = ((DCO+DH2O)-(DCO2+DH2))*1e5

    K0 = np.exp(-deltaG2_at_298/(R*T0))
    K1 = np.exp((deltaH0b/(R*T0))*(1-T0/T))
    K2 = np.exp(deltaA*(np.log(T/T0)-(T-T0)/T)+0.5*deltaB*(((T-T0)**2)/T)+1/6*deltaC*(((T-T0)**2)
                *(T+2*T0))/T + 0.5*deltaD*((T-T0)**2)/(T**2*T0**2))
    Knew = K0*K1*K2
    return Knew

def epssolver(eps,K):
    x = 4 
    return ((1+eps)/x * ((1+eps)/x)) / (((1-eps)/x) * (1-eps)/x) - 35*K

x = np.linspace(298,400,102)
y = calcK2a(x)
z = np.zeros_like(y)
for i in range(len(y)):
    z[i] = so.fsolve(epssolver,0.8995,args = y[i])

#compositions
y_co2 = (1-z)/4
y_h2 = (1-z)/4
y_co = (1+z)/4
y_h2o = (1+z)/4

plt.figure(figsize=(6,5))
plt.plot(x,y,'-',color='blue')
plt.xlabel(r'Temperature, K',fontsize=14)
plt.ylabel('K',fontsize=14)

ax2 = plt.gca().twinx()
ax2.plot(x,z,'-',color='r',label='epsilon')
ax2.plot(x,y_co2,'-',color='g',label='CO2')
ax2.plot(x,y_h2o,'-',color='k',label='H2O')
ax2.plot(x,y_h2,'-',color='m',label='H2',alpha=0.5)
ax2.plot(x,y_co,'-',color='y',label='CO',alpha = 0.5)

ymin, ymax = ax2.get_ylim()
ax2.set_ylim((ymin,ymax))
ax2.set_ylabel(r'$\epsilon_e, y_{x}$',fontsize=14)
plt.title('RWGS reaction, P= 35 bar')
plt.show()



#Combination 1:1 2CO2 + 5H2 <--> CH4 + CO + 3H2O
deltaG3_at_298 = np.round((3*GH2O + 1*GCO + 1*GCH4)-(2*GCO2 + 5*GH2),2)
K3_at_298 = np.exp(-deltaG3_at_298/(R*T))
deltaH0c = (3*HH2O + 1*HCO + 1*HCH4)-(2*HCO2 + 5*HH2)

#K calculation
def calcK3(x):
    T0 = 298.15
    return K3_at_298*np.exp(-(deltaH0c/R)*(1/x - 1/T0))

def calcK3a(x):
    T0 = 298.15
    T = x
    deltaA = (3*AH2O + 1*ACO + 1*ACH4)-(2*ACO2 + 5*AH2)
    deltaB = ((3*BH2O + 1*BCO + 1*BCH4)-(2*BCO2 + 5*BH2))*1e-3
    deltaC = ((3*CH2O + 1*CCO + 1*CCH4)-(2*CCO2 + 5*CH2))*1e-6
    deltaD = ((3*DH2O + 1*DCO + 1*DCH4)-(2*DCO2 + 5*DH2))*1e5

    K0 = np.exp(-deltaG3_at_298/(R*T0))
    K1 = np.exp((deltaH0c/(R*T0))*(1-T0/T))
    K2 = np.exp(deltaA*(np.log(T/T0)-(T-T0)/T)+0.5*deltaB*(((T-T0)**2)/T)+1/6*deltaC*(((T-T0)**2)
                *(T+2*T0))/T + 0.5*deltaD*((T-T0)**2)/(T**2*T0**2))
    Knew = K0*K1*K2
    return Knew

def epssolver(eps,K):
    x = 12-2*eps
    return ((1+eps)/x * ((1+eps)/x) * ((3+3*eps)/x)**3) / ((((2-2*eps)/x)**2) * ((5-5*eps)/x)**5) - 35*K

x = np.linspace(298,400,102)
y = calcK3a(x)
z = np.zeros_like(y)
guess = np.linspace(0.876,0.773,len(x))
for i in range(len(y)):
    z[i] = so.fsolve(epssolver,guess[i],args = y[i])

#compositions
total = 12-2*z
y_co2 = (2-2*z)/total
y_h2 = (5-5*z)/total
y_ch4 = (1+z)/total
y_co = (1+z)/total
y_h2o = (3+3*z)/total

fig, axes = plt.subplots(2, 2,figsize=(10,8)) 
plt.subplot(2,2,1)
plt.plot(x,y,'-',color='blue')
plt.xlabel(r'Temperature, K',fontsize=14)
plt.ylabel('K',fontsize=14)

ax2 = plt.gca().twinx()
ax2.plot(x,z,'-',color='r',label='epsilon')
ax2.plot(x,y_co2,'-',color='g',label='CO2')
ax2.plot(x,y_h2,'-',color='k',label='H2')
ax2.plot(x,y_ch4,'-',color='m',label='CH4')
ax2.plot(x,y_co,'-',color='y',label='CO',alpha=0.5)
ax2.plot(x,y_h2o,'-',color='aqua',label='H2O')

ymin, ymax = ax2.get_ylim()
ax2.set_ylim((ymin,ymax))
ax2.set_ylabel(r'$\epsilon_e, y_{x}$',fontsize=14)
plt.title('1:1',fontsize=14,fontweight='bold')



# # Combination 1:2 3CO2 + 6H2 <--> CH4 + 2CO + 4H2O
deltaG4_at_298 = np.round((4*GH2O + 2*GCO + 1*GCH4)-(3*GCO2 + 6*GH2), 2)
K4_at_298 = np.exp(-deltaG4_at_298/(R*T))
deltaH0d = (4*HH2O + 2*HCO + 1*HCH4)-(3*HCO2 + 6*HH2)

# K calculation
def calcK4(x):
    T0 = 298.15
    return K4_at_298*np.exp(-(deltaH0d/R)*(1/x - 1/T0))

def calcK4a(x):
    T0 = 298.15  
    T = x
    deltaA = (4*AH2O + 2*ACO + 1*ACH4)-(3*ACO2 + 6*AH2)
    deltaB = ((4*BH2O + 2*BCO + 1*BCH4)-(3*BCO2 + 6*BH2))*1e-3
    deltaC = ((4*CH2O + 2*CCO + 1*CCH4)-(3*CCO2 + 6*CH2))*1e-6
    deltaD = ((4*DH2O + 2*DCO + 1*DCH4)-(3*DCO2 + 6*DH2))*1e5

    K0 = np.exp(-deltaG4_at_298/(R*T0))
    K1 = np.exp((deltaH0d/(R*T0))*(1-T0/T))
    K2 = np.exp(deltaA*(np.log(T/T0)-(T-T0)/T)+0.5*deltaB*(((T-T0)**2)/T)+1/6*deltaC*(((T-T0)**2)
                * (T+2*T0))/T + 0.5*deltaD*((T-T0)**2)/(T**2*T0**2))
    Knew = K0*K1*K2
    return Knew

def epssolver(eps, K):
    x = 16-2*eps
    return ((1+eps)/x * ((2+2*eps)/x)**2 * ((4+4*eps)/x)**4) / ((((3-3*eps)/x)**3) * ((6-6*eps)/x)**6) - 35*K

x = np.linspace(298, 400, 102)
y = calcK4a(x)
z = np.zeros_like(y)
guess = np.linspace(0.27, 0.226, len(x))
for i in range(len(y)):
    z[i] = so.fsolve(epssolver, guess[i], args=y[i])

# compositions
total = 16-2*z
y_co2 = (3-3*z)/total
y_h2 = (6-6*z)/total
y_ch4 = (1+z)/total
y_co = (2+2*z)/total
y_h2o = (4+4*z)/total

#plt.figure(figsize=(6, 5))
plt.subplot(2,2,2)
plt.plot(x, y, '-', color='blue')
plt.xlabel(r'Temperature, K',fontsize=14)
plt.ylabel('K',fontsize=14)

ax2 = plt.gca().twinx()
ax2.plot(x, z, '-', color='r', label='epsilon')
ax2.plot(x, y_co2, '-', color='g', label='CO2')
ax2.plot(x, y_h2, '-', color='k', label='H2')
ax2.plot(x, y_ch4, '-', color='m', label='CH4')
ax2.plot(x, y_co, '-', color='y', label='CO')
ax2.plot(x, y_h2o, '-', color='aqua', label='H2O')

ymin, ymax = ax2.get_ylim()
ax2.set_ylim((ymin, ymax))
ax2.set_ylabel(r'$\epsilon_e, y_{x}$',fontsize=14)
plt.title('1:2',fontsize=14,fontweight='bold')


# #Combination 1:3 4CO2 + 7H2 <--> CH4 + 3CO + 5H2O
deltaG5_at_298 = np.round((5*GH2O + 3*GCO + 1*GCH4)-(4*GCO2 + 7*GH2),2)
K5_at_298 = np.exp(-deltaG5_at_298/(R*T))
deltaH0e = (5*HH2O + 3*HCO + 1*HCH4)-(4*HCO2 + 7*HH2)

#K calculation
def calcK5(x):
    T0 = 298.15
    return K5_at_298*np.exp(-(deltaH0e/R)*(1/x - 1/T0))

def calcK5a(x):
    T0 = 298.15 # Kelvin
    T = x
    deltaA = (5*AH2O + 3*ACO + 1*ACH4)-(4*ACO2 + 7*AH2)
    deltaB = ((5*BH2O + 3*BCO + 1*BCH4)-(4*BCO2 + 7*BH2))*1e-3
    deltaC = ((5*CH2O + 3*CCO + 1*CCH4)-(4*CCO2 + 7*CH2))*1e-6
    deltaD = ((5*DH2O + 3*DCO + 1*DCH4)-(4*DCO2 + 7*DH2))*1e5

    K0 = np.exp(-deltaG5_at_298/(R*T0))
    K1 = np.exp((deltaH0e/(R*T0))*(1-T0/T))
    K2 = np.exp(deltaA*(np.log(T/T0)-(T-T0)/T)+0.5*deltaB*(((T-T0)**2)/T)+1/6*deltaC*(((T-T0)**2)
                *(T+2*T0))/T + 0.5*deltaD*((T-T0)**2)/(T**2*T0**2))
    Knew = K0*K1*K2
    return Knew

def epssolver(eps,K):
    x = 20-2*eps
    return ((1+eps)/x * ((3+3*eps)/x)**3 * ((5+5*eps)/x)**5) / ((((4-4*eps)/x)**4) * ((7-7*eps)/x)**7) - 35*K

x = np.linspace(298,400,102)
y = calcK5a(x)
z = np.zeros_like(y)
guess = np.linspace(0.41,0.34,len(x))
for i in range(len(y)):
    z[i] = so.fsolve(epssolver,guess[i],args = y[i])

#compositions
total = 20-2*z
y_co2 = (4-4*z)/total
y_h2 = (7-7*z)/total
y_ch4 = (1+z)/total
y_co = (3+3*z)/total
y_h2o = (5+5*z)/total

#plt.figure(figsize=(6,5))
plt.subplot(2,2,3)
plt.plot(x,y,'-',color='blue')
plt.xlabel(r'Temperature, K',fontsize=14)
plt.ylabel('K',fontsize=14)

ax2 = plt.gca().twinx()
ax2.plot(x,z,'-',color='r',label='epsilon')
ax2.plot(x,y_co2,'-',color='g',label='CO2')
ax2.plot(x,y_h2,'-',color='k',label='H2')
ax2.plot(x,y_ch4,'-',color='m',label='CH4')
ax2.plot(x,y_co,'-',color='y',label='CO')
ax2.plot(x,y_h2o,'-',color='aqua',label='H2O')

ymin, ymax = ax2.get_ylim()
ax2.set_ylim((ymin,ymax))
ax2.set_ylabel(r'$\epsilon_e, y_{x}$',fontsize=14)
plt.title('1:3',fontsize=14,fontweight='bold')


# # excess co2, 1:2
deltaG6_at_298 = np.round((4*GH2O + 2*GCO + 1*GCH4)-(3*GCO2 + 6*GH2),2)
K6_at_298 = np.exp(-deltaG6_at_298/(R*T))
deltaH0f = (4*HH2O + 2*HCO + 1*HCH4)-(3*HCO2 + 6*HH2)

#K calculation
def calcK6(x):
    T0 = 298.15
    return K6_at_298*np.exp(-(deltaH0f/R)*(1/x - 1/T0))

def calcK6a(x):
    T0 = 298.15 # Kelvin
    T = x
    deltaA = (4*AH2O + 2*ACO + 1*ACH4)-(3*ACO2 + 6*AH2)
    deltaB = ((4*BH2O + 2*BCO + 1*BCH4)-(3*BCO2 + 6*BH2))*1e-3
    deltaC = ((4*CH2O + 2*CCO + 1*CCH4)-(3*CCO2 + 6*CH2))*1e-6
    deltaD = ((4*DH2O + 2*DCO + 1*DCH4)-(3*DCO2 + 6*DH2))*1e5

    K0 = np.exp(-deltaG6_at_298/(R*T0))
    K1 = np.exp((deltaH0f/(R*T0))*(1-T0/T))
    K2 = np.exp(deltaA*(np.log(T/T0)-(T-T0)/T)+0.5*deltaB*(((T-T0)**2)/T)+1/6*deltaC*(((T-T0)**2)
                *(T+2*T0))/T + 0.5*deltaD*((T-T0)**2)/(T**2*T0**2))
    Knew = K0*K1*K2
    return Knew

def epssolver(eps,K):
    x = 22-2*eps
    return ((1+eps)/x * ((2+2*eps)/x)**2 * ((4+4*eps)/x)**4) / ((((9-3*eps)/x)**3) * ((6-6*eps)/x)**6) - 35*K

x = np.linspace(298,400,102)
y = calcK6a(x)
z = np.zeros_like(y)
guess = np.linspace(0.855,0.768,len(x))
for i in range(len(y)):
    z[i] = so.fsolve(epssolver,guess[i],args = y[i])

#compositions
total = 22-2*z
y_co2 = (9-3*z)/total
y_h2 = (6-6*z)/total
y_ch4 = (1+z)/total
y_co = (2+2*z)/total
y_h2o = (4+4*z)/total

plt.subplot(2,2,4)
plt.plot(x,y,'-',color='blue')
plt.xlabel(r'Temperature, K',fontsize=14)
plt.ylabel('K',fontsize=14)

ax2 = plt.gca().twinx()
ax2.plot(x,z,'-',color='r',label='epsilon')
ax2.plot(x,y_co2,'-',color='g',label='CO2')
ax2.plot(x,y_h2,'-',color='k',label='H2')
ax2.plot(x,y_ch4,'-',color='m',label='CH4')
ax2.plot(x,y_co,'-',color='y',label='CO')
ax2.plot(x,y_h2o,'-',color='aqua',label='H2O')

ymin, ymax = ax2.get_ylim()
ax2.set_ylim((ymin,ymax))
ax2.set_ylabel(r'$\epsilon_e, y_{x}$',fontsize=14) 
plt.legend(loc="lower center", bbox_to_anchor=(-0.1,-0.3), ncol=6)
plt.title('1:2, Excess CO2',fontsize=14,fontweight='bold')
plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(wspace=0.35)
plt.suptitle('Reaction Equilibria of Ratios of Sabatier and RWGS Reactions, P = 35 bar',fontsize=16,fontweight='bold')
plt.show()
plt.tight_layout()


#plot the error between K(Cp) and K, one reaction at a time
def error_singular(rxn):
    x = np.linspace(298,400,102)
    i = rxn - 1
    z = np.array([calcK1a(x)-calcK1(x),calcK2a(x)-calcK2(x),calcK3a(x)-calcK3(x),
                  calcK4a(x)-calcK4(x),calcK5a(x)-calcK5(x),calcK6a(x)-calcK6(x)])

    min_y = np.min(z[i])
    min_index = np.argmin(z[i])
    min_x = x[min_index]
    plt.plot(min_x,min_y,'ro',label=f'Min Error @ {min_x:.2f} C')
        
    max_y = np.max(z[i])
    max_index = np.argmax(z[i])
    max_x = x[max_index]
    plt.plot(max_x,max_y,'ro',label=f'Max Error @ {max_x:.2f} C')
    
    plt.plot(x,z[i])
    plt.xlabel('Temperature, K')
    plt.ylabel('Difference')
    plt.title(f'Associated Error of Rxn {rxn} assuming constant heat capacity')
    plt.legend()
    plt.show()
    

#plot the error of all reactions in a subplot
def error_all(figsize=(10, 6),nrows=3,ncols=2, sharex=True, sharey=False):

  x = np.linspace(298, 400, 102)
  z = np.array([calcK1a(x) - calcK1(x), calcK2a(x) - calcK2(x), calcK3a(x) - calcK3(x),
                calcK4a(x) - calcK4(x), calcK5a(x) - calcK5(x), calcK6a(x) - calcK6(x)])

  fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
  fig.suptitle('Associated Error of Reactions assuming constant heat capacity', fontsize=14,fontweight='bold')  # Add overall title

  # Loop through each subplot and plot the error for each reaction
  for idx, ax in enumerate(axes.ravel()):
    if idx >= len(z):  
      break
    rxn_num = idx + 1 

    min_y = np.min(z[idx])
    min_index = np.argmin(z[idx])
    min_x = x[min_index]

    max_y = np.max(z[idx])
    max_index = np.argmax(z[idx])
    max_x = x[max_index]

    ax.plot(min_x, min_y, 'ro', label=f'Min Error @ {min_x:.2f} C')
    ax.plot(max_x, max_y, 'ro', label=f'Max Error @ {max_x:.2f} C')
    ax.plot(x, z[idx])
    ax.set_xlabel('Temperature, K')
    ax.set_ylabel('Difference')
    ax.set_title(f'Reaction {rxn_num}')
    ax.legend()

  plt.tight_layout()
  plt.show()

error_all()





















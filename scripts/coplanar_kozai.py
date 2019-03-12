'''
This script carries out the high-eccentricity migration of a coplanar 
two-planet system, following the example illustrated in figure 5 of
Petrovich (2015), ApJ (805:75)

'''

import kozaipy as kp
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    mjup= kp.constants.Mjup
    trip = kp.Triple(m0=1.0,m1=2*mjup,m2=3.3*mjup,
                     a1=1.0,a2=8.0,e1=0.51,e2=0.51,\
                     I=5.0 * np.pi/180.0, g1=0,g2=0,h1=0,h2=np.pi,
                     type0='star',type1='planet',
                     spinorbit_align1=True,
                     spin_rate0 = 2 * np.pi/10, spin_rate1 = 2 * np.pi/0.417,
                     R0=kp.constants.Rsun,R1=kp.constants.Rsun/10,
                     k2_0 = 0.014, k2_1=0.25,
                     tv0=1.825e4,tv1=10.95,
                     rg_0=0.08,rg_1=0.25)

    sol = trip.integrate(timemin=0.0,timemax=100.0e6*365.25,
                         Nevals=30000,octupole_potential=True,
                         short_range_forces_conservative=True, \
                         short_range_forces_dissipative=True)
    sol.to_elements()
    time = sol.vectordata.time
    incl1= sol.elementdata.I1
    ecc1 = sol.elementdata.e1
    incl2= sol.elementdata.I2
    ecc2 = sol.elementdata.e2

    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(121)
    ax.set_xlabel("time[Myr]")
    ax.set_ylabel("eccentricity")
    ax.plot(time/365.25e6,ecc1)
    ax.plot(time/365.25e6,ecc2)

    ax = plt.subplot(122)
    ax.set_xlabel("time[Myr]")
    ax.set_ylabel(r'inclination[$^\circ$]')
    ax.plot(time/365.25e6,incl1+incl2)
    
    plt.show()

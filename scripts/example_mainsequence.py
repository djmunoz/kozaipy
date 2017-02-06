import numpy as np
import kozaipy as kp
import matplotlib.pyplot as plt

if __name__ == "__main__":

    trip = kp.Triple(m0=1.1,m1=0.007443,m2=1.1,a1=5.0,a2=1000.0,e1=0.1,e2=0.5,I=85.6 * np.pi/180.0, \
                     g1=45.0 * np.pi/180.0,g2=0.0,\
                     type0='star',type1='planet',\
                     spinorbit_align0=False,\
                     spinorbit_align1=True,\
                     spin_rate0 = 2 * np.pi/20, spin_rate1 = 2 * np.pi/0.417,
                     R0=kp.constants.Rsun,R1=kp.constants.Rsun/10,
                     k2_0 = 0.014, k2_1=0.25, tv0=2.0e4, tv1=0.365242,rg_0=0.08,rg_1=0.25)
    

        
    
    sol = trip.integrate(timemin=0,timemax=4.1e9*365.25,Nevals=12000,\
                         octupole_potential=False,\
                         short_range_forces_conservative=True, \
                         short_range_forces_dissipative=True,\
                         solve_for_spin_vector=False,
                         version='tides')

    sol.to_elements()
    time = sol.vectordata.time
    ecc1 = sol.elementdata.e1
    ecc2 = sol.elementdata.e2
    incl1 = sol.elementdata.I1
    incl2 = sol.elementdata.I2
    a1 = sol.elementdata.a1
    try:
        spin_period0 = 2*np.pi/np.sqrt(sol.vectordata.Omega0x**2 + sol.vectordata.Omega0y**2 +sol.vectordata.Omega0z**2)
    except TypeError:
        spin_period0 = 2*np.pi/sol.vectordata.Omega0
    try:
        spin_period1 = 2*np.pi/np.sqrt(sol.vectordata.Omega0x**2 + sol.vectordata.Omega0y**2 +sol.vectordata.Omega0z**2)
    except TypeError:
        spin_period1 = 2*np.pi/sol.vectordata.Omega1

    try:
        obliq0 = np.arccos((sol.vectordata.Omega0x * sol.vectordata.l1x + sol.vectordata.Omega0y * sol.vectordata.l1y + sol.vectordata.Omega0z * sol.vectordata.l1z)/np.sqrt(sol.vectordata.Omega0x**2 + sol.vectordata.Omega0y**2 +sol.vectordata.Omega0z**2)/np.sqrt(sol.vectordata.l1x**2 + sol.vectordata.l1y**2 +sol.vectordata.l1z**2))
    except TypeError:
        obliq0 = np.repeat(0,time.shape[0])
        
    fig = plt.figure(figsize=(16,12))
    fig.subplots_adjust(right=0.97,left=0.1,top=0.94,bottom=0.1)
    ax = fig.add_subplot(321)
    ax.plot(time/365.25,ecc1)
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel('eccentricity',size=20)
    #
    ax = fig.add_subplot(322)
    ax.plot(time/365.25,a1)
    ax.plot(time/365.25,a1*(1-ecc1))
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel(r'$a,\,a(1-e)$',size=20)
    #
    ax = fig.add_subplot(323)
    ax.plot(time/365.25,incl1)
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel('inclination[deg]',size=20)
    #
    ax = fig.add_subplot(324)    
    ax.plot(time/365.25,obliq0*180.0/np.pi)
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel('spin-orbit angle [deg]',size=20)

 #
    ax = fig.add_subplot(325)    
    ax.plot(time/365.25,spin_period0)
    f2 = 1 + 7.5 * ecc1**2 + 5.625 * ecc1**4 + 0.3125 * ecc1**6
    f5 = 1 + 3.0 * ecc1**2 + 0.375 * ecc1**4
    spin_period1_ps = 2*np.pi/(f2/f5/(1-ecc1**2)**1.5 * np.sqrt(kp.constants.G*(trip.m0+trip.m1)/a1**3))
    ax.plot(time/365.25,spin_period1)
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel('spin period',size=20)
    ax.set_ylim(0,25)
    
    fig.savefig('migration_mainsequence.png')


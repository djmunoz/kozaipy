import numpy as np
import kozaipy as kp
import matplotlib.pyplot as plt

if __name__ == "__main__":


    radius_function,  dradius_dt_function = kp.interpolate_radius()
    gyroradius_function,  dgyroradius_dt_function = kp.interpolate_gyroradius()

    lagtime_function, dlagtime_dt_function =  kp.interpolate_lagtime()

    
    trip0 = kp.Triple(m0=1.0,m1=0.003,m2=0.050,a1=2.0,a2=40.0,e1=0.5,e2=0.5,I=85.0 * np.pi/180.0, \
                      g1=5.1381,g2=1.6382,\
                      type0='star',type1='planet',\
                      spinorbit_align1 = True,\
                      spin_rate0 = 2 * np.pi/20, spin_rate1 = 2 * np.pi/0.417)

    protostar = kp.Body(mass=1.0,radius=radius_function,dradius_dt=dradius_dt_function,
                        gyroradius=gyroradius_function,dgyroradius_dt=dgyroradius_dt_function,
                        apsidal_constant=0.07,tidal_lag_time=lagtime_function,mass_type='star')

    planet = kp.Body(mass=trip0.m1,radius = kp.constants.Rsun/10,
                     apsidal_constant=0.255,tidal_lag_time=1.0e-8*365.25,gyroradius=0.25,mass_type='planet')
    star = kp.Body(mass=trip0.m0,radius = 1.0/215,
                   apsidal_constant=0.014,tidal_lag_time=1.0e-8*365.25,gyroradius=0.08,mass_type='star')
    
    
    #trip0.properties0 = protostar
    trip0.properties0 = star
    trip0.properties1 = planet
    
    sol = trip0.integrate(timemin=1.0e6*365.25,timemax=10.0e6*365.25,Nevals=100000,\
                          octupole_potential=False,\
                          short_range_forces_conservative=True, \
                          short_range_forces_dissipative=True,\
                          solve_for_spin_vector=False,
                          version='full')

    sol.to_elements()
    sol.save_to_file("contracting_star.txt")
    
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
        spin_period1 = 2*np.pi/np.sqrt(sol.vectordata.Omega1x**2 + sol.vectordata.Omega1y**2 +sol.vectordata.Omega1z**2)
    except TypeError:
        spin_period1 = 2*np.pi/sol.vectordata.Omega1

    try:
        obliq0 = np.arccos((sol.vectordata.Omega0x * sol.vectordata.l1x + sol.vectordata.Omega0y * sol.vectordata.l1y + sol.vectordata.Omega0z * sol.vectordata.l1z)/np.sqrt(sol.vectordata.Omega0x**2 + sol.vectordata.Omega0y**2 +sol.vectordata.Omega0z**2)/np.sqrt(sol.vectordata.l1x**2 + sol.vectordata.l1y**2 +sol.vectordata.l1z**2))
    except TypeError:
        obliq0 = np.repeat(0,time.shape[0])
        
    fig = plt.figure(figsize=(16,12))
    fig.subplots_adjust(right=0.97,left=0.1,top=0.94,bottom=0.1)
    ax = fig.add_subplot(321)
    ax.plot(time/365.25,1-ecc1)
    ax.set_yscale('log')
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel('eccentricity',size=20)
    #
    ax = fig.add_subplot(322)
    ax.plot(time/365.25,a1)
    ax.plot(time/365.25,a1*(1-ecc1))
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel(r'$a,\,a(1-e)$',size=20)
    ax.set_yscale('log')
    #
    ax = fig.add_subplot(323)
    ax.plot(time/365.25,incl1+incl2)
    #ax.set_ylim(83,85.3)
    ax.set_ylim(45,47)
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
    orbital_period1= 2*np.pi/np.sqrt(kp.constants.G*(trip0.m0+trip0.m1)/a1**3)
    spin_period1_ps =  orbital_period1/(f2/f5/(1-ecc1**2)**1.5)
    ax.plot(time/365.25,spin_period1)
    ax.plot(time/365.25,orbital_period1,'k--')
    ax.set_xlabel('time[yr]',size=20)
    ax.set_ylabel('spin period',size=20)
    ax.set_yscale('log')
    ax.set_ylim(0,25)
    
    fig.savefig('kctf_standard.pdf')



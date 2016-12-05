import numpy as np
import secular_triple_systems as ts
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #t = ts.Triple(m0=1.0,m1=0.001,m2=0.04,a1=6.0,a2=100.0,e1=0.001,e2=0.6,I=65.0*np.pi/180.0,
    #              g1=45.0*np.pi/180.0,g2=0)
    #t = ts.Triple(m0=1.0,m1=0.001,m2=0.002,a1=4.0,a2=45.0,e1=0.01,e2=0.6,I=67.0*np.pi/180.0,
    #              g1=180.0*np.pi/180.0,g2=0)
    #t = ts.Triple(m0=1.0,m1=0.25,m2=0.6,a1=60.0,a2=800.0,e1=0.01,e2=0.6,I=98.0*np.pi/180.0,
    #              g1=0.0*np.pi/180.0,g2=0)
    #t = ts.Triple(m0=1.0,m1=1.0e-10,m2=0.001,a1=2.0,a2=5.0,e1=0.2,e2=0.05,I=65.0*np.pi/180.0,
    #              g1=0,g2=0,h1=np.pi)
    t = ts.Triple(m0=1.0,m1=1.0e-3,m2=1.1,a1=5.0,a2=1000.0,e1=0.001,e2=0.5,I=86.5*np.pi/180.0,
                  g1=45*np.pi/180,g2=0,h1=np.pi,R0=4.6491E-3,R1=4.6491E-4)
    
    #sol = t.integrate(timemax=2.5e7*365.25,Nevals=20000,octupole_potential=True)
    #sol = t.integrate(timemax=4.e8*365.25,Nevals=30000,octupole_potential=True)
    #sol = t.integrate(timemax=8.e7*365.25,Nevals=40000,octupole_potential=True)
    #sol = t.integrate(timemax=4.0e6*365.25,Nevals=50000,octupole_potential=True)
    sol = t.integrate(timemax=1.5e9*365.25,Nevals=70000,octupole_potential=True,
                      short_range_forces_conservative=True, short_range_forces_dissipative=True)

    sol.to_elements()
    sol.compute_potential(octupole=True)
    sol.save_to_file("test.txt",Nlines=None)
    time = sol.vectordata.time
    ecc1 = sol.elementdata.e1
    ecc2 = sol.elementdata.e2
    incl1 = sol.elementdata.I1
    incl2 = sol.elementdata.I2

    print incl1
    print incl2
    print incl1.max()
    plt.plot(time/365.25,incl1)#+incl2)
    plt.show()
    plt.plot(time/365.25,incl2)
    print ecc2
    #plt.plot(time,ecc1)
    #plt.plot(time,np.repeat(np.sqrt(1-5.0/3*np.cos(t.I)**2),time.shape))
    #plt.plot(time,1-ecc1)
    #plt.yscale('log')
    plt.show()
    plt.plot(time,sol.potential)
    plt.show()
    print sol.potential

"""
This module defines data structure for hierarchical triple systems.

"""

import numpy as np
import numpy.random as rd
from triples_integrate import integrate_triple_system



#__all__ = ['Triple','TripleSolution','PhysicalProperties']


class Constants(object):

    def __init__(self,*args,**kwargs):

        # Units relative to GGS
        self.G = 2.959122081E-4
        self.CLIGHT = 1.73144633E+2
        self.Rsun = 4.6491E-3


        # Set defaults
        #self.MassUnit = 
        #self.VelocityUnit
        
constants = Constants()

class VectorData(object):
    def __init__(self,*args,**kwargs):
        self.time = kwargs.get("time")
        # for the inner orbit
        self.e1x = kwargs.get("e1x")
        self.e1y = kwargs.get("e1y")
        self.e1z = kwargs.get("e1z")
        self.l1x = kwargs.get("l1x")
        self.l1y = kwargs.get("l1y")
        self.l1z = kwargs.get("l1z")        
        self.spin1x = kwargs.get("spin1x")
        self.spin1y = kwargs.get("spin1y")
        self.spin1z = kwargs.get("spin1z")
        self.spin2x = kwargs.get("spin2x")
        self.spin2y = kwargs.get("spin2y")
        self.spin2z = kwargs.get("spin2z")        
        # for the outer orbit
        self.e2x = kwargs.get("e2x")
        self.e2y = kwargs.get("e2y")
        self.e2z = kwargs.get("e2z")
        self.l2x = kwargs.get("l2x")
        self.l2y = kwargs.get("l2y")
        self.l2z = kwargs.get("l2z")

class ElementData(object):
    def __init__(self,*args,**kwargs):
        self.time = kwargs.get("time")
        # for the inner orbit
        self.e1 = kwargs.get("e1")
        self.a1 = kwargs.get("a1")
        self.I1 = kwargs.get("I1")
        self.g1 = kwargs.get("g1")
        self.h1 = kwargs.get("h1")        
        self.spin1x = kwargs.get("spin1x")
        self.spin1y = kwargs.get("spin1y")
        self.spin1z = kwargs.get("spin1z")
        self.spin2x = kwargs.get("spin2x")
        self.spin2y = kwargs.get("spin2y")
        self.spin2z = kwargs.get("spin2z")        
        # for the outer orbit
        self.e2 = kwargs.get("e2")
        self.a2 = kwargs.get("a2")
        self.I2 = kwargs.get("I2")
        self.g2 = kwargs.get("g2")
        self.h2 = kwargs.get("h2")

    
class TripleSolution(object):

    """
    Class for storing the numerical solution to the triple evolution equations

    """

    def __init__(self,Triple,vector_solution=None,*args,**kwargs):
        """
        Load orbital vector data as a function of time, obtained from a numerical solution
        of the secular equations of motion.

        """

        self.triple = Triple
        self.vectordata =  VectorData()
        self.elementdata =  ElementData()
        if (vector_solution is None):
            # individually supplied orbital data
            # data *arrays*
            self.vectordata.time = kwargs.get("time")
            # for the inner orbit
            self.vectordata.e1x = kwargs.get("e1x")
            self.vectordata.e1y = kwargs.get("e1y")
            self.vectordata.e1z = kwargs.get("e1z")
            self.vectordata.l1x = kwargs.get("l1x")
            self.vectordata.l1y = kwargs.get("l1y")
            self.vectordata.l1z = kwargs.get("l1z")        
            self.vectordata.spin1x = kwargs.get("spin1x")
            self.vectordata.spin1y = kwargs.get("spin1y")
            self.vectordata.spin1z = kwargs.get("spin1z")
            self.vectordata.spin2x = kwargs.get("spin2x")
            self.vectordata.spin2y = kwargs.get("spin2y")
            self.vectordata.spin2z = kwargs.get("spin2z")        
            # for the outer orbit
            self.vectordata.e2x = kwargs.get("e2x")
            self.vectordata.e2y = kwargs.get("e2y")
            self.vectordata.e2z = kwargs.get("e2z")
            self.vectordata.l2x = kwargs.get("l2x")
            self.vectordata.l2y = kwargs.get("l2y")
            self.vectordata.l2z = kwargs.get("l2z")
        else:
            # data supplied at once as a matrix
            self.vectordata.time = vector_solution[:,0] 
            # for the inner orbit
            self.vectordata.e1x = vector_solution[:,1]
            self.vectordata.e1y = vector_solution[:,2]
            self.vectordata.e1z = vector_solution[:,3]
            self.vectordata.l1x = vector_solution[:,4]
            self.vectordata.l1y = vector_solution[:,5]
            self.vectordata.l1z = vector_solution[:,6]
            self.vectordata.spin1x = vector_solution[:,13] 
            self.vectordata.spin1y = vector_solution[:,14]
            self.vectordata.spin1z = vector_solution[:,15]
            self.vectordata.spin2x = vector_solution[:,16]
            self.vectordata.spin2y = vector_solution[:,17]
            self.vectordata.spin2z = vector_solution[:,18]
            # for the outer orbit
            self.vectordata.e2x = vector_solution[:,7]
            self.vectordata.e2y = vector_solution[:,8]
            self.vectordata.e2z = vector_solution[:,9]
            self.vectordata.l2x = vector_solution[:,10]
            self.vectordata.l2y = vector_solution[:,11]
            self.vectordata.l2z = vector_solution[:,12]

    def to_elements(self,to_degrees = True):
        """
        Convert the orbital vector quantities to conventional orbital elements.

        """

        # for the inner orbit
        self.elementdata.e1 = np.sqrt(self.vectordata.e1x[:]**2 + self.vectordata.e1y[:]**2 + self.vectordata.e1z[:]**2)
        self.elementdata.a1 = (self.vectordata.l1x[:]**2 + self.vectordata.l1y[:]**2 + self.vectordata.l1z[:]**2)/constants.G /\
                              (self.triple.m0 + self.triple.m1) / (1 - self.elementdata.e1[:] * self.elementdata.e1[:])
        self.elementdata.I1 = np.arccos(self.vectordata.l1z[:]/np.sqrt(self.vectordata.l1x[:]**2 + self.vectordata.l1y[:]**2 + self.vectordata.l1z[:]**2))
        self.elementdata.h1 = np.arctan2(self.vectordata.l1x[:],-self.vectordata.l1y[:])
        self.elementdata.g1 = np.arctan2((-self.vectordata.e1x[:] * np.sin(self.elementdata.h1[:]) + \
                                           self.vectordata.e1y[:] * np.cos(self.elementdata.h1[:]))* np.cos(self.elementdata.I1[:]) +\
                                          self.vectordata.e1z[:] * np.sin(self.elementdata.I1[:]),
                                          self.vectordata.e1x[:] * np.cos(self.elementdata.h1[:]) + self.vectordata.e1y * np.sin(self.elementdata.h1[:]))

        # for the outer orbit
        self.elementdata.e2 = np.sqrt(self.vectordata.e2x[:]**2 + self.vectordata.e2y[:]**2 + self.vectordata.e2z[:]**2)
        self.elementdata.a2 = (self.vectordata.l2x[:]**2 + self.vectordata.l2y[:]**2 + self.vectordata.l2z[:]**2)/constants.G /\
                              (self.triple.m0 + self.triple.m1 + self.triple.m2) / (2 - self.elementdata.e2[:] * self.elementdata.e2[:])
        self.elementdata.I2 = np.arccos(self.vectordata.l2z[:]/np.sqrt(self.vectordata.l2x[:]**2 + self.vectordata.l2y[:]**2 + self.vectordata.l2z[:]**2))
        self.elementdata.h2 = np.arctan2(self.vectordata.l2x[:],-self.vectordata.l2y[:])
        self.elementdata.g2 = np.arctan2((-self.vectordata.e2x[:] * np.sin(self.elementdata.h2[:]) + \
                                           self.vectordata.e2y[:] * np.cos(self.elementdata.h2[:]))* np.cos(self.elementdata.I2[:]) +\
                                          self.vectordata.e2z[:] * np.sin(self.elementdata.I2[:]),
                                          self.vectordata.e2x[:] * np.cos(self.elementdata.h2[:]) + self.vectordata.e2y * np.sin(self.elementdata.h2[:]))

        
        if (to_degrees):
            self.elementdata.I1 *= 180.0 / np.pi 
            self.elementdata.h1 *= 180.0 / np.pi 
            self.elementdata.g1 *= 180.0 / np.pi 
            self.elementdata.I2 *= 180.0 / np.pi 
            self.elementdata.h2 *= 180.0 / np.pi 
            self.elementdata.g2 *= 180.0 / np.pi              

        self.elementdata.spin1x = self.vectordata.spin1x
        self.elementdata.spin1y = self.vectordata.spin1y            
        self.elementdata.spin1z = self.vectordata.spin1z            
        self.elementdata.spin2x = self.vectordata.spin2x
        self.elementdata.spin2y = self.vectordata.spin2y            
        self.elementdata.spin2z = self.vectordata.spin2z

    def compute_potential(self,octupole = True):
        self.potential = compute_quadrupole_potential(self.triple.m0,self.triple.m1,self.triple.m2,
                                                      self.vectordata.e1x,self.vectordata.e1y,self.vectordata.e1z,
                                                      self.vectordata.l1x,self.vectordata.l1y,self.vectordata.l1z,
                                                      self.vectordata.e2x,self.vectordata.e2y,self.vectordata.e2z,
                                                      self.vectordata.l2x,self.vectordata.l2y,self.vectordata.l2z)

        if (octupole is True):
            print "hehe"
            oct_pot = compute_octupole_potential(self.triple.m0,self.triple.m1,self.triple.m2,
                                                 self.vectordata.e1x,self.vectordata.e1y,self.vectordata.e1z,
                                                 self.vectordata.l1x,self.vectordata.l1y,self.vectordata.l1z,
                                                 self.vectordata.e2x,self.vectordata.e2y,self.vectordata.e2z,
                                                 self.vectordata.l2x,self.vectordata.l2y,self.vectordata.l2z)
            self.potential[:] = self.potential[:] + oct_pot[:]
        
        
        
        
    def save_to_file(self,filename,header=None,Nlines=None):
        """
        Function to save the solution into an ASCII file

        """
        if (Nlines is None):
            Nlines = len(self.vectordata.time)

        print Nlines
        f = open(filename,'w')
        fmt_list = []
        f.write("# time\t\t")

        data = self.vectordata.time.reshape(len(self.vectordata.time),1)
        fmt_list.append('%13.8e  ')

        for k,v in self.elementdata.__dict__.items():
            if (v is not None):
                if (k == 'time'):
                    continue
                if (k == 'a1'): fmt_list.append('%12.8f  ')
                if (k == 'a2'): fmt_list.append('%12.8f  ')
                if (k == 'e1'): fmt_list.append('%10.8f  ')
                if (k == 'e2'): fmt_list.append('%10.8f  ')
                if (k == 'I1'): fmt_list.append('%10.6f  ')
                if (k == 'I2'): fmt_list.append('%10.6f  ')
                if (k == 'g1'): fmt_list.append('%10.6f  ')
                if (k == 'g2'): fmt_list.append('%10.6f  ')
                if (k == 'h1'): fmt_list.append('%10.6f  ')
                if (k == 'h2'): fmt_list.append('%10.6f  ')
                if ('spin' in k): fmt_list.append('%12.6f  ')
                f.write(k+"\t\t")
                data = np.column_stack((data,v))
        f.write("\n")
        f.write("#--------------------------------------------------------------------------\n")
        np.savetxt(f,data[::int(len(self.vectordata.time)/Nlines),:],fmt="".join(fmt_list))
        

        
        

class Triple(object):

    """"
    Class  for generating a hierarchical triple system

    """
    
    def __init__(self, *args,**kwargs):

        # Main variables
        self.m0 = kwargs.get("m0") # mass of central object
        self.m1 = kwargs.get("m1") # mass of secondary
        self.m2 = kwargs.get("m2") # mass of tertiary
        self.a1 = kwargs.get("a1") # semimajor axis of inner orbit
        self.a2 = kwargs.get("a2") # semimajor axis of outer orbit
        self.e1 = kwargs.get("e1") # eccentricity of inner orbit
        self.e2 = kwargs.get("e2") # eccentricity of outer orbit
        self.I  = kwargs.get("I")  # mutual inclination
        self.g1 = kwargs.get("g1") # inner orbit argument of pericenter
        self.g2 = kwargs.get("g2") # outer orbit argument of pericenter
        self.h1 = kwargs.get("h1")  # inner orbit line of nodes
        self.h2 = None  # outer orbit line of nodes

        self.spin_rate0 = kwargs.get("spin_rate0") # Spin rate/angular frequency of first body
        self.spin_rate1 = kwargs.get("spin_rate1") # Spin rate of second body
        self.spin0 = kwargs.get("Spin0") # Spin *vector* for the first body
        self.spin1 = kwargs.get("Spin1") # Spin *vector* for the second body

        self.R0 = kwargs.get("R0") # radius of primary
        self.R1 = kwargs.get("R1") # radius of secondary
        
        # Secondary variables
        self.masstype0 = kwargs.get("masstype0")
        self.masstype1 = kwargs.get("masstype1")
        self.masstype2 = kwargs.get("masstype2")

        
        # Set default values
        if (self.m0 is None): self.m0 = 1.0
        if (self.m1 is None): self.m1 = 1.0
        if (self.m2 is None): self.m2 = 1.0
        if (self.a1 is None): self.a1 = 1.0
        if (self.a2 is None): self.a2 = 100.0            
        if (self.e1 is None): self.e1 = 1.0e-6
        if (self.e2 is None): self.e2 = 1.0e-6
        if (self.I is None): self.I = 0.0
        if (self.g1 is None): self.g1 = rd.random() * 2 * np.pi
        if (self.g2 is None): self.g2 = rd.random() * 2 * np.pi
        if (self.h1 is None): self.h1 = rd.random() * 2 * np.pi
        if (self.h2 is None): self.h2 = self.h1 - np.pi
        if (self.spin_rate0 is None): self.spin_rate0 = 0.0
        if (self.spin_rate1 is None): self.spin_rate1 = 0.0
       

        if (self.R0 is None): self.R0 = constants.Rsun
        if (self.R1 is None): self.R1 = constants.Rsun

        # Compute individual inclinations
        L1 = self.m0 * self.m1/(self.m0 + self.m1) * np.sqrt((self.m0 + self.m1) * self.a1 * (1 - self.e1 * self.e1))
        L2 = (self.m0 + self.m1) * self.m2/(self.m0 + self.m1 + self.m2) * np.sqrt((self.m0 + self.m1 + self.m2) * self.a2 * (1 - self.e2 * self.e2))
        print L1,L2,self.m0,self.m1,self.m2
        L  = np.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * np.cos(self.I))
        self.I2 = np.arccos((L2 + L1 * np.cos(self.I))/L)
        self.I1 = self.I - self.I2

        # Compute the default spin *orientations*
        if (self.spin0 is None):
            spin0 = 
            self.spin0 = [self.spin_rate0 * np.sin(self.h1) * np.sin(self.I1),
                          -self.spin_rate0 * np.cos(self.h1) * np.sin(self.I1), 
                          self.spin_rate0 * np.cos(self.h1)]
        if (self.spin1 is None): 
            self.spin1 = [self.spin_rate1 * np.sin(self.h1) * np.sin(self.I1),
                          -self.spin_rate1 * np.cos(self.h1) * np.sin(self.I1), 
                          self.spin_rate1 * np.cos(self.h1)] 
        
    def vector_form(self):
        """
        Coordinate transformation from orbital elements to orbital vectors
        
        """
        
        # For the inner orbit
        e1x = self.e1 * (np.cos(self.g1) * np.cos(self.h1) - np.sin(self.g1) * np.sin(self.h1) * np.cos(self.I1))
        e1y = self.e1 * (np.cos(self.g1) * np.sin(self.h1) + np.sin(self.g1) * np.cos(self.h1) * np.cos(self.I1))
        e1z = self.e1 * np.sin(self.g1) * np.sin(self.I1)
        print e1x,e1y,e1z,np.sqrt(e1x**2+ e1y**2 + e1z**2)
        print self.h1,self.g1,self.I1
        
        l1  = np.sqrt(constants.G * self.a1 * (self.m0 + self.m1) * (1 - self.e1 * self.e1))
        l1x = l1 * np.sin(self.h1) * np.sin(self.I1)
        l1y = -l1 * np.cos(self.h1) * np.sin(self.I1)
        l1z = l1 * np.cos(self.I1)

        # For the outer orbit
        e2x = self.e2 * (np.cos(self.g2) * np.cos(self.h2) - np.sin(self.g2) * np.sin(self.h2) * np.cos(self.I2))
        e2y = self.e2 * (np.cos(self.g2) * np.sin(self.h2) + np.sin(self.g2) * np.cos(self.h2) * np.cos(self.I2))
        e2z = self.e2 * np.sin(self.g2) * np.sin(self.I2)

        l2  = np.sqrt(constants.G * self.a2 * (self.m0 + self.m1 + self.m2) * (1 - self.e2 * self.e2))
        l2x = l2 * np.sin(self.h2) * np.sin(self.I2)
        l2y = -l2 * np.cos(self.h2) * np.sin(self.I2)
        l2z = l2 * np.cos(self.I2)

        # Including the inner pair's spin angular momenta
        
        return [e1x,e1y,e1z,l1x,l1y,l1z,e2x,e2y,e2z,l2x,l2y,l2z,self.spin0[0],self.spin0[1],self.spin0[2],self.spin1[0],self.spin1[1],self.spin1[2]]
    
        
    def integrate(self,timemax,Nevals,octupole_potential= True,
                  short_range_forces_conservative=False,
                  short_range_forces_dissipative = False):
        """
        Integrate the system forward in time

        """

        vector_ics = self.vector_form()
        
        solution = integrate_triple_system(vector_ics,timemax,Nevals, self.m0,self.m1,self.m2,self.R0,self.R1,
                                           octupole_potential,\
                                           short_range_forces_conservative,short_range_forces_dissipative)
        

        triple_solution = TripleSolution(self,vector_solution = solution)
        
        return triple_solution


    
def compute_quadrupole_potential(m0,m1,m2,e1x,e1y,e1z,l1x,l1y,l1z,e2x,e2y,e2z,l2x,l2y,l2z):

    e1 = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    e2 = np.sqrt(e2x**2 + e2y**2 + e2z**2)
    l1 = np.sqrt(l1x**2 + l1y**2 + l1z**2)
    l2 = np.sqrt(l2x**2 + l2y**2 + l2z**2)

    u1x, u1y, u1z = e1x/e1, e1y/ e1, e1z/e1
    n1x, n1y, n1z = l1x/l1, l1y/ l1, l1z/l1
    n2x, n2y, n2z = l2x/l2, l2y/ l2, l2z/l2
    
    
    n1dotn2 = n1x * n2x + n1y * n2y + n1z * n2z
    u1dotn2 = u1x * n2x + u1y * n2y + u1z * n2z

    
    a1 = l1 * l1 / (1 - e1 * e1) / (constants.G * (m0 + m1))
    a2 = l2 * l2 / (1 - e2 * e2) /  (constants.G * (m0 + m1 + m2))
    
    pot0 = -constants.G * m0 * m1 * m2 / (m0 + m1) * a1 * a1 / a2 / a2 /a2\
          / np.sqrt(1 - e2 * e2) / (1 - e2 * e2) 


    pot = pot0 / 8 * (1 - 6 * e1 * e1 - 3 * (1 - e1 * e1) * n1dotn2 * n1dotn2 +\
                      15 * e1 * e1 * u1dotn2 * u1dotn2)

    return pot


def compute_octupole_potential(m0,m1,m2,e1x,e1y,e1z,l1x,l1y,l1z,e2x,e2y,e2z,l2x,l2y,l2z):

    e1 = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    e2 = np.sqrt(e2x**2 + e2y**2 + e2z**2)
    l1 = np.sqrt(l1x**2 + l1y**2 + l1z**2)
    l2 = np.sqrt(l2x**2 + l2y**2 + l2z**2)


    u1x, u1y, u1z = e1x/e1, e1y/ e1, e1z/e1
    u2x, u2y, u2z = e2x/e2, e2y/ e2, e2z/e2
    n1x, n1y, n1z = l1x/l1, l1y/ l1, l1z/l1
    n2x, n2y, n2z = l2x/l2, l2y/ l2, l2z/l2

    n1dotn2 = n1x * n2x + n1y * n2y + n1z * n2z
    u1dotn2 = u1x * n2x + u1y * n2y + u1z * n2z
    u1dotu2 = u1x * u2x + u1y * u2y + u1z * u2z
    n1dotn2 = n1x * n2x + n1y * n2y + n1z * n2z
    n1dotu2 = n1x * u2x + n1y * u2y + n1z * u2z

    a1 = l1 * l1 / (1 - e1 * e1) / (constants.G * (m0 + m1))
    a2 = l2 * l2 / (1 - e2 * e2) /  (constants.G * (m0 + m1 + m2))
    
    pot0 = -constants.G * m0 * m1 * m2 / (m0 + m1) * a1 * a1 / a2 / a2 /a2\
          / np.sqrt(1 - e2 * e2) / (1 - e2 * e2) 
    pot0 *= (m0 - m1)/(m0 + m1) * a1 / a2 * e2 / (1 - e2 * e2)

    pot = pot0 * 15.0 / 64 *\
          (e1 * u1dotu2 * (8 * e1 * e1 - 1 - 35 * e1 * e1 * u1dotn2 * u1dotn2 +\
                      5 * (1 - e1 * e1) * n1dotn2 * n1dotn2) +\
           10 * e1 * (1 - e1 * e1) * u1dotn2 * n1dotu2 * n1dotn2)

    return pot

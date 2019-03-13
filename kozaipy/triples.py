"""
This module defines data structure for hierarchical triple systems.

"""

import numpy as np
import numpy.random as rd
from triples_integrate import integrate_triple_system


rd.seed(42)

#__all__ = ['Triple','TripleSolution','PhysicalProperties']


triple_keys = {"e1x": None,
               "e1y": None,
               "e1z": None,
               "l1x": None,
               "l1y": None,
               "l1z": None,
               "e2x": None,
               "e2y": None,
               "e2z": None,
               "l2x": None,
               "l2y": None,
               "l2z": None,                     
               "spin0x": None,
               "spin0y": None,
               "spin0z": None,
               "spin1x": None,
               "spin1y": None,
               "spin1z": None,
               "Omega0": None,
               "Omega0x": None,
               "Omega0y": None,
               "Omega0z": None,
               "Omega1": None,
               "Omega1x": None,
               "Omega1y": None,
               "Omega1z": None,
               "m0": None,
               "m1": None,
               "m2": None,
               "R0": None,
               "R1": None}

triple_derivative_keys = {"de1x_dt": None,
                          "de1y_dt": None,
                          "de1z_dt": None,
                          "dl1x_dt": None,
                          "dl1y_dt": None,
                          "dl1z_dt": None,
                          "de2x_dt": None,
                          "de2y_dt": None,
                          "de2z_dt": None,
                          "dl2x_dt": None,
                          "dl2y_dt": None,
                          "dl2z_dt": None
                          }


element_keys = {"a1": 0,
                "e1": 1,
                "I1": 2,
                "g1": 3,
                "h1": 4,
                "a2": 5,
                "e2": 6,
                "I2": 7,
                "g2": 8,
                "h2": 9,
                "Omega0": 10,
                "Omega0x": 11,
                "Omega0y": 12,
                "Omega0z": 13,
                "Omega0x": 14,
                "Omega1": 15,
                "Omega1x": 16,
                "Omega1y": 17,
                "Omega1z": 18
}


triple_data = {"m0": False,
               "m1": False,
               "m2": False,
               "inner_orbit": True,
               "outer_orbit": True,
               "spin0": False,
               "spin1": False,
               "spinorbit_align0": False,
               "spinorbit_align1": False,
               "pseudosynch0": False,
               "pseudosynch1": False,               
               }

               
class Constants(object):

    def __init__(self,*args,**kwargs):

        # Units relative to GGS
        self.G = 2.959122081E-4
        self.CLIGHT = 1.73144633E+2
        self.Rsun = 4.6491E-3
        self.Rjup = 4.7789E-4
        self.Mjup = 9.543E-4
        self.stefan = 1.83864E-23
        
        # Set defaults
        #self.MassUnit = 
        #self.VelocityUnit
        
constants = Constants()


class Body(object):
    def __init__(self,*args,**kwargs):
        self.mass_type = kwargs.get("mass_type")
        self.mass = kwargs.get("mass")
        self.radius = kwargs.get("radius")
        self.gyroradius = kwargs.get("gyroradius")
        self.apsidal_constant = kwargs.get("apsidal_constant")
        self.viscous_time = kwargs.get("viscous_time")
        self.convective_time = kwargs.get("convective_time")
        self.tidal_lag_time = kwargs.get("tidal_lag_time")

        self.dradius_dt = kwargs.get("dradius_dt")
        self.dgyroradius_dt = kwargs.get("dgyroradius_dt")
        
        # Set defaults
        if (self.mass_type is None): self.mass_type = 'pointmass'

        
        
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
        self.spin0x = kwargs.get("spin0x")
        self.spin0y = kwargs.get("spin0y")
        self.spin0z = kwargs.get("spin0z")
        self.spin1x = kwargs.get("spin1x")
        self.spin1y = kwargs.get("spin1y")
        self.spin1z = kwargs.get("spin1z")
        self.Omega0x = kwargs.get("Omega0x")
        self.Omega0y = kwargs.get("Omega0y")
        self.Omega0z = kwargs.get("Omega0z")
        self.Omega1x = kwargs.get("Omega1x")
        self.Omega1y = kwargs.get("Omega1y")
        self.Omega1z = kwargs.get("Omega1z") 
        # for the outer orbit
        self.e2x = kwargs.get("e2x")
        self.e2y = kwargs.get("e2y")
        self.e2z = kwargs.get("e2z")
        self.l2x = kwargs.get("l2x")
        self.l2y = kwargs.get("l2y")
        self.l2z = kwargs.get("l2z")

        # Vector derivatives
        self.de1x_dt = None
        self.de1y_dt = None
        self.de1z_dt = None 
        self.dl1x_dt = None 
        self.dl1y_dt = None 
        self.dl1z_dt = None 
        self.de2x_dt = None 
        self.de2y_dt = None 
        self.de2z_dt = None 
        self.dl2x_dt = None 
        self.dl2y_dt = None 
        self.dl2z_dt = None 
        
class ElementData(object):
    def __init__(self,*args,**kwargs):
        self.time = kwargs.get("time")
        # for the inner orbit
        self.e1 = kwargs.get("e1")
        self.a1 = kwargs.get("a1")
        self.I1 = kwargs.get("I1")
        self.g1 = kwargs.get("g1")
        self.h1 = kwargs.get("h1")        
        self.spin0x = kwargs.get("spin0x")
        self.spin0y = kwargs.get("spin0y")
        self.spin0z = kwargs.get("spin0z")
        self.spin1x = kwargs.get("spin1x")
        self.spin1y = kwargs.get("spin1y")
        self.spin1z = kwargs.get("spin1z")
        self.Omega0x = kwargs.get("Omega0x")
        self.Omega0y = kwargs.get("Omega0y")
        self.Omega0z = kwargs.get("Omega0z")
        self.Omega1x = kwargs.get("Omega1x")
        self.Omega1y = kwargs.get("Omega1y")
        self.Omega1z = kwargs.get("Omega1z") 
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

    def __init__(self,Triple,vector_solution=None,vector_derivatives=None,*args,**kwargs):
        """
        Load orbital vector data as a function of time, obtained from a numerical solution
        of the secular equations of motion.

        """

        self.triple = Triple
        self.vectordata =  VectorData()
        self.elementdata =  ElementData()

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
        self.vectordata.spin0x = kwargs.get("spin0x")
        self.vectordata.spin0y = kwargs.get("spin0y")
        self.vectordata.spin0z = kwargs.get("spin0z")
        self.vectordata.spin1x = kwargs.get("spin1x")
        self.vectordata.spin1y = kwargs.get("spin1y")
        self.vectordata.spin1z = kwargs.get("spin1z")
        self.vectordata.Omega0 = kwargs.get("Omega0")
        self.vectordata.Omega0x = kwargs.get("Omega0x")
        self.vectordata.Omega0y = kwargs.get("Omega0y")
        self.vectordata.Omega0z = kwargs.get("Omega0z")
        self.vectordata.Omega1 = kwargs.get("Omega1")
        self.vectordata.Omega1x = kwargs.get("Omega1x")
        self.vectordata.Omega1y = kwargs.get("Omega1y")
        self.vectordata.Omega1z = kwargs.get("Omega1z")
        # for the outer orbit
        self.vectordata.e2x = kwargs.get("e2x")
        self.vectordata.e2y = kwargs.get("e2y")
        self.vectordata.e2z = kwargs.get("e2z")
        self.vectordata.l2x = kwargs.get("l2x")
        self.vectordata.l2y = kwargs.get("l2y")
        self.vectordata.l2z = kwargs.get("l2z")

        if (vector_solution is not None):
            # data supplied at once as a matrix
            self.vectordata.time = vector_solution[:,0] 
            
            print '\nNumerical solution carried out for:'
            for key,val in self.vectordata.__dict__.items():
                if key not in triple_keys: continue
                if (key == 'time'): continue
                if triple_keys[key] is not None:
                    print key,
                    setattr(self.vectordata, key,  vector_solution[:,1+triple_keys[key]])
                    
        if (vector_derivatives is not None):
            nquant = vector_solution.shape[1]
            print '\nTime derivatives availabile for:'
            for key,val in self.vectordata.__dict__.items():
                if key not in triple_derivative_keys: continue
                related_key = key[1:-3]
                if triple_derivative_keys[key] is not None:
                    print related_key,
                    setattr(self.vectordata, key,  vector_derivatives[:,triple_derivative_keys[key]-nquant])




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
        #self.elementdata.g1 = np.arctan2(self.vectordata.e1z[:] * np.sqrt(1 - self.elementdata.e1**2),\
        #                                 np.sqrt((1 - self.elementdata.e1**2) * (self.vectordata.e1x[:]**2 + self.vectordata.e1y[:]**2) -\
        #                                         self.elementdata.e1**2 * self.vectordata.l1z[:]**2/constants.G /(self.triple.m0 + self.triple.m1) / \
        #                                         self.elementdata.a1))
        
        # for the outer orbit
        self.elementdata.e2 = np.sqrt(self.vectordata.e2x[:]**2 + self.vectordata.e2y[:]**2 + self.vectordata.e2z[:]**2)
        self.elementdata.a2 = (self.vectordata.l2x[:]**2 + self.vectordata.l2y[:]**2 + self.vectordata.l2z[:]**2)/constants.G /\
                              (self.triple.m0 + self.triple.m1 + self.triple.m2) / (1 - self.elementdata.e2[:] * self.elementdata.e2[:])
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

        if (self.vectordata.spin0x is not None):
            self.elementdata.spin0x = self.vectordata.spin0x
        if (self.vectordata.spin0y is not None):
            self.elementdata.spin0y = self.vectordata.spin0y
        if (self.vectordata.spin0z is not None):
            self.elementdata.spin0z = self.vectordata.spin0z
        if (self.vectordata.spin1x is not None):
            self.elementdata.spin1x = self.vectordata.spin1x
        if (self.vectordata.spin1y is not None):
            self.elementdata.spin1y = self.vectordata.spin1y
        if (self.vectordata.spin1z is not None):
            self.elementdata.spin1z = self.vectordata.spin1z

        if (self.vectordata.Omega0 is not None):
            self.elementdata.Omega0 = self.vectordata.Omega0
        if (self.vectordata.Omega0x is not None):
            self.elementdata.Omega0x = self.vectordata.Omega0x
        if (self.vectordata.Omega0y is not None):
            self.elementdata.Omega0y = self.vectordata.Omega0y
        if (self.vectordata.Omega0z is not None):
            self.elementdata.Omega0z = self.vectordata.Omega0z
        if (self.vectordata.Omega1 is not None):
            self.elementdata.Omega1 = self.vectordata.Omega1
        if (self.vectordata.Omega1x is not None):
            self.elementdata.Omega1x = self.vectordata.Omega1x
        if (self.vectordata.Omega1y is not None):
            self.elementdata.Omega1y = self.vectordata.Omega1y
        if (self.vectordata.Omega1z is not None):
            self.elementdata.Omega1z = self.vectordata.Omega1z
            
    def add_body_properties(self, body_index, quantity):

        bodies = [self.triple.properties0, self.triple.properties1]
        
        if (quantity == 'InertiaMoment'):            
            if callable(bodies[body_index].radius):
                radius  = np.asarray([bodies[body_index].radius(r) for r in self.vectordata.time])
            else:
                radius = np.repeat(bodies[body_index].radius,self.vectordata.time.shape[0])
            if callable(bodies[body_index].gyroradius):
                gyroradius  = np.asarray([bodies[body_index].gyroradius(r) for r in self.vectordata.time])
            else:
                gyroradius = np.repeat(bodies[body_index].gyroradius,self.vectordata.time.shape[0])

            if (body_index == 0): self.InertiaMoment0 = gyroradius * self.triple.m0 * radius * radius
            if (body_index == 1): self.InertiaMoment1 = gyroradius * self.triple.m1 * radius * radius
        

    def compute_potential(self,octupole = True):
        self.potential = compute_quadrupole_potential(self.triple.m0,self.triple.m1,self.triple.m2,
                                                      self.vectordata.e1x,self.vectordata.e1y,self.vectordata.e1z,
                                                      self.vectordata.l1x,self.vectordata.l1y,self.vectordata.l1z,
                                                      self.vectordata.e2x,self.vectordata.e2y,self.vectordata.e2z,
                                                      self.vectordata.l2x,self.vectordata.l2y,self.vectordata.l2z)

        if (octupole is True):
            oct_pot = compute_octupole_potential(self.triple.m0,self.triple.m1,self.triple.m2,
                                                 self.vectordata.e1x,self.vectordata.e1y,self.vectordata.e1z,
                                                 self.vectordata.l1x,self.vectordata.l1y,self.vectordata.l1z,
                                                 self.vectordata.e2x,self.vectordata.e2y,self.vectordata.e2z,
                                                 self.vectordata.l2x,self.vectordata.l2y,self.vectordata.l2z)
            self.potential[:] = self.potential[:] + oct_pot[:]
        
        
    def compute_total_angular_momentun(self):

        if ('InertiaMoment0' not in self.vectordata.__dict__.keys()):
            self.add_body_properties(0, 'InertiaMoment')

        if ('InertiaMoment1' not in self.vectordata.__dict__.keys()):
            self.add_body_properties(1, 'InertiaMoment')

        if ((triple_keys['Omega0x'] is None) | (triple_keys['Omega0y'] is None) | (triple_keys['Omega0z'] is None)):
             self.vectordata.Omega0,self.vectordata.Omega0y,self.vectordata.Omega0z = None ,None ,None
            
        if ((triple_keys['Omega1x'] is None) | (triple_keys['Omega1y'] is None) | (triple_keys['Omega1z'] is None)):
             self.vectordata.Omega1x, self.vectordata.Omega1y,self.vectordata.Omega1z = None ,None ,None
             
        angular_momentum = compute_angular_momentum(self.triple.m0,self.triple.m1,self.triple.m2,
                                                    self.vectordata.l1x,self.vectordata.l1y,self.vectordata.l1z,
                                                    self.vectordata.l2x,self.vectordata.l2y,self.vectordata.l2z,
                                                    self.InertiaMoment0,self.InertiaMoment1,
                                                    self.vectordata.Omega0x,self.vectordata.Omega0y,self.vectordata.Omega0z,
                                                    self.vectordata.Omega1x,self.vectordata.Omega1y,self.vectordata.Omega1z)

        print angular_momentum.shape
        self.angular_momentumx = angular_momentum[:,0]
        self.angular_momentumy = angular_momentum[:,1]
        self.angular_momentumz = angular_momentum[:,2]
        
    def save_to_file(self,filename,header=None,Nlines=None):
        """
        Function to save the solution into an ASCII file

        """
        if (Nlines is None):
            Nlines = len(self.vectordata.time)

        f = open(filename,'w')
        fmt_list = []
        head_list = []
        index_list = []
        f.write("# time\t\t")

        data = self.vectordata.time.reshape(len(self.vectordata.time),1)
        fmt_list.append('%13.8e  ')

        if (self.elementdata.a1 is not None):
            for k,v in self.elementdata.__dict__.items():
                if (v is not None):
                    if (k == 'time'):
                        continue
                    if (k == 'a1'):
                        fmt_list.append('%12.8f  ')
                        index_list.append(element_keys['a1'])
                    if (k == 'a2'):
                        fmt_list.append('%12.8f  ')
                        index_list.append(element_keys['a2'])
                    if (k == 'e1'):
                        fmt_list.append('%10.8f  ')
                        index_list.append(element_keys['e1'])
                    if (k == 'e2'):
                        fmt_list.append('%10.8f  ')
                        index_list.append(element_keys['e2'])
                    if (k == 'I1'):
                        fmt_list.append('%10.6f  ')
                        index_list.append(element_keys['I1'])
                    if (k == 'I2'):
                        fmt_list.append('%10.6f  ')
                        index_list.append(element_keys['I2'])
                    if (k == 'g1'):
                        fmt_list.append('%10.6f  ')
                        index_list.append(element_keys['g1'])
                    if (k == 'g2'):
                        fmt_list.append('%10.6f  ')
                        index_list.append(element_keys['g2'])
                    if (k == 'h1'):
                        fmt_list.append('%10.6f  ')
                        index_list.append(element_keys['h1'])
                    if (k == 'h2'):
                        fmt_list.append('%10.6f  ')
                        index_list.append(element_keys['h2'])
                    if ('spin' in k):
                        fmt_list.append('%12.6f  ')
                        index_list.append(element_keys[k])
                    if ('Omega' in k):
                        fmt_list.append('%12.6f  ')
                        index_list.append(element_keys[k])
                    head_list.append(k+"\t\t")
                    data = np.column_stack((data,v))
        else:
            for k,v in self.vectordata.__dict__.items():
                if (v is not None):
                    if (k == 'time'):
                        continue
                    try:
                        if (triple_keys[k] is None):
                            continue
                    except KeyError:
                        if (triple_derivative_keys[k] is None):
                            continue
                        
                    if (k in triple_keys):
                        if (triple_keys[k] is None):
                            continue
                        if (k == 'e1x'):
                            fmt_list.append('%12.8f  ')
                            index_list.append(triple_keys['e1x'])
                        if (k == 'e1y'):
                            fmt_list.append('%12.8f  ')
                            index_list.append(triple_keys['e1y'])
                        if (k == 'e1z'):
                            fmt_list.append('%10.8f  ')
                            index_list.append(triple_keys['e1z'])
                        if (k == 'l1x'):
                            fmt_list.append('%10.8f  ')
                            index_list.append(triple_keys['l1x'])
                        if (k == 'l1y'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['l1y'])
                        if (k == 'l1z'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['l1z'])
                        if (k == 'e2x'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['e2x'])
                        if (k == 'e2y'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['e2y'])
                        if (k == 'e2z'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['e2z'])
                        if (k == 'l2x'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['l2x'])
                        if (k == 'l2y'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['l2y'])
                        if (k == 'l2z'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['l2z'])
                        if (k == 'Omega0'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega0'])
                        if (k == 'Omega0x'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega0x'])
                        if (k == 'Omega0y'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega0y'])
                        if (k == 'Omega0z'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega0z'])
                        if (k == 'Omega1'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega1'])                    
                        if (k == 'Omega1x'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega1x'])
                        if (k == 'Omega1y'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega1y'])
                        if (k == 'Omega1z'):
                            fmt_list.append('%10.6f  ')
                            index_list.append(triple_keys['Omega1z'])

                    if (k in triple_derivative_keys):
                        if (triple_derivative_keys[k] is None):
                            continue
                        if (k == 'de1x_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['de1x_dt'])
                        if (k == 'de1y_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['de1y_dt'])
                        if (k == 'de1z_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['de1z_dt'])
                        if (k == 'dl1x_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['dl1x_dt'])
                        if (k == 'dl1y_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['dl1y_dt'])
                        if (k == 'dl1z_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['dl1z_dt'])
                        if (k == 'de2x_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['de2x_dt'])
                        if (k == 'de2y_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['de2y_dt'])
                        if (k == 'de2z_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['de2z_dt'])
                        if (k == 'dl2x_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['dl2x_dt'])
                        if (k == 'dl2y_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['dl2y_dt'])
                        if (k == 'dl2z_dt'):
                            fmt_list.append('%12.8e  ')
                            index_list.append(triple_derivative_keys['dl2z_dt'])


                
                
                    head_list.append(k+"\t\t")
                    data = np.column_stack((data,v)) 

        

        # Sort the columns to obtain a consistently organized output
        f.write("".join([x for (y,x) in sorted(zip(index_list,head_list))]))

        data[:,1:] = data[:,1:][:,np.asarray(index_list).argsort()]
        fmt_list = np.asarray(fmt_list)
        fmt_list[1:] = fmt_list[1:][np.asarray(index_list).argsort()]
        f.write("\n")
        f.write("#--------------------------------------------------------------------------\n")
        np.savetxt(f,data[::int(len(self.vectordata.time)/Nlines),:],fmt="".join(fmt_list))
        

        
        

class Triple(object):

    """"
    Class  for generating a hierarchical triple system

    """
    
    def __init__(self, *args,**kwargs):
        self.properties0 = Body()
        self.properties1 = Body()
        self.properties2 = Body()
        
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
        self.spin0 = kwargs.get("spin0") # Spin *vector* for the first body
        self.spin1 = kwargs.get("spin1") # Spin *vector* for the second body
        self.Omega0 = kwargs.get("Omega0") # Spin rotation frequency *vector* for the first body
        self.Omega1 = kwargs.get("Omega1") # Spin rotation frequency *vector* for the second body
        
        self.pseudosynch0 =  kwargs.get("pseudosynch0")
        self.pseudosynch1 =  kwargs.get("pseudosynch1")

        self.spinorbit_align0 =  kwargs.get("spinorbit_align0")
        self.spinorbit_align1 =  kwargs.get("spinorbit_align1")
        
        # Bodies' properties
        self.properties0.radius = kwargs.get("R0") # radius of primary
        self.properties1.radius = kwargs.get("R1") # radius of secondary
        self.properties0.gyroradius = kwargs.get("rg0") # gyroradius of primary
        self.properties1.gyroradius = kwargs.get("rg1") # gyroradius of secondary
        self.properties0.apsidal_constant = kwargs.get("k2_0") 
        self.properties1.apsidal_constant = kwargs.get("k2_1") 
        self.properties0.viscous_time = kwargs.get("tv0") 
        self.properties1.viscous_time = kwargs.get("tv1")
        self.properties0.convective_time = kwargs.get("tauconv0") 
        self.properties1.convective_time = kwargs.get("tauconv1") 
        self.properties0.tidal_lag_time = kwargs.get("tlag0") 
        self.properties1.tidal_lag_time = kwargs.get("tlag1") 

        
        self.properties0.mass = self.m0
        self.properties1.mass = self.m1
        self.properties2.mass = self.m2

        if (kwargs.get("type0") is not None):
            self.properties0.mass_type = kwargs.get("type0")
        if (kwargs.get("type1") is not None):
            self.properties1.mass_type = kwargs.get("type1")
        if (kwargs.get("type2") is not None):
            self.properties2.mass_type = kwargs.get("type2")

        # Time derivatives
        self.properties0.dradius_dt = kwargs.get("dR0dt")
        self.properties1.dradius_dt = kwargs.get("dR1dt")
        self.properties0.dgyroradius_dt = kwargs.get("drg0dt")
        self.properties1.dgyroradius_dt = kwargs.get("drg1dt")
            
        # Set default values for the orbits
        if (self.a1 is None): self.a1 = 1.0
        if (self.a2 is None): self.a2 = 100.0            
        if (self.e1 is None): self.e1 = 1.0e-6
        if (self.e2 is None): self.e2 = 1.0e-6
        if (self.I is None): self.I = 0.0
        if (self.g1 is None): self.g1 = rd.random() * 2 * np.pi
        if (self.g2 is None): self.g2 = rd.random() * 2 * np.pi
        if (self.h1 is None): self.h1 = rd.random() * 2 * np.pi
        if (self.h2 is None): self.h2 = self.h1 - np.pi
        if (self.spin_rate0 is None): self.spin_rate0 = 1.0-9
        if (self.spin_rate1 is None): self.spin_rate1 = 1.0-9

        if (self.pseudosynch0 is None): self.pseudosynch0 = False
        if (self.pseudosynch1 is None): self.pseudosynch1 = False

        if (self.spinorbit_align0 is None): self.spinorbit_align0 = False
        if (self.spinorbit_align1 is None): self.spinorbit_align1 = False

        
        # Compute individual inclinations
        m0, m1, m2 = self.m0, self.m1, self.m2
        L1 = m0 * m1/(m0 + m1) * np.sqrt((m0 + m1) * self.a1 * (1 - self.e1 * self.e1))
        L2 = (m0 + m1) * m2/(m0 + m1 + m2) * np.sqrt((m0 + m1 + m2) * self.a2 * (1 - self.e2 * self.e2))
        L  = np.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * np.cos(self.I))
        self.I2 = np.arccos((L2 + L1 * np.cos(self.I))/L)
        self.I1 = self.I - self.I2

        # Moments of inertia
        if (self.properties0.gyroradius is None):
            if (self.properties0.mass_type == 'star'): self.properties0.gyroradius = get_gyroradius_star(self.properties0.mass)
            elif (self.properties0.mass_type == 'planet'): self.properties0.gyroradius = get_gyroradius_planet(self.properties0.mass)

        if (self.properties1.gyroradius is None):
            if (self.properties1.mass_type == 'star'): self.properties1.gyroradius = get_gyroradius_star(self.properties1.mass)
            elif (self.properties1.mass_type == 'planet'): self.properties1.gyroradius = get_gyroradius_planet(self.properties1.mass)
        
        # Compute the default spin *orientations*
        if (self.spin0 is None) & (self.properties0.mass_type != 'pointmass'):
            if callable(self.properties0.radius):
                radius0 = self.properties0.radius(0)
            else:
                radius0 = self.properties0.radius
            if callable(self.properties0.gyroradius):
                gyroradius0 = self.properties0.gyroradius(0)
            else:
                gyroradius0 = self.properties0.gyroradius

            deltah_0, deltaI_0 = rd.random()*np.pi*0.001,rd.random()*np.pi*0.001
            self.Omega0 = [self.spin_rate0 * np.sin(self.h1 + deltah_0) * np.sin(self.I1 + deltaI_0),
                          -self.spin_rate0 * np.cos(self.h1 + deltah_0) * np.sin(self.I1 + deltaI_0), 
                           self.spin_rate0 * np.cos(self.I1 + deltaI_0)]
                
            if (radius0 is not None):
                I0 = gyroradius0 * self.m0 * radius0 * radius0
                self.spin0 = [self.Omega0[0] * I0, self.Omega0[1] * I0, self.Omega0[2] * I0]
                
        if (self.spin1 is None) & (self.properties1.mass_type != 'pointmass'):
            if callable(self.properties1.radius):
                radius1 = self.properties1.radius(0)
            else:
                radius1 = self.properties1.radius
            if callable(self.properties1.gyroradius):
                gyroradius1 = self.properties1.gyroradius(0)
            else:
                gyroradius1 = self.properties1.gyroradius
            if (radius1 is not None):
                I1 = gyroradius1 * self.m1 * radius1 * radius1
                spin1 = self.spin_rate1 * I1
                deltah_1, deltaI_1 = rd.random()*np.pi*0.001,rd.random()*np.pi*0.001
                self.spin1 = [spin1 * np.sin(self.h1 + deltah_1) * np.sin(self.I1 + deltaI_1),
                              -spin1 * np.cos(self.h1 + deltah_1) * np.sin(self.I1 + deltaI_1), 
                              spin1 * np.cos(self.I1 + deltaI_1)] 
                self.Omega1 = [self.spin1[0] / I1, self.spin1[1] / I1, self.spin1[2] / I1]


        
    def vector_form(self):
        """
        Coordinate transformation from orbital elements to orbital vectors
        
        """
        m0, m1, m2 = self.m0, self.m1, self.m2
        # For the inner orbit
        e1x = self.e1 * (np.cos(self.g1) * np.cos(self.h1) - np.sin(self.g1) * np.sin(self.h1) * np.cos(self.I1))
        e1y = self.e1 * (np.cos(self.g1) * np.sin(self.h1) + np.sin(self.g1) * np.cos(self.h1) * np.cos(self.I1))
        e1z = self.e1 * np.sin(self.g1) * np.sin(self.I1)
        
        l1  = np.sqrt(constants.G * self.a1 * (m0 + m1) * (1 - self.e1 * self.e1))
        l1x = l1 * np.sin(self.h1) * np.sin(self.I1)
        l1y = -l1 * np.cos(self.h1) * np.sin(self.I1)
        l1z = l1 * np.cos(self.I1)

        # For the outer orbit
        e2x = self.e2 * (np.cos(self.g2) * np.cos(self.h2) - np.sin(self.g2) * np.sin(self.h2) * np.cos(self.I2))
        e2y = self.e2 * (np.cos(self.g2) * np.sin(self.h2) + np.sin(self.g2) * np.cos(self.h2) * np.cos(self.I2))
        e2z = self.e2 * np.sin(self.g2) * np.sin(self.I2)

        l2  = np.sqrt(constants.G * self.a2 * (m0 + m1 + m2) * (1 - self.e2 * self.e2))
        l2x = l2 * np.sin(self.h2) * np.sin(self.I2)
        l2y = -l2 * np.cos(self.h2) * np.sin(self.I2)
        l2z = l2 * np.cos(self.I2)

        return [e1x,e1y,e1z,
                l1x,l1y,l1z,
                e2x,e2y,e2z,
                l2x,l2y,l2z]
    

    def set_ics(self,spin_vector=False):
        """
        Set the initial conditions to be used by the integrations routines.

        """
        vector = self.vector_form()
        jj = 0
        triple_keys['e1x'],triple_keys['e1y'],triple_keys['e1z'] =  jj + 0, jj + 1, jj + 2
        triple_keys['l1x'],triple_keys['l1y'],triple_keys['l1z'] =  jj + 3, jj + 4, jj + 5
        jj+=6
        triple_keys['e2x'],triple_keys['e2y'],triple_keys['e2z'] =  jj + 0, jj + 1, jj + 2
        triple_keys['l2x'],triple_keys['l2y'],triple_keys['l2z'] =  jj + 3, jj + 4, jj + 5
        jj+=6
        
        if not (self.properties0.mass_type == 'pointmass'):
            if not (self.pseudosynch0): 
                triple_data['spin0'] = True
                if not (self.spinorbit_align0):
                    if (spin_vector):
                        triple_keys['spin0x'],triple_keys['spin0y'],triple_keys['spin0z'] = jj + 0, jj + 1, jj + 2
                        jj+=3
                        vector +=  [self.spin0[0],self.spin0[1],self.spin0[2]]
                    else:
                        triple_keys['Omega0x'],triple_keys['Omega0y'],triple_keys['Omega0z'] = jj + 0, jj + 1, jj + 2
                        jj+=3
                        vector +=  [self.Omega0[0],self.Omega0[1],self.Omega0[2]]
                else:
                    triple_data['spinorbit_align0'] = True
                    triple_keys['Omega0'] = jj
                    jj+=1
                    vector +=  [self.spin_rate0]
            else: 
                triple_data['pseudosynch0'] = True
                
        if not (self.properties1.mass_type == 'pointmass'):
            if not (self.pseudosynch1): 
                triple_data['spin1'] = True
                if not (self.spinorbit_align1): 
                    if (spin_vector):
                        triple_keys['spin1x'],triple_keys['spin1y'],triple_keys['spin1z'] =  jj + 0, jj + 1, jj + 2
                        jj+=3
                        vector +=  [self.spin1[0],self.spin1[1],self.spin1[2]]
                    else:
                        triple_keys['Omega1x'],triple_keys['Omega1y'],triple_keys['Omega1z'] = jj + 0, jj + 1, jj + 2
                        jj+=3
                        vector +=  [self.Omega1[0],self.Omega1[1],self.Omega1[2]]
                else:
                    triple_data['spinorbit_align1'] = True
                    triple_keys['Omega1'] = jj
                    jj+=1
                    vector +=  [self.spin_rate1]       
            else: 
                triple_data['pseudosynch1'] = True

        #if (self.properties0.dradius_dt is not None):
        #    if (self.properties0.dradius_dt(0) != 0):
        #        vector += [self.properties0.dradius_dt(0)]
        #        triple_keys['R0'] = jj 
        #        jj+=1
            
        #if (self.properties1.dradius_dt is not None):
        #    if (self.properties1.dradius_dt(0) != 0):
        #        vector += [self.properties1.dradius_dt(0)]
        #        triple_keys['R1'] = jj 
        #        jj+=1
        
        jj+=1
        triple_derivative_keys['de1x_dt'],triple_derivative_keys['de1y_dt'],triple_derivative_keys['de1z_dt'] =  jj + 0, jj + 1, jj + 2
        triple_derivative_keys['dl1x_dt'],triple_derivative_keys['dl1y_dt'],triple_derivative_keys['dl1z_dt'] =  jj + 3, jj + 4, jj + 5
        jj+=6
        triple_derivative_keys['de2x_dt'],triple_derivative_keys['de2y_dt'],triple_derivative_keys['de2z_dt'] =  jj + 0, jj + 1, jj + 2
        triple_derivative_keys['dl2x_dt'],triple_derivative_keys['dl2y_dt'],triple_derivative_keys['dl2z_dt'] =  jj + 3, jj + 4, jj + 5
        
        return vector
               
    def integrate(self,timemin,timemax,Nevals,octupole_potential= True,
                  short_range_forces_conservative=False,
                  short_range_forces_dissipative = False,
                  solve_for_spin_vector = False,
                  version = 'tides',
                  add_derivatives = False):
        """
        Integrate the system forward in time

        """

        #self.check_validity_bodies()
        
        vector_ics = self.set_ics()

        
        solution = integrate_triple_system(vector_ics,timemin,timemax,Nevals,
                                           self.properties0,self.properties1,self.properties2,
                                           octupole_potential=octupole_potential,\
                                           short_range_forces_conservative=short_range_forces_conservative,
                                           short_range_forces_dissipative=short_range_forces_dissipative,
                                           solve_for_spin_vector = solve_for_spin_vector,
                                           version=version,
                                           add_derivatives = add_derivatives)

        if (add_derivatives):
            triple_solution = TripleSolution(self,vector_solution = solution[0],vector_derivatives = solution[1])
        else:
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


def compute_angular_momentum(m0,m1,m2,l1x,l1y,l1z,l2x,l2y,l2z,
                             InertiaMoment0,InertiaMoment1,Omega0x,Omega0y,Omega0z,Omega1x,Omega1y,Omega1z):

    mu1 = m0 * m1 /(m0 + m1)
    mu2 = (m0 + m1) * m2 / (m0 + m1 +m2)

    l1 = np.sqrt(l1x**2 + l1y**2 + l1z**2)
    l2 = np.sqrt(l2x**2 + l2y**2 + l2z**2)
    if ((Omega0x is None) |(Omega0y is None) |(Omega0z is None)):
        Omega0x, Omega0y, Omega0z = 0, 0, 0
    if ((Omega1x is None) |(Omega1y is None) |(Omega1z is None)):
        Omega1x, Omega1y, Omega1z = 0, 0, 0
        
    Omega0 = np.sqrt(Omega0x * Omega0x + Omega0y * Omega0y + Omega0z * Omega0z)
    Omega1 = np.sqrt(Omega1x * Omega1x + Omega1y * Omega1y + Omega1z * Omega1z)
    Jx = mu1 * l1x + mu2 * l2x + InertiaMoment0 * Omega0x +  InertiaMoment1 * Omega1x
    Jy = mu1 * l1y + mu2 * l2y + InertiaMoment0 * Omega0y +  InertiaMoment1 * Omega1y
    Jz = mu1 * l1z + mu2 * l2z + InertiaMoment0 * Omega0z +  InertiaMoment1 * Omega1z

    return np.array([Jx,Jy,Jz]).T



def get_gyroradius_star(m):
    return 0.08


def get_gyroradius_planet(m):
    return 0.25

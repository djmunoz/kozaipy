KOZAIPY - Secular equations of motion for the evolution of triple systems with tidal friction
==================================================



Welcome to KozaiPy!


Examples
--------

To run a secular triple (e.g. Fig.3 of Naoz et al 2013). First setup the triple
as:

.. code:: python
	  
   import kozaipy as kp
   import numpy as np

   trip = kp.Triple(m0=1.0,m1=0.001,m2=0.04,a1=6.0,a2=100.0,e1=0.001,e2=0.6,I=65.0 * np.pi/180.0, g1=45.0 * np.pi/180.0,g2=0)

And then evolve the system for 2.5e7 years

.. code:: python
	  
   sol = trip.integrate(timemax=2.5e7*365.25,Nevals=30000,octupole_potential=True)

To plot the inclination and eccentricity of the inner orbit, simply do

.. code:: python
	  
   import matplotlib.pyplot as plt

   incl1= sol.elementdata.I1
   ecc1 = sol.elementdata.e1

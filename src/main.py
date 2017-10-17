'''
************************************************************************************************************************************************************
File: main.py
Written By: Luke Burks
October 2017

Coding up the partial differential equations governing 
Reaction-Diffusion systems as described by Alan Turing
for animal stripe pattern formation

Inspired by: https://www.youtube.com/watch?v=alH3yc6tX98
Coding help drawn from: http://ipython-books.github.io/featured-05/

Simulating the equations:
du/dt = a(\delta)u + u - u^3 - v + k
(\tau)dv/dt = b(\delta)v + u - v


************************************************************************************************************************************************************
'''


from __future__ import division
__author__ = "Luke Burks"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"


import numpy as np
import matplotlib.pyplot as plt



class TuringPattern:

	def __init__(self):
		#set up parameters
		self.a = 2.8e-4;
		self.b = 5e-3;
		self.tau = .1; 
		self.k = .005; 

		#set up discrete grid
		self.size = 100; 
		self.dx = 2/self.size;
		self.time = 10; 
		self.dt = .9*self.dx**2/2;
		self.n = int(self.time/self.dt); 

		#set up grid values for variables
		self.U = np.random.rand(self.size,self.size); 
		self.V = np.random.rand(self.size,self.size); 

	#uses five-point stencil finite difference method
	def discreteLaplace(self,Z):
		Ztop = Z[0:-2,1:-1]; 
		Zleft = Z[1:-1,0:-2]; 
		Zbottom = Z[2:,1:-1]; 
		Zright = Z[1:-1,2:]; 
		Zcenter = Z[1:-1,1:-1]; 
		return ((Ztop+Zleft+Zbottom+Zright - 4*Zcenter)/self.dx**2)


	def simulate(self,visualize = False):

		fig,ax = plt.subplots(); 

		for i in range(self.n):

			#print("Loop {} of {}".format(i,self.n)); 

			#Get laplacians
			deltaU = self.discreteLaplace(self.U); 
			deltaV = self.discreteLaplace(self.V); 

			#Grab values
			Uc = self.U[1:-1,1:-1]; 
			Vc = self.V[1:-1,1:-1]; 

			#Update Values
			self.U[1:-1,1:-1] = Uc + self.dt*(self.a*deltaU+Uc-Uc**3 - Vc + self.k); 
			self.V[1:-1,1:-1] = Vc + self.dt*(self.b*deltaV + Uc - Vc)/self.tau; 

			#Account for boundry conditions, derivatives null at edges
			for Z in (self.U,self.V):
				Z[0,:] = Z[1,:]; 
				Z[-1,:] = Z[-2,:]; 
				Z[:,0] = Z[:,1]; 
				Z[:,-1] = Z[:,-2]; 

			if(visualize and i%1000 == 1):
				ax.cla(); 
				ax.imshow(self.U,cmap='inferno',extent=[-1,1,-1,1],interpolation='gaussian'); 
				ax.set_xticks([]); 
				ax.set_yticks([]);
				plt.pause(0.01);  




	def display(self):
		plt.figure(); 
		plt.imshow(self.U,cmap='inferno',extent=[-1,1,-1,1],interpolation='gaussian'); 
		plt.xticks([]); 
		plt.yticks([]); 
		plt.show(); 


if __name__ == '__main__':

	print("Creating Pattern"); 
	A = TuringPattern(); 
	print("Simulating Pattern"); 
	A.simulate(visualize=True); 
	print("Displaying Pattern"); 
	A.display(); 
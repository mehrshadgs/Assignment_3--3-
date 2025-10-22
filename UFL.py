# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np


class UFL_Problem:
    """
    Class that represent a problem instance of the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    f : numpy array
        the yearly fixed operational costs of all facilities
    c : numpy 2-D array (matrix)
        the yearly transportation cost of delivering all demand from markets to facilities
    n_markets : int
        number of all markets.
    n_facilities : int
        number of all available locations.
    """

    def __init__(self, f, c, n_markets, n_facilities):

        self.f = f
        self.c = c
        self.n_markets = n_markets
        self.n_facilities = n_facilities
        #print(self.f[0])
        #print(self.c[0])

    def __str__(self):
        return f" Uncapacitated Facility Location Problem: {self.n_markets} markets, {self.n_facilities} facilities"

    def readInstance(fileName):
        """
        Read the instance fileName

        Parameters
        ----------
        fileName : str
            instance name in the folder Instance.

        Returns
        -------
        UFL Object

        """
        # Read filename
        f = open(f"Instances/{fileName}")
        n_line = 0
        n_markets = 0
        n_facilities = 0
        n_row = 0
        for line in f.readlines():
            asList = line.replace(" ", "_").split("_")
            if line:
                if n_line == 0:
                    n_markets = int(asList[0])
                    n_facilities = int(asList[1])
                    f_j = np.empty(n_markets)
                    c_ij = np.empty((n_markets, n_facilities))
                elif n_line <= n_markets:  # For customers
                    index = n_line - 1
                    f_j[index] = asList[1]
                else:
                    if len(asList) == 1:
                        n_row += 1
                        demand_i = float(asList[0])
                        n_column = 0
                    else:
                        for i in range(len(asList)-1):
                            c_ij[n_row-1, n_column] = demand_i * \
                                float(asList[i])
                            n_column += 1
            n_line += 1
       # print(f_j[0])
       # print(c_ij[0])
        return UFL_Problem(f_j, c_ij, n_markets, n_facilities)    

class UFL_Solution: 
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    y : numpy array
        binary array indicating whether facilities are open
    x : numpy 2-D array (matrix)
        fraction of demand from markets sourced from facilities
    instance: UFL_Problem
        the problem instance
    """ 
    
    def __init__(self, y, x, instance):
        self.y = y
        self.x = x
        self.instance = instance
        #print(self.y)
        #print(self.x)

    def isFeasible(self): 
        """
        Method that checks whether the solution is feasible
        
        Returns true if feasible, false otherwise
        """
        tol = 1e-5 # sum is not exactly 1 
        Markets = self.instance.n_markets
        Facilities = self.instance.n_facilities
        
        # Shape checks
        if len(self.y) != Facilities: 
            return False
        if self.x.shape != (Markets, Facilities):
            return False
        if self.x.ndim != 2:
            return False
        if self.y.ndim != 1:
            return False
        
        
        sums = np.sum(self.x, axis=1)
        # print(len(self.x) )
        # print(len(self.x[0]))
        # print(len(self.x[1]) )
        # print(len(self.y) )
        # print(len(sums))
        # print(sums[0])
        # for s in sums:
        #     if abs(s - 1.0) > tol:
        #         print(s)
        #         return False
        # https://www.geeksforgeeks.org/python/numpy-allclose-in-python/
        if not np.all(np.isclose(sums, 1.0, atol=tol)):
            return False

        #print(len(self.x - self.y[np.newaxis, :] ))
        #####
        
        if np.any(self.x - self.y[np.newaxis, :] > tol): # np.any from AI
            return False
        
        ##########
        
        if np.any(self.x < -tol) or np.any(self.x > 1.0 + tol):
            return False

        if not np.all((np.isclose(self.y, 0.0, atol=tol)) | (np.isclose(self.y, 1.0, atol=tol))):
            return False
        #print("Y values:")
        
        #print(self.y)

        #print("Sums:")
        #print(sums)
        

        return True

    def getCosts(self): 
        """
        Method that computes and returns the costs of the solution
        """
        opening_costs = np.sum(self.y * self.instance.f)
        transportation_costs = np.sum(self.x * self.instance.c)
        #print(self.y)
        #print(self.instance.f)
        #print(self.instance.c[0])
        #print(len(self.instance.c[0]))
        
        #print("Opening costs:", opening_costs)
        print("total  costs:", transportation_costs + opening_costs)
        return opening_costs + transportation_costs
    
class LagrangianHeuristic: 
    """
    Class used for the Lagrangian Heuristic
        
    Attributes
    ----------
    instance : UFL_Problem
        the problem instance
    """
    
    def __init__(self,instance):
        self.instance = instance
        
    def computeTheta(self,labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the optimal value of the Lagrangian problem
        """
        return
    
    def computeLagrangianSolution(self,labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the Lagrangian solution (as a UFL_Solution)
        """
        return
    
    def convertToFeasibleSolution(self,lagr_solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """
        return
        
    def updateMultipliers(self,labda_old,lagr_solution):
        """
        Method that, given the previous Lagrangian multipliers and Lagrangian Solution 
        updates and returns a new array of Lagrangian multipliers
        """
        return
    
    def runHeuristic(self):
        """
        Method that performs the Lagrangian Heuristic. 
        """
        
        
read_instance = UFL_Problem.readInstance("MO5")
n_markets = read_instance.n_markets
n_facilities = read_instance.n_facilities

# open all facilities and split each market equally among them (feasible)
y = np.ones(n_facilities)
x = np.ones((n_markets, n_facilities)) / n_facilities

#print(x.sum(axis=1))
solution = UFL_Solution(y, x, read_instance)

#solution.isFeasible()
solution.getCosts()
#print(read_instance)
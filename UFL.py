# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np
import matplotlib.pyplot as plt


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
        #print("total  costs:", transportation_costs + opening_costs)
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

        f = self.instance.f
        c = self.instance.c
        
        
        # if cost new is negative, it is cheaper to server than to pay the fine,
        # if cost new is positive, pay the fine
        #print(c[0])
        #print(np.sum(c[0]))
        costNew = c - labda[:, np.newaxis]
        #print(costNew[0])
        #print(np.sum(costNew[0]))
        profits = np.sum(np.minimum(0, costNew), axis=0) + f
        # proftis = costNew /  cost - labda
        #print("Profits:", profits)
        
        facility_cost = np.minimum(0, profits)

        #print("Facility costs:", facility_cost)
        # lower bound is labda + min(0, cost - labda) so
        # lower bound is , labda or cost. 
        lower_bound = np.sum(labda) + np.sum(facility_cost)
        

        #print("Lower bound:", lower_bound)
        
        
        
        return lower_bound
    
    def computeLagrangianSolution(self,labda):
            """
            Method that, given an array of Lagrangian multipliers computes and returns
            the Lagrangian solution (as a UFL_Solution), following the logic from slides.
            """
            c = self.instance.c
            f = self.instance.f
            tol = 1e-5 # the sume is not exactly 1

            # Calculate c_ij - lambda_i
            costnew = c - labda[:, np.newaxis] 
            #print( costnew[0])
            profits = f + np.sum(np.minimum(0, costnew), axis=0) 
            #print(profits)
            # Open if profit is negative
            y = np.where(profits < -tol, 1.0, 0.0) 
            #print(y)



            assign_if_cost_neg = costnew < -tol
            assign_if_facility_open = y == 1.0

   
            x = np.where(assign_if_cost_neg & assign_if_facility_open, 1.0, 0.0)


            return UFL_Solution(y, x, self.instance)
        
    def convertToFeasibleSolution(self,lagr_solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """
        
        y = lagr_solution.y.copy()
        c = lagr_solution.instance.c
        #print("c:", c.shape)
        f = lagr_solution.instance.f
        n_markets = lagr_solution.instance.n_markets
        n_facilities = lagr_solution.instance.n_facilities
        
        Xnew = np.zeros((n_markets, n_facilities))
        open_facilities = np.where(y == 1)[0]
        
        if len(open_facilities) == 0:
            total_cost = f + np.sum(c, axis=0)
            cheapest_facility = np.argmin(total_cost)
            
            y[cheapest_facility] = 1
            
            open_facilities = np.array([cheapest_facility])
        
        for i in range(n_markets):

                
            costs_open = c[i, open_facilities]
            cheapest_open_index = np.argmin(costs_open)
            cheapest_open_facility = open_facilities[cheapest_open_index]
            Xnew[i, cheapest_open_facility] = 1.0
        
        # print("Lagrangian x:", lagr_solution.x)
        # print("Feasible x:", Xnew)
        # print("Feasible y:", y)
        return UFL_Solution(y, Xnew, lagr_solution.instance)
        
        
    def updateMultipliers(self, labda_old, lagr_solution, iteration, initial_step=1.0):
            """
            Method that updates Lagrangian multipliers based on the rule from the slides,
            using an additive step that decreases with iterations.
            """
            tol = 1e-5
            #print(labda_old)
            
            sum_x_over_j = np.sum(lagr_solution.x, axis=1)

            # AI idea: Step size decreases with sqrt of iteration number
            step = initial_step / np.sqrt(iteration) 


            labda_new = labda_old.copy() 

            increase_indices = sum_x_over_j < 1.0 - tol
            labda_new[increase_indices] = labda_old[increase_indices] + step

            decrease_indices = sum_x_over_j > 1.0 + tol
            labda_new[decrease_indices] = labda_old[decrease_indices] - step



            labda_new = np.maximum(0, labda_new)
            #print(labda_new)
            return labda_new

    def runHeuristic(self, max_iterations, initial_step=2.0, gap_tolerance=0.1):
            """
            Method that performs the Lagrangian Heuristic.
            """

            labda = np.zeros(self.instance.n_markets)
            best_lb = -np.inf
            best_up = np.inf  
            best_feasible_solution = None 

            lower_bounds_history = []
            Best_upper_bounds_history = []
            upper_bounds_history = []

            print(f"Parameters: Max Iterations={max_iterations}, Initial Step={initial_step}, Gap Tol={gap_tolerance}")


            for k in range(1, max_iterations + 1): # To avoid division by zero in the step size

                current_lb = self.computeTheta(labda)
                lower_bounds_history.append(current_lb)

                
                if current_lb > best_lb:
                    best_lb = current_lb

                lagrangian_sol = self.computeLagrangianSolution(labda)


                feasible_sol = self.convertToFeasibleSolution(lagrangian_sol)

                current_up = np.inf 
                if feasible_sol.isFeasible():
                    current_up = feasible_sol.getCosts()
                    upper_bounds_history.append(current_up)
                    if current_up < best_up:
                        best_up = current_up
                        best_feasible_solution = feasible_sol
                else:
                    print("solution is not feasible!")


                Best_upper_bounds_history.append(best_up)
 
                
                if best_up < np.inf and best_lb > -np.inf and best_up > 0: # A
                    gap = (best_up - best_lb) / best_up
                    #print(gap)



                
                if gap <= gap_tolerance:
                    print(f"\nTermination criterion met: Gap ({gap*100:.2f}%) <= Tolerance ({gap_tolerance*100:.2f}%) at iteration {k}.")
                    break
                
                #print(labda)
                labda = self.updateMultipliers(labda, lagrangian_sol, k, initial_step)
                #print(labda)
                #break



            print("\nLagrangian Heuristic Finished.")
            #print("gap_tolerance :",gap_tolerance)
            print(f"Final Best Lower Bound: {best_lb:.2f}")
            print(f"Final Best Upper Bound: {best_up:.2f}")
            final_gap = (best_up - best_lb) / best_up if best_up > 0 and best_up < np.inf else np.inf
            print(f"Final Gap: {final_gap*100:.2f}%")

            iterations = range(1, k + 1) 

            plt.figure(figsize=(10, 6)) 


            plt.plot(iterations, lower_bounds_history, label='Best Lower Bound', color='blue', linestyle='-')

            plt.plot(iterations, Best_upper_bounds_history, label='Best Upper Bound', color='red', linestyle='-')

            plt.plot(iterations, upper_bounds_history, label='Upper Bound', color='black', linestyle='-',linewidth=0.5,alpha=0.5)


            plt.xlabel('Iteration')
            plt.ylabel('Objective Value (Cost)')
            plt.title('Lagrangian Heuristic Bounds Progression')
            plt.legend()
            plt.grid(True)
            plt.tight_layout() 
            plt.show()

            return best_feasible_solution, best_lb, best_up, lower_bounds_history, upper_bounds_history
        
        
        
        
read_instance = UFL_Problem.readInstance("MO3")
n_markets = read_instance.n_markets
n_facilities = read_instance.n_facilities


y = np.ones(n_facilities)
x = np.ones((n_markets, n_facilities)) / n_facilities

#print(x.sum(axis=1))
lagrangian_heuristic = LagrangianHeuristic(read_instance)
lagrangian_heuristic.computeTheta(np.ones(n_markets))
solution = lagrangian_heuristic.computeLagrangianSolution(np.zeros(n_markets) * 20)
#print(UFL_Solution.isFeasible(solution))
#print(solution.getCosts())
#solution = lagrangian_heuristic.convertToFeasibleSolution(solution)
#print(UFL_Solution.isFeasible(solution))
#print(solution.getCosts())

#lagrangian_heuristic.updateMultipliers(np.ones(n_markets) * 2, solution, 100)
#print("Running Heuristic:")
#print(lagrangian_heuristic.runHeuristic())
#lagrangian_heuristic.runHeuristic(max_iterations=2500, initial_step=1.0, gap_tolerance=0.1)
#print(read_instance)
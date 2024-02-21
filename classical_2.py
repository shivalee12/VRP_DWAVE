import pulp
import pandas as pd
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
import time
import copy
import numpy as np
from data_preparation import DataCalculator
import itertools as it
import streamlit as st
import matplotlib.patches as patches


N = 11
data = DataCalculator("data.xlsx", "locations", "vehicle_data_size", N)
c = data.get_cost_matrix(k=400, max_row=N, max_column=N)
t = data.get_time_matrix(max_row=N, max_column=N)
Q = data.get_max_capacity()
q = data.get_order_size()
N_0 = len(c[0])-1  # Number of locations (excluding depot)
K = len(c) # number of vehicles
total_vehicle_count=K
customer_count = N
cost = c
lat, long = data.get_lat_long()
T_bound = 4*np.max(t)
q.insert(0, 0)

#Iterates over the different vehicles, to see if the optimal solution can be obtained with less than the max number of vehicles
def classical_form(customer_count, total_vehicle_count, cost, lat, long):
    # st.write("You selected classical.")

    for vehicle_count in range(1,total_vehicle_count+1): 
        
        # definition of LpProblem instance
        problem = pulp.LpProblem("CVRP", pulp.LpMinimize)

        # definition of variables which are 0/1
        x = [[[pulp.LpVariable("x%s_%s,%s"%(i,j,k), cat="Binary") if i != j else None for k in range(vehicle_count)]for j in range(customer_count)] for i in range(customer_count)]

        start_time = time.time()
        # add objective function
        problem += pulp.lpSum(cost[k][i][j] * x[i][j][k] if i != j else 0
                            for k in range(vehicle_count) 
                            for j in range(customer_count) 
                            for i in range (customer_count))

        # # Time constraint
        # for k in range(vehicle_count):
            
        #     problem += pulp.lpSum(t[k][i][j] * x[i][j][k] if i != j else 0                   
        #                       for j in range(customer_count) 
        #                       for i in range (customer_count))<=T_bound
            
            

        # constraints
        # Only one vehicle visits each node
        for j in range(1, customer_count):
            problem += (pulp.lpSum(x[i][j][k] if i != j else 0 
                                for i in range(customer_count) 
                                for k in range(vehicle_count)) == 1) 

        # Every vehicle has only one outgoing and incoming link from depot
        for k in range(vehicle_count):
            problem += (pulp.lpSum(x[0][j][k] for j in range(1,customer_count)) == 1)
            problem += (pulp.lpSum(x[i][0][k] for i in range(1,customer_count)) == 1)

        # Each node has one vehicle coming in and one vehicle outgoing
        for k in range(vehicle_count):
            for j in range(customer_count):
                problem += (pulp.lpSum(x[i][j][k] if i != j else 0 
                                    for i in range(customer_count)) -  pulp.lpSum(x[j][i][k] if i != j else 0 for i in range(customer_count)) == 0)

        #Delivery Capacity of each vehicle should not exceed maximum capacity
        for k in range(vehicle_count):
            problem += (pulp.lpSum(q[j] * x[i][j][k] if i != j else 0 for i in range(customer_count) for j in range (1,customer_count)) <= Q[k]) 


        #Subtour constraint
        subtours = []
        for i in range(2,customer_count):
            subtours += it.combinations(range(1,customer_count), i)

        for s in subtours:
            problem+=(pulp.lpSum(x[i][j][k] if i !=j else 0 for i, j in it.permutations(s,2) for k in range(vehicle_count)) <= len(s) - 1)

        status = problem.solve()
        print("feasibility of solution with",k+1,"vehicles","=",status) #If status =1 then solution is feasible, -1 is infeasible
        # print vehicle_count which needed for solving problem
        # print calculated minimum cost value
        if status == 1:
            print('Vehicle Requirements:', vehicle_count)
            print('Cost:', pulp.value(problem.objective))
            st.write('Cost:', pulp.value(problem.objective))
            end_time = time.time()
            execution_time = end_time - start_time
            cost = pulp.value(problem.objective)
            print("Execution time:", execution_time, "seconds")
            st.write("Execution time:", execution_time*1000, "ms")          
            for k in range(vehicle_count):
                vehicle_travel_time = 0  # Variable to store the travel time for the current vehicle

                for i in range(customer_count):
                    for j in range(customer_count):
                        if i != j and pulp.value(x[i][j][k]) == 1:
                            vehicle_travel_time += t[k][i][j]

                print("Total travel time for vehicle", k+1, ":", vehicle_travel_time)


            
    #        print("i","  ","j","  ","k")


    #        for k in range(vehicle_count):
    #            for i in range(customer_count):
    #                for j in range(customer_count):


    #                    if(i!=j):
    #                       if pulp.value(x[i][j][k])==1:
    #                          print(i,"  ",j,"  ",k)
    #To display the path of the vehicle as Vehicle: 2 0 -> 2 -> 3 , etc                   
            for k in range(vehicle_count):
                path = []  # List to store the path for vehicle 0
                current_node = 0  # Starting node
                visited = set()  # Set to keep track of visited nodes

                while current_node not in visited:
                    visited.add(current_node)  # Mark the current node as visited
                    path.append(current_node)  # Add the current node to the path

                    next_node = None  # Variable to store the next node to visit

                    for j in range(customer_count):
                        if current_node != j and pulp.value(x[current_node][j][k]) == 1:
                            next_node = j  # Update the next node to visit
                            break

                    if next_node is None:
                        break  # Exit the loop if no valid next node is found

                    current_node = next_node  # Update the current node to the next node
            
            
                path.append(0)

                print("Vehicle ", k+1,":", " -> ".join(str(node) for node in path))
                st.write("Vehicle ", k+1,":", " -> ".join(str(node) for node in path))
            
                        
            #Plotting using matplotlib
            fig, ax = plt.subplots()
            for i in range(customer_count):    
                if i == 0:
                    ax.scatter(lat[i], long[i], c='green', s=200)
                    ax.text(lat[i], long[i], "depot", fontsize=12)
                else:
                    ax.scatter(lat[i], long[i], c='orange', s=50)
                    ax.text(lat[i], long[i], str(i), fontsize=12)

            for k in range(vehicle_count):
                for i in range(customer_count):
                    for j in range(customer_count):
                        if i != j and pulp.value(x[i][j][k]) == 1:
                            color = "black" if k == 0 else "blue" if k == 1 else "red"
                            ax.plot([lat[i], lat[j]], [long[i], long[j]], color=color)
                            # if k==0 and pulp.value(x[i][j][k]) == 1:
                            #     plt.plot([lat[i], lat[j]], [long[i], long[j]], color="black")
                                
                        
                            # if k==1 and pulp.value(x[i][j][k])==1:
                            #     plt.plot([lat[i], lat[j]], [long[i], long[j]], color="blue")
                    

                            # if k==2 and pulp.value(x[i][j][k])==1:
                            #     plt.plot([lat[i], lat[j]], [long[i], long[j]], color="red")
                            arrow = patches.FancyArrow(lat[i], long[i], lat[j] - lat[i], long[j] - long[i],
                                          width=0.0001, edgecolor=color, facecolor=color, alpha=1,head_width=0.005, head_length=0.005)
                            ax.add_patch(arrow)
                        

                        
            plt.savefig(f"{customer_count-1}_locations")
            image_path=f"{customer_count-1}_locations.png"
            st.image(image_path, caption=f"{customer_count-1}_locations and {vehicle_count} vehicles", use_column_width=True)

            plt.close()   
            break
    
    return execution_time*1000, cost
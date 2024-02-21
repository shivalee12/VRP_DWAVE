import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.patches as patches

def compute_x_matrix(y):
    """
    Compute the x matrix based on the input array y.

    Args:
        y (ndarray): Input array of shape (k, n, n), where k is the number of samples,
                     n is the dimension, and n-1 is the index range.

    Returns:
        ndarray: The computed x matrix of shape (k, n+1, n+1).
    """
    k = y.shape[0]
    n = y.shape[1]
    x = np.zeros((k, n + 1, n + 1))
    
    for s in range(k):
        for i in range(n):
            for j in range(n):
                for alpha in range(n - 1):
                    x[s, i+1, j+1] += y[s, i, alpha] * y[s, j, alpha + 1]

    
    for s in range(k):
        for i in range(n):
            x[s, 0, i+1] = y[s, i, 0]
            for alpha in range(1, n):
                x[s, 0, i+1] += (1 - np.sum(y[s, :, alpha - 1])) * y[s, i, alpha]

    for s in range(k):
        for i in range(n):
            x[s, i+1, 0] = y[s, i, n - 1]
            for alpha in range(n - 1):
                x[s, i+1, 0] += y[s, i, alpha] * (1 - np.sum(y[s, :, alpha + 1]))

    return x

def path(x):
    K = x.shape[0]
    N_0 = x.shape[1]-1
    for k in range(K):
        path = []  # List to store the path for vehicle 0
        current_node = 0  # Starting node
        visited = set()  # Set to keep track of visited nodes

        while current_node not in visited:
            visited.add(current_node)  # Mark the current node as visited
            path.append(current_node)  # Add the current node to the path

            next_node = None  # Variable to store the next node to visit

            for j in range(N_0+1):
                if current_node != j and (x[k][current_node][j]) == 1:
                    next_node = j  # Update the next node to visit
                    break

            if next_node is None:
                break  # Exit the loop if no valid next node is found

            current_node = next_node  # Update the current node to the next node

        path.append(0)

        print("Vehicle ", k+1, ":", " -> ".join(str(node) for node in path))
        st.write("Vehicle ", k+1, ":", " -> ".join(str(node) for node in path))

def find_paths_in_binary_matrix(matrix):
    """
    Find paths in a binary matrix and return a list of paths for each vehicle.

    Args:
        matrix (ndarray): Input binary matrix of shape (k, n, n), where k is the number of vehicles,
                          and n is the dimension of the square matrix.

    Returns:
        list: A list of paths for each vehicle, where each path is a string in the format "{start} -> {end}",
              representing the indices of the matrix where the value is 1.

    Example:
    >>> matrix = np.array([[[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]],
                              
                              [[0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 0]]])
    >>> find_paths_in_binary_matrix(matrix)
    Vehicle_1 path: ['1 -> 0', '0 -> 1', '1 -> 2', '2 -> 1']
    Vehicle_2 path: ['0 -> 1', '0 -> 2', '1 -> 0']
    Out: [['1 -> 0', '0 -> 1', '1 -> 2', '2 -> 1'], ['0 -> 1', '0 -> 2', '1 -> 0']]

    Note:
        - The binary matrix represents connections between indices, where a value of 1 indicates a connection
          between two indices.
        - Each vehicle has its own set of paths, and the paths are stored in a list.
        - The function prints the paths for each vehicle and returns all paths as the result.
    """

    k = matrix.shape[0]
    n = matrix.shape[1]
    all_paths = []
    for s in range(k):
        paths = []
        for i in range(n):
            for j in range(n):
                if matrix[s, i, j] == 1:
                    path = f"{i} -> {j}"
                    paths.append(path)
        print(f"Vehicle_{s+1} path: {paths}")
        all_paths.append(paths)
    return paths

def calculate_cost(K, N_0, solutions, c):
    """
    Calculate the cost based on the given parameters and solutions.

    Args:
        K (int): Number of vehicles.
        N_0 (int): Dimension of the matrix.
        solutions (dict): Dictionary of solutions with keys representing indices and values representing values.
                          The keys are in the format 'y_ialphav', where 'v' represents the vehicle number, 'i' and 'alpha'
                          represent the indices of the matrix
        c (ndarray): Coefficient matrix of shape (N_0+1, N_0+1).
    Returns:
        tuple: A tuple containing the cost, x_matrix, and matrix.
            - cost (float): The calculated cost based on the solutions.
            - x_matrix (ndarray): The computed x matrix of shape (K, N_0+1, N_0+1) or (N_0+1, N_0+1) depending on K.
            - matrix (ndarray): The matrix generated from the solutions of shape (K, N_0, N_0) or (N_0, N_0) depending on K.

    Note:
        - The function calculates the cost by computing the x matrix using the given solutions and the coefficient matrix.
        - If K is greater than 1, the matrix is of shape (K, N_0, N_0), otherwise it is of shape (N_0, N_0).
        - The x matrix is computed using the compute_x_matrix function.
        - The cost is the sum of element-wise multiplication between c and x_matrix.
        - The function returns the cost, x_matrix, and y-matrix as a tuple.
    """
    if K>1:
        matrix = np.zeros((K, N_0, N_0), dtype=int)
        
        for key, value in solutions.items():
            components = key.split('_')
            # print(components)
            i = components[1]
            alpha = components[2]
            v = components[3]
            x = int(i) - 1
            y = int(alpha) - 1
            z = int(v)
            # print(x,y,z)
            # print(value)
            matrix[z, x, y] = value
            
            
    else:
        matrix = np.zeros((N_0, N_0), dtype=int)
        for key, value in solutions.items():
            components = key.split('_')
            i = components[1]
            alpha = components[2]
            
            x = i - 1
            y = alpha - 1
            
                
            matrix[x, y] = value
    x_matrix = compute_x_matrix(np.array(matrix))
    cost = np.sum(c * x_matrix)
    return cost, x_matrix, matrix

def calculate_time(K, N_0, solutions, t):
    """
    Calculate the time taken for each vehicle based on the given parameters and solutions.

    Args:
        K (int): Number of vehicles.
        N_0 (int): Dimension of the matrix.
        solutions (dict): Dictionary of solutions with keys representing indices and values representing values.
                          The keys are in the format 'y_ialphav', where 'v' represents the vehicle number, 'i' and 'alpha'
                          represent the indices of the matrix
        t (ndarray): Time matrix of shape (K, N_0+1, N_0+1), where K is the number of vehicles, and N_0+1 is the size of the matrix.

    Returns:
        None: The function does not return any value. It prints the time taken for each vehicle.

    Note:
        - The function calculates the time taken by each vehicle based on the given solutions and the time matrix.
        - The matrix is of shape (K, N_0, N_0) if K is greater than 1, otherwise it is of shape (1, N_0, N_0).
        - The time taken by each vehicle is computed by element-wise multiplication between t and the corresponding x_matrix for the vehicle.
        - The time is printed for each vehicle in minutes.
        - The function does not return any value.
    """
    matrix = np.zeros((K, N_0, N_0), dtype=int)
    for key, value in solutions.items():
        x = int(key[2]) - 1
        y = int(key[3]) - 1
        z = int(key[4])
        if K-1:
            matrix[z, x, y] = value
        else:
            matrix[0, x, y] = value 
    x_matrix = compute_x_matrix(np.array(matrix))
    for i in range(x_matrix.shape[0]):
        time_taken = np.sum(t[i] * x_matrix[i])
        print(f'Vehicle_{i+1} time: {time_taken*60:.2f}mins')
    
def cost_optim(K, N_0, solution, c,t , A,B,C,D,E,Q,q,T_bound):
    """
    Calculates the cost of a given solution.

    Args:
        K: The number of vehicles.
        N_0: The number of nodes.
        solution: The solution as a dictionary of binary variables.
        c: The cost matrix.
        t: The travel time matrix.
        A: The weight of the cost term.
        B: The weight of the flow conservation term.
        C: The weight of the connectivity term.
        D: The weight of the capacity constraint term.
        E: The weight of the travel time term.
        Q: The capacity of the vehicles.
        q: The demand at each node.
        T_bound: The upper bound on the total travel time.

    Returns:
        The cost of the solution, equality penalties, and capacity penalty
    """
    final_value = 0.0

    for v in range(K):
        for i in range(1,N_0+1):
            for j in range(1,N_0+1):
                for alpha in range(1,N_0):
                    final_value += A * c[v][i][j] * int(solution[f"y_{i}_{alpha}_{v}"]) * int(solution[f"y_{j}_{alpha + 1}_{v}"])

    for v in range(K):
        for i in range(1,N_0+1):
            final_value += A * c[v][0][i] * (int(solution[f"y_{i}_1_{v}"]) + sum((1 - sum(int(solution[f"y_{j}_{alpha-1}_{v}"]) for j in range(1,N_0+1) if j != i)) * int(solution[f"y_{i}_{alpha}_{v}"]) for alpha in range(2, N_0 + 1)))
            final_value += A * c[v][i][0] * (int(solution[f"y_{i}_{N_0}_{v}"]) + sum(int(solution[f"y_{i}_{alpha}_{v}"]) * (1 - sum(int(solution[f"y_{j}_{alpha+1}_{v}"]) for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))
    cost = final_value
    for i in range(1,N_0+1):
        final_value += B * (1 - sum(int(solution[f"y_{i}_{alpha}_{v}"]) for alpha in range(1, N_0 + 1) for v in range(K)))**2

    cost_b = final_value-cost

    for alpha in range(1, N_0 + 1):
        final_value += C * (1 - sum(int(solution[f"y_{i}_{alpha}_{v}"]) for i in range(1,N_0 + 1) for v in range(K)))**2
    cost_c = final_value - cost - cost_b

    for v in range(K):
        final_value += D*(sum(q[i-1]*int(solution[f"y_{i}_{alpha}_{v}"]) for alpha in range(1, N_0 + 1) for i in range(1,N_0+1))-Q[v])
        
        # *(sum(q[i-1]*int(solution[f"y_{i}{alpha}{v}"]) for alpha in range(1, N_0 + 1) for i in range(1,N_0+1))-Q[v]*0.75)
    cost_cap = final_value-cost-cost_b-cost_c

    # Encourage low total travel time
    # for v in range(K):
    #     for i in range(1,N_0+1):
    #         for j in range(1,N_0+1):
    #             for alpha in range(1,N_0):
    #                 final_value += E * t[v][i][j] * int(solution[f"y_{i}{alpha}{v}"]) * int(solution[f"y_{j}{alpha+1}{v}"])
    #         final_value += E * t[v][0][i] * (int(solution[f"y_{i}1{v}"]) + sum((1 - sum(int(solution[f"y_{j}{alpha-1}{v}"]) for j in range(1,N_0+1) if j != i)) * int(solution[f"y_{i}{alpha}{v}"]) for alpha in range(2, N_0 + 1)))
    #         final_value += E * t[v][i][0] * (int(solution[f"y_{i}{N_0}{v}"]) + sum(int(solution[f"y_{i}{alpha}{v}"]) * (1 - sum(int(solution[f"y_{j}{alpha+ 1}{v}"]) for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))
        
    
    return final_value , cost, cost_b, cost_c, cost_cap #- K*E*T_bound

# def output(N,K,lat,long,x):
#     for i in range(N):    
#         if i == 0:
#             plt.scatter(lat[i], long[i], c='green', s=200)
#             plt.text(lat[i], long[i], "depot", fontsize=12)
#         else:
#             plt.scatter(lat[i], long[i], c='orange', s=50)
#             plt.text(lat[i], long[i], str(i), fontsize=12)

#     for k in range(K):
#         for i in range(N):
#             for j in range(N):
#                 if i != j and (x[k][i][j]) == 1:
#                     if k==0 and (x[k][i][j]) == 1:
#                         plt.plot([lat[i], lat[j]], [long[i], long[j]], color="black")
                
#                     if k==1 and (x[k][i][j])==1:
#                         plt.plot([lat[i], lat[j]], [long[i], long[j]], color="blue")
                

#                     if k==2 and (x[k][i][j])==1:
#                         plt.plot([lat[i], lat[j]], [long[i], long[j]], color="red")
                

                
#     plt.savefig(f"{N-1} locations")
#     plt.show()   
#     plt.axis('off')  # Hide axis ticks and labels
#     st.pyplot()


def output(N, K, lat, long, x):
    fig, ax = plt.subplots()

    for i in range(N):
        if i == 0:
            ax.scatter(lat[i], long[i], c='green', s=200)
            ax.text(lat[i], long[i], "depot", fontsize=12)
        else:
            ax.scatter(lat[i], long[i], c='orange', s=50)
            ax.text(lat[i], long[i], str(i), fontsize=12)

    for k in range(K):
        for i in range(N):
            for j in range(N):
                if i != j and (x[k][i][j]) == 1:
                    color = "black" if k == 0 else "blue" if k == 1 else "red"
                    ax.plot([lat[i], lat[j]], [long[i], long[j]], color=color)

                    # if k == 0 and (x[k][i][j]) == 1:
                    #     ax.plot([lat[i], lat[j]], [long[i], long[j]], color="black")

                    # if k == 1 and (x[k][i][j]) == 1:
                    #     ax.plot([lat[i], lat[j]], [long[i], long[j]], color="blue")

                    # if k == 2 and (x[k][i][j]) == 1:
                    #     ax.plot([lat[i], lat[j]], [long[i], long[j]], color="red")
                    arrow = patches.FancyArrow(lat[i], long[i], lat[j] - lat[i], long[j] - long[i],
                                          width=0.0001, edgecolor=color, facecolor=color, alpha=1,head_width=0.005, head_length=0.005)
                    ax.add_patch(arrow)

    # plt.axis('off')  # Hide axis ticks and labels

    return fig


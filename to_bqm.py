import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
from data_preparation import DataCalculator
from y_to_x import cost_optim, calculate_cost,path, find_paths_in_binary_matrix, output
import streamlit as st
import dwave.cloud as dc

def binarybqm_form(N_0,K, c, t, Q, q, A, B, C, D, E, T_bound,lat,long,num_reads=3500):
    # Create a QUBO object
    # st.write("You selected bqm.")
    qubo = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Minimization of distance costs
    for v in range(K):
        for i in range(1,N_0+1):
            for j in range(1,N_0+1):
                for alpha in range(1,N_0):
                    qubo += A * c[v][i][j] * dimod.Binary(f"y_{i}_{alpha}_{v}") * dimod.Binary(f"y_{j}_{alpha + 1}_{v}")

    # Minimization of distance costs from depot to locations and vice versa
    for v in range(K):
        for i in range(1,N_0+1):
            qubo += A * c[v][0][i] * (dimod.Binary(f"y_{i}_1_{v}") + sum((1 - sum(dimod.Binary(f"y_{j}_{alpha - 1}_{v}") for j in range(1,N_0+1) if j != i)) * dimod.Binary(f"y_{i}_{alpha}_{v}") for alpha in range(2, N_0 + 1)))
            qubo += A * c[v][i][0] * (dimod.Binary(f"y_{i}_{N_0}_{v}") + sum(dimod.Binary(f"y_{i}_{alpha}_{v}") * (1 - sum(dimod.Binary(f"y_{j}_{alpha + 1}_{v}") for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))

    # Encourage each location to be visited exactly once across all vehicles
    for i in range(1,N_0+1):
        qubo += B * (1 - sum(dimod.Binary(f"y_{i}_{alpha}_{v}") for alpha in range(1, N_0 + 1) for v in range(K)))**2


    # Encourage each time step to be visited exactly once across all vehicles
    for alpha in range(1, N_0 + 1):
        qubo += C * (1 - sum(dimod.Binary(f"y_{i}_{alpha}_{v}") for i in range(1,N_0 + 1) for v in range(K)))**2

    # Encourage capacity constraints for each vehicle
    for v in range(K):
        # qubo += D * (sum(q[i-1] ** 2 * dimod.Binary(f"y_{i}{alpha}{v}") ** 2 for i in range(1,N_0+1) for alpha in range(1, N_0 + 1))
        #              - Q[v] * sum(q[i-1] * dimod.Binary(f"y_{i}{alpha}{v}") for i in range(1,N_0+1) for alpha in range(1, N_0 + 1))
        #              + (Q[v] ** 2)* 0.75)
        qubo += D*(sum(q[i-1]*dimod.Binary(f"y_{i}_{alpha}_{v}") for alpha in range(1, N_0 + 1) for i in range(1,N_0+1))-Q[v])
        # *(sum(q[i-1]*dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(1, N_0 + 1) for i in range(1,N_0+1))-Q[v]*0.75)

    # Encourage low total travel time
    # for v in range(K):
    #     for i in range(1,N_0+1):
    #         for j in range(1,N_0+1):
    #             for alpha in range(1,N_0):
    #                 qubo += E * t[v][i][j] * dimod.Binary(f"y_{i}{alpha}{v}") * dimod.Binary(f"y_{j}{alpha + 1}{v}")
    #         qubo += E * t[v][0][i] * (dimod.Binary(f"y_{i}1{v}") + sum((1 - sum(dimod.Binary(f"y_{j}{alpha - 1}{v}") for j in range(1,N_0+1) if j != i)) * dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(2, N_0 + 1)))
    #         qubo += E * t[v][i][0] * (dimod.Binary(f"y_{i}{N_0}{v}") + sum(dimod.Binary(f"y_{i}{alpha}{v}") * (1 - sum(dimod.Binary(f"y_{j}{alpha + 1}{v}") for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))



    # # Upper bound constraint on total travel time
    # qubo += -T_bound

    # Get the quadratic and linear coefficients
    quadratic = qubo.quadratic
    linear = qubo.linear

    # Print the coefficients
    # print("Quadratic Coefficients:")
    # for (u, v), coeff in quadratic.items():
    #     print(f"{u}: {v} = {coeff}")

    # print("\nLinear Coefficients:")
    # for v, coeff in linear.items():
    #     print(f"{v} = {coeff}")

    # # Get the list of variables
    # variables = list(qubo.variables)
    # print(variables)
    # # Create an empty matrix
    # matrix_size = len(variables)
    # qubo_matrix = np.zeros((matrix_size, matrix_size))

    # # Populate the matrix with quadratic coefficients
    # for (u, v), coeff in qubo.quadratic.items():
    #     u_idx = variables.index(u)
    #     v_idx = variables.index(v)
    #     qubo_matrix[u_idx, v_idx] = coeff

    # print(qubo_matrix)
    # Define the solver and sampler
    sampler = EmbeddingComposite(DWaveSampler())

    # Solve the BQM
    response = sampler.sample(qubo, num_reads=num_reads, label =f"{N_0} locations,B,C={B}, D={D}")
    qpu_access_time = response.info['timing']['qpu_access_time']

    # Retrieve the optimal solution(s)
    solutions = response.first.sample
    # Print the QPU access time
    # solver = dc.get_solver(sampler)
    # qpu_access_time = response.info['timing']['qpu_access_time']
    # Process and analyze the solutions
    # route = [0, 2, 3, 1, 0]  # Route indices: 0 -> 2 -> 3 -> 1 -> 0

    # for variable, value in solutions.items():
    #     print(variable, value)

    s, cost, costb, costc, costcap = cost_optim(K, N_0, solutions, c, t, A,
                        B, C, D, E, Q, q, T_bound)
    _, x, y = calculate_cost(K, N_0, solutions, c)
    find_paths_in_binary_matrix(x)
    path(x)
    print(f"cost_b={costb}")
    print(f"cost_c={costc}")
    print(f"capacity cost = {costcap}")
    print(f"cost of path is {cost}")
    print(f"hamiltonian cost:{s}")
    fig=output(N_0+1,K,lat,long,x)
    st.write(f"Equality penalty={costb +costc}")
    # st.write(f"cost_c={costc}")
    st.write(f"capacity cost = {costcap}")
    st.write(f"cost of path is {cost}")
    st.write(f"QPU time:{qpu_access_time/1000}ms")
    st.write(f"hamiltonian cost:{s}")
    st.pyplot(fig)
    execution_time = qpu_access_time/1000
    
    return  execution_time, cost

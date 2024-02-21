import dimod
from dimod import Binary, ConstrainedQuadraticModel, quicksum
from dwave.system import LeapHybridCQMSampler
import numpy as np
from y_to_x import compute_x_matrix, find_paths_in_binary_matrix, calculate_cost, cost_optim, output,path
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from minorminer.busclique import find_clique_embedding
import numpy as np
import streamlit as st
import dwave.cloud as dc

def binarycqm_form(N_0, K, c, t, Q, q, A, B, C, D, E, T_bound,time,lat,long):
    # st.write("You selected cqm.")
    print("\nBuilding constrained quadratic model...")
    cqm = ConstrainedQuadraticModel()

    for v in range(K):
        for i in range(1,N_0+1):
            for alpha in range(1,N_0+1):
                cqm.add_variable('BINARY', f'y_{i}_{alpha}_{v}')

    #Defining the binary variables
    y  = {(i, alpha, v): dimod.Binary(f'y_{i}_{alpha}_{v}')
                            for v in range(K)
                            for i in range(1,N_0+1)
                            for alpha in range(1,N_0+1)}
    # cqm = dimod.ConstrainedQuadraticModel()

    # for v in range(K):
    #     for i in range(1,N_0+1):
    #         for alpha in range(1,N_0+1):
    #             cqm.add_variable("BINARY",)


    #Calculate cost of travelling between locations
    
    print("\nSetting objective of constrained quadratic model...")
    qubo = dimod.BinaryQuadraticModel.empty("BINARY")
    for v in range(K):
        for i in range(1,N_0+1):
            for alpha in range(1,N_0+1):
                qubo.add_variable(f'y_{i}_{alpha}_{v}',1)

    for v in range(K):
        for i in range(1,N_0+1):
            for j in range(1,N_0+1):
                for alpha in range(1,N_0):
                    #qubo.add_quadratic(dimod.Binary(f"y_{i}{alpha}{v}"),dimod.Binary(f"y_{j}{alpha + 1}{v}"),c[v][i][j])
                    qubo += c[v][i][j]*y[i,alpha,v]*y[j,alpha+1,v]

    #Calculate cost of travelling to and from depot
    for v in range(K):
        for i in range(1,N_0+1):
            qubo +=  c[v][0][i] * (y[i,1,v] + quicksum((1 - quicksum(y[j,alpha-1,v] for j in range(1,N_0+1) if j != i)) * y[i,alpha,v] for alpha in range(2, N_0 + 1)))
            qubo +=  c[v][i][0] * (y[i,N_0,v] + quicksum(y[i,alpha,v] * (1 - quicksum(y[j,alpha+1,v] for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))


    cqm.set_objective(qubo) #Define Objective function

    print("\nAdding Constraints...")
    #Ensures that every node has a unique position in cycle
    for i in range(1,N_0+1):
        # qubo+=(sum(y[i,alpha,v] for alpha in range(1, N_0 + 1) for v in range(K))-1)**2
        cqm.add_constraint((quicksum(y[i,alpha,v] for alpha in range(1, N_0 + 1) for v in range(K)))==1)

    #Ensures that every node is visited once
    for alpha in range(1,N_0+1):
        # qubo+=(sum(y[i,alpha,v] for i in range(1,N_0 + 1) for v in range(K))-1)**2
        cqm.add_constraint((quicksum(y[i,alpha,v] for i in range(1,N_0 + 1) for v in range(K)))==1)

    #Ensures that the capacity constraint is satisfied
    for v in range(K):
        cqm.add_constraint(quicksum(q[i-1]*y[i,alpha,v] for alpha in range(1, N_0 + 1) for i in range(1,N_0+1)) - Q[v]<=0 )

    # print(cqm)
    # presolver = Presolver(cqm)
    # presolver.load_default_presolvers()
    # presolver.apply()
    # print(presolver.copy_model())

    sampler = LeapHybridCQMSampler()  
    sampleset = sampler.sample_cqm(cqm,time_limit=time)
    
    # solver = dc.get_solver(sampler)
    # qpu_access_time = solver.qpu_access_time
    # print("\nSampleset is:\n")
    # print(sampleset)
    # print("Best Sample\n")
    # best = sampleset.first.sample
    # s, cost, costb, costc, costcap = cost_optim(K, N_0, best, c, t, A,
    #                     B, C, D, E, Q, q, T_bound)
    # _, x, y = calculate_cost(K, N_0, best, c)
    # find_paths_in_binary_matrix(x)
    # print(f"cost_b={costb}")
    # print(f"cost_c={costc}")
    # print(f"capacity cost = {costcap}")
    # print(f"cost of path is {cost}")
    # print(f"hamiltonian cost:{s}")
    # print(sampleset.first.sample)
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)  
    print(feasible_sampleset.info) 
    qpu_access_time = feasible_sampleset.info['qpu_access_time']

    print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
    try:
        best = feasible_sampleset.first.sample
        s, cost, costb, costc, costcap = cost_optim(K, N_0, best, c, t, A,
                        B, C, D, E, Q, q, T_bound)
        _, x, y = calculate_cost(K, N_0, best, c)
        find_paths_in_binary_matrix(x)
        path(x)
        print(f"cost_b={costb}")
        print(f"cost_c={costc}")
        print(f"capacity cost = {costcap}")
        print(f"cost of path is {cost}")
        # print(f"hamiltonian cost:{s}")
        fig=output(N_0+1,K,lat,long,x)
        # st.write(f"cost_b={costb}")
        # st.write(f"cost_c={costc}")
        # st.write(f"capacity cost = {costcap}")
        st.write(f"cost of path is {cost}")
        st.write(f"QPU time:{qpu_access_time/1000} ms")
        # st.write(f"hamiltonian cost:{s}")
        st.pyplot(fig)
        execution_time = qpu_access_time/1000


    except:
        print("no feasible solution")
        st.write("no feasible solution")
        # time+=10
        # binarycqm_form(N_0, K, c, t, Q, q, A, B, C, D, E, T_bound,time)
    # print("best solution=", best)
    
    #Checks and sees whether constraints are violated or not
    #for label, violation in cqm.iter_violations(best, clip=False): #(if clip = true, then negative violations are rounded to 0)
    #    print(label, violation)

    return execution_time, cost

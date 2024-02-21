from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, ADAM
from qiskit.utils import algorithm_globals
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.primitives import Sampler
import numpy as np
from data_preparation import DataCalculator
from to_qubo import creat_qubo
# Define the VRP parameters
file_path = 'examples/example2.xlsx'
data=DataCalculator(file_path, "locations", "vehicle_data_size")
c = data.get_cost_matrix(k=1)
t = data.get_time_matrix()
Q = data.get_max_capacity()
q = data.get_order_size()
N_0 = len(c[0])-1  # Number of locations (excluding depot)
A = 1.0  # Coefficient for distance costs keep it 1
B = 1000000.0  # Coefficient for location visitation -shud be kept high
C = 1000000.0  # Coefficient for time step visitation -shud be kept high
D = 100.0 # Coefficient for order delivery
E = 100.0  # Coefficient for total travel time-less than b and c
T_bound = 10000.0  # Upper bound constraint on total travel time
K = len(c) # number of vehicles
class QuantumOptimizer:
    def __init__(self, instance, n, K):

        self.instance = instance
        self.n = n
        self.K = K

    def binary_representation(self, x_sol=0):

        instance = self.instance
        n = self.n
        K = self.K

        A = 1 #np.max(instance)   # A parameter of cost function
        B = 100 #np.max(instance) * 10
        C = 100 #np.max(instance) * 10
        # Determine the weights w
        instance_vec = instance.reshape(n**2)
        w_list = [instance_vec[x] for x in range(n**2) if instance_vec[x] > 0]
        w = np.zeros(n * (n - 1))
        for ii in range(len(w_list)):
            w[ii] = w_list[ii]

        # Some variables I will use
        Id_n = np.eye(n)
        Im_n_1 = np.ones([n - 1, n - 1])
        Iv_n_1 = np.ones(n)
        Iv_n_1[0] = 0
        Iv_n = np.ones(n - 1)
        neg_Iv_n_1 = np.ones(n) - Iv_n_1

        v = np.zeros([n, n * (n - 1)])
        for ii in range(n):
            count = ii - 1
            for jj in range(n * (n - 1)):

                if jj // (n - 1) == ii:
                    count = ii

                if jj // (n - 1) != ii and jj % (n - 1) == count:
                    v[ii][jj] = 1.0

        vn = np.sum(v[1:], axis=0)

        try:
            max(x_sol)
            # Evaluates the cost distance from a binary representation of a path
            fun = (
                lambda x: np.dot(np.around(x), np.dot(Q, np.around(x)))
                + np.dot(g, np.around(x))
                + c
            )
            cost = fun(x_sol)
        except:
            cost = 0

        return Q, g, c, cost

    def construct_problem(self, Q, g, c) -> QuadraticProgram:
        qp = QuadraticProgram()
        for i in range((n-1) * (n-1)):
            qp.binary_var(str(i))
        qp.objective.quadratic = Q
        qp.objective.linear = g
        qp.objective.constant = c
        return qp

    # def solve_problem(self, qp):
    #     algorithm_globals.random_seed = 10598
    #     vqe = SamplingVQE(sampler=Sampler(), optimizer=SPSA(), ansatz=RealAmplitudes())
    #     optimizer = MinimumEigenOptimizer(min_eigen_solver=vqe)
    #     result = optimizer.solve(qp)
    #     # compute cost of the obtained result
    #     _, _, _, level = self.binary_representation(x_sol=result.x)
    #     return result.x, level

    def solve_problem(self, qp):
        algorithm_globals.random_seed = 10598
        # quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
        #                                    seed_simulator=algorithm_globals.random_seed,
        #                                    seed_transpiler=algorithm_globals.random_seed)
        # sampler = Aer.get_backend('qasm_simulator')
        # Define the classical optimizer
        optimizer = COBYLA()
        qaoa = QAOA(sampler=Sampler(), optimizer=SPSA())
        optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        result = optimizer.solve(qp)
        print(f'result is:{result}')
        # compute cost of the obtained result
        _,_,_,level = self.binary_representation(x_sol=result.x)
        return result.x, level
    
# Instantiate the quantum optimizer class with parameters:
quantum_optimizer = QuantumOptimizer(c, N_0+1, K)
Q,g,constant = creat_qubo(c,t,Q,q,N_0, A,B,C,D,E,T_bound, K)
qp = quantum_optimizer.construct_problem(Q, g, constant)
quantum_solution, quantum_cost = quantum_optimizer.solve_problem(qp)

print(quantum_solution, quantum_cost)

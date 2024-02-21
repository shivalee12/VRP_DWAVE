import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from minorminer.busclique import find_clique_embedding
import numpy as np

def creat_qubo(c,t,Q,q,N_0, A,B,C,D,E,T_bound, K):
    # Create a QUBO object
    qubo = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Minimization of distance costs
    for v in range(K):
        for i in range(1,N_0+1):
            for j in range(1,N_0+1):
                for alpha in range(1,N_0):
                    qubo += A * c[v][i][j] * dimod.Binary(f"y_{i}{alpha}{v}") * dimod.Binary(f"y_{j}{alpha + 1}{v}")

    # Minimization of distance costs from depot to locations and vice versa
    for v in range(K):
        for i in range(1,N_0+1):
            qubo += A * c[v][0][i] * (dimod.Binary(f"y_{i}1{v}") + sum((1 - sum(dimod.Binary(f"y_{j}{alpha - 1}{v}") for j in range(1,N_0+1) if j != i)) * dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(2, N_0 + 1)))
            qubo += A * c[v][i][0] * (dimod.Binary(f"y_{i}{N_0}{v}") + sum(dimod.Binary(f"y_{i}{alpha}{v}") * (1 - sum(dimod.Binary(f"y_{j}{alpha + 1}{v}") for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))

    # Encourage each location to be visited exactly once across all vehicles
    for i in range(1,N_0+1):
        qubo += B * (1 - sum(dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(1, N_0 + 1) for v in range(K)))**2


    # Encourage each time step to be visited exactly once across all vehicles
    for alpha in range(1, N_0 + 1):
        qubo += C * (1 - sum(dimod.Binary(f"y_{i}{alpha}{v}") for i in range(1,N_0 + 1) for v in range(K)))**2

    # Encourage capacity constraints for each vehicle
    for v in range(K):
        qubo += D*(sum(q[i-1]*dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(1, N_0 + 1) for i in range(1,N_0+1))-Q[v])*(sum(q[i-1]*dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(1, N_0 + 1) for i in range(1,N_0+1))-Q[v]*0.75)

    # Encourage low total travel time
    for v in range(K):
        for i in range(1,N_0+1):
            for j in range(1,N_0+1):
                for alpha in range(1,N_0):
                    qubo += E * t[v][i][j] * dimod.Binary(f"y_{i}{alpha}{v}") * dimod.Binary(f"y_{j}{alpha + 1}{v}")
            qubo += E * t[v][0][i] * (dimod.Binary(f"y_{i}1{v}") + sum((1 - sum(dimod.Binary(f"y_{j}{alpha - 1}{v}") for j in range(1,N_0+1) if j != i)) * dimod.Binary(f"y_{i}{alpha}{v}") for alpha in range(2, N_0 + 1)))
            qubo += E * t[v][i][0] * (dimod.Binary(f"y_{i}{N_0}{v}") + sum(dimod.Binary(f"y_{i}{alpha}{v}") * (1 - sum(dimod.Binary(f"y_{j}{alpha + 1}{v}") for j in range(1,N_0+1) if j != i)) for alpha in range(1, N_0)))



    # Upper bound constraint on total travel time
    qubo += -T_bound

    variables = list(qubo.variables)
    # print(variables)
    def get_quadratic(qubo, variables):
        # Get the quadratic and linear coefficients
        quadratic = qubo.quadratic
        # print(quadratic,linear)
        # Extract the unique row and column indices
        rows = columns = variables
        rows = sorted(set(rows))
        columns = sorted(set(columns))
        # Create an empty matrix
        matrix = np.zeros((len(rows), len(columns)))

        # Fill the matrix with values from the dictionary
        for key, value in quadratic.items():
            row_idx = rows.index(key[0])
            col_idx = columns.index(key[1])
            matrix[row_idx, col_idx] = round(value, 2)

        # Print the matrix
        np.set_printoptions(precision=2)
        return matrix

    def get_linaer(qubo,variables):
        linear = qubo.linear
        linear = list(linear.values())
        return linear

    def get_constant(qubo):
        return qubo.offset
    Q = get_quadratic(qubo, variables)
    g = get_linaer(qubo, variables)
    c = get_constant(qubo)
    # print(Q.shape)
    # print(g)
    return Q,g,c


from data_preparation import DataCalculator
from y_to_x import cost_optim, calculate_cost,path
# from to_csp import binarycsp_form
from to_cqm import binarycqm_form
from to_bqm import binarybqm_form
from classical_2 import classical_form
# Define the VRP parameters

N = 6
data = DataCalculator("data.xlsx", "locations", "vehicle_data_size", N)
c = data.get_cost_matrix(k=400, max_row=N, max_column=N)
t = data.get_time_matrix(max_row=N, max_column=N)
Q = data.get_max_capacity()
q = data.get_order_size()
N_0 = len(c[0])-1  # Number of locations (excluding depot)
A = 1.0  # Coefficient for distance costs keep it 1
B = 1000.0  # Coefficient for location visitation -shud be kept high
C = 1000.0  # Coefficient for time step visitation -shud be kept high
D = 100.0 # Coefficient for order delivery
E = 100.0  # Coefficient for total travel time-less than b and c
T_bound = 10000.0  # Upper bound constraint on total travel time
K = len(c) # number of vehicles
lat, long =data.get_lat_long()
num_reads=4000
# print("CSP result:")
# binarycsp_form(N_0,K, c, t, Q, q, A, B, C, D, E, T_bound)
# # print(csp)
# print("CQM result:")
# binarycqm_form(N_0,K, c, t, Q, q, A, B, C, D, E, T_bound,5,lat, long)

# print("BQM result:")
# binarybqm_form(N_0,K, c, t, Q, q, A, B, C, D, E, T_bound,lat, long,num_reads)

print("classical result:")
classical_form(N, K, c, lat, long)
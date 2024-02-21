import streamlit as st
import pandas as pd
import numpy as np
from to_cqm import binarycqm_form
from to_bqm import binarybqm_form
from classical_2 import classical_form
from data_preparation import DataCalculator

def main():
    session_state = st.session_state
    if "last_results" not in session_state:
        session_state.last_results = []
    # results = []
    st.title('Vehicle routing problem')
    st.write("Enter a value for N below:")

    # Create an input field for the user to enter N
    N = st.number_input(label="Enter N:", min_value=0, max_value=490, step=1)

    # Display the value of N
    # st.write("You entered:", N)
    # capacity_constraint = st.number_input("Enter Capacity Constraint:", min_value=0, step=1)
    # equality_constraint = st.number_input("Enter Equality Constraint:", min_value=0, step=1)
    st.subheader("Choose a Function")
    data = DataCalculator("data.xlsx", "locations", "vehicle_data_size", N)
    c = data.get_cost_matrix(k=400, max_row=N, max_column=N)
    t = data.get_time_matrix(max_row=N, max_column=N)
    Q = data.get_max_capacity()
    q = data.get_order_size()
    N_0 = len(c[0])-1  # Number of locations (excluding depot)
    A = 1.0  # Coefficient for distance costs keep it 1
    B = 0  # Coefficient for location visitation -shud be kept high
    C = 0  # Coefficient for time step visitation -shud be kept high
    D = 0# Coefficient for order delivery
    E = 0.0  # Coefficient for total travel time-less than b and c
    T_bound = 10000.0  # Upper bound constraint on total travel time
    K = len(c) # number of vehicles
    lat, long =data.get_lat_long()
    # Create a selectbox for the user to choose the function
    chosen_function = st.selectbox("Select a function:", ("cqm", "bqm", "classical"))
    if chosen_function == "bqm":
        capacity_constraint = st.number_input("Enter Capacity Constraint:", min_value=0, step=1)
        equality_constraint = st.number_input("Enter Equality Constraint:", min_value=0, step=1)
        num_reads = st.number_input("Enter num_reads:", min_value=1, max_value=4000, step=1)
        B=C=equality_constraint
        D=capacity_constraint
    enter_button = st.button("Enter")
    if enter_button:
        st.subheader("Results")
        if chosen_function == "cqm":
            # capacity_constraint = st.number_input("Enter Capacity Constraint:", min_value=0, step=1)
            # equality_constraint = st.number_input("Enter Equality Constraint:", min_value=0, step=1)
            # B=C=equality_constraint
            # D=capacity_constraint
            st.write(f"Running on CQM backend for {N_0} locations and {K} vehicles")
            execution_time, cost = binarycqm_form(N_0,K, c, t, Q, q, A, B, C, D, E, T_bound,5,lat, long)
            session_state.last_results.append(
            {
                "vehicle_count": K,
                "cost": cost,
                "execution_time": execution_time,
                "backend": "CQM",
                "customer_count": N_0
            }
        )
        elif chosen_function == "bqm":
            st.write(f"Running on BQM backend for {N_0} locations and {K} vehicles")
            execution_time, cost = binarybqm_form(N_0,K, c, t, Q, q, A, B, C, D, E, T_bound,lat,long,num_reads=num_reads)
            session_state.last_results.append(
            {
                "vehicle_count": K,
                "cost": cost,
                "execution_time": execution_time,
                "backend": "BQM",
                "customer_count": N_0
            }
        )
        elif chosen_function == "classical":
            st.write(f"Running on Classical backend for {N_0} locations and {K} vehicles")
            execution_time, cost = classical_form(N, K, c, lat, long)
            session_state.last_results.append(
            {
                "vehicle_count": K,
                "cost": cost,
                "execution_time": execution_time,
                "backend": "Classical",
                "customer_count": N_0
            }
        )
    # session_state.last_results = results
    
    # Display the last 3 run results
    # st.write(f"{len(session_state.last_results)}")
    # st.write(session_state.last_results)
    last_runs = st.button("Show last runs")
    if last_runs:
        # Store only the last 3 results
        if len(session_state.last_results)>=3:
            st.subheader("Last results:")
            # session_state.last_results = session_state.last_results[-3:]
        else:
            if len(session_state.last_results)==1:
                st.subheader("last result:")
            if len(session_state.last_results)==2:
                st.subheader("last 2 results:")
            # session_state.last_results = session_state.last_results

        for result in session_state.last_results:
            st.write(f"Backend_name: {result['backend']}")
            st.write(f"Customer count: {result['customer_count']}")
            st.write(f"Vehicle Count: {result['vehicle_count']}")
            st.write(f"Cost: {result['cost']}")
            st.write(f"Execution Time: {result['execution_time']} ms")
            # st.write("Vehicle Travel Times:")
            # for k, travel_time in enumerate(result["vehicle_travel_times"], start=1):
            #     st.write(f"Vehicle {k}: {travel_time}")
            st.write("---")  # Add a separator between results

if __name__ == "__main__":
    main()


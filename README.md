# VRP

## Introduction

This repository contains code for solving the Vehicle Routing Problem (VRP) using D-Wave quantum annealers. The VRP is a classic optimization problem in which the goal is to find the most optimal route for a fleet of vehicles to visit a set of locations(customers).

## Code

The code in this repository uses three different solvers to solve the VRP:

-   A classical solver based on LP method.
-   A quantum solver based on a binary quadratic model (BQM).
-   A quantum solver based on a constrained quadratic model (CQM).

The trials.py file in this repository initializes the VRP parameters using the data_preparation file.
### App

This is a Streamlit web application that allows users to input the number of locations and choose the backend for computation: BQM (Quantum Binary Quadratic Model), CQM (Classical Quadratic Model), or Classical. The app provides results in both written and graphical formats.

#### Usage

1. Clone the repository:

```bash
git clone https://gitlab.com/shivaleerks12/VRP-Dwave.git
cd VRP-Dwave
```
2. To run the app
```bash
streamlit run app.py
```


### Data Preparation

This class is used to calculate distance, time, and cost matrices based on location and vehicle data. The class takes two arguments: the file path to the Excel sheet containing location and vehicle data, and the names of the sheets in the Excel file containing location and vehicle data.

The class has the following methods:

-   calculate_distance(): Calculates the distance between two coordinates.
-   calculate_distance_matrix(): Calculates the distance matrix between all pairs of locations.
-   calculate_time_matrix(): Calculates the time matrix based on the distance matrix and speed matrix.
-   calculate_cost_matrix(): Calculates the cost matrix based on the distance matrix, fuel cost, and constant factor(k).
-   get_cost_matrix(): Returns a list of cost matrices for each vehicle type based on the number of vehicles.
-   get_time_matrix(): Returns a list of time matrices for each vehicle type based on the number of vehicles.
-   get_order_size(): Returns the list of order sizes for each location.
-   get_max_capacity(): Returns the list of maximum load capacities for each vehicle type.

## Running the Code

To run the code in this repository, you will need to have the following installed:

-   D-Wave Ocean software.
-   Python 3.

Once you have installed the required software, you can run the code by typing the following command in a terminal:

`python3 python trials.py`

This will run the code for a number of different problem instances and print the results to the console.

### Table of logs

| N   | K   | classical route: V1                    | V2                                                | classical time | classical cost | CQM cost | BQM cost                  | vehicle routes-v1                                                  | v2                                                                                       | constraint satisfied | time      |
| --- | --- | -------------------------------------- | ------------------------------------------------- | -------------- | -------------- | -------- | ------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | -------------------- | --------- |
| 4   | 2   | Vehicle_1 path: []                     | 0 ->2 -> 4 -> 1 -> 3 -> 0                         | 0.080123901    | 112.7471745    | 153.757  | 112.747 (D= 10000, E=100) | Vehicle_1 path: []                                                 | Vehicle 2 : 0 -> 2 -> 4 -> 3 -> 1 -> 0                                                   | yes                  | 42.519 ms |
| 5   | 2   | Vehicle_1 path: []                     | 0 -> 2 -> 4 -> 5 -> 3 -> 1 -> 0                   | 0.1208         | 110.3649677    | 152.6    | 126.389 (D= 1000,E=100)   | Vehicle_1 path: []                                                 | Vehicle 2 : 0 -> 2 -> 3 -> 1 -> 4 -> 5 -> 0                                              | yes                  | 73.678 ms |
| 6   | 2   | Vehicle_1 path: []                     | 0 -> 2 -> 4 -> 5 -> 3 -> 1 -> 6 -> 0              | 0.248588133    | 107.23425      | 146.82   | 254.172                   | Vehicle_1 path: []                                                 | Vehicle_2 path: ['0 -> 1', '1 -> 3', '2 -> 6', '3 -> 5', '4 -> 2', '5 -> 4', '6 -> 0']   | yes                  | 31.822         |
| 7   | 2   | 0 -> 2 -> 4 -> 5 -> 0                  | 0 -> 7 -> 3 -> 1 -> 6 -> 0                        | 0.253687143    | 222.1871036    | 265.50   | 312.579                   | Vehicle_1 path: ['0 -> 3', '1 -> 5', '3 -> 1', '5 -> 0']           | Vehicle_2 path: ['0 -> 2', '2 -> 4', '4 -> 7', '6 -> 0', '7 -> 6']                       | yes                  | 15.927         |
| 8   | 2   | 0 -> 2 -> 8 -> 0                       | 0 -> 7 -> 4 -> 5 -> 3 -> 1 -> 6 -> 0              | 0.369401932    | 225.1299349    | 269.27   | -                         | Vehicle_1 path: ['0 -> 3', '1 -> 5', '3 -> 1', '5 -> 0']           | Vehicle_2 path: ['0 -> 2', '2 -> 8', '4 -> 7', '6 -> 0', '7 -> 6', '8 -> 4']             | yes                  | 31.849         |
| 9   | 2   | 0 -> 2 -> 8 -> 0                       | 0 -> 7 -> 9 -> 4 -> 5 -> 3 -> 1 -> 6 -> 0         | 0.54723405     | 228.7424815    | 264.382  | -                         | Vehicle_1 path: ['0 -> 1', '1 -> 3', '3 -> 5', '5 -> 0']           | Vehicle_2 path: ['0 -> 2', '2 -> 9', '4 -> 8', '6 -> 0', '7 -> 6', '8 -> 7', '9 -> 4']   | yes                  | 31.852         |
| 10  | 2   | Vehicle 1 : 0 -> 2 -> 3 -> 1 -> 6 -> 0 | Vehicle 2 : 0 -> 7 -> 4 -> 5 -> 8 -> 9 -> 10 -> 0 | 0.899271       | 240.227        | 245.618  | -                         | Vehicle_1 path: ['0 -> 6', '4 -> 5', '5 -> 8', '6 -> 4', '8 -> 0'] | Vehicle_2 path: ['0 -> 7', '1 -> 3', '2 -> 10', '3 -> 9', '7 -> 1', '9 -> 2', '10 -> 0'] | yes                  | 31.856         |

### Demo

link to demo: https://youtu.be/0iEFFggV8A8

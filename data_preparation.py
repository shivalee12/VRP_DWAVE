import pandas as pd
import geopy
from geopy.distance import geodesic
import string

class DataCalculator:
    """
    A class for calculating distance, time, and cost matrices based on location and vehicle data.

    Args:
        file_path (str): The file path to the Excel sheet containing location and vehicle data.
        locations (str): The name of the sheet in the Excel file containing location data.
        vehicles (str): The name of the sheet in the Excel file containing vehicle data.

    Attributes:
        file_path (str): The file path to the Excel sheet containing location and vehicle data.
        locations (list): A list of location names.
        latitudes (list): A list of latitude values corresponding to the locations.
        longitudes (list): A list of longitude values corresponding to the locations.
        order (list): A list of order sizes for each location.
        vehicle_types (list): A list of vehicle types.
        cost (list): A list of costs for each vehicle type.
        load_capacity (list): A list of load capacities (in volume) for each vehicle type.
        number_of_vehicles (list): A list of the number of vehicles available for each vehicle type.
        speed_matrix (list): A list of sheet names containing speed matrices for each vehicle type.

    Methods:
        calculate_distance(coord1, coord2):
            Calculates the distance between two coordinates.
        
        calculate_distance_matrix():
            Calculates the distance matrix between all pairs of locations.
        
        calculate_time_matrix(speed_matrix):
            Calculates the time matrix based on the distance matrix and speed matrix.
        
        calculate_cost_matrix(speed_matrix, fuel_cost=60, k=1):
            Calculates the cost matrix based on the distance matrix, fuel cost, and constant factor(k).
        
        get_cost_matrix(k=1,fuel_cost=60, max_row=1, max_column=1):
            Returns a list of cost matrices for each vehicle type based on the number of vehicles.
        
        get_time_matrix(max_row=1, max_column=1):
            Returns a list of time matrices for each vehicle type based on the number of vehicles.
        
        get_order_size():
            Returns the list of order sizes for each location.
        
        get_max_capacity():
            Returns the list of maximum load capacities for each vehicle type.
    """

    def __init__(self, file_path, locations, vehicles,N):
        self.file_path = file_path
        self.N = N
        self.locations, self.latitudes, self.longitudes, self.order = self._read_location_data(file_path, locations)
        self.vehicle_types, self.cost, self.load_capacity, self.number_of_vehicles, self.speed = self._read_vehicle_data(file_path, vehicles)

    def _read_location_data(self, file_path, sheet_name):
        # Read the Excel sheet
        df = pd.read_excel(file_path, sheet_name = sheet_name)

        # Extract location, latitude, and longitude columns
        locations = df["Location Name"].tolist()[:self.N]
        latitudes = df["Latitude"].tolist()[:self.N]
        longitudes = df["Longitude"].tolist()[:self.N]
        order_sizes = df["order sizes"].to_list()[:self.N]

        # Validate latitude and longitude values
        valid_latitudes = []
        valid_longitudes = []
        for order in  order_sizes:
            if pd.isna(order):
                raise ValueError("Order Value is NaN")
        for latitude, longitude in zip(latitudes, longitudes):
            if pd.isna(latitude) or pd.isna(longitude):
                raise ValueError("Latitude or longitude value is NaN.")

            if not (-90 <= latitude <= 90):
                raise ValueError("Invalid latitude value: {}".format(latitude))

            if not (-180 <= longitude <= 180):
                raise ValueError("Invalid longitude value: {}".format(longitude))

            valid_latitudes.append(latitude)
            valid_longitudes.append(longitude)

        return locations, valid_latitudes, valid_longitudes, order_sizes
    
    def _read_vehicle_data(self,file_path, sheet_name):
        df = pd.read_excel(file_path, sheet_name = sheet_name)
        
        required_columns = ['vehicle type', 'cost', 'load capacity in volume', 'no of vehicles', 'speed matrix']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError("Missing columns in the vehicle data: {}".format(", ".join(missing_columns)))
        vehicle_types = df["vehicle type"].tolist()
        cost = df["cost"].tolist()
        load_capacity = df["load capacity in volume"].tolist()
        number_of_vehicles = df["no of vehicles"].tolist()
        speed = df["speed matrix"].tolist()

        return vehicle_types, cost, load_capacity, number_of_vehicles, speed
    
    def calculate_distance(self, coord1, coord2):
        return geodesic(coord1, coord2).kilometers
    
    def calculate_distance_matrix(self):
        dist_matrix = pd.DataFrame(index=self.locations, columns=self.locations)

        for i in range(len(self.locations)):
            for j in range(i, len(self.locations)):
                place1 = (self.latitudes[i], self.longitudes[i])
                place2 = (self.latitudes[j], self.longitudes[j])
                distance1 = self.calculate_distance(place1, place2)
                distance2 = self.calculate_distance(place2, place1)
                dist_matrix.iloc[i, j] = distance1
                dist_matrix.iloc[j, i] = distance2

        return dist_matrix
    
    def calculate_time_matrix(self, speed_matrix):
        distance_matrix = self.calculate_distance_matrix()
        time_matrix = pd.DataFrame(index=self.locations, columns=self.locations)

        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                if speed_matrix[i][j] == 0:
                    time_matrix.iloc[i, j] = 0  # Set cost as 0 when speed is 0
                else:
                    time_matrix.iloc[i, j] = distance_matrix.iloc[i, j] / speed_matrix[i][j]
        return time_matrix.values.tolist()
    
    def calculate_cost_matrix(self, speed_matrix, fuel_cost, k):
        distance_matrix = self.calculate_distance_matrix()
        cost_matrix = pd.DataFrame(index=self.locations, columns=self.locations)

        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                if speed_matrix[i][j] == 0:
                    cost_matrix.iloc[i, j] = 0  # Set cost as 0 when speed is 0
                else:
                    cost_matrix.iloc[i, j] = (speed_matrix[i][j] * fuel_cost * distance_matrix.iloc[i, j]) / k
        return cost_matrix.values.tolist()

    def column_number_to_letter(self,column_number):
        """
        Convert column number to corresponding Excel column letter.
        """
        letters = string.ascii_uppercase
        result = []
        while column_number:
            column_number, remainder = divmod(column_number - 1, 26)
            result.append(letters[remainder])
        return ''.join(reversed(result))

    def get_cost_matrix(self,k=1,fuel_cost=60, max_row=1, max_column=1):
        cost_matrices = []
        # print(self.number_of_vehicles)
        max_column_letter = self.column_number_to_letter(max_column)
        num_vehicles =  [round(num) for num in self.number_of_vehicles]
        for i in range(len(self.speed)):
            speed_matrix_file = self.speed[i]
            speed_matrix = pd.read_excel(self.file_path, sheet_name=speed_matrix_file, header=None, nrows=max_row, usecols=lambda x: x < max_column).values
            # speed_matrix = pd.read_excel(self.file_path, sheet_name = speed_matrix_file, header=None).values
            cost_matrix = self.calculate_cost_matrix(speed_matrix,fuel_cost, k)
            
            # num_vehicles = self.number_of_vehicles[i]
            num_v = num_vehicles[i]
            # print(num_vehicles)
            cost_matrices.extend([cost_matrix] * num_v)
        return cost_matrices
    
    def get_time_matrix(self, max_row=1, max_column=1):
        time_matrices = []
        max_column_letter = self.column_number_to_letter(max_column)
        num_vehicles =  [round(num) for num in self.number_of_vehicles]
        for i in range(len(self.speed)):
            speed_matrix_file = self.speed[i] 
            speed_matrix = pd.read_excel(self.file_path, sheet_name=speed_matrix_file, header=None, nrows=max_row, usecols=lambda x: x < max_column).values
            # speed_matrix = pd.read_excel(self.file_path, sheet_name = speed_matrix_file, header=None).values
            time_matrix = self.calculate_time_matrix(speed_matrix)
            num_v = num_vehicles[i]
            

            time_matrices.extend([time_matrix] * num_v)
        return time_matrices
    
    def get_order_size(self):
        return self.order[1:]
    
    def get_max_capacity(self):
        Q=[]
        for i in range(len(self.load_capacity)):
            num_vehicles = self.number_of_vehicles[i]
            Q.extend([self.load_capacity[i]] * num_vehicles)
        return Q
    
    def get_lat_long(self):
        return self.latitudes, self.longitudes



import os
import sys
import time
import random
import copy
import math
import heapq
import itertools
from datetime import datetime

def read_file_into_string(input_file, ord_range):
    the_file = open(input_file, 'r') 
    current_char = the_file.read(1) 
    file_string = ""
    length = len(ord_range)
    while current_char != "":
        i = 0
        while i < length:
            if ord(current_char) >= ord_range[i][0] and ord(current_char) <= ord_range[i][1]:
                file_string = file_string + current_char
                i = length
            else:
                i = i + 1
        current_char = the_file.read(1)
    the_file.close()
    return file_string

def remove_all_spaces(the_string):
    length = len(the_string)
    new_string = ""
    for i in range(length):
        if the_string[i] != " ":
            new_string = new_string + the_string[i]
    return new_string

def integerize(the_string):
    length = len(the_string)
    stripped_string = "0"
    for i in range(0, length):
        if ord(the_string[i]) >= 48 and ord(the_string[i]) <= 57:
            stripped_string = stripped_string + the_string[i]
    resulting_int = int(stripped_string)
    return resulting_int

def convert_to_list_of_int(the_string):
    list_of_integers = []
    location = 0
    finished = False
    while finished == False:
        found_comma = the_string.find(',', location)
        if found_comma == -1:
            finished = True
        else:
            list_of_integers.append(integerize(the_string[location:found_comma]))
            location = found_comma + 1
            if the_string[location:location + 5] == "NOTE=":
                finished = True
    return list_of_integers

def build_distance_matrix(num_cities, distances, city_format):
    dist_matrix = []
    i = 0
    if city_format == "full":
        for j in range(num_cities):
            row = []
            for k in range(0, num_cities):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    elif city_format == "upper_tri":
        for j in range(0, num_cities):
            row = []
            for k in range(j):
                row.append(0)
            for k in range(num_cities - j):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    else:
        for j in range(0, num_cities):
            row = []
            for k in range(j + 1):
                row.append(0)
            for k in range(0, num_cities - (j + 1)):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    if city_format == "upper_tri" or city_format == "strict_upper_tri":
        for i in range(0, num_cities):
            for j in range(0, num_cities):
                if i > j:
                    dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

input_file = "...txt"

if len(sys.argv) > 1:
    input_file = sys.argv[1]

path_for_city_files = os.path.join("..", "..")

path_to_input_file = os.path.join(path_for_city_files, input_file)
if os.path.isfile(path_to_input_file):
    ord_range = [[32, 126]]
    file_string = read_file_into_string(path_to_input_file, ord_range)
    file_string = remove_all_spaces(file_string)
    print("I have found and read the input file " + input_file + ":")
else:
    print("*** error: The city file " + input_file + " does not exist in the city-file folder.")
    sys.exit()

location = file_string.find("SIZE=")
if location == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()
    
comma = file_string.find(",", location)
if comma == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()
    
num_cities_as_string = file_string[location + 5:comma]
num_cities = integerize(num_cities_as_string)
print("   the number of cities is stored in 'num_cities' and is " + str(num_cities))

comma = comma + 1
stripped_file_string = file_string[comma:]
distances = convert_to_list_of_int(stripped_file_string)

counted_distances = len(distances)
if counted_distances == num_cities * num_cities:
    city_format = "full"
elif counted_distances == (num_cities * (num_cities + 1))/2:
    city_format = "upper_tri"
elif counted_distances == (num_cities * (num_cities - 1))/2:
    city_format = "strict_upper_tri"
else:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()

dist_matrix = build_distance_matrix(num_cities, distances, city_format)
print("   the distance matrix 'dist_matrix' has been built.")

start_time = time.time()

class BasicPSOTSP:
    """
        Standard PSO initialization as per lecture pseudocode.

        Args:
            dist_matrix (list[list[int]]): Distance matrix for TSP.
            num_cities (int): Number of cities.
            num_parts (int): Number of particles.
            max_it (int): Max iterations.
            theta (float): Inertia coefficient.
            alpha (float): Cognitive component coefficient.
            beta (float): Social component coefficient.
            time_limit (int): Time limit for optimization in seconds.
        """
    def __init__(self, dist_matrix, num_parts=30, max_it=1000, theta=0.5, alpha=1.0, beta=1.0, time_limit=60):
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.num_particles = num_parts
        self.max_iter = max_it
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.time_limit = time_limit

    def tour_length(self, tour):
        return sum(self.dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)) + self.dist_matrix[tour[-1]][tour[0]]

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            tour = list(range(1, self.num_cities))
            random.shuffle(tour)
            tour = [0] + tour
            particles.append({
                'position': tour,
                'velocity': [],
                'pbest': tour[:],
                'pbest_cost': self.tour_length(tour)
            })
        return particles

    def subtract_tours(self, a, b):
        a = a[:]
        swaps = []
        for i in range(1, len(a)):
            if a[i] != b[i]:
                j = a.index(b[i])
                while j > i:
                    swaps.append((j-1, j))
                    a[j], a[j-1] = a[j-1], a[j]
                    j -= 1
        return swaps

    def apply_velocity(self, tour, velocity):
        new_tour = tour[:]
        for i, j in velocity:
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def multiply_velocity(self, velocity, coef):
        k = int(coef * len(velocity))
        return velocity[:k]

    def add_velocities(self, *velocities):
        result = []
        for v in velocities:
            result.extend(v)
        return result

    def optimize(self):
        start_time = time.time()
        swarm = self.initialize_particles()
        global_best = min(swarm, key=lambda p: p['pbest_cost'])

        for _ in range(self.max_iter):
            if time.time() - start_time > self.time_limit:
                break

            for particle in swarm:
                r1, r2 = random.random(), random.random()

                v_old = self.multiply_velocity(particle['velocity'], self.theta)
                v_cognitive = self.multiply_velocity(self.subtract_tours(particle['position'], particle['pbest']), self.alpha * r1)
                v_social = self.multiply_velocity(self.subtract_tours(particle['position'], global_best['pbest']), self.beta * r2)

                new_velocity = self.add_velocities(v_old, v_cognitive, v_social)
                new_position = self.apply_velocity(particle['position'], new_velocity)
                new_cost = self.tour_length(new_position)

                if new_cost < particle['pbest_cost']:
                    particle['pbest'] = new_position
                    particle['pbest_cost'] = new_cost

                if new_cost < global_best['pbest_cost']:
                    global_best = {
                        'position': new_position,
                        'pbest': new_position,
                        'pbest_cost': new_cost
                    }

                particle['position'] = new_position
                particle['velocity'] = new_velocity

        return global_best['pbest'], global_best['pbest_cost']
    
# Initialize the PSO solver
solver = BasicPSOTSP(dist_matrix, num_parts=30, max_it=1000, theta=0.5, alpha=1.0, beta=1.0)

# Run the optimization
tour, tour_length = solver.optimize()

end_time = time.time()
elapsed_time = round(end_time - start_time, 1)

added_note = "\nRUN-TIME = " + str(elapsed_time) + " seconds.\n"
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
added_note = added_note + "DATE-TIME = " + dt_string + ".\n"

flag = "good"
length = len(tour)
for i in range(0, length):
    if isinstance(tour[i], int) == False:
        flag = "bad"
    else:
        tour[i] = int(tour[i])
if flag == "bad":
    print("*** error: Your tour contains non-integer values.")
    sys.exit()
if isinstance(tour_length, int) == False:
    print("*** error: The tour-length is a non-integer value.")
    sys.exit()
tour_length = int(tour_length)
if len(tour) != num_cities:
    print("*** error: The tour does not consist of " + str(num_cities) + " cities as there are, in fact, " + str(len(tour)) + ".")
    sys.exit()
flag = "good"
for i in range(0, num_cities):
    if not i in tour:
        flag = "bad"
if flag == "bad":
    print("*** error: Your tour has illegal or repeated city names.")
    sys.exit()
check_tour_length = 0
for i in range(0, num_cities - 1):
    check_tour_length = check_tour_length + dist_matrix[tour[i]][tour[i + 1]]
check_tour_length = check_tour_length + dist_matrix[tour[num_cities - 1]][tour[0]]
if tour_length != check_tour_length:
    print("*** error: The length of your tour is not " + str(tour_length) + "; it is actually " + str(check_tour_length) + ".")
    sys.exit()
print("You have successfully built a tour of length " + str(tour_length) + "!")
len_dt_string = len(dt_string)
date_time_number = 0
for i in range(0, len_dt_string):
    date_time_number = date_time_number + ord(dt_string[i])
tour_diff = abs(tour[0] - tour[num_cities - 1])
for i in range(0, num_cities - 1):
    tour_diff = tour_diff + abs(tour[i + 1] - tour[i])
local_time = time.asctime(time.localtime(time.time()))
output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
output_file_time = output_file_time.replace(" ", "0")
script_name = os.path.basename(sys.argv[0])
if len(sys.argv) > 2:
    output_file_time = sys.argv[2]
output_file_name = script_name[0:len(script_name) - 3] + "_" + input_file[0:len(input_file) - 4] + "_" + output_file_time + ".txt"

f = open(output_file_name,'w')
f.write(str(tour[0]))
for i in range(1,num_cities):
    f.write(",{0}".format(tour[i]))
f.close()
print("I have successfully written your tour to the tour file:\n   " + output_file_name + ".")


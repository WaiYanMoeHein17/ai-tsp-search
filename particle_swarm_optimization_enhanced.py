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

path_for_city_files = os.path.join("..", "city-files")

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

class PSOTSP:
    def __init__(self, dist_matrix, num_parts=30, max_it=1000, theta=0.5, alpha=1.0, beta=1.0, stagnation_limit=30, elitism_size=2, early_stop_patience=100, time_limit=60):
        """
        Initializes the PSOTSP class with all parameters needed for the algorithm.

        Args:
            dist_matrix (list[list[int]]): Symmetric distance matrix representing the TSP instance.
            num_parts (int): Number of particles in the swarm.
            max_it (int): Maximum number of iterations.
            theta (float): Inertia weight.
            alpha (float): Personal best influence weight.
            beta (float): Global best influence weight.
            stagnation_limit (int): Max iterations before a particle is reset.
            elitism_size (int): Number of top solutions retained as elites.
            early_stop_patience (int): Number of iterations without improvement before early stop.
            time_limit (int): Max allowed runtime (in seconds) for the optimizer.

        Enhancements:
            - Adaptive parameters (theta, alpha, beta)
            - Time-limited execution
            - Elitism tracking
        """
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.num_particles = num_parts
        self.max_iter = max_it
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.initial_theta = theta
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.stagnation_limit = stagnation_limit
        self.elitism_size = elitism_size
        self.early_stop_patience = early_stop_patience
        self.time_limit = time_limit

    def _create_particle(self, tour):
        return {
            'position': tour,
            'velocity': [],
            'pbest': tour,
            'pbest_cost': self.tour_length(tour),
            'no_improve': 0
        }

    def initialize_particles(self):
        """
        Initializes the particle swarm using three heuristics:
        - Nearest Neighbour (NN)
        - Random Greedy (RG)
        - Pure Random

        Returns:
            list: Initialized particle list with diverse starting tours.

        Enhancement:
            Smart initialization for improved exploration and swarm diversity.
        """
        particles = []
        third = self.num_particles // 3

        for _ in range(third):
            tour = self.nearest_neighbour_tour()
            particles.append(self._create_particle(tour))

        for _ in range(third):
            tour = self.random_greedy_tour()
            particles.append(self._create_particle(tour))

        for _ in range(third):
            tour = list(range(1, self.num_cities))
            random.shuffle(tour)
            tour = [0] + tour
            particles.append(self._create_particle(tour))

        return particles

    def two_opt(self, tour, max_swaps=10):
        """
        Applies a limited 2-opt local search to remove crossing edges and improve tour quality.

        Args:
            tour (list[int]): The tour to be optimized.
            max_swaps (int): Max allowed improving swaps.

        Returns:
            list[int]: Locally refined tour.

        Enhancement:
            Hybrid local search applied selectively to promising solutions.
        """
        best = tour
        swaps_done = 0
        improved = True
        while improved and swaps_done < max_swaps:
            if start_time and time.time() - start_time > self.time_limit:
                break   
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    if j - i == 1: continue
                    new_tour = best[:]
                    new_tour[i:j+1] = reversed(new_tour[i:j+1])
                    if self.tour_length(new_tour) < self.tour_length(best):
                        best = new_tour
                        improved = True
                        swaps_done += 1
                        break
                if improved: break
        return best

    def crossover_elimination(self, tour):
        """
        Performs one-pass crossover elimination on a tour to remove intersecting edges.
        This is a lightweight method similar to 2-opt but less computationally expensive.

        Args:
            tour (list[int]): Tour to apply crossover cleaning.

        Returns:
            list[int]: Refined tour with crossing edges removed.

        Enhancement:
            Global best refinement through edge-structure improvement.
        """
        # One pass crossover cleaner (like 2-opt but only one pass, only first found crossing)
        for i in range(len(tour) - 2):
            for j in range(i + 2, len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[j + 1]
                if (self.dist_matrix[a][b] + self.dist_matrix[c][d] >
                    self.dist_matrix[a][c] + self.dist_matrix[b][d]):
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    return tour  # Stop after first fix
        return tour

    def tour_length(self, tour):
        """
        Computes the total length of a tour.

        Args:
            tour (list[int]): A list of city indices forming a complete tour.

        Returns:
            int: Total length of the tour based on the distance matrix.
        """
        return sum(self.dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)) + self.dist_matrix[tour[-1]][tour[0]]

    def nearest_neighbour_tour(self):
        """
        Constructs a tour using the Nearest Neighbour heuristic.

        Returns:
            list[int]: A complete tour starting from city 0 and extending by always choosing the nearest unvisited city.
        """
        unvisited = list(range(1, self.num_cities))
        tour = [0]
        while unvisited:
            last_city = tour[-1]
            next_city = min(unvisited, key=lambda x: self.dist_matrix[last_city][x])
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def random_greedy_tour(self, k=3):
        """
        Constructs a tour using a Random Greedy heuristic where the next city is chosen randomly among the k nearest unvisited cities.

        Args:
            k (int): Number of nearest cities to consider for randomized selection. Defaults to 3.

        Returns:
            list[int]: A complete randomized greedy tour starting from city 0.
        """
        unvisited = list(range(1, self.num_cities))
        tour = [0]
        while unvisited:
            last = tour[-1]
            candidates = sorted(unvisited, key=lambda x: self.dist_matrix[last][x])[:k]
            next_city = random.choice(candidates)
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def subtract_tours(self, a, b):
        """
        Computes a sequence of swaps to transform tour a into tour b using direct swaps (non-BSS).

        Args:
            a (list[int]): Source tour.
            b (list[int]): Target tour.

        Returns:
            list[tuple]: List of (i, j) swap operations.
        """
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

    def subtract_tours_bss(self, a, b):
        
        a_target = a[:]
        b_working = b[:]
        swaps = []
        for i in range(1, len(a_target)):
            if b_working[i] != a_target[i]:
                j = b_working.index(a_target[i])
                while j > i:
                    swaps.append((j - 1, j))
                    b_working[j], b_working[j - 1] = b_working[j - 1], b_working[j]
                    j -= 1
        return swaps

    def apply_velocity(self, tour, velocity):
        """
        Applies a sequence of swaps (velocity) to a tour.

        Args:
            tour (list[int]): Current tour.
            velocity (list[tuple]): List of swap operations.

        Returns:
            list[int]: Modified tour after swaps.
        """
        new_tour = tour[:]
        for i, j in velocity:
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def probabilistic_scale(self, velocity, probability):
        """
        Randomly selects swaps from a velocity sequence with a given probability.

        Args:
            velocity (list[tuple]): Swap operations.
            probability (float): Probability of retaining each swap.

        Returns:
            list[tuple]: Filtered list of swap operations.
        """
        return [swap for swap in velocity if random.random() < probability]

    def multiply_velocity(self, velocity, coef):
        """
        Truncates a velocity sequence to a fraction of its original length.

        Args:
            velocity (list[tuple]): Swap operations.
            coef (float): Fraction to retain.

        Returns:
            list[tuple]: Truncated velocity.
        """
        k = int(coef * len(velocity))
        return velocity[:k]

    def normalize_velocity(self, original, new):
        """
        Computes normalized velocity (as BSS) needed to transform original tour into new tour.

        Args:
            original (list[int]): Starting tour.
            new (list[int]): Target tour.

        Returns:
            list[tuple]: Swap operations representing the velocity.
        """
        return self.subtract_tours_bss(new, original)

    def add_velocities(self, *velocities):
        """
        Concatenates multiple velocity sequences into one.

        Args:
            *velocities (list[tuple]): Variable number of velocity lists.

        Returns:
            list[tuple]: Combined velocity.
        """
        result = []
        for v in velocities:
            result.extend(v)
        return result

    def optimize(self):
        """
        Executes the PSO optimization loop.

        Returns:
            tuple: Best tour (list[int]) and its cost (int).

        Core Algorithm:
            - Initializes diverse swarm using heuristics.
            - Updates particles using adaptive velocity blending (probabilistic scaling).
            - Applies BSS-based swap computation for efficient tour updates.
            - Selectively refines new solutions using 2-opt.
            - Detects and resets stagnant particles.
            - Tracks and returns elite tours.
            - Applies early stopping based on improvement plateau.
            - Applies crossover elimination to global best each iteration.
            - Applies final 2-opt to elite set before output.

        Enhancements:
            - Probabilistic velocity scaling
            - Basic Swap Sequence (BSS)
            - Adaptive parameters (theta, alpha, beta)
            - Selective 2-opt hybridization
            - Elitism and final 2-opt polishing
            - One-pass crossover cleaning on global best
            - Stagnation handling and early stopping
            - Time-limited execution
        """
        swarm = self.initialize_particles()
        global_best = min(swarm, key=lambda x: x['pbest_cost'])
        elites = []
        no_global_improve = 0

        for t in range(self.max_iter):
            if time.time() - start_time > self.time_limit:
                break

            theta = max(0.4, self.initial_theta * (0.99**t))
            alpha = self.initial_alpha * (1 - t / self.max_iter)
            beta = self.initial_beta * (1 - t / self.max_iter)
            prev_best_cost = global_best['pbest_cost']

            # Apply one-pass crossover cleaner to global best tour
            global_best['pbest'] = self.crossover_elimination(global_best['pbest'])
            global_best['pbest_cost'] = self.tour_length(global_best['pbest'])

            for particle in swarm:
                if time.time() - start_time > self.time_limit:
                    break
                r1 = random.random()
                r2 = random.random()

                v_old = self.probabilistic_scale(particle['velocity'], self.theta)
                v_cognitive = self.probabilistic_scale(self.subtract_tours(particle['position'], particle['pbest']), self.alpha * r1)
                v_social = self.probabilistic_scale(self.subtract_tours(particle['position'], global_best['pbest']), self.beta * r2)

                combined_velocity = self.add_velocities(v_old, v_cognitive, v_social)
                new_position = self.apply_velocity(particle['position'], combined_velocity)
                if particle['no_improve'] < 5 and new_position != particle['position']:
                    new_position = self.two_opt(new_position)
                new_velocity = self.normalize_velocity(particle['position'], new_position)
                new_cost = self.tour_length(new_position)

                if new_cost < particle['pbest_cost']:
                    particle['pbest'] = new_position
                    particle['pbest_cost'] = new_cost
                    particle['no_improve'] = 0
                else:
                    particle['no_improve'] += 1

                if new_cost < global_best['pbest_cost']:
                    global_best = {
                        'position': new_position,
                        'pbest': new_position,
                        'pbest_cost': new_cost
                    }

                if particle['no_improve'] >= self.stagnation_limit:
                    particle['position'] = self.random_greedy_tour()
                    particle['pbest'] = particle['position'][:]
                    particle['pbest_cost'] = self.tour_length(particle['position'])
                    particle['velocity'] = []
                    particle['no_improve'] = 0
                else:
                    particle['position'] = new_position
                    particle['velocity'] = new_velocity

            if global_best['pbest_cost'] >= prev_best_cost:
                no_global_improve += 1
            else:
                no_global_improve = 0

            if no_global_improve >= self.early_stop_patience:
                break

            heapq.heappush(elites, (global_best['pbest_cost'], global_best['pbest']))
            elites = heapq.nsmallest(self.elitism_size, elites)

        # Final 2-opt polish on elite tours
        elites = [] 
        for _, t in elites:
            if time.time() - start_time > self.time_limit:
                break
            t = self.two_opt(t)
            cost = self.tour_length(t)
            elites.append((cost, t))
        
        if not elites:
            return global_best['pbest'], global_best['pbest_cost']
        elites = sorted(elites, key=lambda x: x[0])
        return elites[0][1], elites[0][0]

max_it = 600
num_parts = 50
num_runs = 1
best_tour = None
best_length = float('inf')

for _ in range(num_runs):
    solver = PSOTSP(dist_matrix, num_parts=num_parts, max_it=max_it, theta=0.4, alpha=1.4, beta=1.4, time_limit=55) # set time limit here
    candidate_tour, candidate_length = solver.optimize()
    if candidate_length < best_length:
        best_tour = candidate_tour
        best_length = candidate_length

tour = best_tour
tour_length = best_length

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


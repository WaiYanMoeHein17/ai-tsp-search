import os
import sys
import time
import random
import copy
import math
import heapq
import itertools
#from numba import njit
from functools import partial
from datetime import datetime
from multiprocessing import Pool 

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

# Enhanced ACO with Iteration-Best Pheromones, 3-opt, Restart Mechanism
max_it = 100 # maximum number of iterations
num_ants = num_cities # number of ants = number of cities
alpha = 1.0 # pheromone importance
beta = 5.0 # heuristic importance
base_rho = 0.3 # base pheromone evaporation rate
q = 100 # pheromone deposit constant 
elite_weight = 2.0 
random_set = 0.9 # probability of choosing the best neighbor
k = min(15, num_cities - 1) # number of nearest neighbors
bl_size = min(30, num_cities - 1) # backup list size
min_new_edges = 8 # minimum new edges to trigger checklist update
stagnation_limit = 20 # stagnation limit for pheromone restart
entropy_threshold = 0.05 # entropy threshold for pheromone restart
pop_size = 50 # number of tours to retain
tour_population = [] # list of best tours
T0 = 100.0
cooling_rate = 0.95 # cooling rate for TSP

tau_min = 0.1 # minimum pheromone level
tau_max = 10.0 # maximum pheromone level

# Initial Pheromone Matrix and Nearest Neighbors
pheromone = [[1.0 for _ in range(num_cities)] for _ in range(num_cities)]
nearest_neighbors = [sorted(range(num_cities), key=lambda j: dist_matrix[i][j])[:k] for i in range(num_cities)]
backup_lists = [sorted(range(num_cities), key=lambda j: dist_matrix[i][j])[k:k+bl_size] for i in range(num_cities)]

def distance(tour):
    """Calculate the total distance of a tour.
    This function computes the total distance of a given tour by summing the distances between consecutive cities in the tour.

    Args:
        tour : A list of integers representing the tour, where each integer is a city index.

    Returns:
        int : The total distance of the tour.
    """
    return sum(dist_matrix[tour[i]][tour[(i+1)%num_cities]] for i in range(num_cities))

def quick_gain(tour, i, j, k):
    """
    Estimates whether removing and reconnecting three edges in the tour
    would result in a shorter path length.

    Args:
        tour (list[int]): Current tour of city indices.
        i (int): First split index.
        j (int): Second split index.
        k (int): Third split index.

    Returns:
        bool: True if the new edge arrangement improves the tour length, else False.
    """
    a, b = tour[i - 1], tour[i]
    c, d = tour[j - 1], tour[j]
    e, f = tour[k - 1], tour[k % len(tour)]
    removed = dist_matrix[a][b] + dist_matrix[c][d] + dist_matrix[e][f]
    added = dist_matrix[a][c] + dist_matrix[b][e] + dist_matrix[d][f]
    return added < removed

# Enhancement: 3-opt local search
def three_opt(tour):
    """
    Performs 3-opt local search on the tour to improve its total distance.
    Tries all combinations of 3 edges and applies the best reconnection.

    Args:
        tour (list[int]): The input tour to optimize.

    Returns:
        list[int]: A new tour with improved path length (or the original if no improvements).
    """
    best = tour[:]
    best_dist = distance(best)
    improved = True
    while improved:
        improved = False
        for i, j, k in itertools.combinations(range(1, len(tour) - 1), 3):
            if time.time() - start_time > time_limit:
                print("[Terminated] Time limit reached (60 seconds). Returning best tour found.")
                return best
            if not quick_gain(best, i, j, k):
                continue
            a = best[:i] + best[i:j][::-1] + best[j:k][::-1] + best[k:]
            b = best[:i] + best[j:k] + best[i:j] + best[k:]
            c = best[:i] + best[j:k] + best[i:j][::-1] + best[k:]
            for candidate in [a, b, c]:
                candidate_dist = distance(candidate)
                if candidate_dist < best_dist:
                    best = list(map(int, candidate))
                    best_dist = candidate_dist
                    improved = True
                    break
            if improved:
                break
    return best

# Enhancement: 2-opt with a checklist of priority cities
def two_opt_with_checklist(tour, checklist):
    """
    Applies 2-opt optimization driven by a checklist of priority cities,
    then applies 3-opt for further improvement.

    Args:
        tour (list[int]): Current TSP tour.
        checklist (list[int]): Cities to prioritize for 2-opt swaps.

    Returns:
        list[int]: Optimized tour after local search.
    """
    improved = True
    best = tour[:]
    best_dist = distance(best)  # Cache current tour distance

    while improved and checklist:
        if time.time() - start_time > time_limit:
            print("[Terminated] Time limit reached (60 seconds). Returning best tour found.")
            return best
        improved = False
        a = checklist.pop(0)
        for b in nearest_neighbors[a]:
            if b not in best:
                continue
            i, j = best.index(a), best.index(b)
            if i > j:
                i, j = j, i
            if j - i < 2:
                continue
            new_tour = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
            new_dist = distance(new_tour)
            if new_dist < best_dist:
                best = new_tour
                best_dist = new_dist
                checklist += [best[i], best[j]]
                improved = True
                break
    return three_opt(best)

def compute_probabilities(current_city, visited):
    """
    Computes the probability of transitioning from the current city
    to each unvisited city, using pheromone and distance heuristics.

    Args:
        current_city (int): Index of the current city.
        visited (list[int]): List of already visited cities.

    Returns:
        list[float]: Normalized probability distribution over next cities.
    """
    visited_mask = [True] * num_cities
    for v in visited:
        visited_mask[v] = False

    virtual_pheromone = compute_edge_frequency_pheromones()
    pher = virtual_pheromone[current_city]
    dists = dist_matrix[current_city]

    probs = []
    for i in range(num_cities):
        if visited_mask[i]:
            tau = pher[i] ** alpha
            eta = (1.0 / dists[i]) ** beta if dists[i] > 0 else 0
            probs.append(tau * eta)
        else:
            probs.append(0)

    total = sum(probs)
    if total == 0:
        return [1.0 / num_cities] * num_cities
    return [p / total for p in probs]

def lin_kernighan_local_search(tour: list[int], max_depth: int = 3, time_limit: int = 60) -> list[int]:
    """
    Lightweight Lin-Kernighan-inspired local optimization with a time limit.

    Args:
        tour (list[int]): Tour to refine.
        max_depth (int): How deeply to try recursive improvements.
        time_limit (int): Maximum time allowed for the search in seconds.

    Returns:
        list[int]: Improved tour.
    """
    current = tour[:]
    current_length = distance(current)
    improved = True
    depth = 0

    while improved and depth < max_depth:
        if time.time() - start_time > time_limit:
            print("[Terminated] Time limit reached during Lin-Kernighan local search. Returning best tour found.")
            break
        improved = False
        for i in range(num_cities - 1):
            for j in range(i + 2, num_cities):
                if time.time() - start_time > time_limit:
                    print("[Terminated] Time limit reached during Lin-Kernighan local search. Returning best tour found.")
                    return current
                if j == num_cities - 1 and i == 0:
                    continue  # skip end-to-start swap
                new_tour = current[:i+1] + current[i+1:j+1][::-1] + current[j+1:]
                new_length = distance(new_tour)
                if new_length < current_length:
                    current = new_tour
                    current_length = new_length
                    improved = True
                    break
            if improved:
                break
        depth += 1

    return current


def construct_solution(source_solution):
    """
    Constructs one full solution per ant, optionally applies local search.

    Args:
        source_solution (list[int]): A guide tour used to encourage edge novelty.

    Returns:
        list[tuple[list[int], list[int]]]: List of tuples containing optimized tours and their checklists.
    """
    solutions = []
    starts = random.sample(range(num_cities), num_ants)
    for start in starts:
        if time.time() - start_time > time_limit:
            print("[Terminated] Time limit reached (60 seconds). Returning best tour found.")
            break
        visited = [int(start)]
        checklist = [int(start)]
        current_city = int(start)
        new_edges = 0
        while len(visited) < num_cities:
            probs = compute_probabilities(current_city, visited)
            q = random.random()
            if q < random_set:
                next_city = max(enumerate(probs), key=lambda x: x[1])[0]
            else:
                next_city = random.choices(range(num_cities), weights=probs)[0]
            if next_city in visited:
                candidates = list(set(range(num_cities)) - set(visited))
                next_city = int(random.choice(candidates))
            # Enhancement: Edge novelty check
            if (current_city, next_city) not in zip(source_solution, source_solution[1:] + [source_solution[0]]):
                new_edges += 1
                checklist.append(next_city)
            visited.append(next_city)
            current_city = next_city
            if new_edges >= min_new_edges:
                current_idx = source_solution.index(current_city)
                for offset in range(1, num_cities):
                    next_node = source_solution[(current_idx + offset) % num_cities]
                    if next_node not in visited:
                        visited.append(next_node)
                break
        if len(set(visited)) != num_cities:
            raise ValueError("Tour is invalid: contains duplicates or is incomplete.")

        # Enhancement: selective local search
        if num_cities <= 40 or random.random() < 0.3:
            optimized = two_opt_with_checklist(visited, checklist)
        else:
            optimized = visited

        solutions.append((optimized, checklist))
    return solutions

# Enhancement: pheromone update with global and iteration-best tours
def update_pheromones(solutions, best_tour, use_global=True, rho=0.3):
    """
    Updates the pheromone matrix based on either global or iteration-best tour.

    Args:
        solutions (list[tuple]): The list of constructed tours and checklists.
        best_tour (list[int]): The best-so-far tour for reinforcement.
        use_global (bool, optional): Whether to update using global best. Defaults to True.
        rho (float, optional): Pheromone evaporation rate. Defaults to 0.3.

    Returns:
        None
    """
    for i in range(num_cities):
        for j in range(num_cities):
            pheromone[i][j] *= (1 - rho)
            pheromone[i][j] = min(max(pheromone[i][j], tau_min), tau_max)

    target_tour = best_tour if use_global else min([s[0] for s in solutions], key=distance)
    for i in range(num_cities):
        a, b = target_tour[i], target_tour[(i + 1) % num_cities]
        delta = q / distance(target_tour)
        pheromone[a][b] += delta
        pheromone[b][a] += delta

    if not use_global:
        for s in solutions:
            tour = s[0]
            if tour != target_tour:
                for i in range(num_cities):
                    a, b = tour[i], tour[(i + 1) % num_cities]
                    pheromone[a][b] *= (1 - rho)
                    pheromone[b][a] *= (1 - rho)

def compute_edge_frequency_pheromones() -> list[list[float]]:
    """
    Builds a virtual pheromone matrix based on edge frequencies in the tour population.

    Returns:
        list[list[float]]: Symmetric pheromone matrix.
    """
    freq = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    for tour in tour_population:
        for i in range(num_cities):
            a, b = tour[i], tour[(i + 1) % num_cities]
            freq[a][b] += 1
            freq[b][a] += 1
    total = len(tour_population)
    if total == 0:
        return [[1.0 for _ in range(num_cities)] for _ in range(num_cities)]
    return [[(f / total + tau_min) for f in row] for row in freq]

# Enhancement: compute entropy of pheromone distribution
def compute_entropy():
    """
    Computes Shannon entropy over the pheromone matrix to measure search diversity.

    Returns:
        float: Entropy of pheromone distribution.
    """
    flat_pheromones = [value for row in pheromone for value in row]
    total = sum(flat_pheromones)
    probs = [p / total for p in flat_pheromones if total > 0]

    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy

best_tour = None
best_length = float('inf')
source_solution = random.sample(range(num_cities), num_cities)
stagnation_counter = 0
#length_history = []
time_limit = 55 # seconds
start_time = time.time()

for iteration in range(max_it):
    if time.time() - start_time > time_limit:
        print("[Terminated] Time limit reached (60 seconds). Returning best tour found.")
        break

    constructed = construct_solution(source_solution)
    iteration_best = min(constructed, key=lambda x: distance(x[0]))[0]
    iteration_best_length = distance(iteration_best)
    
    # Enhancement: Tour population management (PACO)
    if iteration_best not in tour_population:
        tour_population.append(iteration_best)
        if len(tour_population) > pop_size:
            worst = max(tour_population, key=distance)
            tour_population.remove(worst)

    #length_history.append(iteration_best_length)
    # Enhancement: Adaptive pheromone evaporation rate
    improvement = (best_length - iteration_best_length) / best_length if best_length > 0 else 1.0
    rho = base_rho * (1 - improvement)

    # Enhancement: Simulated Annealing
    T = T0 * (cooling_rate ** iteration) # cooling schedule
    delta = iteration_best_length - best_length
    if delta < 0:
        # better solution found
        best_length = iteration_best_length
        best_tour = iteration_best
        stagnation_counter = 0
    else:
        # worse solution found 
        acceptance_probability = math.exp(-delta / T) if T > 0 else 0
        if random.random() < acceptance_probability:
            best_length = iteration_best_length
            best_tour = iteration_best
            stagnation_counter = 0
        else: 
            stagnation_counter += 1
    
    """if iteration_best_length < best_length:
        best_length = iteration_best_length
        best_tour = iteration_best
        stagnation_counter = 0
    else:
        stagnation_counter += 1"""
    # Enhancement: Pheromone update with global and iteration-best tours
    update_pheromones(constructed, best_tour, use_global=(iteration % 2 == 0), rho=rho)
    source_solution = best_tour
    # Enhancement: Entropy-based pheromone restart
    entropy = compute_entropy()
    if entropy < entropy_threshold:
        pheromone = [[1.0 for _ in range(num_cities)] for _ in range(num_cities)]
        stagnation_counter = 0
        source_solution = random.sample(range(num_cities), num_cities)
        #print(f"[Restart] Entropy dropped below threshold ({entropy:.4f}). Restarting pheromone.")

tour = list(map(int, best_tour))
tour_length = best_length

# Enhancement: Lin-Kernighan Local Search
if 100 <= num_cities <= 600:
    start_refinement_time = time.time()
    #print("Applying LK-inspired final refinement...")
    improved_tour = lin_kernighan_local_search(tour, max_depth=5)
    if time.time() - start_refinement_time > time_limit:
        print("[Terminated] Time limit reached during final refinement. Returning best tour found.")
    else:
        improved_length = distance(improved_tour)
        if improved_length < tour_length:
            #print(f"Tour improved: {tour_length} → {improved_length}")
            tour = improved_tour
            tour_length = improved_length


"""plt.plot(length_history)
plt.xlabel("Iteration")
plt.ylabel("Tour Length")
plt.title("Tour Quality Over Time")
plt.grid(True)
plt.show()"""

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


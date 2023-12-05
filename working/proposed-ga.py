# Imports
import pandas as pd
import numpy as np
import random
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi as gd
import math

population_size = 50
max_generations = 100

teacher_type = 'FT' # FT or PT
max_hours_per_week = 30 # Max hours per week
unavailable_days = 'Monday,Tuesday,Wednesday'

# Read data
# df = pd.read_csv(r"./data/preprocessed_data.csv")
df = pd.read_csv(r"./data/preprocessed_data.csv")

try:
    df_sampled = df.sample(n=10000, replace=True) # Sample with replacement
except ValueError as e:
    print(f"Error: Not enough records to sample with replacement: {e}")
    exit()

# Preprocess data
"""
From course, timeslot, and classrom to
course, timeslot, classroom, day, time, start_time, end_time
"""

def split_timeslot(timeslot):
    parts = timeslot.split(' ')
    if len(parts) >= 2:
        return parts[0], ' '.join(parts[1:])
    return timeslot, ''

# Show all rows
pd.set_option('display.max_rows', None)

# Show all columns
pd.set_option('display.max_columns', None)

df_sampled['day'], df_sampled['time'] = zip(*df_sampled['timeslot'].apply(split_timeslot))
df_sampled['start_time'], df_sampled['end_time'] = zip(*df_sampled['time'].apply(lambda x: x.split('-') if x else ('', '')))

location_to_index_map = {classroom: index for index, classroom in enumerate(df['classroom'].unique(), start=1)}

# If FT, then remove all rows that includes PT
if teacher_type == 'FT':
    # Is PT in faculty column?
    is_pt = df_sampled['faculty'].str.contains('PT')

    # Remove all rows that includes PT
    df_sampled = df_sampled[~is_pt]

# If PT, then remove all rows that includes FT
if teacher_type == 'PT':
    # Is FT in faculty column?
    is_ft = df_sampled['faculty'].str.contains('FT')

    # Remove all rows that includes FT
    df_sampled = df_sampled[~is_ft]

# Gets unique values from classroom then maps them to an index.

# {'Com Lab 1': 1, 'Com Lab 2': 2, 'Com Lab 3': 3, 'Com Lab 4': 4, 'GV 301': 5, 'GV 302': 6, 'GV 303': 7, 'GV 304': 8, 'GV 305': 9, 'GV 306': 10, 'GV 307': 11, 'GCA 301': 12, 'GCA 302': 13, 'GCA 303': 14, 'GCA 304': 15, 'GCA 305': 16}

r = 50 # Growth rate parameter. Chaotic behavior occurs at r = 3.99
x = 0.5 # Current value

def logistic_map(r, x):
    """
    Calculate the logistic map value based on the given parameters.

    Parameters:
    - r (float): The growth rate parameter. Expected to be in the range [0, 4].
    - x (float): The current value. Expected to be in the range [0, 1].

    Returns:
    - float: The logistic map value. Expected to be in the range [0, 1].
    """
    result = r * x * (1 - x)
    return result

class Schedule:
    def __init__(self, assignments, teacher_type, ga_type, max_hours_per_week=None, unavailable_days=None, location_to_index_map=None):
        self.assignments = assignments
        self.teacher_type = teacher_type
        self.ga_type = ga_type
        self.max_hours = max_hours_per_week
        self.unavailable_days = unavailable_days or []
        self.location_to_index_map = location_to_index_map
    
    def create_point_cloud(location_to_index_map):
        # Convert each location to a point (x, y) - for simplicity, we can use (index, index)
        point_cloud = [(index, index) for index in location_to_index_map.values()]
        return point_cloud
    
    def compute_persistence_diagram(point_cloud):
        rips_complex = gd.RipsComplex(points=point_cloud)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=5)
        persistence = simplex_tree.persistence()
        return persistence


    # TDA Based Fitness Function
    def fitness(self):
        # Print self.assignments
        # print(f"Self Assignment: {self.assignments}")

        # Preprocess self.assignments
        # Check if start_time and end_time are empty then preprocess them

        fitness_score = 1000

        def get_day(timeslot):
            return timeslot.split(' ')[0] if timeslot else ''
        
        def get_total_hours(start_time, end_time):
            """
            'start_time': '5:00 ', 'end_time': ' 6:00 PM'
            """

            # Remove ':', 'PM', 'AM', and spaces
            start_time = start_time.replace(':', '').replace('PM', '').replace('AM', '').replace(' ', '')
            end_time = end_time.replace(':', '').replace('PM', '').replace('AM', '').replace(' ', '')

            # Then we have integer but still in string format, so convert it to int
            start_time = int(start_time)
            end_time = int(end_time)

            # Calculate the total hours
            total_hours = (end_time - start_time) / 100

            return total_hours

        def create_point_cloud(location_to_index_map):
            # Number of classrooms
            num_classrooms = len(location_to_index_map)

            # Creating points on a circle to ensure distinct positioning
            point_cloud = []
            for index in location_to_index_map.values():
                angle = 2 * math.pi * index / num_classrooms
                point = (math.cos(angle), math.sin(angle))
                point_cloud.append(point)

            return point_cloud
        
        def compute_persistence_diagram(point_cloud):
            rips_complex = gd.RipsComplex(points=point_cloud)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            persistence = simplex_tree.persistence()
            return persistence
        
        def tda_fitness(persistence_diagram, scale_factor=100):
            # Example: Calculate fitness based on the persistence of holes (dimension 1)
            fitness_score = 0
            for feature in persistence_diagram:
                dimension, (birth, death) = feature
                if dimension == 1:  # Looking at holes
                    persistence = death - birth
                    fitness_score += persistence  # More persistent holes, higher fitness
            return fitness_score * scale_factor

        # NOTE: Check if unavailable days are in the schedule
        for assignment in self.assignments:
            day = get_day(assignment['timeslot'])
            if day in self.unavailable_days:
                fitness_score -= 400

        # NOTE: Check if max hours per week is exceeded
        total_hours = 0

        for assignment in self.assignments:
            # Access the start_time and end_time of the assignment
            start_time = assignment.get('start_time')
            end_time = assignment.get('end_time')

            total_hours += get_total_hours(start_time, end_time)

        # NOTE: Check if there are similar timeslots. If there are, then subtract 200 from fitness_score
        timeslots = [assignment['timeslot'] for assignment in self.assignments]
        timeslot_counts = {timeslot: timeslots.count(timeslot) for timeslot in timeslots}
        for timeslot, count in timeslot_counts.items():
            if count > 1:
                fitness_score -= 200

        # TDA Fitness Calculation
        point_cloud = create_point_cloud(self.location_to_index_map)

        persistence_diagram = compute_persistence_diagram(point_cloud)
        
        tda_score = tda_fitness(persistence_diagram)

        # Integrate TDA score with existing fitness score
        fitness_score += tda_score

        return fitness_score

def mutate(schedule, mutation_rate=1):
    if random.random() < mutation_rate:
        # Choose a random assignment index
        mutation_index = random.randint(0, len(schedule.assignments) - 1)
        # Replace it with a new random assignment from the data
        mutated_assignment = df_sampled.sample(1).to_dict('records')[0]
        schedule.assignments[mutation_index] = mutated_assignment
    return schedule

def de_crossover_and_mutation(parent1, parent2, population, F=0.8, CR=0.5):
    # Ensure that parent1 and parent2 have the same number of assignments
    min_length = min(len(parent1.assignments), len(parent2.assignments))

    # Create a trial vector
    trial_vector = []
    for i in range(min_length):
        if random.random() < CR:
            # Differential mutation
            individual1, individual2 = random.sample(population, 2)
            mutated_assignment = {}
            for key in parent1.assignments[i]:
                try:
                    # Attempt to convert to float and apply differential mutation
                    base_value = float(parent1.assignments[i].get(key, 0))
                    diff = float(individual1.assignments[i].get(key, 0)) - float(individual2.assignments[i].get(key, 0))
                    mutated_assignment[key] = base_value + F * diff
                except ValueError:
                    # Non-numeric value, keep the original value
                    mutated_assignment[key] = parent1.assignments[i][key]
            trial_vector.append(mutated_assignment)
        else:
            # Take assignment from parent1
            trial_vector.append(parent1.assignments[i])

    # Create a child Schedule with the trial vector
    child = Schedule(trial_vector, parent1.teacher_type, parent1.ga_type, parent1.max_hours, parent1.unavailable_days, parent1.location_to_index_map)

    # Mutation (optional)
    child = mutate(child)  # Assuming 'mutate' is a predefined function

    return child

# Main (Using Proposed GA)
def main():
    mutation_rate = 0.1
    best_fitnesses = []
    current_generation = 0
    population = []

    all_fitness_scores = []  # Initialize an empty list to store fitness scores of all generations

    while current_generation < max_generations:
        # If population is empty, then create a new population
        if len(population) == 0:
            # Use logistic map then store the result in a variable
            logistic_map_result = logistic_map(r, x)

            # Calculate Raw Sample Size
            raw_sample_size = int(logistic_map_result * len(df))

            # Calculate Sample Size
            raw_sample_size = int(x * len(df))

            # Ensure that the sample size is not larger than the length of the dataframe
            sample_size = min(raw_sample_size, len(df))

            # Use df_sampled instead of df
            sampled_indices = df_sampled.sample(n=sample_size, replace=True).index  # Sample with replacement

            sampled_df = df.loc[sampled_indices].to_dict('records')  # Convert to a list of dictionaries

            # Use len(sampled_df) instead of sample_size
            for i in range(population_size): # // is floor division
                # Create a list of assignments. If population_size is 100, then the list will contain 100 population with divided and distributed assignments.
                assignments = sampled_df[i * (len(sampled_df) // population_size):(i + 1) * (len(sampled_df) // population_size)]
                
                # Apply split_timeslot function to timeslot column of assignments
                for assignment in assignments:
                    assignment['day'], assignment['time'] = split_timeslot(assignment['timeslot'])
                    assignment['start_time'], assignment['end_time'] = assignment['time'].split('-') if assignment['time'] else ('', '')

                # Create a new schedule
                schedule = Schedule(assignments, teacher_type, '2', max_hours_per_week, unavailable_days, location_to_index_map)  # Pass '2' for proposed GA

                population.append(schedule)

         # Break if expected_population is reached
        
        # Check if all are 100 inside population list
        is_all_100 = all(schedule.fitness() == 1000 for schedule in population)

        if is_all_100:
            print(f"Expected Population Reached at Generation {current_generation + 1}")
            break

        fitness_scores = []

        # Loop through population and calculate their fitness scores
        for schedule in population:
            fitness_scores.append(schedule.fitness())

        # Get fitness scores of the current generation and store them
        current_gen_fitness = [schedule.fitness() for schedule in population]
        all_fitness_scores.append(current_gen_fitness)

        # Get the index of the schedule with the highest fitness score
        best_fitness = max(fitness_scores)

        # Select parents to be used for crossover and mutation and put them to next generation
        next_generation = []

        # Sort fitness scores in descending order
        sorted_fitness_scores = sorted(fitness_scores, reverse=True)
        print(f"Current Fitness Scores: {sorted_fitness_scores} | Length: {len(sorted_fitness_scores)}\n")

         # Loop through population and select parents
        for i in range(population_size):  # We'll create one child per pair
            # Select parents based on fitness scores
            # Make sure that the indices are within the range of the population
            parent1_index = fitness_scores.index(sorted_fitness_scores[i % len(sorted_fitness_scores)])
            parent2_index = fitness_scores.index(sorted_fitness_scores[(i + 1) % len(sorted_fitness_scores)])

            parent1 = population[parent1_index]
            parent2 = population[parent2_index]

            # Apply DE crossover and mutation
            child = de_crossover_and_mutation(parent1, parent2, population)

            # Add the child to the next generation
            next_generation.append(child)

        next_generation = sorted(next_generation, key=lambda x: x.fitness(), reverse=True)[:population_size]
 
        # Loop through next generation and mutate them
        next_generation = [mutate(schedule, mutation_rate) for schedule in next_generation]

        # Apply elitism
        # Get the index of the schedule with the highest fitness score
        best_fitness = max(fitness_scores)
        best_fitness_index = fitness_scores.index(best_fitness)

        # Replace the worst schedule with the best schedule
        next_generation[-1] = population[best_fitness_index]

        # Print next generation fitness scores
        next_generation_fitness_scores = []

        # Loop through next generation and calculate their fitness scores
        next_generation_fitness_scores = [schedule.fitness() for schedule in next_generation]

        # Sort next generation fitness scores in descending order
        next_generation_fitness_scores = sorted(next_generation_fitness_scores, reverse=True)

        # Print next generation fitness scores
        print(f"Next Generation Fitness Scores: {next_generation_fitness_scores} | Length: {len(next_generation_fitness_scores)}\n")

        # Find the best fitness in the current generation and add it to best_fitnesses
        best_fitness = max(next_generation_fitness_scores)
        best_fitnesses.append(best_fitness)
        # print(f"Best Fitness Score in Generation {current_generation + 1}: {best_fitness}")

        # Increment current generation
        current_generation += 1

        population = next_generation
  
    # Print the best schedule
    best_index = next_generation_fitness_scores.index(max(next_generation_fitness_scores))
    best_schedule = next_generation[best_index]

    # Display a table of the best schedule
    best_schedule_df = pd.DataFrame(best_schedule.assignments)

    # Sort the dataframe by timeslot
    best_schedule_df = best_schedule_df.sort_values(by=['timeslot'])

    # Display a table of the best schedule
    print(f"\nBest Schedule Dataframe:\n{best_schedule_df}")

    # Sort by faculty
    best_schedule_df = best_schedule_df.sort_values(by=['faculty'])

    # Display a table of the best schedule
    print(f"\nBest Schedule Dataframe (Sorted by Faculty):\n{best_schedule_df}")

    # Group the schedule by faculty into separate dataframes
    grouped_schedule = best_schedule_df.groupby('faculty')

    # Display the schedule for each faculty
    for faculty, faculty_schedule in grouped_schedule:
        # Add color to print of `faculty`
        print(f"\033[1;32;40mFaculty: {faculty}\033[0m")
        print(faculty_schedule)

    # Create a flat file for the fitness scores for proposed GA
    with open('proposed-ga-fitness-scores.csv', 'w') as f:
        for fitness_score in best_fitnesses:
            f.write(f"{fitness_score}\n")

    # After the GA loop, write the fitness scores to a CSV file
    with open('proposed_population_fitness_scores.csv', 'w') as file:
        for gen_scores in all_fitness_scores:
            file.write(','.join(map(str, gen_scores)) + '\n')

    # Plot the best fitness function overtime
    plt.plot(best_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.show()

main()


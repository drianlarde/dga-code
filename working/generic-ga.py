import pandas as pd
import numpy as np
import random
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi as gd

df = pd.read_csv(r"./data/preprocessed_data.csv")

try:
    df_sampled = df.sample(n=10000, replace=True)
except ValueError as e:
    print(f"Error: Not enough records to sample with replacement: {e}")
    exit()

ga_type = 1

teacher_type = 'FT'

population_size = 1000 
max_generations = 100

max_hours = 40
unavailable_days = 'Monday,Wednesday'

target_fitness = 95  # This means we are aiming for a solution with at least a fitness of 95
convergence_threshold = 0.5  # This means we consider the algorithm to have converged if the improvement is less than 0.5

def split_timeslot(timeslot):
    parts = timeslot.split(' ')
    if len(parts) >= 2:
        return parts[0], ' '.join(parts[1:])
    return timeslot, ''

df_sampled['day'], df_sampled['time'] = zip(*df_sampled['timeslot'].apply(split_timeslot))
df_sampled['start_time'], df_sampled['end_time'] = zip(*df_sampled['time'].apply(lambda x: x.split('-') if x else ('', '')))

class Schedule:
    def __init__(self, assignments, teacher_type, ga_type, max_hours=None, unavailable_days=None, location_to_index_map=None):
        self.assignments = assignments
        self.teacher_type = teacher_type
        self.ga_type = ga_type
        self.max_hours = max_hours
        self.unavailable_days = unavailable_days or []
        self.location_to_index_map = location_to_index_map

    def update_occupied_timeslots(self):
        for a in self.assignments:
            Schedule.occupied_timeslots.add((a['faculty'], a['timeslot']))

    def is_timeslot_available(self, faculty, timeslot):
        return (faculty, timeslot) not in Schedule.occupied_timeslots

    def __str__(self):
        schedule_details = "Schedule:\n"
        for assignment in self.assignments:
            schedule_details += f"Course: {assignment['course']}, Faculty: {assignment['faculty']}, Timeslot: {assignment['timeslot']}, Classroom: {assignment['classroom']}\n"
        return schedule_details

    def fitness(self):
        # Initialize variables
        total_hours = 0
        faculty_course_count = {}
        fitness_score = 1000

        for a in self.assignments:
            duration = self.calculate_duration(a['start_time'], a['end_time'])
            total_hours += duration
            faculty_course_count[a['faculty']] = faculty_course_count.get(a['faculty'], 0) + 1

        if self.teacher_type == 'FT' and total_hours > self.max_hours:
            fitness_score -= 300

        if self.teacher_type == 'PT' and total_hours > 12:
            fitness_score -= 100

        # Check if distribution is inequitable
        if len(faculty_course_count) > 1:
            max_courses = max(faculty_course_count.values())
            min_courses = min(faculty_course_count.values())
            error_percentage = (max_courses - min_courses) / max_courses

            if error_percentage > 0.1:
                fitness_score -= int(error_percentage * 10)

        # Consulting Hours Overlap
        for i in range(len(self.assignments)):
            for j in range(i + 1, len(self.assignments)):
                if self.assignments[i]['faculty'] == self.assignments[j]['faculty'] and \
                        self.assignments[i]['timeslot'] == self.assignments[j]['timeslot']:
                    fitness_score -= 500
                    break

        # Faculty Preferences Ignored
        for a in self.assignments:
            if a['day'] in self.unavailable_days:
                fitness_score -= 400

        # Classroom Utilization Inefficiency - Underutilized or overutilized classrooms
        classroom_usage = {}
        for a in self.assignments:
            classroom_usage[a['classroom']] = classroom_usage.get(a['classroom'], 0) + self.calculate_duration(a['start_time'], a['end_time'])

        for classroom, usage in classroom_usage.items():
            if usage < 3:
                fitness_score -= 300
            elif usage > 5:
                fitness_score -= 300

        # Student Schedule Clashes
        for i in range(len(self.assignments)):
            for j in range(i + 1, len(self.assignments)):
                if self.assignments[i]['timeslot'] == self.assignments[j]['timeslot'] and \
                        self.assignments[i]['classroom'] == self.assignments[j]['classroom']:
                    fitness_score -= 200
                    break

        # NOTE: Remember to use fitness_score = 1 / 1 + penalty

        return fitness_score

    @staticmethod
    def calculate_duration(start, end):
        def parse_time(time_str):
            time_str = time_str.strip()
            if ' PM' in time_str or ' AM' in time_str:
                time_str = time_str[:time_str.rfind(' ')]
            for fmt in ('%I:%M %p', '%H:%M'):
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Time data '{time_str}' does not match expected format")

        start_time, end_time = parse_time(start), parse_time(end)
        if end_time < start_time:
            end_time += timedelta(days=1)

        return (end_time - start_time).total_seconds() / 3600.0 # In hours

    @staticmethod
    def convert_time_to_float(time_str):
        # Map each day to a number (e.g., Monday=0, Tuesday=1, etc.)
        day_to_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

        # Split the time string into day and time of day (if day is provided)
        parts = time_str.strip().split(maxsplit=1)
        day_part = parts[0] if len(parts) == 2 else "Monday"  # Default to Monday if day is not provided
        time_of_day = parts[-1]

        day_num = day_to_num.get(day_part, 0)  # Default to 0 if day is not found

        # Convert time to total hours
        time_format = "%H:%M"
        time_delta = datetime.strptime(time_of_day, time_format) - datetime.strptime("00:00", time_format)
        total_hours = 24 * day_num + time_delta.total_seconds() / 3600

        return total_hours

# Initialize population
def initialize_population(size, df, teacher_type, max_hours, unavailable_days, location_to_index_map):
    population = []
    for _ in range(size):
        assignments = df.sample(n=10).to_dict('records')
        population.append(Schedule(assignments, teacher_type, '1', max_hours, unavailable_days, location_to_index_map))

    return population

def select_parents(population):
    # Sort the population by fitness in descending order and assign ranks
    sorted_population = sorted(population, key=lambda x: x.fitness(), reverse=True)
    
    # Calculate selection probabilities based on ranks
    total_ranks = sum(range(1, len(population) + 1))  # Sum of ranks
    selection_probabilities = [rank / total_ranks for rank in range(len(population), 0, -1)]

    # Select parents
    parents = []
    for _ in range(len(population)):
        chosen_parent = random.choices(sorted_population, weights=selection_probabilities, k=1)[0]
        parents.append(chosen_parent)

    return parents

def enhanced_crossover(parent1, parent2, teacher_type, max_hours, unavailable_days):
        """
        Performs enhanced crossover between two parent schedules. Crossover is performed by selecting two random crossover.

        Parameters:
        - parent1 (Schedule): The first parent schedule.
        - parent2 (Schedule): The second parent schedule.
        - teacher_type (str): The type of teacher (FT or PT).
        - max_hours (int): The maximum number of hours for full-time teachers.
        - unavailable_days (list): The list of unavailable days for faculty.

        Returns:
        - child (Schedule): The child schedule generated from crossover.
        """

        crossover_point1 = random.randint(1, len(parent1.assignments) - 2)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1.assignments) - 1)

        new_assignments = parent1.assignments[:crossover_point1] + parent2.assignments[crossover_point1:crossover_point2] + parent1.assignments[crossover_point2:]
        child = Schedule(new_assignments, teacher_type, ga_type, max_hours, unavailable_days)

        return child

def mutate(schedule):
    mutation_rate = 0.1
    if random.random() < mutation_rate:
        mutated_assignment = df_sampled.sample(1).to_dict('records')[0]
        schedule.assignments[random.randint(0, len(schedule.assignments) - 1)] = mutated_assignment

    return schedule

def elitism(population, top_k=1):
    return sorted(population, key=lambda s: s.fitness(), reverse=True)[:top_k]

def logistic_map(r, x):
    """Logistic map function."""
    return r * x * (1 - x)

def check_termination_condition(population, generation, max_generations, target_fitness, convergence_threshold):
    """
    Check if the termination condition for the genetic algorithm is met.

    :param population: The current population of schedules.
    :param generation: The current generation number.
    :param max_generations: The maximum number of generations to run.
    :param target_fitness: The target fitness score to achieve.
    :param convergence_threshold: The threshold for population convergence.

    :return: True if the termination condition is met, False otherwise.
    """
    if generation >= max_generations:
        return True

    return False

def generic_ga(population_size, max_generations, df, teacher_type, max_hours=None, unavailable_days=None):
    # Define the mapping from classroom locations to indices
    location_to_index_map = {classroom: index for index, classroom in enumerate(df['classroom'].unique(), start=1)}

    # Initialize population
    population = initialize_population(population_size, df, teacher_type, max_hours, unavailable_days, location_to_index_map)

    fitness_scores = []  # List to store the best fitness score in each generation

    all_fitness_scores = []  # Initialize an empty list to store fitness scores of all generations


    # Main GA loop
    for generation in range(max_generations):
        # Append the best fitness score in the current generation, inside population.fitnes()
        generation_best_fitness = max(individual.fitness() for individual in population)
        fitness_scores.append(generation_best_fitness)

        current_gen_fitness = [individual.fitness() for individual in population]
        all_fitness_scores.append(current_gen_fitness)

        # Check termination condition
        if check_termination_condition(population, generation, max_generations, target_fitness, convergence_threshold):
            print(f"Termination condition met at generation {generation}.")
            break
        
        # Sort the population by fitness in descending order and assign ranks
        sorted_population = sorted(population, key=lambda x: x.fitness(), reverse=True)
        
        # Calculate selection probabilities based on ranks
        total_ranks = sum(range(1, len(population) + 1))  # Sum of ranks
        selection_probabilities = [rank / total_ranks for rank in range(len(population), 0, -1)]

        # Get sorted population in an array of their fitness values
        sorted_population_fitness = [individual.fitness() for individual in sorted_population]

        # Select parent
        parents = []
        for _ in range(len(population)):
            chosen_parent = random.choices(sorted_population, weights=selection_probabilities, k=1)[0]
            parents.append(chosen_parent)

        # Generate offspring through crossover, handling odd number of parents ---
        children = []
        
        for i in range(0, len(parents), 2):
            # Check if the next parent exists
            if i + 1 < len(parents): # If the next parent exists
                # Print 2 parents and their fitness
                child = enhanced_crossover(parents[i], parents[i+1], teacher_type, max_hours, unavailable_days)

                children.append(child)
            else: # Handle the case where the number of parents is odd
                # Example: Pair the last parent with the first parent
                child = enhanced_crossover(parents[i], parents[0], teacher_type, max_hours, unavailable_days)
                children.append(child)

        # Mutate the offspring
        for child in children:
            mutate(child)

        # Apply elitism to form the next generation, this preserves the best individuals
        elite = elitism(population)
        population = children + elite

    # Find the best solution from the final population
    best_solution = max(population, key=lambda ind: ind.fitness())

    # Plot the evolution of fitness scores
    plt.plot(fitness_scores)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Generic Genetic Algorithm Progress')
    plt.show()

    # Create a flat file of fitness scores for generic GA
    with open('generic-ga-fitness-scores.csv', 'w') as f:
        for score in fitness_scores:
            f.write(f"{score}\n")

    # After the GA loop, write the fitness scores to a CSV file
    with open('population_fitness_scores.csv', 'w') as file:
        for gen_scores in all_fitness_scores:
            file.write(','.join(map(str, gen_scores)) + '\n')

    return best_solution

def run_genetic_algorithm(ga_type, population_size, max_generations, df, teacher_type, max_hours=None, unavailable_days=None, feature_persistence_threshold=0.1):
    return generic_ga(population_size, max_generations, df, teacher_type, max_hours, unavailable_days)

best_schedule_found = run_genetic_algorithm(ga_type, population_size, max_generations, df_sampled, teacher_type, max_hours, unavailable_days)

if best_schedule_found:
    print("Best schedule fitness:", best_schedule_found.fitness())
    print("Best Schedule Details:\n", best_schedule_found)
else:
    print("No suitable schedule was found.")


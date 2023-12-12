import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import gudhi as gd
import numpy as np

def perform_tda_analysis(population):
    # Transform your population of schedules into a point cloud
    point_cloud = np.array([schedule.get_features() for schedule in population])

    # Construct a Rips complex from the point cloud
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=1.0)

    # Create a simplex tree from the Rips complex
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Compute persistent homology
    persistence = simplex_tree.persistence()

    return persistence

initial_crossover_rate = 0.5  # Initial crossover rate
initial_mutation_rate = 0.05  # Initial mutation rate
initial_tournament_size = 3    # Initial tournament size

def adapt_parameters_based_on_tda(tda_results, crossover_rate, mutation_rate, tournament_size):
    persistence_values = [p[1][1] - p[1][0] for p in tda_results if p[1][1] < float('inf')]
    avg_persistence = sum(persistence_values) / len(persistence_values) if persistence_values else 0

    # Adjust mutation rate based on persistence
    if avg_persistence < 0.2:  # Low diversity
        mutation_rate += 0.01
    else:  # High diversity
        mutation_rate -= 0.01

    # Adjust crossover rate and tournament size
    if avg_persistence > 0.4:  # High convergence
        crossover_rate += 0.05
        tournament_size += 1
    else:  # Low convergence
        crossover_rate -= 0.05
        tournament_size -= 1

    # Ensure rates and sizes stay within reasonable ranges
    mutation_rate = max(0.01, min(mutation_rate, 0.2))
    crossover_rate = max(0.3, min(crossover_rate, 0.8))
    tournament_size = max(2, min(tournament_size, 5))

    return crossover_rate, mutation_rate, tournament_size


def logistic_map(r, x, iterations):
    values = []
    for _ in range(iterations):
        x = r * x * (1 - x)
        values.append(x)
    return values

# Explores more in solution space
def initialize_population(size, df, teacher_type, max_hours, unavailable_days, r=3.99):
    population = []
    for _ in range(size):
        x_start = random.random()
        logistic_values = logistic_map(r, x_start, len(df))

        # Normalize logistic values to sum to 1, to use as weights for sampling
        logistic_weights = [lv / sum(logistic_values) for lv in logistic_values]

        # Use weighted sampling to select courses based on logistic map values
        sorted_courses = random.choices(df.to_dict('records'), weights=logistic_weights, k=len(df))

        assignments = []
        total_hours = 0
        
        for course in sorted_courses:
            if total_hours + course['hours'] <= max_hours and course['day'] not in unavailable_days:
                assignments.append(course)
                total_hours += course['hours']
            # Stop adding courses if max hours are reached or exceeded
            if total_hours >= max_hours:
                break

        population.append(Schedule(assignments, teacher_type, '1', max_hours, unavailable_days))

    return population

# Typical GA initialization, no diversity and doesn't explore more in solution space
# Initialize the population
# def initialize_population(size, df, teacher_type, max_hours, unavailable_days):
#     population = []
#     for _ in range(size):
#         shuffled_df = df.sample(frac=1).to_dict('records')
#         assignments = []
#         total_hours = 0

#         for course in shuffled_df:
#             if total_hours + course['hours'] <= max_hours:
#                 # Remove direct classroom assignment here
#                 assignments.append(course)
#                 total_hours += course['hours']
#             else:
#                 break  # Stop adding courses if max hours are reached or exceeded

#         population.append(Schedule(assignments, teacher_type, '1', max_hours, unavailable_days))

#     return population

# Selecting parents
def tournament_selection(population, tournament_size):
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best_in_tournament = max(tournament, key=lambda ind: ind.fitness())
        parents.append(best_in_tournament)
    return parents

# Crossover
def crossover(parent1, parent2):
    child_assignments = []

    split_point = random.randint(0, len(parent1.assignments) - 1)
    child_assignments.extend(parent1.assignments[:split_point])

    for assignment in parent2.assignments[split_point:]:
        # Remove direct classroom assignment
        child_assignments.append(assignment)

    return Schedule(child_assignments, parent1.teacher_type, '1', parent1.max_hours, parent1.unavailable_days)

def adaptive_mutation_rate(best_fitness, diversity, base_rate=0.05, max_rate=0.2):
    """
    Calculates an adaptive mutation rate based on the best fitness score and population diversity.
    The mutation rate increases if the diversity is low or the best fitness is not improving significantly.
    
    :param best_fitness: The best fitness score in the current generation.
    :param diversity: The diversity measure of the current population.
    :param base_rate: The base mutation rate.
    :param max_rate: The maximum allowed mutation rate.
    :return: The calculated mutation rate.
    """
    # Adjust the rate based on diversity and best fitness
    rate = base_rate + (max_rate - base_rate) * (1 - diversity / best_fitness)
    return min(max(rate, base_rate), max_rate)

def de_inspired_mutation(population, best_fitness, diversity):
    # With adaptive mutation rate (SOP 1 & 3)(Much Better)
    mutation_rate = adaptive_mutation_rate(best_fitness, diversity)

    # Typical DE inspired mutation (Somehow Better)
    # mutation_rate = 0.1

    if random.random() < mutation_rate:
        # Select 3 random individuals
        a, b, c = random.sample(population, 3)

        # Choose a random day from one of the individuals
        chosen_day = random.choice([random.choice(a.assignments), 
                                    random.choice(b.assignments), 
                                    random.choice(c.assignments)])['day']

        # Choose a random time slot (start_time, end_time, hours) from one of the individuals
        chosen_time_slot = random.choice([random.choice(a.assignments), 
                                          random.choice(b.assignments), 
                                          random.choice(c.assignments)])

        # Create a new assignment combining chosen day and time slot
        new_assignment = {
            'day': chosen_day,
            'start_time': chosen_time_slot['start_time'],
            'end_time': chosen_time_slot['end_time'],
            'hours': chosen_time_slot['hours'],
            'course': new_df.sample(1).to_dict('records')[0]['course']
        }

        # Replace a random assignment in a random individual's schedule with the new assignment
        random_individual = random.choice(population)
        random_individual.assignments[random.randint(0, len(random_individual.assignments) - 1)] = new_assignment

# Elitism
def elitism(population, top_k=1):
    return sorted(population, key=lambda s: s.fitness(), reverse=True)[:top_k]

# Genetic Algorithm
def run_genetic_algorithm(population_size, max_generations, df, teacher_type, max_hours, unavailable_days, initial_crossover_rate, initial_mutation_rate, initial_tournament_size, target_fitness_level, generation_of_target_fitness):
    # Initialize population
    population = initialize_population(population_size, df, teacher_type, max_hours, unavailable_days)
    all_fitness_scores = []
    diversity_data = []
    best_fitness_scores = []

    crossover_rate = initial_crossover_rate
    mutation_rate = initial_mutation_rate
    tournament_size = initial_tournament_size

    for generation in range(max_generations):
        # TDA analysis and parameter adaptation every 'n' generations
        if generation % 10 == 0:  # For example, every 10 generations
            tda_results = perform_tda_analysis(population)
            crossover_rate, mutation_rate, tournament_size = adapt_parameters_based_on_tda(
                tda_results, crossover_rate, mutation_rate, tournament_size
            )

        # Select parents
        parents = tournament_selection(population, tournament_size)

        # Identify and preserve the elite before creating new offspring
        elite = elitism(population, top_k=1)
        
        # Generate new population through crossover and mutation
        new_population = []
        
        while len(new_population) < population_size - len(elite):
            parent1, parent2 = random.sample(parents, 2)
            if random.random() <= crossover_rate:
                child = crossover(parent1, parent2)
                new_population.append(child)

        # Calculate fitness and diversity before mutation
        for individual in new_population:
            individual.fitness_score = individual.fitness()
        best_fitness = max(individual.fitness() for individual in new_population)
        diversity = calculate_population_diversity(new_population)

        # Mutate the new offspring (not the elite)
        for individual in new_population:
            de_inspired_mutation(new_population, best_fitness, diversity)

        # Combine the elite with the new offspring
        new_population.extend(elite)

        # Update the population
        population = new_population

        # Calculate and store diversity and fitness for this generation
        diversity_data.append({'generation': generation, 'diversity': diversity})
        best_fitness_scores.append(best_fitness)

        # Check for target fitness achievement
        if best_fitness >= target_fitness_level and generation not in generation_of_target_fitness:
            generation_of_target_fitness.append(generation)

    # Return the best solution found
    best_solution = max(population, key=lambda ind: ind.fitness())
    return best_solution, all_fitness_scores, diversity_data, best_fitness_scores

# Get the next faculty identifier
def get_next_faculty_identifier(faculty_type, file_path):
    try:
        existing_df = pd.read_csv(file_path)
        existing_faculties = existing_df['faculty'].dropna().unique()
        highest_number = 0
        for faculty in existing_faculties:
            if faculty.startswith(faculty_type):
                number = int(faculty.replace(faculty_type, ''))
                highest_number = max(highest_number, number)
        return f"{faculty_type}{highest_number + 1}"
    except FileNotFoundError:
        return f"{faculty_type}1"
    
# Save the best schedule
def save_best_schedule(best_schedule, faculty_type, file_path):
    faculty_identifier = get_next_faculty_identifier(faculty_type, file_path)
    for assignment in best_schedule.assignments:
        assignment['faculty'] = faculty_identifier
    df = pd.DataFrame(best_schedule.assignments)
    try:
        existing_df = pd.read_csv(file_path)
        new_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        new_df = df
    new_df.to_csv(file_path, index=False)

# New function for post-processing to assign classrooms
def assign_classrooms(best_schedule, existing_schedules):
    # Print best schedule
    print("\nBest Schedule:")
    for assignment in best_schedule.assignments:
        print(f"{assignment['course']} - {assignment['start_time']} - {assignment['end_time']}")

    # Print existing schedules
    print("\nExisting Schedules:")
    for _, row in existing_schedules.iterrows():
        print(f"{row['course']} - {row['start_time']} - {row['end_time']}")

    # Randomly assign classrooms
    for assignment in best_schedule.assignments:
        if '.1' in assignment['course']:
            assignment['classroom'] = 'Classroom Lab 1'
        else:
            assignment['classroom'] = 'Classroom Lec 1'

    for assignment in best_schedule.assignments:
        for _, row in existing_schedules.iterrows():
            if assignment['day'] == row['day'] and assignment['start_time'] == row['start_time'] and assignment['end_time'] == row['end_time'] and assignment['classroom'] == row['classroom']:
                new_classroom = None

                # Split assignment['classroom'] and get the number
                number = int(assignment['classroom'].split(' ')[-1])

                while True:
                    # Increment the number until there are no conflicts
                    increment_number = number + 1
                    new_classroom = f"Classroom {'Lab' if '.1' in assignment['course'] else 'Lec'} {increment_number}"

                    if new_classroom != row['classroom']:
                        break

                assignment['classroom'] = new_classroom

# Function to read existing schedules or create an empty DataFrame if the file doesn't exist
def read_or_create_existing_schedules(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Define the columns that your schedule DataFrame should have
        columns = ['day', 'start_time', 'end_time', 'course', 'classroom', 'faculty', 'hours']
        return pd.DataFrame(columns=columns)

# New function to calculate population diversity
def calculate_population_diversity(population):
    fitness_scores = [individual.fitness() for individual in population]
    diversity = np.var(fitness_scores)  # Variance of fitness scores as a diversity measure
    return diversity

# Schedule class
class Schedule:
    def __init__(self, assignments, teacher_type, ga_type, max_hours=None, unavailable_days=None):
        self.assignments = assignments
        self.teacher_type = teacher_type
        self.ga_type = ga_type
        self.max_hours = max_hours
        self.unavailable_days = unavailable_days or []

    def get_features(self):
        # Example: Extracting simple features for TDA
        features = []

        # Feature 1: Total hours of assignments
        total_hours = sum(assignment['hours'] for assignment in self.assignments)
        features.append(total_hours)

        # Feature 2: Count of assignments
        count_assignments = len(self.assignments)
        features.append(count_assignments)

        return features

    def fitness(self):
        total_hours = 0
        courses_added = set() # To check for duplicate courses
        penalty = 0

        # Penalize for scheduling on unavailable days and overlapping schedules
        unavailable_days_set = set(self.unavailable_days)

        # Initialize schedule for checking overlapping schedule
        schedule = []

        for assignment in self.assignments:
            total_hours += assignment['hours']

            if assignment['day'] in unavailable_days_set:
                # penalty += 500  # Penalize for scheduling on unavailable days
                penalty += 5  # Penalize for scheduling on unavailable days
            
            # Check for duplicate courses
            course_code = assignment['course']

            if course_code in courses_added:
                # penalty += 500
                penalty += 5

            courses_added.add(course_code)

            # Check for overlapping day, start_time, end_time
            for other in schedule:
                if assignment['day'] == other['day'] and assignment['start_time'] == other['start_time'] and assignment['end_time'] == other['end_time']:
                    # penalty += 500
                    penalty += 5
                    break

            schedule.append({
                'day': assignment['day'],
                'start_time': assignment['start_time'],
                'end_time': assignment['end_time']
            })

        # penalty += abs(total_hours - self.max_hours) * 10

        penalty += abs(total_hours - self.max_hours)

        fitness_score = 1 / (1 + penalty)

        return fitness_score

# Load the new dataset
new_df = pd.read_csv('combinations.csv')

# Extended run_configs with a wider range of GA parameters
run_configs = [
    # For Slow Convergence Simulation
    {'seed': 55, 'teacher_type': 'FT', 'max_hours': 35, 'unavailable_days': ['Tuesday', 'Thursday'], 'population_size': 5, 'crossover_rate': 0.3, 'mutation_rate': 0.01, 'tournament_size': 2},

    # Turn on both to simulate `Sensitivity to Initial Conditions`

    # For Fast Convergence Simulation
    {'seed': 60, 'teacher_type': 'PT', 'max_hours': 20, 'unavailable_days': ['Monday', 'Wednesday'], 'population_size': 12, 'crossover_rate': 0.65, 'mutation_rate': 0.15, 'tournament_size': 5},

    # Other
    {'seed': 65, 'teacher_type': 'FT', 'max_hours': 28, 'unavailable_days': ['Friday'], 'population_size': 8, 'crossover_rate': 0.5, 'mutation_rate': 0.2, 'tournament_size': 6},
    # {'seed': 70, 'teacher_type': 'PT', 'max_hours': 22, 'unavailable_days': ['Monday', 'Friday'], 'population_size': 15, 'crossover_rate': 0.85, 'mutation_rate': 0.05, 'tournament_size': 7},
    # {'seed': 75, 'teacher_type': 'FT', 'max_hours': 30, 'unavailable_days': ['Wednesday', 'Thursday'], 'population_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'tournament_size': 2},
    # {'seed': 80, 'teacher_type': 'PT', 'max_hours': 26, 'unavailable_days': ['Tuesday'], 'population_size': 14, 'crossover_rate': 0.7, 'mutation_rate': 0.12, 'tournament_size': 3},
    # {'seed': 85, 'teacher_type': 'FT', 'max_hours': 32, 'unavailable_days': [], 'population_size': 9, 'crossover_rate': 0.55, 'mutation_rate': 0.18, 'tournament_size': 5},
    # {'seed': 90, 'teacher_type': 'PT', 'max_hours': 24, 'unavailable_days': ['Sunday', 'Saturday'], 'population_size': 13, 'crossover_rate': 0.8, 'mutation_rate': 0.07, 'tournament_size': 4},
    # {'seed': 95, 'teacher_type': 'FT', 'max_hours': 29, 'unavailable_days': ['Monday', 'Tuesday', 'Wednesday'], 'population_size': 11, 'crossover_rate': 0.6, 'mutation_rate': 0.1, 'tournament_size': 6},
    # {'seed': 100, 'teacher_type': 'PT', 'max_hours': 23, 'unavailable_days': ['Thursday', 'Friday'], 'population_size': 16, 'crossover_rate': 0.9, 'mutation_rate': 0.05, 'tournament_size': 2},
]

# Define the target fitness level for comparison
target_fitness_level = 0.9  # This value can be adjusted based on your criteria

# To store the generation at which the target fitness level is first reached
generation_of_target_fitness = []

max_generations = 80

# Initialize collections for results
all_best_fitness_scores = {}
all_diversity_data = {'generation': range(max(config['population_size'] for config in run_configs))}

# Run experiments and collect data in a single loop
for config in run_configs:
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)

    # Run the genetic algorithm
    best_schedule, fitness_data, diversity_data, best_fitness_scores = run_genetic_algorithm(
        config['population_size'], max_generations, new_df, config['teacher_type'], config['max_hours'], config['unavailable_days'],
        config['crossover_rate'], config['mutation_rate'], config['tournament_size'], target_fitness_level, generation_of_target_fitness
    )

    # Post-processing, saving schedules, and printing
    existing_schedules = read_or_create_existing_schedules('generic-db.csv')
    assign_classrooms(best_schedule, existing_schedules)
    save_best_schedule(best_schedule, config['teacher_type'], 'generic-db.csv')
    print("Best Schedule Fitness:", best_schedule.fitness())

    # Store the diversity and fitness data for plotting
    diversity_df = pd.DataFrame(diversity_data)
    config_label = f"seed_{seed}_type_{config['teacher_type']}_hours_{config['max_hours']}_CR_{config['crossover_rate']}_MR_{config['mutation_rate']}_TS_{config['tournament_size']}"
    all_diversity_data[config_label] = diversity_df.set_index('generation')['diversity'].tolist()
    all_best_fitness_scores[config_label] = best_fitness_scores

# Calculating the average number of generations needed to reach the target fitness level
if generation_of_target_fitness:
    average_generations_to_target = sum(generation_of_target_fitness) / len(generation_of_target_fitness)
else:
    average_generations_to_target = None

print("Average Generations to reach target fitness level:", average_generations_to_target)

# Get `all_best_fitness_scores` then turn them into .csv
all_best_fitness_scores_df = pd.DataFrame(all_best_fitness_scores)
all_best_fitness_scores_df.to_csv('best_fitness_scores.csv', index=False)

# After all runs are complete, combine the diversity data
max_gen_count = max(len(diversity) for diversity in all_diversity_data.values())
all_diversity_data['generation'] = range(max_gen_count)

# Create the DataFrame with all diversity data
all_diversity_df = pd.DataFrame({k: v + [None]*(max_gen_count - len(v)) if len(v) < max_gen_count else v for k, v in all_diversity_data.items()})

# Save the diversity data to a CSV file
all_diversity_df.to_csv('diversity_across_generations.csv', index=False)

# Create a figure with two subplots, side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# Plot Diversity Across Generations in the first subplot (ax1)
for column in all_diversity_df.columns:
    if 'seed_' in column:
        ax1.plot(all_diversity_df['generation'], all_diversity_df[column], label=column)
ax1.set_title('Diversity Across Generations for Different Initial Conditions')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Diversity (Variance of Fitness Scores)')
# ax1.legend(title='Configurations', loc='best')

# Plot Best Fitness Scores Across Generations in the second subplot (ax2)
for config, fitness_scores in all_best_fitness_scores.items():
    ax2.plot(range(len(fitness_scores)), fitness_scores, label=config)
ax2.set_title('Best Fitness Scores Across Generations for Different Initial Conditions')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Best Fitness Score')
# ax2.legend(title='Configurations', loc='best')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the figure with both subplots
plt.show()
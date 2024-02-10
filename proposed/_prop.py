import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import threading
import time

start_time = time.time()

# Parameters for Genetic Algorithm
pc = 0.8  # Crossover probability
pm = 0.1  # Mutation probability
nPop = 50  # Population size for each subpopulation
nDist = 2  # Number of subpopulations
nGen = 100  # Number of generations

# Initialize global variables
initial_crossover_rate = pc
initial_mutation_rate = pm
initial_tournament_size = 3
global_migration_pool = []

def calculate_mae_mse(predicted_fitness, optimal_fitness_values):
    if len(predicted_fitness) != len(optimal_fitness_values):
        raise ValueError("Length of predicted fitness values and optimal fitness values must be the same.")

    total_absolute_error = 0
    total_squared_error = 0

    for predicted, optimal in zip(predicted_fitness, optimal_fitness_values):
        total_absolute_error += abs(predicted - optimal)
        total_squared_error += (predicted - optimal) ** 2

    mae = total_absolute_error / len(predicted_fitness)
    mse = total_squared_error / len(predicted_fitness)

    return mae, mse

def logistic_map(r, x, iterations):
    values = []
    for _ in range(iterations):
        x = 2 * x * (1 - x)
        values.append(x)
    return values

# Explores more in solution space
def initialize_population(size, df, teacher_type, max_hours, unavailable_days, r=3.99, diversity_threshold=0.5):
    population = []
    while len(population) < size:
        x_start = random.random()
        logistic_values = logistic_map(r, x_start, len(df))

        # Shuffle the dataframe based on logistic values to ensure high diversity
        shuffled_df = df.sample(frac=1, weights=logistic_values, replace=False)

        assignments = []
        total_hours = 0

        # Iterate over the shuffled dataframe to create unique assignments for each individual
        for _, course in shuffled_df.iterrows():
            if total_hours + course['hours'] <= max_hours and course['day'] not in unavailable_days:
                assignments.append(course)
                total_hours += course['hours']
            # Stop adding courses if max hours are reached or exceeded
            if total_hours >= max_hours:
                break

        # Check if the diversity of the generated individual is above the threshold
        individual_hours = [course['hours'] for course in assignments]
        if np.var(individual_hours) < diversity_threshold:
            # If not diverse enough, discard and regenerate this individual
            continue

        # If diverse enough, add to the population
        population.append(Schedule(assignments, teacher_type, '1', max_hours, unavailable_days))

    return population


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
    rate = base_rate + (max_rate - base_rate) * (1 - diversity / best_fitness) # meaning: rate is equals
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

# Define a lock for thread-safe operations
migration_lock = threading.Lock()

# Genetic Algorithm
def run_genetic_algorithm(population_size, max_generations, df, teacher_type, max_hours, unavailable_days, crossover_rate, mutation_rate, tournament_size, subpopulation_id, migration_interval, target_fitness_level, generation_of_target_fitness, global_migration_pool):
    population = initialize_population(population_size, df, teacher_type, max_hours, unavailable_days)
    all_fitness_scores = []
    diversity_data = []
    best_fitness_scores = []

    for generation in range(max_generations):
        # Select parents
        parents = tournament_selection(population, tournament_size)

        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            if random.random() <= crossover_rate:
                child = crossover(parent1, parent2)
                new_population.append(child)

        # Calculate fitness and diversity for the new population
        for individual in new_population:
            individual.fitness_score = individual.fitness()
        best_fitness = max(individual.fitness_score for individual in new_population)
        diversity = calculate_population_diversity(new_population)

        # Apply mutation
        for individual in new_population:
            de_inspired_mutation(new_population, best_fitness, diversity)

        # Update population
        population = new_population

        # Check if it's time to migrate
        if generation % migration_interval == 0:
            with migration_lock:
                selective_migration(population, global_migration_pool, subpopulation_id)

        # Store fitness and diversity data
        diversity_data.append({'generation': generation, 'diversity': diversity})
        best_fitness_scores.append(best_fitness)

        # Check for target fitness
        if best_fitness >= target_fitness_level and generation not in generation_of_target_fitness:
            generation_of_target_fitness.append(generation)

    best_solution = max(population, key=lambda ind: ind.fitness())

    # Get final population
    final_population = population

    # Get the final population fitness scores
    final_population_fitness_scores = [individual.fitness() for individual in final_population]

    return best_solution, all_fitness_scores, diversity_data, best_fitness_scores, final_population, final_population_fitness_scores

def selective_migration(population, global_migration_pool, subpopulation_id):
    # Select top individuals to migrate
    top_individuals = sorted(population, key=lambda x: x.fitness(), reverse=True)[:5]  # Example: top 5
    # Add them to the global migration pool with their subpopulation ID
    for individual in top_individuals:
        global_migration_pool.append((subpopulation_id, individual))

    # Retrieve and integrate migrants from other subpopulations
    for subpop_id, migrant in global_migration_pool:
        if subpop_id != subpopulation_id:
            population.append(migrant)
            # Remove the migrant from the pool
            global_migration_pool.remove((subpop_id, migrant))

    # # Ensure population size remains constant (optional)
    # population = population[:population_size]

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

def subpopulation_thread(df, config, subpopulation_id, global_migration_pool):
    run_genetic_algorithm(
        population_size=config['population_size'],
        max_generations=nGen,
        df=df,
        teacher_type=config['teacher_type'],
        max_hours=config['max_hours'],
        unavailable_days=config['unavailable_days'],
        crossover_rate=config['crossover_rate'],
        mutation_rate=config['mutation_rate'],
        tournament_size=config['tournament_size'],
        subpopulation_id=subpopulation_id,
        migration_interval=config['migration_interval'],
        target_fitness_level=config['target_fitness_level'],
        generation_of_target_fitness=[],
        global_migration_pool=global_migration_pool
    )

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
        # Define weights for each constraint
        weights = {
            'unavailable_day_penalty': 5,  # replace 5 with the actual weight for scheduling on unavailable days
            'duplicate_course_penalty': 5,  # replace 5 with the actual weight for duplicate courses
            'overlap_penalty': 5,  # replace 5 with the actual weight for overlapping classes
            'hours_penalty': 1  # replace 1 with the actual weight for total hours
        }

        # Initialize penalties
        unavailable_day_penalty = 0
        duplicate_course_penalty = 0
        overlap_penalty = 0
        hours_penalty = 0

        # Initialize schedule for checking overlapping schedule
        schedule = []

        # Set to track courses added to avoid duplicates
        courses_added = set()

        # Calculate penalties
        for assignment in self.assignments:
            # Check for scheduling on unavailable days
            if assignment['day'] in self.unavailable_days:
                unavailable_day_penalty += 1

            # Check for duplicate courses
            if assignment['course'] in courses_added:
                duplicate_course_penalty += 1
            else:
                courses_added.add(assignment['course'])

            # Check for overlapping classes
            for other in schedule:
                if assignment['day'] == other['day'] and assignment['start_time'] == other['start_time'] and assignment['end_time'] == other['end_time']:
                    overlap_penalty += 1
                    break  # Only need to find one overlap to penalize

            # Add the assignment to the schedule for overlap checking
            schedule.append({
                'day': assignment['day'],
                'start_time': assignment['start_time'],
                'end_time': assignment['end_time']
            })

        # Calculate the penalty for not meeting the total hours
        total_hours = sum(assignment['hours'] for assignment in self.assignments)
        hours_penalty = abs(total_hours - self.max_hours)

        # Calculate total weighted penalty
        total_penalty = (weights['unavailable_day_penalty'] * unavailable_day_penalty +
                         weights['duplicate_course_penalty'] * duplicate_course_penalty +
                         weights['overlap_penalty'] * overlap_penalty +
                         weights['hours_penalty'] * hours_penalty)
        
        print(f"Total Weighted Penalty: {total_penalty}")

        # Calculate fitness score (assuming lower penalty is better)
        fitness_score = 1 / (1 + total_penalty)

        return fitness_score

# Load the new dataset
new_df = pd.read_csv('combinations.csv')

# Extended run_configs with a wider range of GA parameters
run_configs = [
    # For Slow Convergence Simulation
    {'seed': 55, 'teacher_type': 'FT', 'max_hours': 35, 'unavailable_days': ['Tuesday', 'Thursday'], 'population_size': 5, 'crossover_rate': 0.3, 'mutation_rate': 0.5, 'tournament_size': 2, 'subpopulation_id': 0}, 

    # Turn on both to simulate `Sensitivity to Initial Conditions`

    # For Fast Convergence Simulation
    # {'seed': 60, 'teacher_type': 'PT', 'max_hours': 20, 'unavailable_days': ['Monday', 'Wednesday'], 'population_size': 12, 'crossover_rate': 0.65, 'mutation_rate': 0.15, 'tournament_size': 5, 'subpopulation_id': 1},

    # Other
    # {'seed': 65, 'teacher_type': 'FT', 'max_hours': 28, 'unavailable_days': ['Friday'], 'population_size': 8, 'crossover_rate': 0.5, 'mutation_rate': 0.2, 'tournament_size': 6, 'subpopulation_id': 1},
    {'seed': 70, 'teacher_type': 'PT', 'max_hours': 22, 'unavailable_days': ['Monday', 'Friday'], 'population_size': 15, 'crossover_rate': 0.85, 'mutation_rate': 0.05, 'tournament_size': 7, 'subpopulation_id': 0},
    # {'seed': 75, 'teacher_type': 'FT', 'max_hours': 30, 'unavailable_days': ['Wednesday', 'Thursday'], 'population_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'tournament_size': 2, 'subpopulation_id': 1},
    # {'seed': 80, 'teacher_type': 'PT', 'max_hours': 26, 'unavailable_days': ['Tuesday'], 'population_size': 14, 'crossover_rate': 0.7, 'mutation_rate': 0.12, 'tournament_size': 3, 'subpopulation_id': 0},
    # {'seed': 85, 'teacher_type': 'FT', 'max_hours': 32, 'unavailable_days': [], 'population_size': 9, 'crossover_rate': 0.55, 'mutation_rate': 0.18, 'tournament_size': 5, 'subpopulation_id': 1},
    # {'seed': 90, 'teacher_type': 'PT', 'max_hours': 24, 'unavailable_days': ['Sunday', 'Saturday'], 'population_size': 13, 'crossover_rate': 0.8, 'mutation_rate': 0.07, 'tournament_size': 4, 'subpopulation_id': 0},
    # {'seed': 95, 'teacher_type': 'FT', 'max_hours': 29, 'unavailable_days': ['Monday', 'Tuesday', 'Wednesday'], 'population_size': 11, 'crossover_rate': 0.6, 'mutation_rate': 0.1, 'tournament_size': 6, 'subpopulation_id': 1},
    # {'seed': 100, 'teacher_type': 'PT', 'max_hours': 23, 'unavailable_days': ['Thursday', 'Friday'], 'population_size': 16, 'crossover_rate': 0.9, 'mutation_rate': 0.05, 'tournament_size': 2, 'subpopulation_id': 0},
]

# Global list to store threads for each subpopulation
thread_list = []

migration_interval = 10  # Example interval

# Run GA for each subpopulation
for subpop_id in range(nDist):
    for config in run_configs:
        # Create a separate thread for each subpopulation and configuration
        thread = threading.Thread(
            target=subpopulation_thread,
            args=(new_df, config, subpop_id, global_migration_pool)
        )
        thread_list.append(thread)
        thread.start()

# Wait for all threads to complete
for thread in thread_list:
    thread.join()

# Define the target fitness level for comparison
target_fitness_level = 0.9  # This value can be adjusted based on your criteria

# To store the generation at which the target fitness level is first reached
generation_of_target_fitness = []

max_generations = 80

# Initialize collections for results
all_best_fitness_scores = {}
all_diversity_data = {'generation': range(max(config['population_size'] for config in run_configs))}

all_final_populations = []
all_final_population_fitness_scores = []

# Run experiments and collect data in a single loop
for config in run_configs:
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)

    # Run the genetic algorithm
    best_schedule, fitness_data, diversity_data, best_fitness_scores, final_population, final_population_fitness_scores = run_genetic_algorithm(
        config['population_size'], max_generations, new_df, config['teacher_type'], config['max_hours'], config['unavailable_days'],
        config['crossover_rate'], config['mutation_rate'], config['tournament_size'], config['subpopulation_id'], migration_interval, target_fitness_level, generation_of_target_fitness, global_migration_pool
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

    # Append the final population
    all_final_populations.append(best_schedule)

    # Append the final population fitness scores
    all_final_population_fitness_scores.append(final_population_fitness_scores)

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

# Place this just before your plotting code and after the main execution ends.
end_time = time.time()
total_time = end_time - start_time

print(f"Total execution time: {total_time} seconds")

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

# Get MAE and MSE using final population fitness scores for each config
for config, final_population_fitness_scores in zip(run_configs, all_final_population_fitness_scores):
    mae, mse = calculate_mae_mse(final_population_fitness_scores, [1] * len(final_population_fitness_scores))
    print(f"MAE: {mae}, MSE: {mse} for {config}\n\n")



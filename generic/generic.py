import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the new dataset
new_df = pd.read_csv('combinations.csv')

# Initialize the population
def initialize_population(size, df, teacher_type, max_hours, unavailable_days):
    population = []
    for _ in range(size):
        shuffled_df = df.sample(frac=1).to_dict('records')
        assignments = []
        total_hours = 0

        for course in shuffled_df:
            if total_hours + course['hours'] <= max_hours:
                # Remove direct classroom assignment here
                assignments.append(course)
                total_hours += course['hours']
            else:
                break  # Stop adding courses if max hours are reached or exceeded

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

# Mutation
def mutate(schedule, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutated_assignment = new_df.sample(1).to_dict('records')[0]
        # Remove the classroom assignment from mutate
        schedule.assignments[random.randint(0, len(schedule.assignments) - 1)] = mutated_assignment

# Elitism
def elitism(population, top_k=1):
    return sorted(population, key=lambda s: s.fitness(), reverse=True)[:top_k]

# Genetic Algorithm
def run_genetic_algorithm(population_size, max_generations, df, teacher_type, max_hours=None, unavailable_days=None):
    # Initialize population
    population = initialize_population(population_size, df, teacher_type, max_hours, unavailable_days)
    all_fitness_scores = []  # To store fitness scores of all individuals across generations
    diversity_data = []  # To store diversity of each generation
    best_fitness_scores = []  # To store the best fitness score of each generation


    for generation in range(max_generations):
        # Select parents
        parents = tournament_selection(population, tournament_size=5)

        # Generate new population through crossover and mutation
        new_population = []

        # Keep the best individual from the current generation
        while len(new_population) < population_size: # While new population is not full
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

        # Update fitness score with existing schedules
        for individual in new_population:
            individual.fitness_score = individual.fitness()

        # Append fitness scores of all individuals to list
        generation_scores = [{'generation': generation, 'individual_id': idx, 'fitness_score': individual.fitness()} for idx, individual in enumerate(population)]
        all_fitness_scores.extend(generation_scores)

        # Calculate and store diversity for this generation
        diversity = calculate_population_diversity(population)
        diversity_data.append({
            'generation': generation,
            'diversity': diversity
        })

        # Apply elitism - include top individuals from the current generation
        elite = elitism(population)
        new_population.extend(elite)

        # Print existing population fitness in sorted in descending order
        sorted_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
        print(f'Existing population fitness: {[individual.fitness() for individual in sorted_population]}')

        # Replace the old population with the new one
        population = new_population
        sorted_new_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
        # print(f'New population fitness: {[individual.fitness() for individual in sorted_new_population]}')

        # Optional: Print the best fitness of the current generation
        best_fitness = max(individual.fitness() for individual in population)

        best_fitness_scores.append(best_fitness)

        # Print all fitness scores sorted in descending order
        # print(f"\nGeneration {generation}: Best Fitness = {best_fitness}\n")

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

    def fitness(self):
        fitness_score = 1000
        total_hours = 0
        courses_added = set() # To check for duplicate courses

        # Penalize for scheduling on unavailable days and overlapping schedules
        unavailable_days_set = set(self.unavailable_days)

        # Initialize schedule for checking overlapping schedule
        schedule = []

        for assignment in self.assignments:
            total_hours += assignment['hours']

            if assignment['day'] in unavailable_days_set:
                fitness_score -= 500
            
            # Check for duplicate courses
            course_code = assignment['course']

            if course_code in courses_added:
                fitness_score -= 500

            courses_added.add(course_code)

            # Check for overlapping day, start_time, end_time
            for other in schedule:
                if assignment['day'] == other['day'] and assignment['start_time'] == other['start_time'] and assignment['end_time'] == other['end_time']:
                    fitness_score -= 500  # Penalize for overlapping schedule
                    break

            schedule.append({
                'day': assignment['day'],
                'start_time': assignment['start_time'],
                'end_time': assignment['end_time']
            })

        # Penalize when total hours is far from the maximum hours
        fitness_score -= abs(total_hours - self.max_hours) * 10

        return fitness_score

fitness_scores = []

# Define different configurations for runs
run_configs = [
    {'seed': 42, 'teacher_type': 'FT', 'max_hours': 30, 'unavailable_days': ['Saturday', 'Monday'], 'population_size': 10},
    {'seed': 51, 'teacher_type': 'PT', 'max_hours': 25, 'unavailable_days': ['Sunday', 'Friday', 'Wednesday'], 'population_size': 10},
    {'seed': 77, 'teacher_type': 'FT', 'max_hours': 20, 'unavailable_days': ['Saturday', 'Sunday'], 'population_size': 10},
    {'seed': 62, 'teacher_type': 'PT', 'max_hours': 15, 'unavailable_days': ['Saturday', 'Sunday', 'Tuesday'], 'population_size': 10},
]

# Prepare to collect best fitness scores for different configurations
all_best_fitness_scores = {}

all_diversity_data = {'generation': range(max(config['population_size'] for config in run_configs))}

for config in run_configs:
    seed = config['seed']
    teacher_type = config['teacher_type']
    max_hours = config['max_hours']
    unavailable_days = config['unavailable_days']
    population_size = config['population_size']
    max_generations = 100  # Set a common number for all runs

    np.random.seed(seed)
    random.seed(seed)

    best_schedule, fitness_data, diversity_data, best_fitness_scores = run_genetic_algorithm(
        population_size, max_generations, new_df, teacher_type, max_hours, unavailable_days
    )

    existing_schedules = read_or_create_existing_schedules('generic-db.csv')

    # Post-processing
    assign_classrooms(best_schedule, existing_schedules)

    # Save the best schedule
    save_best_schedule(best_schedule, teacher_type, 'generic-db.csv')
    print("Best Schedule Fitness:", best_schedule.fitness())

    # Print the best schedule
    print("\nBest Schedule:")
    for assignment in best_schedule.assignments:
        print(assignment)

    # Print total hours
    total_hours = 0

    for assignment in best_schedule.assignments:
        total_hours += assignment['hours']
    print("\nTotal Hours:", total_hours)

    # Store the diversity data for plotting
    diversity_df = pd.DataFrame(diversity_data)
    config_label = f'seed_{seed}_type_{teacher_type}_hours_{max_hours}'
    all_diversity_data[config_label] = diversity_df.set_index('generation')['diversity'].tolist()
    all_best_fitness_scores[config_label] = best_fitness_scores

# After all runs are complete, combine the diversity data
max_gen_count = max(len(diversity) for diversity in all_diversity_data.values())
all_diversity_data['generation'] = range(max_gen_count)

# Create the DataFrame with all diversity data
all_diversity_df = pd.DataFrame({k: v + [None]*(max_gen_count - len(v)) if len(v) < max_gen_count else v for k, v in all_diversity_data.items()})

# Now all_diversity_df can be saved to CSV or used for plotting directly
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
ax1.legend(title='Configurations', loc='best')

# Plot Best Fitness Scores Across Generations in the second subplot (ax2)
for config, fitness_scores in all_best_fitness_scores.items():
    ax2.plot(range(len(fitness_scores)), fitness_scores, label=config)
ax2.set_title('Best Fitness Scores Across Generations for Different Initial Conditions')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Best Fitness Score')
ax2.legend(title='Configurations', loc='best')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the figure with both subplots
plt.show()

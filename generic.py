import pandas as pd
import random
import matplotlib.pyplot as plt
import os


# Load the new dataset
file_path = 'combinations.csv'  # Replace with the correct path to your CSV file
new_df = pd.read_csv(file_path)

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

        # Append best fitness to list of fitness scores
        fitness_scores.append(best_fitness)

        # Print all fitness scores sorted in descending order
        # print(f"\nGeneration {generation}: Best Fitness = {best_fitness}\n")

    # Return the best solution found
    best_solution = max(population, key=lambda ind: ind.fitness())
    return best_solution

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

# Parameters for the genetic algorithm
population_size = 30
max_generations = 100
teacher_type = 'PT'
max_hours = 14
unavailable_days = ['Saturday', 'Sunday']
fitness_scores = []

# Run the genetic algorithm
best_schedule = run_genetic_algorithm(population_size, max_generations, new_df, teacher_type, max_hours, unavailable_days)

# Existing schedules
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

plt.plot(fitness_scores)
plt.title("Fitness Scores")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
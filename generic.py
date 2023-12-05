import pandas as pd
import random
import matplotlib.pyplot as plt


# Load the new dataset
file_path = 'combinations.csv'  # Replace with the correct path to your CSV file
new_df = pd.read_csv(file_path)

def initialize_population(size, df, teacher_type, max_hours, unavailable_days):
    population = []
    for _ in range(size):
        # Sort the dataframe to have different course options
        shuffled_df = df.sample(frac=1).to_dict('records')
        assignments = []
        total_hours = 0

        for course in shuffled_df:
            if total_hours + course['hours'] <= max_hours:
                assignments.append(course)
                total_hours += course['hours']
            else:
                break  # Stop adding courses if max hours are reached or exceeded

        population.append(Schedule(assignments, teacher_type, '1', max_hours, unavailable_days))

    return population

def tournament_selection(population, tournament_size=3):
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best_in_tournament = max(tournament, key=lambda ind: ind.fitness())
        parents.append(best_in_tournament)
    return parents

# Crossover
def crossover(parent1, parent2):
    child_assignments = []

    # Choose a random point to split the schedules
    split_point = random.randint(0, len(parent1.assignments) - 1)

    # Add the first part of the first parent to the child
    child_assignments.extend(parent1.assignments[:split_point])

    # Add the second part of the second parent to the child
    child_assignments.extend(parent2.assignments[split_point:])

    # Return the child schedule
    return Schedule(child_assignments, parent1.teacher_type, '1', parent1.max_hours, parent1.unavailable_days)

def mutate(schedule, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutated_assignment = new_df.sample(1).to_dict('records')[0]
        schedule.assignments[random.randint(0, len(schedule.assignments) - 1)] = mutated_assignment

def elitism(population, top_k=1):
    return sorted(population, key=lambda s: s.fitness(), reverse=True)[:top_k]

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

class Schedule:
    def __init__(self, assignments, teacher_type, ga_type, max_hours=None, unavailable_days=None):
        self.assignments = assignments
        self.teacher_type = teacher_type
        self.ga_type = ga_type
        self.max_hours = max_hours
        self.unavailable_days = unavailable_days or []

    def fitness(self):
        # Initialize the fitness score
        fitness_score = 1000

        # Initialize total hours
        total_hours = 0

        for assignment in self.assignments:
            total_hours += assignment['hours']

        # Penalize schedules that exceed maximum hours
        if total_hours > self.max_hours:
            # Print
            # print('Penalized for exceeding maximum hours')
            fitness_score -= 1000

        # Penalize for scheduling on unavailable days
        for assignment in self.assignments:
            if assignment['day'] in self.unavailable_days:
                # Print
                # print('Penalized for scheduling on unavailable days')
                fitness_score -= 500  # Adjust penalty as needed

        # Penalize for overlapping schedules
        for i in range(len(self.assignments)):
            for j in range(i + 1, len(self.assignments)):
                if self.assignments[i]['day'] == self.assignments[j]['day'] and self.is_overlapping(self.assignments[i], self.assignments[j]):
                    fitness_score -= 1000  # Penalty for each overlapping schedule

        return fitness_score

    # Other methods of the Schedule class can be added here as needed
    @staticmethod
    def is_overlapping(assignment1, assignment2):
        start_time1, end_time1 = Schedule.convert_to_minutes(assignment1['start_time']), Schedule.convert_to_minutes(assignment1['end_time'])
        start_time2, end_time2 = Schedule.convert_to_minutes(assignment2['start_time']), Schedule.convert_to_minutes(assignment2['end_time'])

        return max(start_time1, start_time2) < min(end_time1, end_time2)

    @staticmethod
    def convert_to_minutes(time_str):
        # Split time into components
        time_part, meridiem = time_str.split()
        hours, minutes = map(int, time_part.split(':'))

        # Convert to 24-hour format if it's PM
        if meridiem == 'PM' and hours < 12:
            hours += 12
        elif meridiem == 'AM' and hours == 12:
            hours = 0

        return hours * 60 + minutes

# Parameters for the genetic algorithm
population_size = 50
max_generations = 300
teacher_type = 'FT'
max_hours = 5
unavailable_days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Friday']
fitness_scores = []

# Run the genetic algorithm
best_schedule = run_genetic_algorithm(population_size, max_generations, new_df, teacher_type, max_hours, unavailable_days)
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
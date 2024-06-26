import random
from datetime import datetime, timedelta
from multiprocessing import Process, Manager, Event
import matplotlib.pyplot as plt
import time
import numpy as np

# ------------------------ Helper Functions ------------------------

# Function to randomly assign course durations, ensuring a balanced distribution of 1, 2, and 3-hour courses
def balanced_course_duration_assignment(courses_units):
    duration_choices = [1, 2, 3]  # Allowed durations
    assigned_durations = {course: random.choice(duration_choices) for course in courses_units.keys()}
    return assigned_durations

def adjust_time_slot_selection(course_duration, time_slots):
    """Select a start time slot that can accommodate the course duration."""
    # Convert course duration to a number of 30-minute slots
    num_slots_needed = course_duration * 2
    valid_start_slots = [slot for slot in time_slots if time_slots.index(slot) <= len(time_slots) - num_slots_needed]

    if not valid_start_slots:
        return None  # If no slot can accommodate the duration, return None

    start_slot = random.choice(valid_start_slots)
    start_index = time_slots.index(start_slot)
    end_index = start_index + num_slots_needed - 1
    return f"{time_slots[start_index].split(' - ')[0]} - {time_slots[end_index].split(' - ')[1]}"

# ------------------------ Initialization Functions ------------------------

def logistic_map(x, r=4.0):
    """Logistic map function for chaotic mapping."""
    return r * x * (1 - x)

def initialize_population(population_size, faculty_data, courses_units, courses, rooms, days, time_slots):
    population = []
    x0 = random.uniform(0, 1)  # Initial value for chaotic sequence

    for _ in range(population_size):
        chromosome = []
        x = x0  # Reset to initial value for each chromosome

        for faculty in faculty_data:
            assigned_courses_with_details = []
            remaining_units = faculty['max_units']
            shuffled_courses = random.sample(courses, len(courses))

            for course in shuffled_courses:
                x = logistic_map(x)  # Update x using chaotic mapping
                if courses_units[course] <= remaining_units and x > 0.5:  # Use chaotic condition
                    room = random.choice(rooms)
                    available_days = faculty['availability']
                    day = random.choice(available_days)
                    adjusted_time_slot = adjust_time_slot_selection(courses_units[course], time_slots)
                    if adjusted_time_slot:
                        assigned_courses_with_details.append((course, room, day, adjusted_time_slot))
                        remaining_units -= courses_units[course]

            chromosome.append({
                'id': faculty['id'],
                'max_units': faculty['max_units'],
                'total_assigned_units': faculty['max_units'] - remaining_units,
                'assigned_courses_with_details': assigned_courses_with_details
            })
        population.append(chromosome)
    return population


def print_population(population, faculty_data):
    for chromosome_index, chromosome in enumerate(population, start=1):
        print(f"Chromosome {chromosome_index}:")
        for faculty_schedule in chromosome:
            faculty_id = faculty_schedule['id']
            # Extract additional faculty details from the faculty_data
            faculty_details = next((faculty for faculty in faculty_data if faculty['id'] == faculty_id), None)
            if faculty_details:
                availability = ", ".join(faculty_details['availability'])
                is_part_time = "Yes" if faculty_details['is_part_time'] else "No"
                consulting_hours = ", ".join([" - ".join(times) for times in faculty_details['consulting_hours']])

                print(f"  Faculty ID: {faculty_schedule['id']}, Max Units: {faculty_schedule['max_units']}, Total Assigned Units: {faculty_schedule['total_assigned_units']}")
                print(f"    Availability: {availability}")
                print(f"    Part-Time: {is_part_time}")
                print(f"    Consulting Hours: {consulting_hours}")
                for course_detail in faculty_schedule['assigned_courses_with_details']:
                    course, room, day, time_slot = course_detail
                    print(f"      - Course: {course}, Room: {room}, Day: {day}, Time Slot: {time_slot}")
                print("  " + "-" * 50)  # Separator for readability within a chromosome
        print("\n" + "=" * 60)  # Separator between chromosomes


# ------------------------ Fitness Functions ------------------------

def calculate_fitness(chromosome, faculty_data, availability_penalty, overlap_penalty, overload_penalty, consulting_conflict_penalty, max_hours_per_day=4):
    """
    Calculate the fitness for a single chromosome, incorporating penalties for:
    - Assigning courses outside of faculty availability.
    - Overlaps in room assignments (same room, same time, same day).
    - Daily overloads exceeding a specified number of teaching hours.
    - Scheduling courses during faculty consulting hours.

    Returns a fitness score for the chromosome, where a lower score indicates a worse solution.
    """
    penalty = 0

    # Penalty for assigning courses outside of faculty availability
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        faculty_availability = [faculty['availability'] for faculty in faculty_data if faculty['id'] == faculty_id][0]

        for course_detail in faculty_schedule['assigned_courses_with_details']:
            course_day = course_detail[2]
            if course_day not in faculty_availability:
                penalty += availability_penalty  # Use the passed penalty value

    # Penalty for overlaps
    overlaps = check_for_overlaps(chromosome)
    penalty += len(overlaps) * overlap_penalty  # Use the passed penalty value

    # Penalty for daily overloads
    daily_overloads = check_for_daily_overloads(chromosome, max_hours_per_day)
    penalty += len(daily_overloads) * overload_penalty  # Use the passed penalty value

    # Penalty for consulting hour conflicts
    consulting_conflicts = check_for_consulting_hour_conflicts(chromosome, faculty_data)
    penalty += len(consulting_conflicts) * consulting_conflict_penalty  # Use the passed penalty value

    # Fitness score calculation
    fitness_score = -penalty  # Using negative score because lower (more negative) is worse

    return fitness_score


def calculate_fitness_detailed(chromosome, faculty_data, max_hours_per_day=4):
    """
    Calculate the fitness for a single chromosome, incorporating penalties for various constraints,
    and provide a detailed breakdown of the penalties and the specific solutions that got penalized.
    """
    penalties = {
        'availability_violations': [],
        'room_overlaps': [],
        'daily_overloads': [],
        'consulting_hour_conflicts': [],
        'lab_subjects': []
    }
    
    PENALTY_VALUES = {
        'availability': 10,
        'overlap': 10,
        'overload': 10,
        'consulting_conflict': 10,
        'lab_subject': 0
    }
    
    # Penalty for assigning courses outside of faculty availability
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        faculty_availability = [faculty['availability'] for faculty in faculty_data if faculty['id'] == faculty_id][0]
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            course_day = course_detail[2]
            if course_day not in faculty_availability:
                penalties['availability_violations'].append({
                    'faculty_id': faculty_id,
                    'course': course_detail[0],
                    'day': course_day
                })
    
    # Penalty for overlaps
    overlaps = check_for_overlaps(chromosome)
    penalties['room_overlaps'].extend(overlaps)
    
    # Penalty for daily overloads
    daily_overloads = check_for_daily_overloads(chromosome, max_hours_per_day)
    penalties['daily_overloads'].extend(daily_overloads)
    
    # Penalty for consulting hour conflicts
    consulting_conflicts = check_for_consulting_hour_conflicts(chromosome, faculty_data)
    penalties['consulting_hour_conflicts'].extend(consulting_conflicts)
    
    # Penalty for (Lab) subjects
    for faculty_schedule in chromosome:
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            course = course_detail[0]
            if '(Lab)' in course:
                penalties['lab_subjects'].append({
                    'faculty_id': faculty_schedule['id'],
                    'course': course
                })
    
    total_penalty = (
        len(penalties['availability_violations']) * PENALTY_VALUES['availability'] +
        len(penalties['room_overlaps']) * PENALTY_VALUES['overlap'] +
        len(penalties['daily_overloads']) * PENALTY_VALUES['overload'] +
        len(penalties['consulting_hour_conflicts']) * PENALTY_VALUES['consulting_conflict'] +
        len(penalties['lab_subjects']) * PENALTY_VALUES['lab_subject']
    )
    
    fitness_score = -total_penalty
    
    return fitness_score, penalties


# ------------------------ Selection Functions ------------------------

def rank_selection(population, faculty_data, availability_penalty, overlap_penalty, overload_penalty, consulting_conflict_penalty):
    """
    Selects two parents using rank selection based on fitness.

    Parameters:
        population (list): The population from which to select parents.
        faculty_data (list): Faculty data for fitness calculation.

    Returns:
        tuple: The top two chromosomes based on fitness.
    """
    # Calculate fitness for each chromosome in the population
    population_with_fitness = [
        (
            chromosome,
            calculate_fitness(
                chromosome,
                faculty_data,
                availability_penalty,
                overlap_penalty,
                overload_penalty,
                consulting_conflict_penalty,
            ),  # Pass penalty values
        )
        for chromosome in population
    ]
    # Sort the population based on fitness in descending order (higher fitness is better)
    sorted_population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
    # Select the top two chromosomes as parents
    parent1, parent2 = sorted_population[0][0], sorted_population[1][0]
    return parent1, parent2, sorted_population[0][1], sorted_population[1][1]

def tournament_selection(population, faculty_data, tournament_size=3):
    """
    Selects two parents using tournament selection.

    Parameters:
        population (list): The population from which to select parents.
        faculty_data (list): Faculty data for fitness calculation.
        tournament_size (int): The number of individuals participating in each tournament.

    Returns:
        tuple: Two selected parent chromosomes along with their fitness scores.
    """
    # Ensure the tournament size does not exceed the population size
    tournament_size = min(tournament_size, len(population))

    # Randomly select individuals for the tournament
    tournament_individuals = random.sample(population, tournament_size)

    # Calculate fitness for each selected individual
    tournament_with_fitness = [(individual, calculate_fitness(individual, faculty_data)) for individual in tournament_individuals]

    # Sort the selected individuals based on fitness in descending order (higher fitness is better)
    sorted_tournament = sorted(tournament_with_fitness, key=lambda x: x[1], reverse=True)

    # Select the top two individuals from the tournament as parents
    parent1, parent2 = sorted_tournament[0][0], sorted_tournament[1][0]
    fitness1, fitness2 = sorted_tournament[0][1], sorted_tournament[1][1]

    return parent1, parent2, fitness1, fitness2

def print_chromosome_details(chromosome):
    """
    Prints the details of a chromosome.

    Parameters:
        chromosome (dict): The chromosome to print.
    """
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        print(f"  Faculty ID: {faculty_id}, Max Units: {faculty_schedule['max_units']}, Total Assigned Units: {faculty_schedule['total_assigned_units']}")
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            print(f"    - Course: {course_detail[0]}, Room: {course_detail[1]}, Day: {course_detail[2]}, Time Slot: {course_detail[3]}")

def print_selected_parents_with_fitness(parent1, parent2, fitness1, fitness2):
    """
    Prints the selected parents and their fitness in a `ructured format.

    Parameters:
        parent1, parent2 (dict): The selected parent chromosomes.
        fitness1, fitness2 (float): The fitness scores of the selected parents.
    """
    print("Selected Parent 1 (Fitness: {:.2f}):".format(fitness1))
    # print_chromosome_details(parent1)
    print("Selected Parent 2 (Fitness: {:.2f}):".format(fitness2))
    # print_chromosome_details(parent2)

# ------------------------ Crossover Functions ------------------------

def crossover(parent1, parent2, faculty_data):
    """
    Performs a crossover between two parent chromosomes to generate two offspring,
    with awareness of faculty consulting hours to avoid scheduling conflicts,
    and room overlap checks to avoid double-booking of rooms.

    Parameters:
        parent1 (list): The first parent chromosome.
        parent2 (list): The second parent chromosome.
        faculty_data (list): List containing faculty availability and consulting hours.

    Returns:
        tuple: Two new chromosomes (offspring) resulting from the crossover,
               avoiding consulting hour conflicts as much as possible.
    """
    # Clone parents to create offspring that are initially copies of the parents
    offspring1 = [faculty.copy() for faculty in parent1]
    offspring2 = [faculty.copy() for faculty in parent2]

    # Determine crossover points randomly
    cross_points = random.sample(range(len(parent1)), 2)
    cross_point1, cross_point2 = min(cross_points), max(cross_points)

    # Perform the swap between the two cross points for each offspring
    for i in range(cross_point1, cross_point2 + 1):
        # Before swapping, check if the swap would cause consulting hour conflicts
        if not will_cause_consulting_conflict(offspring1[i], faculty_data) and not will_cause_consulting_conflict(offspring2[i], faculty_data):
            # Swap assigned courses, rooms, days, and time slots between parents if no conflicts
            offspring1[i]['assigned_courses_with_details'], offspring2[i]['assigned_courses_with_details'] = \
                offspring2[i]['assigned_courses_with_details'], offspring1[i]['assigned_courses_with_details']

    # Perform the swap between the two cross points for each offspring, with additional room overlap checks
    for i in range(cross_point1, cross_point2 + 1):
        if not will_cause_room_overlap(offspring1[i], offspring2[i], offspring1 + offspring2):
            offspring1[i]['assigned_courses_with_details'], offspring2[i]['assigned_courses_with_details'] = \
                offspring2[i]['assigned_courses_with_details'], offspring1[i]['assigned_courses_with_details']

    return offspring1, offspring2

def will_cause_consulting_conflict(faculty_schedule, faculty_data):
    """
    Checks if assigning courses from one parent to another would cause a conflict
    with the faculty's consulting hours.

    Parameters:
        faculty_schedule (dict): The faculty schedule to check.
        faculty_data (list): List containing faculty consulting hours.

    Returns:
        bool: True if the swap will cause a consulting hour conflict, False otherwise.
    """
    faculty_id = faculty_schedule['id']
    faculty_info = next((f for f in faculty_data if f['id'] == faculty_id), None)
    if not faculty_info:
        return False  # Proceed with the swap if faculty info is not found

    consulting_hours = faculty_info['consulting_hours']
    for course_detail in faculty_schedule['assigned_courses_with_details']:
        _, _, course_day, course_time_slot = course_detail
        for consulting_day, consulting_slot in consulting_hours:
            if course_day == consulting_day and times_overlap(course_time_slot, consulting_slot):
                return True  # Conflict detected

    return False  # No conflict detected

def will_cause_room_overlap(faculty_schedule1, faculty_schedule2, full_population):
    """
    Checks if swapping assigned courses between two faculty schedules would cause a room overlap
    within the entire population's schedules.

    Parameters:
        faculty_schedule1 (dict): The faculty schedule to potentially swap from.
        faculty_schedule2 (dict): The faculty schedule to potentially swap to.
        full_population (list): The full population of schedules to check for potential room overlaps.

    Returns:
        bool: True if swapping would cause a room overlap, False otherwise.
    """
    # Combine details for a hypothetical swap scenario
    combined_details = faculty_schedule1['assigned_courses_with_details'] + faculty_schedule2['assigned_courses_with_details']

    # Check for overlaps in the combined details against all schedules
    for schedule in full_population:
        for detail1 in combined_details:
            for detail2 in schedule['assigned_courses_with_details']:
                if detail1 != detail2:  # Avoid comparing the course to itself
                    if detail1[1] == detail2[1] and detail1[2] == detail2[2] and times_overlap(detail1[3], detail2[3]):
                        return True  # Overlap detected

    return False  # No overlap detected

def times_overlap(time_slot1, time_slot2):
    """
    Determines if two time slots overlap.

    Parameters:
        time_slot1 (str): The first time slot in the format 'HH:MM AM/PM - HH:MM AM/PM'.
        time_slot2 (str): The second time slot in the same format.

    Returns:
        bool: True if the time slots overlap, False otherwise.
    """
    start1, end1 = [datetime.strptime(time, "%I:%M %p") for time in time_slot1.split(' - ')]
    start2, end2 = [datetime.strptime(time, "%I:%M %p") for time in time_slot2.split(' - ')]
    return max(start1, start2) < min(end1, end2)

# ------------------------ Mutation Functions ------------------------

def mutate(chromosome, population, rooms, days, time_slots, faculty_data, F=0.5):
    """
    Objective 1: Adaptive mutation strategy inspired by Differential Evolution (DE) to apply changes to a single chromosome within the context of a scheduling problem.
    Applies a mutation inspired by Differential Evolution (DE) to a single chromosome within the context of a scheduling problem.
    The mutation randomly targets aspects of course assignments: room, day, or time slot, and applies changes based on the DE strategy,
    using differences between other randomly selected chromosomes to guide the mutation.
    Parameters:
    - chromosome: The individual chromosome to mutate, representing a faculty schedule.
    - population: The current population from which to select individuals for DE operations.
    - rooms, days, time_slots: Lists of available rooms, days, and time slots for scheduling.
    - faculty_data: Data containing faculty preferences and constraints.
    - F: DE scaling factor controlling the intensity of mutations.
    Returns:
    - Mutated chromosome with potentially altered course assignments.
    """
    # Ensure there are enough individuals for DE operation, excluding the current chromosome
    eligible_population = [ind for ind in population if ind != chromosome]
    if len(eligible_population) < 3:
        print("Not enough unique chromosomes for DE-inspired mutation. Skipping mutation for this chromosome.")
        return chromosome  # Return the chromosome unchanged
    # Safe to proceed with sampling
    r1, r2, r3 = random.sample(eligible_population, 3)

    for faculty_index, faculty_schedule in enumerate(chromosome):
        # Protect against index errors when faculty_index exceeds the length of r1, r2, or r3
        if faculty_index >= len(r1) or faculty_index >= len(r2) or faculty_index >= len(r3):
            continue

        for course_detail_index, course_detail in enumerate(faculty_schedule['assigned_courses_with_details']):
            # Select a mutation aspect randomly
            mutation_choice = random.choice(['room', 'day', 'time_slot'])

            if mutation_choice == 'room':
                # DE-inspired mutation for room, considering non-overlapping constraints
                new_room = random.choice([room for room in rooms if room != course_detail[1]])
                if not causes_room_overlap(faculty_schedule['assigned_courses_with_details'], course_detail_index, new_room, chromosome):
                    course_detail = (course_detail[0], new_room, course_detail[2], course_detail[3])

            elif mutation_choice == 'day':
                # Get available days for mutation, excluding the current day
                available_days = [day for day in days if day != course_detail[2] and day in faculty_data[faculty_index]['availability']]

                if available_days:  # Check if there are valid days for mutation
                    new_day = random.choice(available_days)
                    course_detail = (course_detail[0], course_detail[1], new_day, course_detail[3])
                else:
                    # Handle the case where no valid days are found
                    print(f"No valid days for mutation found for faculty {faculty_index}, course {course_detail[0]}. Skipping day mutation.")
                    # You can choose alternative actions here, such as:
                    # - Trying a different mutation aspect (room or time slot)
                    # - Skipping the mutation for this course entirely

            elif mutation_choice == 'time_slot':
                # DE-inspired mutation for time slot, ensuring no overlap with consulting hours
                new_time_slot = random.choice([ts for ts in time_slots if ts != course_detail[3]])
                if not conflicts_with_consulting_hours(course_detail[2], new_time_slot, faculty_data[faculty_index]['consulting_hours']):
                    course_detail = (course_detail[0], course_detail[1], course_detail[2], new_time_slot)

            # Safely update the course detail after mutation
            faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail

    return chromosome

# def mutate(chromosome, population, rooms, days, time_slots, faculty_data, F=0.5):
#     """
#     Applies a mutation inspired by Differential Evolution (DE) to a single chromosome within the context of a scheduling problem.
#     The mutation prioritizes mutating aspects causing conflicts or constraints.

#     Parameters:
#     - chromosome: The individual chromosome to mutate, representing a faculty schedule.
#     - population: The current population from which to select individuals for DE operations.
#     - rooms, days, time_slots: Lists of available rooms, days, and time slots for scheduling.
#     - faculty_data: Data containing faculty preferences and constraints.
#     - F: DE scaling factor controlling the intensity of mutations.

#     Returns:
#     - Mutated chromosome with potentially altered course assignments.
#     """

#     # Ensure there are enough individuals for DE operation, excluding the current chromosome
#     eligible_population = [ind for ind in population if ind != chromosome]
#     if len(eligible_population) < 3:
#         print("Not enough unique chromosomes for DE-inspired mutation. Skipping mutation for this chromosome.")
#         return chromosome  # Return the chromosome unchanged

#     # Safe to proceed with sampling
#     r1, r2, r3 = random.sample(eligible_population, 3)

#     # Mutation logic
#     for faculty_index, faculty_schedule in enumerate(chromosome):
#         # Protect against index errors when faculty_index exceeds the length of r1, r2, or r3
#         if faculty_index >= len(r1) or faculty_index >= len(r2) or faculty_index >= len(r3):
#             continue

#         for course_detail_index, course_detail in enumerate(faculty_schedule['assigned_courses_with_details']):
#             # Check for conflicts or constraints
#             room_conflict = causes_room_overlap(faculty_schedule['assigned_courses_with_details'], course_detail_index, course_detail[1], chromosome)
#             day_conflict = course_detail[2] not in faculty_data[faculty_index]['availability']
#             time_conflict = conflicts_with_consulting_hours(course_detail[2], course_detail[3], faculty_data[faculty_index]['consulting_hours'])

#             # Prioritize mutating aspects causing conflicts or constraints
#             if room_conflict:
#                 new_room = random.choice([room for room in rooms if room != course_detail[1]])
#                 course_detail = (course_detail[0], new_room, course_detail[2], course_detail[3])
#             elif day_conflict:
#                 new_day = random.choice([day for day in days if day != course_detail[2] and day in faculty_data[faculty_index]['availability']])
#                 course_detail = (course_detail[0], course_detail[1], new_day, course_detail[3])
#             elif time_conflict:
#                 new_time_slot = random.choice([ts for ts in time_slots if ts != course_detail[3]])
#                 if not conflicts_with_consulting_hours(course_detail[2], new_time_slot, faculty_data[faculty_index]['consulting_hours']):
#                     course_detail = (course_detail[0], course_detail[1], course_detail[2], new_time_slot)

#             # Safely update the course detail after mutation
#             faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail

#     return chromosome




def conflicts_with_consulting_hours(day, time_slot, consulting_hours):
    """
    Checks if a given day and time slot conflicts with any of the consulting hours.

    Parameters:
        day (str): The day of the course.
        time_slot (str): The time slot of the course.
        consulting_hours (list): List of consulting hour tuples (day, time_slot) for a faculty.

    Returns:
        bool: True if there is a conflict, False otherwise.
    """
    for consulting_day, consulting_time_slot in consulting_hours:
        if day == consulting_day and times_overlap(time_slot, consulting_time_slot):
            return True
    return False

def causes_room_overlap(course_details, index, new_room, chromosome):
    """
    Checks if assigning a new room to a course results in an overlap with other courses in the same room.

    Parameters:
        course_details (list): List of course details for a faculty.
        index (int): Index of the course being mutated.
        new_room (str): The new room being assigned to the course.
        chromosome (list): The entire chromosome to check for potential room overlaps.

    Returns:
        bool: True if changing to the new room causes an overlap, False otherwise.
    """
    day, time_slot = course_details[index][2], course_details[index][3]
    for faculty_schedule in chromosome:
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            if course_detail[1] == new_room and course_detail[2] == day and times_overlap(course_detail[3], time_slot):
                return True  # Overlap detected
    return False  # No overlap detected

def calculate_population_diversity(population):
    total_assigned_units = [faculty['total_assigned_units'] for chromosome in population for faculty in chromosome]
    diversity = np.var(total_assigned_units)
    return diversity

# ------------------------ Elitism Functions ------------------------

def select_elites(population, faculty_data, availability_penalty, overlap_penalty, overload_penalty, consulting_conflict_penalty, n_elites=2):
    """
    Selects the top n_elites chromosomes from the population based on their fitness scores.
    """
    # Calculate fitness for each chromosome in the population
    population_with_fitness = [
        (
            chromosome,
            calculate_fitness(
                chromosome,
                faculty_data,
                availability_penalty,
                overlap_penalty,
                overload_penalty,
                consulting_conflict_penalty,
            ), 
        )
        for chromosome in population
    ]
    # Sort the population based on fitness in descending order (higher fitness is better)
    sorted_population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
    # Select the top n_elites chromosomes as elites
    elites = [item[0] for item in sorted_population[:n_elites]]
    elite_fitness_scores = [item[1] for item in sorted_population[:n_elites]]
    return elites, elite_fitness_scores

def print_elites_with_fitness(elites, elite_fitness_scores):
    """
    Prints the elite chromosomes and their fitness in a structured format.

    Parameters:
        elites (list): The selected elite chromosomes.
        elite_fitness_scores (list): The fitness scores of the selected elites.
    """
    for index, (elite, fitness) in enumerate(zip(elites, elite_fitness_scores), start=1):
        print(f"Elite {index} (Fitness: {fitness:.2f}):")
        # print_chromosome_details(elite)

# ------------------------------ Migration Functions ------------------------------

def migrate_selected_individuals_between_islands(islands, num_migrants=1, migration_rate=0.1):
    """
    Migrates a specified number of individuals between islands based on a migration rate, allowing for selective genetic material exchange across different populations. This function enhances genetic diversity across the islands, potentially leading to better solutions in a distributed genetic algorithm framework.

    Parameters:
        islands (dict): A dictionary where keys are island names and values are the populations (lists of chromosomes) of each island.
        num_migrants (int): The number of individuals (chromosomes) to migrate from each island. Defaults to 1.
        migration_rate (float): The probability of migration happening between each pair of islands. Defaults to 0.1.

    Each island will send out a selected number of its individuals to the next island in sequence, determined by the migration rate. If the migration occurs (based on the migration rate), the selected individuals are exchanged in a circular pattern among all islands. The function prints details of the migration process, including which chromosomes are being migrated and between which islands.
    """
    island_names = list(islands.keys())
    num_islands = len(island_names)

    # Perform migration based on the migration rate
    if random.random() < migration_rate:
        for i, island_name in enumerate(island_names):
            next_island = island_names[(i + 1) % num_islands]

            # Selecting migrants
            migrants_from_current = random.sample(islands[island_name], min(num_migrants, len(islands[island_name])))
            migrants_from_next = random.sample(islands[next_island], min(num_migrants, len(islands[next_island])))

            # Exchanging migrants
            for migrant in migrants_from_current:
                islands[island_name].remove(migrant)
                islands[next_island].append(migrant)

            for migrant in migrants_from_next:
                islands[next_island].remove(migrant)
                islands[island_name].append(migrant)

            print(f"Migrated {num_migrants} individuals between {island_name} and {next_island}.")
    else:
        print("Migration did not occur this generation due to migration rate threshold.")

# ------------------------ Diversity Functions ------------------------

def calculate_diversity(population):
    """
    Calculates the diversity of the population based on the uniqueness of course-day-time slot-room combinations.

    Parameters:
        population (list): The current population of chromosomes.

    Returns:
        int: The diversity metric.
    """
    unique_assignments = set()
    for chromosome in population:
        for faculty_schedule in chromosome:
            for course_detail in faculty_schedule['assigned_courses_with_details']:
                unique_assignments.add((course_detail[0], course_detail[2], course_detail[3], course_detail[1]))  # (course, day, time_slot, room)
    return len(unique_assignments)

# ------------------------ Checker + Printing Functions ------------------------

def check_for_overlaps(chromosome):
    # Extend the assignment details to include faculty ID for tracking overlaps between faculties
    assignment_details = []
    for faculty_schedule in chromosome:
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            course, room, day, time_slot = course_detail
            assignment_details.append({
                'faculty_id': faculty_schedule['id'],
                'course': course,
                'room': room,
                'day': day,
                'time_slot': time_slot
            })

    # Check for overlaps, including the faculty ID in the overlap information
    overlaps = []
    for i, assignment in enumerate(assignment_details):
        for other_assignment in assignment_details[i+1:]:
            if (assignment['day'] == other_assignment['day'] and
                assignment['room'] == other_assignment['room'] and
                assignment['time_slot'] == other_assignment['time_slot']):
                overlaps.append({
                    'faculty1': assignment['faculty_id'],
                    'faculty2': other_assignment['faculty_id'],
                    'details': (assignment, other_assignment)
                })

    return overlaps

def print_overlaps(overlaps):
    if not overlaps:
        print("  No overlaps detected.")
    else:
        print("  Overlaps detected:")
        for overlap in overlaps:
            faculty1, faculty2 = overlap['faculty1'], overlap['faculty2']
            assignment1, assignment2 = overlap['details']
            if faculty1 == faculty2:
                print(f"    - {faculty1} has overlapping schedules:")
            else:
                print(f"    - {faculty1} and {faculty2} have overlapping schedules:")
            print(f"      Course {assignment1['course']} and Course {assignment2['course']} overlap in Room {assignment1['room']}, on {assignment1['day']} during {assignment1['time_slot']}.")

# --

def check_for_daily_overloads(chromosome, max_hours_per_day=4):
    """
    Checks for daily overloads where a faculty is scheduled to teach more than `max_hours_per_day`.
    Returns a list of overload incidents, including faculty ID and the day overloaded.
    """
    daily_overloads = []

    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        # Initialize a dictionary to track daily teaching hours
        daily_hours = {day: 0 for day in days}

        for course_detail in faculty_schedule['assigned_courses_with_details']:
            _, _, day, time_slot = course_detail
            start_time_str, end_time_str = time_slot.split(' - ')
            start_time = datetime.strptime(start_time_str, '%I:%M %p')
            end_time = datetime.strptime(end_time_str, '%I:%M %p')
            # Calculate the duration in hours
            duration_hours = (end_time - start_time).seconds / 3600
            daily_hours[day] += duration_hours

        # Check for overloads
        for day, hours in daily_hours.items():
            if hours > max_hours_per_day:
                daily_overloads.append({
                    'faculty_id': faculty_id,
                    'day': day,
                    'hours': hours
                })

    return daily_overloads

def print_daily_overloads(daily_overloads):
    if not daily_overloads:
        print("  No daily overloads detected.")
    else:
        print("  Daily overloads detected:")
        for overload in daily_overloads:
            print(f"    - Faculty {overload['faculty_id']} is overloaded on {overload['day']}, scheduled for {overload['hours']:.2f} hours.")

# --

def check_for_consulting_hour_conflicts(chromosome, faculty_data):
    conflicts = []
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        faculty_info = next((f for f in faculty_data if f['id'] == faculty_id), None)
        if not faculty_info:
            continue  # Skip if faculty not found in data

        consulting_hours = faculty_info['consulting_hours']
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            _, _, course_day, course_time_slot = course_detail
            for consulting_time in consulting_hours:
                consulting_day, consulting_slot = consulting_time
                if course_day == consulting_day and times_overlap(course_time_slot, consulting_slot):
                    conflicts.append({
                        'faculty_id': faculty_id,
                        'course_time_slot': course_time_slot,
                        'consulting_time_slot': consulting_slot
                    })
    return conflicts

def times_overlap(course_slot, consulting_slot):
    course_start, course_end = [datetime.strptime(time, "%I:%M %p") for time in course_slot.split(' - ')]
    consulting_start, consulting_end = [datetime.strptime(time, "%I:%M %p") for time in consulting_slot.split(' - ')]
    return max(course_start, consulting_start) < min(course_end, consulting_end)

def print_consulting_hour_conflicts(conflicts):
    if not conflicts:
        print("  No consulting hour conflicts detected.")
    else:
        print("  Consulting hour conflicts detected:")
        for conflict in conflicts:
            print(f"    - Faculty {conflict['faculty_id']} has a teaching slot at {conflict['course_time_slot']} conflicting with consulting hours at {conflict['consulting_time_slot']}.")

# ------------------------ Main Evolutionary Loop ------------------------

NUM_GENERATIONS = 100  # Number of generations to evolve the population
MUTATION_RATE = 0.1  # Mutation rate
POPULATION_SIZE = 20
NUM_ISLANDS = 2
CHROMOSOMES_PER_ISLAND = POPULATION_SIZE // NUM_ISLANDS  # 20 chromosomes per island

faculty_data = [
    {'id': 'faculty1', 'availability': ['Tuesday', 'Thursday'], 'max_units': 16, 'is_part_time': False, 'consulting_hours': [('Tuesday', '01:00 PM - 02:00 PM'), ('Thursday', '01:00 PM - 02:00 PM')]},
    {'id': 'faculty2', 'availability': ['Monday', 'Wednesday'], 'max_units': 12, 'is_part_time': True, 'consulting_hours': [('Monday', '10:00 AM - 11:00 AM')]},
    {'id': 'faculty3', 'availability': ['Wednesday', 'Friday'], 'max_units': 18, 'is_part_time': False, 'consulting_hours': [('Friday', '03:00 PM - 04:00 PM')]},
    {'id': 'faculty4', 'availability': ['Tuesday', 'Thursday', 'Friday'], 'max_units': 20, 'is_part_time': True, 'consulting_hours': [('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]},
    {'id': 'faculty5', 'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'max_units': 20, 'is_part_time': False, 'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]},
    {
        'id': 'faculty6',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty7',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty8',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty9',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty10',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty11',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty12',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty13',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty14',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty15',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty16',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty17',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty18',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty19',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty20',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty21',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty22',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    },
    {
        'id': 'faculty23',
        'availability': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'max_units': 20,
        'is_part_time': False,
        'consulting_hours': [('Monday', '10:00 AM - 11:00 AM'), ('Tuesday', '10:00 AM - 11:00 AM'), ('Wednesday', '10:00 AM - 11:00 AM'), ('Thursday', '10:00 AM - 11:00 AM'), ('Friday', '10:00 AM - 11:00 AM')]
    }
]

# Limit faculty data to 10 only
faculty_data = faculty_data[:25]

# courses_units = {
#     'Subject 1': 3, 'Subject 2': 3, 'Subject 3': 4, 'Subject 4': 2, 'Subject 5': 3, 'Subject 6': 5,
#     'Subject 7': 3, 'Subject 8': 4, 'Subject 9': 2, 'Subject 10': 3, 'Subject 11': 4, 'Subject 12': 5
# }

courses_units = {
    'Subject 1 (Lab)': 3, 'Subject 1 (Lec)': 3, 'Subject 2 (Lab)': 3, 'Subject 2 (Lec)': 3,
    'Subject 3 (Lab)': 1, 'Subject 3 (Lec)': 2, 'Subject 4 (Lab)': 2, 'Subject 4 (Lec)': 2,
    'Subject 5 (Lab)': 3, 'Subject 5 (Lec)': 3, 'Subject 6 (Lab)': 1, 'Subject 6 (Lec)': 2,
    # Add more subjects as needed
}

courses = list(courses_units.keys())

# rooms = ['room' + str(i) for i in range(1, 24)]  # 23 rooms available

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# Generate time slots with 30-minute intervals from 7:00 AM to 9:00 PM
start_time = datetime.strptime("07:00 AM", "%I:%M %p")

end_time = datetime.strptime("09:00 PM", "%I:%M %p")

time_slots = []

while start_time < end_time:
    end_interval = start_time + timedelta(minutes=30)
    time_slots.append(f"{start_time.strftime('%I:%M %p')} - {end_interval.strftime('%I:%M %p')}")
    start_time += timedelta(minutes=30)

courses_units_adjusted = {course: min(units, 3) for course, units in courses_units.items()}

# Generate a balanced assignment of course durations
balanced_course_durations = balanced_course_duration_assignment(courses_units_adjusted)

# rooms = [f'room{i}' for i in range(1, 24)]

rooms = ['GCA 306', 'GCA 307', 'GV 306', 'GC 307', 'Computer Lab 2', 'Computer Lab 3', 'Computer Lab 4']

# Initialize populations for each island
islands = {f"Island_{i+1}": [] for i in range(NUM_ISLANDS)}

migration_chances = [0.1, 0.2, 0.3, 0.4, 0.5]  # List of chances for migrating candidates

optimal_fitness_threshold = 0  # Target fitness indicating an optimal solution

# Initialize a dictionary to track the best solution for each island
best_solutions = {island_name: None for island_name in islands}

# Generating initial populations for each island
for island_name in islands.keys():
    islands[island_name] = initialize_population(CHROMOSOMES_PER_ISLAND, faculty_data, courses_units_adjusted, courses, rooms, days, time_slots)

    


# ------------------------------ Main Distributed Genetic Algorithm Loop ------------------------------

def island_process(
    island_name,
    island_population,
    faculty_data,
    num_generations,
    mutation_rate,
    migration_pool,
    migration_log,
    shared_dict,
    fitness_scores,
    diversity_scores,
    global_best_fitness,
    optimal_found_event,
    lock,  # Add the lock as an argument
    current_generation,
    availability_penalty,
    overlap_penalty,
    overload_penalty,
    consulting_conflict_penalty,
):
    """
    Executes the genetic algorithm for a single island.
    """
    for generation in range(1, num_generations + 1):
        # Check if the optimal solution is found, terminate if so
        if optimal_found_event.is_set():
            print(f"{island_name} terminating early due to global optima found.")
            # Update shared_dict with the best solution found by the island before terminating
            best_solution = max(
                island_population,
                key=lambda x: calculate_fitness(
                    x,
                    faculty_data,
                    availability_penalty,  # Pass penalty values
                    overlap_penalty,
                    overload_penalty,
                    consulting_conflict_penalty,
                ),
            )
            shared_dict[island_name] = best_solution
            return

        # Selection: Rank Selection
        parent1, parent2, fitness1, fitness2 = rank_selection(
            island_population,
            faculty_data,
            availability_penalty,  # Pass penalty values
            overlap_penalty,
            overload_penalty,
            consulting_conflict_penalty,
        )

        # Crossover
        offspring1, offspring2 = crossover(parent1, parent2, faculty_data)

        # Mutation
        mutated_offspring1 = mutate(
            offspring1, island_population, rooms, days, time_slots, faculty_data
        )
        mutated_offspring2 = mutate(
            offspring2, island_population, rooms, days, time_slots, faculty_data
        )

        # Elitism and Populations Update
        elites, elite_fitness_scores = select_elites(
            island_population,
            faculty_data,
            availability_penalty,  # Pass penalty values
            overlap_penalty,
            overload_penalty,
            consulting_conflict_penalty,
        )
        island_population[-4:-2] = [mutated_offspring1, mutated_offspring2]
        island_population[-2:] = elites

        # Fitness Re-evaluation for logging
        updated_fitness_scores = [
            calculate_fitness(
                chromosome,
                faculty_data,
                availability_penalty,  # Pass penalty values
                overlap_penalty,
                overload_penalty,
                consulting_conflict_penalty,
            )
            for chromosome in island_population
        ]

        # Calculate and log best fitness for the generation
        best_fitness = max(
            [
                calculate_fitness(
                    chromosome,
                    faculty_data,
                    availability_penalty,  # Pass penalty values
                    overlap_penalty,
                    overload_penalty,
                    consulting_conflict_penalty,
                )
                for chromosome in island_population
            ]
        )
        fitness_scores[island_name].append(best_fitness)

        # Update global best fitness if necessary (using the lock)
        if best_fitness > global_best_fitness.value:
            with lock:  # Acquire the lock before updating
                global_best_fitness.value = best_fitness

        # Check for optimal solution and signal early stopping
        if global_best_fitness.value == optimal_fitness_threshold:
            optimal_found_event.set()

        # Calculate and log diversity for the generation
        diversity = calculate_diversity(island_population)
        diversity_scores[island_name].append(diversity)

        # Migration handling
        while not migration_pool.empty():
            try:
                migrant_info = migration_pool.get_nowait()
                if migrant_info["destination"] == island_name:
                    # Proceed with migration logic
                    least_fit_idx = island_population.index(
                        min(
                            island_population,
                            key=lambda x: calculate_fitness(
                                x,
                                faculty_data,
                                availability_penalty,  # Pass penalty values
                                overlap_penalty,
                                overload_penalty,
                                consulting_conflict_penalty,
                            ),
                        )
                    )
                    island_population[least_fit_idx] = migrant_info["chromosome"]
                    migration_log.put(
                        f"{migrant_info['source']} -> {island_name}: {migrant_info['chromosome']}"
                    )
            except Exception as e:
                break  # If the pool is empty or an error occurs

        # Contribute the best chromosome to the migration pool every few generations
        if generation % 5 == 0:
            best_chromosome = max(
                island_population, key=lambda x: calculate_fitness(x, faculty_data, availability_penalty, overlap_penalty, overload_penalty, consulting_conflict_penalty)
            )
            destinations = [name for name in shared_dict.keys() if name != island_name]
            if destinations:
                destination = random.choice(destinations)
                migration_pool.put(
                    {
                        "source": island_name,
                        "chromosome": best_chromosome,
                        "destination": destination,
                    }
                )

        # Update shared_dict with the best chromosome found by the island
        best_chromosome = max(
            island_population,
            key=lambda x: calculate_fitness(
                x,
                faculty_data,
                availability_penalty,  # Pass penalty values
                overlap_penalty,
                overload_penalty,
                consulting_conflict_penalty,
            ),
        )
        shared_dict[island_name] = best_chromosome

        current_generation.value += 1


# Main Distributed Genetic Algorithm Loop
def run_dga(
    islands,
    num_generations=100,
    mutation_rate=0.1,
    num_migrants=2,
    faculty_data=None,
    availability_penalty=10,
    overlap_penalty=10,
    overload_penalty=10,
    consulting_conflict_penalty=10,
):
    start_time = time.time()

    manager = Manager()
    shared_dict = manager.dict()
    migration_pool = manager.Queue()
    migration_log = manager.Queue()
    fitness_scores = manager.dict({island_name: manager.list() for island_name in islands})
    diversity_scores = manager.dict({island_name: manager.list() for island_name in islands})
    global_best_fitness = manager.Value('i', float('-inf'))
    optimal_found_event = Event()
    current_generation = manager.Value('i', 0)  # Create a shared variable for current generation
    processes = []

    # Create a lock for synchronization
    lock = manager.Lock() 

    # Define your optimal fitness threshold here (e.g., for 0 penalty):
    optimal_fitness_threshold = 0



    for island_name, island_population in islands.items():
        p = Process(
            target=island_process,
            args=(
                island_name,
                island_population,
                faculty_data,
                num_generations,
                mutation_rate,
                migration_pool,
                migration_log,
                shared_dict,
                fitness_scores,
                diversity_scores,
                global_best_fitness,
                optimal_found_event,
                lock,  # Pass the lock to the process
                current_generation,
                availability_penalty,  # Pass penalty values
                overlap_penalty,
                overload_penalty,
                consulting_conflict_penalty,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    migration_logs = []
    while not migration_log.empty():
        log_entry = migration_log.get()
        migration_logs.append(log_entry)

    end_time = time.time()
    total_time = end_time - start_time

    # Collect fitness scores over generations
    fitness_scores_over_generations = {island_name: list(scores) for island_name, scores in fitness_scores.items()}

    # Collect island diversity scores over generations
    island_diversity_scores_over_generations = {island_name: list(scores) for island_name, scores in diversity_scores.items()}

    # Calculate overall diversity scores over generations
    diversity_scores_over_generations = []
    for generation in range(num_generations):
        generation_diversity = sum(island_diversity_scores_over_generations[island_name][generation] for island_name in islands if generation < len(island_diversity_scores_over_generations[island_name]))
        diversity_scores_over_generations.append(generation_diversity)

    # Find the global best solution and its fitness
    global_best_solution = None
    global_best_fitness = float("-inf")
    for island_name, island_solution in shared_dict.items():
        island_fitness = calculate_fitness(
            island_solution,
            faculty_data,
            availability_penalty,  # Pass penalty values
            overlap_penalty,
            overload_penalty,
            consulting_conflict_penalty,
        )
        if island_fitness > global_best_fitness:
            global_best_solution = island_solution
            global_best_fitness = island_fitness

    global_best_fitness_value = global_best_fitness

    # Calculate penalties for the global best solution
    _, global_best_penalties = calculate_fitness_detailed(global_best_solution, faculty_data) 

    # Initialize the results dictionary here
    results = {        
        'total_time': total_time,
        'global_best_fitness': global_best_fitness_value,
        'global_best_solution': global_best_solution,
        'fitness_scores_over_generations': fitness_scores_over_generations,
        'island_diversity_scores_over_generations': island_diversity_scores_over_generations,
        'diversity_scores_over_generations': diversity_scores_over_generations,
        'last_generation': current_generation.value  
    }

    # Include penalties in the results dictionary
    results['global_best_penalties'] = global_best_penalties 

    return results

if __name__ == "__main__":
    proposed_results = run_dga(islands, faculty_data, num_generations=NUM_GENERATIONS, mutation_rate=MUTATION_RATE, migration_rate=0.5, num_migrants=2)

    generation = 7

    # Extract the diversity score for the specified generation from the proposed model's results
    proposed_diversity = proposed_results['diversity_scores_over_generations'][generation - 1]

    print(f"Proposed Model Diversity Score at Generation {generation}: {proposed_diversity}")

    # Extract the fitness scores for the specified generation from the proposed model's results
    proposed_fitness_scores = proposed_results['fitness_scores_over_generations']

    # Extract the fitness score for the specified generation from the proposed model's results
    proposed_fitness = proposed_fitness_scores['Island_1'][generation - 1]

    print(f"Proposed Model Fitness Score at Generation {generation}: {proposed_fitness}")




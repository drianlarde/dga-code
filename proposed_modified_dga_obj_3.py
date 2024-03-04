import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time

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

def initialize_population(population_size, faculty_data, courses_units, courses, rooms, days, time_slots):
    """
    Initializes the population with chromosomes, considering faculty availability when assigning courses.
    Ensures that courses are assigned to days when the faculty member is available.

    Parameters:
    - population_size (int): The size of the population to be initialized.
    - faculty_data (list): List containing faculty availability and other details.
    - courses_units (dict): A dictionary mapping each course to its unit value.
    - courses (list): List of all courses.
    - rooms (list): List of available rooms.
    - days (list): List of all possible days.
    - time_slots (list): List of available time slots.

    Returns:
    - list: A list of chromosomes representing the initial population.
    """
    population = []
    for _ in range(population_size):
        chromosome = []
        for faculty in faculty_data:
            assigned_courses_with_details = []
            remaining_units = faculty['max_units']
            shuffled_courses = random.sample(courses, len(courses))
            
            for course in shuffled_courses:
                if courses_units[course] <= remaining_units:
                    room = random.choice(rooms)
                    available_days = faculty['availability']  # Only use faculty's available days
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

def calculate_fitness(chromosome, faculty_data, max_hours_per_day=4):
    """
    Calculate the fitness for a single chromosome, incorporating penalties for:
    - Assigning courses outside of faculty availability.
    - Overlaps in room assignments (same room, same time, same day).
    - Daily overloads exceeding a specified number of teaching hours.
    - Scheduling courses during faculty consulting hours.
    
    Returns a fitness score for the chromosome, where a lower score indicates a worse solution.
    """
    penalty = 0

    # Define penalty values for each type of violation
    # PENALTY_VALUE_FOR_AVAILABILITY = 10
    # PENALTY_VALUE_FOR_OVERLAP = 20
    # PENALTY_VALUE_FOR_DAILY_OVERLOAD = 15
    # PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT = 25

    PENALTY_VALUE_FOR_AVAILABILITY = 10
    PENALTY_VALUE_FOR_OVERLAP = 10
    PENALTY_VALUE_FOR_DAILY_OVERLOAD = 10
    PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT = 10

    # Penalty for assigning courses outside of faculty availability
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        faculty_availability = [faculty['availability'] for faculty in faculty_data if faculty['id'] == faculty_id][0]
        
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            course_day = course_detail[2]
            if course_day not in faculty_availability:
                penalty += PENALTY_VALUE_FOR_AVAILABILITY

    # Penalty for overlaps
    overlaps = check_for_overlaps(chromosome)
    penalty += len(overlaps) * PENALTY_VALUE_FOR_OVERLAP

    # Penalty for daily overloads
    daily_overloads = check_for_daily_overloads(chromosome, max_hours_per_day)
    penalty += len(daily_overloads) * PENALTY_VALUE_FOR_DAILY_OVERLOAD

    # Penalty for consulting hour conflicts
    consulting_conflicts = check_for_consulting_hour_conflicts(chromosome, faculty_data)
    penalty += len(consulting_conflicts) * PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT

    # Fitness score calculation
    fitness_score = -penalty  # Using negative score because lower (more negative) is worse
    return fitness_score

def calculate_fitness_detailed(chromosome, faculty_data, max_hours_per_day=4):
    """
    Calculate the fitness for a single chromosome, incorporating penalties for various constraints,
    and provide a detailed breakdown of the penalties.
    
    Parameters:
        chromosome (list): The chromosome to evaluate.
        faculty_data (list): Faculty data for fitness calculation.
        max_hours_per_day (int): Maximum hours a faculty can teach per day.
        
    Returns:
        tuple: Fitness score (negative value where lower is worse) and a dictionary of penalty details.
    """
    penalties = {
        'availability_violations': 0,
        'room_overlaps': 0,
        'daily_overloads': 0,
        'consulting_hour_conflicts': 0
    }

    # PENALTY_VALUES = {
    #     'availability': 10,
    #     'overlap': 20,
    #     'overload': 15,
    #     'consulting_conflict': 25
    # }

    PENALTY_VALUES = {
        'availability': 10,
        'overlap': 10,
        'overload': 10,
        'consulting_conflict': 10
    }

    # Penalty for assigning courses outside of faculty availability
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        faculty_availability = [faculty['availability'] for faculty in faculty_data if faculty['id'] == faculty_id][0]
        
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            course_day = course_detail[2]
            if course_day not in faculty_availability:
                penalties['availability_violations'] += PENALTY_VALUES['availability']

    # Penalty for overlaps
    overlaps = check_for_overlaps(chromosome)
    penalties['room_overlaps'] += len(overlaps) * PENALTY_VALUES['overlap']

    # Penalty for daily overloads
    daily_overloads = check_for_daily_overloads(chromosome, max_hours_per_day)
    penalties['daily_overloads'] += len(daily_overloads) * PENALTY_VALUES['overload']

    # Penalty for consulting hour conflicts
    consulting_conflicts = check_for_consulting_hour_conflicts(chromosome, faculty_data)
    penalties['consulting_hour_conflicts'] += len(consulting_conflicts) * PENALTY_VALUES['consulting_conflict']

    total_penalty = sum(penalties.values())
    fitness_score = -total_penalty
    
    return fitness_score, penalties

# ------------------------ Selection Functions ------------------------

def rank_selection(population, faculty_data):
    """
    Selects two parents using rank selection based on fitness.
    
    Parameters:
        population (list): The population from which to select parents.
        faculty_data (list): Faculty data for fitness calculation.
    
    Returns:
        tuple: The top two chromosomes based on fitness.
    """
    # Calculate fitness for each chromosome in the population
    population_with_fitness = [(chromosome, calculate_fitness(chromosome, faculty_data)) for chromosome in population]
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
    Prints the selected parents and their fitness in a structured format.
    
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

def mutate(chromosome, mutation_rate, courses, rooms, days, time_slots, faculty_data):
    """
    Applies mutation to a given chromosome based on the mutation rate, with mutations considering faculty availability,
    avoiding conflicts with consulting hours, and preventing room overlaps. This ensures that course days are only
    mutated to days when the faculty member is available, do not overlap with their consulting hours, and avoid
    double booking of rooms.

    Parameters:
        chromosome (list): The chromosome to mutate.
        mutation_rate (float): The probability of mutation for each element of the chromosome.
        courses (list): List of all possible courses.
        rooms (list): List of available rooms.
        days (list): List of all possible days.
        time_slots (list): List of available time slots.
        faculty_data (list): List containing faculty availability and other details including consulting hours.

    Returns:
        list: The mutated chromosome with consideration to faculty availability, consulting hours, and room booking.
    """
    for faculty_schedule in chromosome:
        faculty_id = faculty_schedule['id']
        faculty_info = next((item for item in faculty_data if item['id'] == faculty_id), None)
        if not faculty_info:
            continue  # Skip if faculty info is not found
        
        if random.random() < mutation_rate:
            for course_detail_index, course_detail in enumerate(faculty_schedule['assigned_courses_with_details']):
                mutation_choice = random.choice(['course', 'room', 'day', 'time_slot'])

                if mutation_choice == 'room':
                    new_room = random.choice(rooms)
                    faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail[:1] + (new_room,) + course_detail[2:]

                if mutation_choice == 'day':
                    new_day = random.choice([day for day in days if day in faculty_info['availability']])
                    faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail[:2] + (new_day,) + course_detail[3:]

                if mutation_choice == 'time_slot':
                    new_time_slot = random.choice(time_slots)
                    faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail[:3] + (new_time_slot,)
                
                # if mutation_choice == 'room':
                #     # Attempt to select a new room that does not result in overlaps
                #     for _ in range(10):  # Try up to 10 times to find a non-overlapping room
                #         new_room = random.choice(rooms)
                #         if not causes_room_overlap(faculty_schedule['assigned_courses_with_details'], course_detail_index, new_room, chromosome):
                #             faculty_schedule['assigned_courses_with_details'][course_detail_index] = (course_detail[0], new_room) + course_detail[2:]
                #             break

                # if mutation_choice == 'day':
                #     new_day = random.choice([day for day in days if day in faculty_info['availability']])
                #     # Check for consulting hour conflicts before applying the mutation
                #     if not conflicts_with_consulting_hours(new_day, course_detail[3], faculty_info['consulting_hours']):
                #         faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail[:2] + (new_day,) + course_detail[3:]
                
                # if mutation_choice == 'time_slot':
                #     # Attempt to select a new time slot that does not conflict with consulting hours
                #     for _ in range(10):  # Try up to 10 times to find a non-conflicting time slot
                #         new_time_slot = random.choice(time_slots)
                #         if not conflicts_with_consulting_hours(course_detail[2], new_time_slot, faculty_info['consulting_hours']):
                #             faculty_schedule['assigned_courses_with_details'][course_detail_index] = course_detail[:3] + (new_time_slot,)
                #             break

                # Other mutation logic for 'course' and 'room' remains unchanged
                
    return chromosome

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

# ------------------------ Elitism Functions ------------------------

def select_elites(population, faculty_data, n_elites=2):
    """
    Selects the top n_elites chromosomes from the population based on their fitness scores.

    Parameters:
        population (list): The current population from which to select elites.
        faculty_data (list): Faculty data for fitness calculation.
        n_elites (int): The number of elite chromosomes to select.
    
    Returns:
        list: The list of elite chromosomes.
    """
    # Calculate fitness for each chromosome in the population
    population_with_fitness = [(chromosome, calculate_fitness(chromosome, faculty_data)) for chromosome in population]
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
POPULATION_SIZE = 10
NUM_ISLANDS = 4
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

courses_units = {
    'course1': 3, 'course2': 3, 'course3': 4, 'course4': 2, 'course5': 3, 'course6': 5, 
    'course7': 3, 'course8': 4, 'course9': 2, 'course10': 3, 'course11': 4, 'course12': 5
}

courses = list(courses_units.keys())

rooms = ['room' + str(i) for i in range(1, 24)]  # 23 rooms available

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

rooms = [f'room{i}' for i in range(1, 24)]

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

from multiprocessing import Process, Manager

def island_process(island_name, island_population, faculty_data, num_generations, mutation_rate, migration_pool, migration_log, shared_dict):
    """
    Executes the genetic algorithm for a single island.
    """
    for generation in range(1, num_generations + 1):
        # Selection: Rank Selection
        parent1, parent2, fitness1, fitness2 = rank_selection(island_population, faculty_data)

        # Crossover
        offspring1, offspring2 = crossover(parent1, parent2, faculty_data)

        # Mutation
        mutated_offspring1 = mutate(offspring1, mutation_rate, courses, rooms, days, time_slots, faculty_data)
        mutated_offspring2 = mutate(offspring2, mutation_rate, courses, rooms, days, time_slots, faculty_data)

        # Elitism and Populations Update
        elites, elite_fitness_scores = select_elites(island_population, faculty_data)
        island_population[-4:-2] = [mutated_offspring1, mutated_offspring2]
        island_population[-2:] = elites

        # Fitness Re-evaluation for logging
        updated_fitness_scores = [calculate_fitness(chromosome, faculty_data) for chromosome in island_population]
        print(f'Best Fitness in {island_name}: {max(updated_fitness_scores)}')

        # Migration handling
        while not migration_pool.empty():
            try:
                migrant_info = migration_pool.get_nowait()
                if migrant_info['destination'] == island_name:  # Ensure the migrant is intended for this island
                    # Proceed with migration logic
                    least_fit_idx = island_population.index(min(island_population, key=lambda x: calculate_fitness(x, faculty_data)))
                    island_population[least_fit_idx] = migrant_info['chromosome']
                    print(f"Migrated to {island_name} from {migrant_info['source']}: {migrant_info['chromosome']}")
                    migration_log.put(f"{migrant_info['source']} -> {island_name}: {migrant_info['chromosome']}")
            except Exception as e:
                break  # If the pool is empty or an error occurs

        # Contribute the best chromosome to the migration pool every few generations
        if generation % 5 == 0:
            best_chromosome = max(island_population, key=lambda x: calculate_fitness(x, faculty_data))
            # Randomly select a destination island different from the current one
            destinations = [name for name in shared_dict.keys() if name != island_name]
            if destinations:  # Ensure there's at least one possible destination
                destination = random.choice(destinations)
                migration_pool.put({"source": island_name, "chromosome": best_chromosome, "destination": destination})

        # Optionally update shared_dict with the best chromosome at the end
        if generation == num_generations:
            shared_dict[island_name] = best_chromosome

# Main Distributed Genetic Algorithm Loop
def run_dga(islands, faculty_data, num_generations=100, mutation_rate=0.1, num_migrants=2):
    """
    Executes the Distributed Genetic Algorithm across multiple islands, evolving solutions over generations.
    
    Parameters:
        islands (dict): Dictionary of islands, where each key is an island name and its value is the island's population.
        faculty_data (list): Detailed information about faculty members, including their availability and constraints.
        num_generations (int): The total number of generations to evolve.
        mutation_rate (float): Probability of mutation for each gene within a chromosome.
        migration_rate (float): Probability of migration event between islands per generation.
        num_migrants (int): Number of individuals to migrate between islands during a migration event.
    """
    # Start time of the algorithm
    start_time = time.time()
    print("Start Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # Initialize multiprocessing manager and shared dictionary
    manager = Manager()
    shared_dict = manager.dict()  # For storing final results or best solutions
    migration_pool = manager.Queue()  # For chromosome migrations
    migration_log = manager.Queue()  # For logging migrations

    # List to keep track of processes
    processes = []

    for island_name, island_population in islands.items():
        p = Process(target=island_process, args=(island_name, island_population, faculty_data, num_generations, mutation_rate, migration_pool, migration_log, shared_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Log migrations
    while not migration_log.empty():
        log_entry = migration_log.get()
        print(log_entry)

    end_time = time.time()
    print("End Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print("Total Time:", end_time - start_time)

if __name__ == "__main__":
    # Assuming `islands` and `faculty_data` are already defined
    run_dga(islands, faculty_data, num_generations=100, mutation_rate=0.1, num_migrants=2)
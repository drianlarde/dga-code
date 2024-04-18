# modified_dga_streamlit.py

import streamlit as st
from datetime import datetime, time, timedelta
from modified_dga_main import run_dga, initialize_population, balanced_course_duration_assignment
import random
import string
import pandas as pd

if 'faculty_list' not in st.session_state:
    st.session_state['faculty_list'] = []
if 'reset_form' not in st.session_state:
    st.session_state['reset_form'] = False
if 'editing_index' not in st.session_state:
    st.session_state['editing_index'] = None
if 'show_settings' not in st.session_state:
    st.session_state['show_settings'] = False
if 'subjects_units' not in st.session_state:
    # st.session_state['subjects_units'] = {
    #     'Subject 1': 3, 'Subject 2': 3, 'Subject 3': 2, 'Subject 4': 2, 'Subject 5': 3, 'Subject 6': 1,
    #     'Subject 7': 3, 'Subject 8': 2, 'Subject 9': 2, 'Subject 10': 3, 'Subject 11': 1, 'Subject 12': 1
    # }
    st.session_state['subjects_units'] = {
        'Subject 1 (Lab)': 3, 'Subject 1 (Lec)': 3, 'Subject 2 (Lab)': 3, 'Subject 2 (Lec)': 3,
        'Subject 3 (Lab)': 1, 'Subject 3 (Lec)': 2, 'Subject 4 (Lab)': 2, 'Subject 4 (Lec)': 2,
        'Subject 5 (Lab)': 3, 'Subject 5 (Lec)': 3, 'Subject 6 (Lab)': 1, 'Subject 6 (Lec)': 2,
        # Add more subjects as needed
    }

def add_faculty_form(editing_index=None):
    with st.form(f"faculty_form"):
        unique_key_suffix = f"_reset_{editing_index}" if st.session_state['reset_form'] else f"_{editing_index}"

        faculty_id = st.text_input("Faculty ID", value=st.session_state.faculty_list[editing_index]['id'] if editing_index is not None else "", key=f"faculty_id{unique_key_suffix}")
        availability = st.multiselect("Availability", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], default=st.session_state.faculty_list[editing_index]['availability'] if editing_index is not None else [], key=f"availability{unique_key_suffix}")
        max_units = st.number_input("Max Units", min_value=0, max_value=30, step=1, value=st.session_state.faculty_list[editing_index]['max_units'] if editing_index is not None else 0, key=f"max_units{unique_key_suffix}")
        is_part_time = st.checkbox("Is Part Time?", value=st.session_state.faculty_list[editing_index]['is_part_time'] if editing_index is not None else False, key=f"is_part_time{unique_key_suffix}")
        consulting_day, consulting_time = st.session_state.faculty_list[editing_index]['consulting_hours'][0] if editing_index is not None else ('Monday', '10:00 AM - 11:00 AM')
        consulting_day = st.selectbox("Consulting Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'].index(consulting_day), key=f"consulting_day{unique_key_suffix}")

        start_time_str, end_time_str = consulting_time.split(' - ')
        start_time = st.time_input("Start Time", value=datetime.strptime(start_time_str, '%I:%M %p').time(), key=f"start_time{unique_key_suffix}")
        end_time = st.time_input("End Time", value=datetime.strptime(end_time_str, '%I:%M %p').time(), key=f"end_time{unique_key_suffix}")

        col1, col2 = st.columns(2)
        with col1:
            if editing_index is None:
                submitted = st.form_submit_button("Add Faculty Member")
            else:
                submitted = st.form_submit_button("Update Faculty Member")
        with col2:
            if editing_index is not None:
                if st.form_submit_button("Cancel"):
                    st.session_state['editing_index'] = None
                    st.experimental_rerun()

        if submitted:
            faculty_data = {
                'id': faculty_id,
                'availability': availability,
                'max_units': max_units,
                'is_part_time': is_part_time,
                'consulting_hours': [(consulting_day, f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}")],
            }
            if editing_index is None:
                st.session_state.faculty_list.append(faculty_data)
            else:
                st.session_state.faculty_list[editing_index] = faculty_data
                st.session_state['editing_index'] = None

            st.session_state['reset_form'] = not st.session_state['reset_form']
            st.experimental_rerun()

def remove_faculty(index):
    st.session_state.faculty_list.pop(index)
    st.experimental_rerun()

def edit_faculty(index):
    st.session_state['editing_index'] = index
    st.experimental_rerun()

def clear_faculty_list():
    st.session_state.faculty_list = []
    st.experimental_rerun()

def display_faculty_list():
    if st.session_state.faculty_list:
        for i, faculty in enumerate(st.session_state.faculty_list):
            with st.expander(f"{faculty['id']}"):
                st.markdown(f"""
                - **ID:** {faculty['id']}
                - **Availability:** {', '.join(faculty['availability'])}
                - **Max Units:** {faculty['max_units']}
                - **Part Time:** {'Yes' if faculty['is_part_time'] else 'No'}
                - **Consulting Hours:** {', '.join([f'{ch[0]}: {ch[1]}' for ch in faculty['consulting_hours']])}
                """)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Remove", key=f"remove_{i}"):
                        remove_faculty(i)
                with col2:
                    if st.button("Edit", key=f"edit_{i}"):
                        edit_faculty(i)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Faculty List"):
            clear_faculty_list()

def add_subject():
    st.subheader("Add Subject")
    with st.form("add_subject_form"):
        subject_name = st.text_input("Subject Name")
        subject_units = st.number_input("Subject Units", min_value=1, step=1)
        if st.form_submit_button("Add Subject"):
            st.session_state['subjects_units'][subject_name] = subject_units
            st.experimental_rerun()

def remove_subject(subject):
    del st.session_state['subjects_units'][subject]
    st.experimental_rerun()

def display_subjects():
    st.subheader("Subject Units")
    for subject, units in st.session_state['subjects_units'].items():
        col1, col2 = st.columns([4, 1])
        col1.text(f"{subject}: {units} units")
        if col2.button("Remove", key=f"remove_subject_{subject}"):
            remove_subject(subject)
    add_subject()

def display_dga_progress(global_best_fitness, global_best_solution, fitness_scores_over_generations, island_diversity_scores_over_generations, diversity_scores_over_generations):
    st.subheader("DGA Progress")

    st.write(f"Global Best Fitness: {global_best_fitness}")

    st.write("Global Best Solution:")
    for faculty_schedule in global_best_solution:
        st.write(f"Faculty ID: {faculty_schedule['id']}")
        for subject_detail in faculty_schedule['assigned_subjects_with_details']:
            st.write(f"- Subject: {subject_detail[0]}, Room: {subject_detail[1]}, Day: {subject_detail[2]}, Time Slot: {subject_detail[3]}")

    st.markdown("### Fitness Scores Over Generations")
    st.markdown("This chart shows the fitness scores of the best solution found in each generation.")
    st.line_chart(fitness_scores_over_generations)

    st.markdown("### Island Diversity Scores Over Generations")
    st.markdown("These charts show the diversity scores of each island over the generations.")
    for island_name, diversities in island_diversity_scores_over_generations.items():
        st.write(f"{island_name} Diversity Scores Over Generations")
        st.line_chart({island_name: diversities})

    st.markdown("### Overall Diversity Scores Over Generations")
    st.markdown("This chart shows the overall diversity scores across all islands over the generations.")
    st.line_chart({"Overall Diversity": diversity_scores_over_generations})

def display_dga_results(dga_results):
    st.subheader("DGA Results")

    # Display the total time taken
    st.write(f"Total Time: {dga_results['total_time']:.2f} seconds")

    # Display the global best fitness
    st.write(f"Global Best Fitness: {dga_results['global_best_fitness']}")

    # Display the global best solution
    st.markdown("### Global Best Solution")
    st.markdown("The global best solution is displayed in a toggle format, with each faculty member's schedule shown in a tabular format inside the toggle.")

    # Group the assigned subjects by faculty ID
    faculty_schedules = {}
    for faculty_schedule in dga_results['global_best_solution']:
        faculty_id = faculty_schedule['id']
        if faculty_id not in faculty_schedules:
            faculty_schedules[faculty_id] = []
        faculty_schedules[faculty_id].extend(faculty_schedule['assigned_courses_with_details'])

    # Display each faculty member's schedule in a toggle
    for faculty_id, subject_details in faculty_schedules.items():
        with st.expander(f"Faculty ID: {faculty_id}"):
            data = [[subject_detail[0], subject_detail[1], subject_detail[2], subject_detail[3]] for subject_detail in subject_details]
            df = pd.DataFrame(data, columns=["Subject", "Room", "Day", "Time Slot"])
            st.write(df)

    st.markdown("## DGA Results Over Generations")
    st.markdown("This section shows the progress of the Distributed Genetic Algorithm over the generations.")
    st.line_chart(dga_results['fitness_scores_over_generations'])

    st.markdown("### Island Diversity Scores Over Generations")
    st.markdown("These charts show the diversity scores of each island over the generations.")
    for island_name, diversities in dga_results['island_diversity_scores_over_generations'].items():
        st.write(f"{island_name} Diversity Scores Over Generations")
        st.line_chart({island_name: diversities})

    st.markdown("### Overall Diversity Scores Over Generations")
    st.markdown("This chart shows the overall diversity scores across all islands over the generations.")
    st.line_chart({"Overall Diversity": dga_results['diversity_scores_over_generations']})

    st.markdown("## Penalties in Global Best Solution")
    if dga_results['global_best_penalties'] is None:
        st.warning("No penalty information available.")
    else:
        for penalty_type, penalty_details in dga_results['global_best_penalties'].items():
            st.write(f"### {penalty_type.replace('_', ' ').title()}")
            if penalty_details:
                for detail in penalty_details:
                    if penalty_type == 'availability_violations':
                        st.write(f"- Faculty {detail['faculty_id']}: Course {detail['course']} assigned on {detail['day']}")
                    elif penalty_type == 'room_overlaps':
                        st.write(f"- Faculty {detail['faculty1']} and Faculty {detail['faculty2']}: Courses {detail['details'][0]['course']} and {detail['details'][1]['course']} overlap in Room {detail['details'][0]['room']} on {detail['details'][0]['day']} at {detail['details'][0]['time_slot']}")
                    elif penalty_type == 'daily_overloads':
                        st.write(f"- Faculty {detail['faculty_id']} is overloaded on {detail['day']}, scheduled for {detail['hours']:.2f} hours")
                    elif penalty_type == 'consulting_hour_conflicts':
                        st.write(f"- Faculty {detail['faculty_id']}: Course at {detail['course_time_slot']} conflicts with consulting hours at {detail['consulting_time_slot']}")
                    elif penalty_type == 'lab_subjects':
                        st.write(f"- Faculty {detail['faculty_id']}: Assigned lab course {detail['course']}")
            else:
                st.write("No penalties of this type.")

def display_settings():
    st.subheader("Distributed Genetic Algorithm Settings")
    NUM_GENERATIONS = st.number_input("Number of Generations", min_value=1, value=100, step=1)
    MUTATION_RATE = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    POPULATION_SIZE = st.number_input("Population Size", min_value=1, value=20, step=1)
    NUM_ISLANDS = st.number_input("Number of Islands", min_value=1, value=2, step=1)
    MIGRATION_RATE = st.slider("Migration Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    NUM_MIGRANTS = st.number_input("Number of Migrants", min_value=1, value=2, step=1)

    st.subheader("Penalty Values")
    PENALTY_VALUE_FOR_AVAILABILITY = st.slider("Availability Violation", 0, 100, 10)
    PENALTY_VALUE_FOR_OVERLAP = st.slider("Room Overlap", 0, 100, 10)
    PENALTY_VALUE_FOR_DAILY_OVERLOAD = st.slider("Daily Overload", 0, 100, 10)
    PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT = st.slider("Consulting Hour Conflict", 0, 100, 10)

    display_subjects()

    return NUM_GENERATIONS, MUTATION_RATE, POPULATION_SIZE, NUM_ISLANDS, MIGRATION_RATE, NUM_MIGRANTS, PENALTY_VALUE_FOR_AVAILABILITY, PENALTY_VALUE_FOR_OVERLAP, PENALTY_VALUE_FOR_DAILY_OVERLOAD, PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT

def generate_random_faculty(num_faculty):
    available_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    new_faculty_list = []
    for i in range(num_faculty):
        faculty_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        availability = random.sample(available_days, random.randint(1, len(available_days)))
        max_units = random.randint(1, 30)
        is_part_time = random.choice([True, False])
        consulting_day = random.choice(available_days)
        start_time = datetime.strptime(random.choice(['08:00 AM', '09:00 AM', '10:00 AM', '11:00 AM', '12:00 PM', '01:00 PM', '02:00 PM']), '%I:%M %p')
        end_time = start_time + timedelta(hours=1)
        consulting_hours = [(consulting_day, f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}")]

        faculty_data = {
            'id': faculty_id,
            'availability': availability,
            'max_units': max_units,
            'is_part_time': is_part_time,
            'consulting_hours': consulting_hours
        }
        new_faculty_list.append(faculty_data)

    return new_faculty_list

def main():
    st.title("Modified Distributed Genetic Algorithm for Optimized Faculty Workload and Subject Assignment")
    st.write("This is a modified version of the Distributed Genetic Algorithm for optimized faculty workload and subject assignment. This application allows you to add faculty members, subjects, and run the Distributed Genetic Algorithm to optimize the faculty workload and subject assignment.")

    if st.button("Settings", key="settings_button"):
        st.session_state['show_settings'] = not st.session_state['show_settings']

    NUM_GENERATIONS = 100
    MUTATION_RATE = 0.1
    POPULATION_SIZE = 20
    NUM_ISLANDS = 2
    MIGRATION_RATE = 0.1
    NUM_MIGRANTS = 2

    # Initialize penalty values with defaults
    PENALTY_VALUE_FOR_AVAILABILITY = 10
    PENALTY_VALUE_FOR_OVERLAP = 10
    PENALTY_VALUE_FOR_DAILY_OVERLOAD = 10
    PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT = 10

    if st.session_state['show_settings']:
        NUM_GENERATIONS, MUTATION_RATE, POPULATION_SIZE, NUM_ISLANDS, MIGRATION_RATE, NUM_MIGRANTS, PENALTY_VALUE_FOR_AVAILABILITY, PENALTY_VALUE_FOR_OVERLAP, PENALTY_VALUE_FOR_DAILY_OVERLOAD, PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT = display_settings()  # Get penalty values from sliders

    if st.session_state['editing_index'] is None:
        add_faculty_form()
    else:
        add_faculty_form(editing_index=st.session_state['editing_index'])

    display_faculty_list()

    num_faculty_to_generate = st.number_input("Number of Faculty to Generate", min_value=1, value=1, step=1)

    new_faculty_list = []

    if st.button("Generate Random Faculty"):
        new_faculty_list = generate_random_faculty(num_faculty_to_generate)
        st.session_state.faculty_list.extend(new_faculty_list)

    st.write(new_faculty_list)

    if st.button("Run Distributed Genetic Algorithm", key="run_dga_button"):
        CHROMOSOMES_PER_ISLAND = POPULATION_SIZE // NUM_ISLANDS

        subjects = list(st.session_state['subjects_units'].keys())

        # rooms = ['room' + str(i) for i in range(1, 24)]  # 23 rooms available

        rooms = ['GCA 306', 'GCA 307', 'GV 306', 'GC 307', 'Computer Lab 2', 'Computer Lab 3', 'Computer Lab 4']

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

        start_time = datetime.strptime("07:00 AM", "%I:%M %p")
        end_time = datetime.strptime("09:00 PM", "%I:%M %p")

        time_slots = []
        while start_time < end_time:
            end_interval = start_time + timedelta(minutes=30)
            time_slots.append(f"{start_time.strftime('%I:%M %p')} - {end_interval.strftime('%I:%M %p')}")
            start_time += timedelta(minutes=30)

        subjects_units_adjusted = {subject: min(units, 3) for subject, units in st.session_state['subjects_units'].items()}

        balanced_subject_durations = balanced_course_duration_assignment(subjects_units_adjusted)

        islands = {f"Island_{i+1}": [] for i in range(NUM_ISLANDS)}

        for island_name in islands.keys():
            islands[island_name] = initialize_population(CHROMOSOMES_PER_ISLAND, st.session_state.faculty_list, subjects_units_adjusted, subjects, rooms, days, time_slots)

        terminal_output = st.empty()
        with st.spinner("Running Distributed Genetic Algorithm..."):
            dga_results = run_dga(
                islands,
                st.session_state.faculty_list,
                num_generations=NUM_GENERATIONS,
                mutation_rate=MUTATION_RATE,
                migration_rate=MIGRATION_RATE,
                num_migrants=NUM_MIGRANTS,
                availability_penalty=PENALTY_VALUE_FOR_AVAILABILITY,
                overlap_penalty=PENALTY_VALUE_FOR_OVERLAP,
                overload_penalty=PENALTY_VALUE_FOR_DAILY_OVERLOAD,
                consulting_conflict_penalty=PENALTY_VALUE_FOR_CONSULTING_HOUR_CONFLICT
            )

        display_dga_results(dga_results)


if __name__ == '__main__':
    main()

# proposed_modified_dga_obj_combined_streamlit.py

import streamlit as st
from datetime import datetime, time, timedelta
from proposed_modified_dga_obj_combined_main import run_dga, islands, faculty_data, migration_chances
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
if 'courses_units' not in st.session_state:
    st.session_state['courses_units'] = {
        'course1': 3, 'course2': 3, 'course3': 4, 'course4': 2, 'course5': 3, 'course6': 5,
        'course7': 3, 'course8': 4, 'course9': 2, 'course10': 3, 'course11': 4, 'course12': 5
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

def add_course():
    st.subheader("Add Course")
    with st.form("add_course_form"):
        course_name = st.text_input("Course Name")
        course_units = st.number_input("Course Units", min_value=1, step=1)
        if st.form_submit_button("Add Course"):
            st.session_state['courses_units'][course_name] = course_units
            st.experimental_rerun()

def remove_course(course):
    del st.session_state['courses_units'][course]
    st.experimental_rerun()

def display_courses():
    st.subheader("Course Units")
    for course, units in st.session_state['courses_units'].items():
        col1, col2 = st.columns([4, 1])
        col1.text(f"{course}: {units} units")
        if col2.button("Remove", key=f"remove_course_{course}"):
            remove_course(course)
    add_course()

def display_dga_progress(global_best_fitness, global_best_solution, fitness_scores_over_generations, island_diversity_scores_over_generations, diversity_scores_over_generations):
    st.subheader("DGA Progress")

    st.write(f"Global Best Fitness: {global_best_fitness}")

    st.write("Global Best Solution:")
    for faculty_schedule in global_best_solution:
        st.write(f"Faculty ID: {faculty_schedule['id']}")
        for course_detail in faculty_schedule['assigned_courses_with_details']:
            st.write(f"- Course: {course_detail[0]}, Room: {course_detail[1]}, Day: {course_detail[2]}, Time Slot: {course_detail[3]}")

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

    if dga_results['global_best_solution'] is None:
        st.warning("No global best solution found.")
    else:
        # Group the assigned courses by faculty ID
        faculty_schedules = {}
        for faculty_schedule in dga_results['global_best_solution']:
            faculty_id = faculty_schedule['id']
            if faculty_id not in faculty_schedules:
                faculty_schedules[faculty_id] = []
            faculty_schedules[faculty_id].extend(faculty_schedule['assigned_courses_with_details'])

        # Display each faculty member's schedule in a toggle
        for faculty_id, course_details in faculty_schedules.items():
            with st.expander(f"Faculty ID: {faculty_id}"):
                data = [[course_detail[0], course_detail[1], course_detail[2], course_detail[3]] for course_detail in course_details]
                df = pd.DataFrame(data, columns=["Course", "Room", "Day", "Time Slot"])
                st.write(df)

    st.markdown("## DGA Results Over Generations")
    st.markdown("This section shows the progress of the Distributed Genetic Algorithm over the generations.")
    
    # Check if fitness_scores_over_generations is available in the dga_results
    if 'fitness_scores_over_generations' in dga_results:
        st.line_chart(dga_results['fitness_scores_over_generations'])
    else:
        st.warning("Fitness scores over generations not available.")

    st.markdown("### Island Diversity Scores Over Generations")
    st.markdown("These charts show the diversity scores of each island over the generations.")
    
    # Check if island_diversity_scores_over_generations is available in the dga_results
    if 'island_diversity_scores_over_generations' in dga_results:
        for island_name, diversities in dga_results['island_diversity_scores_over_generations'].items():
            st.write(f"{island_name} Diversity Scores Over Generations")
            st.line_chart({island_name: diversities})
    else:
        st.warning("Island diversity scores over generations not available.")

    st.markdown("### Overall Diversity Scores Over Generations")
    st.markdown("This chart shows the overall diversity scores across all islands over the generations.")
    
    # Check if diversity_scores_over_generations is available in the dga_results
    if 'diversity_scores_over_generations' in dga_results:
        st.line_chart({"Overall Diversity": dga_results['diversity_scores_over_generations']})
    else:
        st.warning("Overall diversity scores over generations not available.")

def display_settings():
    st.subheader("Distributed Genetic Algorithm Settings")
    NUM_GENERATIONS = st.number_input("Number of Generations", min_value=1, value=100, step=1)
    MUTATION_RATE = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    POPULATION_SIZE = st.number_input("Population Size", min_value=1, value=20, step=1)
    NUM_ISLANDS = st.number_input("Number of Islands", min_value=1, value=2, step=1)
    MIGRATION_RATE = st.slider("Migration Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    NUM_MIGRANTS = st.number_input("Number of Migrants", min_value=1, value=2, step=1)

    display_courses()

    return NUM_GENERATIONS, MUTATION_RATE, POPULATION_SIZE, NUM_ISLANDS, MIGRATION_RATE, NUM_MIGRANTS

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
    st.title("Proposed Distributed Genetic Algorithm")

    dga_results = run_dga(islands, faculty_data, num_generations=100, mutation_rate=0.1, num_migrants=2)
    
    st.write(dga_results)

    display_dga_results(dga_results)

if __name__ == '__main__':
    main()

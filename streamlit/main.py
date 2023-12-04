import streamlit as st

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi as gd

import random
import itertools
from tqdm import tqdm
import pandas as pd

st.title(
    "Enhancing Genetic Algorithm For Optimized Faculty Workload And Course Assignment"
)
# Add a subheader in the app
st.subheader("By Joshua D. Bumanlag & Adrian Angelo D. Abelarde")

# Add paragraph
st.write(
    "This web application is used to optimize the faculty workload and course assignment using the Enhanced Genetic Algorithm. The user can generate a dataset .CSV file containing the faculty and course information. The user can also choose which algorithm to use: the Generic Algorithm or the Proposed Algorithm."
)

st.divider()

# Dataset Option

st.subheader("1. Upload or Generate a Dataset")
st.write(
    "Upload a .CSV file containing the faculty and course information or generate a dataset."
)

data_option = st.selectbox(
    "Choose an option",
    ("Generate a dataset", "Upload a file"),
)

if data_option == "Generate a dataset":
    # Add a subheader in the app
    st.subheader("Generate a dataset")
    # Make a button to generate a dataset
    if st.button("Generate"):
        courses = [
            {
                "code": "CSC 0102",
                "title": "Discrete Structures 1",
                "hours": 3,
                "year": 1,
            },
            {
                "code": "CSC 0211",
                "title": "Discrete Structures 2",
                "hours": 3,
                "year": 1,
            },
            {
                "code": "CSC 0212",
                "title": "Object Oriented Programming (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "CSC 0212.1",
                "title": "Object Oriented Programming (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "CSC 0213",
                "title": "Logic Design and Digital Computer Circuits (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "CSC 0213.1",
                "title": "Logic Design and Digital Computer Circuits (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "CSC 0221",
                "title": "Algorithm and Complexity",
                "hours": 3,
                "year": 2,
            },
            {
                "code": "CSC 0222",
                "title": "Architecture and Organization (Lec)",
                "hours": 2,
                "year": 2,
            },
            {
                "code": "CSC 0222.1",
                "title": "Architecture and Organization (Lab)",
                "hours": 1,
                "year": 2,
            },
            {
                "code": "CSC 0223",
                "title": "Human Computer Interaction",
                "hours": 3,
                "year": 2,
            },
            {"code": "CSC 0224", "title": "Operation Research", "hours": 3, "year": 2},
            {
                "code": "CSC 0311",
                "title": "Automata Theory and Formal Languages",
                "hours": 3,
                "year": 3,
            },
            {
                "code": "CSC 0312",
                "title": "Programming Languages (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0312.1",
                "title": "Programming Languages (Lab)",
                "hours": 1,
                "year": 3,
            },
            {
                "code": "CSC 0313",
                "title": "Software Engineering (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0313.1",
                "title": "Software Engineering (Lab)",
                "hours": 1,
                "year": 3,
            },
            {
                "code": "CSC 0314",
                "title": "Operating System (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0314.1",
                "title": "Operating System (Lab)",
                "hours": 1,
                "year": 3,
            },
            {
                "code": "CSC 0315",
                "title": "Intelligent System (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0315.1",
                "title": "Intelligent System (Lab)",
                "hours": 1,
                "year": 3,
            },
            {
                "code": "CSC 0316",
                "title": "Information Assurance Security",
                "hours": 3,
                "year": 3,
            },
            {
                "code": "CSC 0321",
                "title": "Software Engineering 2 (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0321.1",
                "title": "Software Engineering 2 (Lab)",
                "hours": 1,
                "year": 3,
            },
            {
                "code": "CSC 0322",
                "title": "Compiler Design (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0322.1",
                "title": "Compiler Design (Lab)",
                "hours": 1,
                "year": 3,
            },
            {
                "code": "CSC 0323",
                "title": "Computational Science (Lec)",
                "hours": 2,
                "year": 3,
            },
            {
                "code": "CSC 0323.1",
                "title": "Computational Science (Lab)",
                "hours": 1,
                "year": 3,
            },
            {"code": "CSC 0324", "title": "CS Elective 1 (Lec)", "hours": 2, "year": 3},
            {
                "code": "CSC 0324.1",
                "title": "CS Elective 1 (Lab)",
                "hours": 1,
                "year": 3,
            },
            {"code": "CSC 0325", "title": "Research Writing", "hours": 3, "year": 3},
            {
                "code": "CSC 195.1",
                "title": "Practicum (240 Hrs)",
                "hours": 2,
                "year": 5,
            },
            {
                "code": "ICC 0101",
                "title": "Introduction to Computing (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "ICC 0101.1",
                "title": "Introduction to Computing (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "ICC 0102",
                "title": "Fundamentals of Programming (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "ICC 0102.1",
                "title": "Fundamentals of Programming (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "ICC 0103",
                "title": "Intermediate Programming (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "ICC 0103.1",
                "title": "Intermediate Programming (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "ICC 0104",
                "title": "Data Structures and Algorithms (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "ICC 0104.1",
                "title": "Data Structures and Algorithms (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "ICC 0105",
                "title": "Information Management (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "ICC 0105.1",
                "title": "Information Management (Lab)",
                "hours": 1,
                "year": 1,
            },
            {
                "code": "ICC 0106",
                "title": "Applications Development and Emerging Technologies (Lec)",
                "hours": 2,
                "year": 1,
            },
            {
                "code": "ICC 0106.1",
                "title": "Applications Development and Emerging Technologies (Lab)",
                "hours": 1,
                "year": 1,
            },
        ]

        # Full-time faculty - set to exactly 4 courses
        faculty = [
            {"id": f"FT{i}", "type": "Full-time", "max_classes": 4} for i in range(1, 7)
        ]

        # Part-time faculty - set to a max of 3 courses and max 12 units
        # The year level constraint is also applied
        faculty += [
            {
                "id": f"PT{i}",
                "type": "Part-time",
                "max_classes": 3,
                "max_units": 12,
                "year_level": random.choice([1, 2, 3, 4]),
            }
            for i in range(1, 5)
        ]

        classrooms = (
            ["Com Lab 1", "Com Lab 2", "Com Lab 3", "Com Lab 4"]
            + [f"GV 30{i}" for i in range(1, 8)]
            + [f"GCA 30{i}" for i in range(1, 6)]
        )

        # Define Days
        days_of_week = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]

        # Generating Timeslots with Days and AM/PM
        def generate_timeslots():
            timeslots = []
            for day in days_of_week:
                for hour in range(7, 21):  # From 7 AM to 9 PM (21:00)
                    am_pm = "AM" if hour < 12 else "PM"
                    hour_12 = hour if hour <= 12 else hour - 12
                    timeslot = f"{day} {hour_12}:00 - {hour_12+1}:00 {am_pm}"
                    timeslots.append(timeslot)
            return timeslots

        timeslots = generate_timeslots()

        # Function to initialize faculty hours and courses taught
        def initialize_faculty():
            for fac in faculty:
                fac["hours_taught"] = 0
                fac["courses_taught"] = []

        initialize_faculty()

        # Initialize the schedule list before starting the assignment
        schedule = []

        # Function to check if a faculty member can teach a course
        def can_teach(faculty_member, course):
            if faculty_member["type"] == "Part-time":
                return (
                    faculty_member["hours_taught"] + course["hours"]
                    <= faculty_member["max_units"]
                    and len(faculty_member["courses_taught"])
                    < faculty_member["max_classes"]
                    and course["year"] == faculty_member["year_level"]
                )
            else:  # Full-time
                return (
                    len(faculty_member["courses_taught"])
                    < faculty_member["max_classes"]
                )

        # Function to assign a course to a faculty member
        def assign_course(faculty_member, course):
            faculty_member["hours_taught"] += course["hours"]
            faculty_member["courses_taught"].append(course["code"])
            print(
                f"Assigned {course['code']} to {faculty_member['id']}. Total hours taught: {faculty_member['hours_taught']}."
            )

        # Initialize the classroom usage tracker
        classroom_usage = {classroom: 0 for classroom in classrooms}

        # Modified function to assign a course to a faculty member
        def assign_course_to_faculty(faculty_member, course, room):
            faculty_member["hours_taught"] += course["hours"]
            faculty_member["courses_taught"].append(course["code"])
            classroom_usage[room] += 1  # Update classroom usage

        for course in tqdm(courses, desc="Scheduling Courses"):
            for fac in faculty:
                if can_teach(fac, course):
                    for timeslot in timeslots:
                        # Find the least used classroom
                        least_used_classroom = min(
                            classroom_usage, key=classroom_usage.get
                        )
                        schedule.append(
                            {
                                "course": course["code"],
                                "faculty": fac["id"],
                                "timeslot": timeslot,
                                "classroom": least_used_classroom,
                            }
                        )
                        assign_course_to_faculty(fac, course, least_used_classroom)
                        # No break here, we continue to assign to other classrooms

        for course in tqdm(courses, desc="Scheduling Courses"):
            for fac in faculty:
                if can_teach(fac, course):
                    for timeslot in timeslots:
                        for room in classrooms:
                            schedule.append(
                                {
                                    "course": course["code"],
                                    "faculty": fac["id"],
                                    "timeslot": timeslot,
                                    "classroom": room,
                                }
                            )
                            assign_course(fac, course)
                            break  # Break to ensure one room per course per timeslot

        # Convert the schedule to a DataFrame and save as CSV
        schedule_df = pd.DataFrame(schedule)
        schedule_df.to_csv("./data/preprocessed_data.csv", index=False)

        print("Full schedule generated and saved.")

        # Display the output file
        st.dataframe(schedule_df)


if data_option == "Upload a file":
    st.subheader("Upload a file")
    uploaded_file = st.file_uploader("Choose a file")

    # Read the file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    # Display the file
    if uploaded_file is not None:
        st.dataframe(df)

st.divider()

# Algorithm Option
st.subheader("2. Choose an Algorithm")
st.write(
    "Choose which algorithm to use: the Generic Algorithm or the Proposed Algorithm."
)

algorithm_option = st.selectbox(
    "Choose an option",
    ("Generic Genetic Algorithm", "Proposed Genetic Algorithm"),
)

st.divider()


# Full-time or Part-time Option
st.subheader("3. Choose a Faculty Type")
st.write("Choose which faculty type to use: Full-time or Part-time.")

faculty_option = st.selectbox(
    "Choose an option",
    ("Full-time", "Part-time"),
)

# If Full-time, make an option to choose 'Enter maximum hours you can teach per week'
if faculty_option == "Full-time":
    st.subheader("Enter maximum hours you can teach per week")
    max_hours = st.number_input("Enter a number", min_value=0, max_value=40, value=8)

# If Part-time, make an option to choose 'Enter unavailable days (comma-separated, e.g., Monday,Wednesday):'
if faculty_option == "Part-time":
    st.subheader("Enter unavailable days")
    st.write("Comma-separated, e.g., Monday,Wednesday")
    unavailable_days = st.text_input("Enter a string", value="")

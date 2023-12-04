import random
import itertools
from tqdm import tqdm
import pandas as pd

max_hours_per_week = 20

courses = [
    {"code": "CSC 0102", "title": "Discrete Structures 1", "hours": 3, "year": 1},
    {"code": "CSC 0211", "title": "Discrete Structures 2", "hours": 3, "year": 1},
    {"code": "CSC 0212", "title": "Object Oriented Programming (Lec)", "hours": 2, "year": 1},
    {"code": "CSC 0212.1", "title": "Object Oriented Programming (Lab)", "hours": 1, "year": 1},
    {"code": "CSC 0213", "title": "Logic Design and Digital Computer Circuits (Lec)", "hours": 2, "year": 1},
    {"code": "CSC 0213.1", "title": "Logic Design and Digital Computer Circuits (Lab)", "hours": 1, "year": 1},
    {"code": "CSC 0221", "title": "Algorithm and Complexity", "hours": 3, "year": 2},
    {"code": "CSC 0222", "title": "Architecture and Organization (Lec)", "hours": 2, "year": 2},
    {"code": "CSC 0222.1", "title": "Architecture and Organization (Lab)", "hours": 1, "year": 2},
    {"code": "CSC 0223", "title": "Human Computer Interaction", "hours": 3, "year": 2},
    {"code": "CSC 0224", "title": "Operation Research", "hours": 3, "year": 2},
    {"code": "CSC 0311", "title": "Automata Theory and Formal Languages", "hours": 3, "year": 3},
    {"code": "CSC 0312", "title": "Programming Languages (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0312.1", "title": "Programming Languages (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0313", "title": "Software Engineering (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0313.1", "title": "Software Engineering (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0314", "title": "Operating System (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0314.1", "title": "Operating System (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0315", "title": "Intelligent System (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0315.1", "title": "Intelligent System (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0316", "title": "Information Assurance Security", "hours": 3, "year": 3},
    {"code": "CSC 0321", "title": "Software Engineering 2 (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0321.1", "title": "Software Engineering 2 (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0322", "title": "Compiler Design (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0322.1", "title": "Compiler Design (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0323", "title": "Computational Science (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0323.1", "title": "Computational Science (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0324", "title": "CS Elective 1 (Lec)", "hours": 2, "year": 3},
    {"code": "CSC 0324.1", "title": "CS Elective 1 (Lab)", "hours": 1, "year": 3},
    {"code": "CSC 0325", "title": "Research Writing", "hours": 3, "year": 3},
    {"code": "CSC 195.1", "title": "Practicum (240 Hrs)", "hours": 2, "year": 5},
    {"code": "ICC 0101", "title": "Introduction to Computing (Lec)", "hours": 2, "year": 1},
    {"code": "ICC 0101.1", "title": "Introduction to Computing (Lab)", "hours": 1, "year": 1},
    {"code": "ICC 0102", "title": "Fundamentals of Programming (Lec)", "hours": 2, "year": 1},
    {"code": "ICC 0102.1", "title": "Fundamentals of Programming (Lab)", "hours": 1, "year": 1},
    {"code": "ICC 0103", "title": "Intermediate Programming (Lec)", "hours": 2, "year": 1},
    {"code": "ICC 0103.1", "title": "Intermediate Programming (Lab)", "hours": 1, "year": 1},
    {"code": "ICC 0104", "title": "Data Structures and Algorithms (Lec)", "hours": 2, "year": 1},
    {"code": "ICC 0104.1", "title": "Data Structures and Algorithms (Lab)", "hours": 1, "year": 1},
    {"code": "ICC 0105", "title": "Information Management (Lec)", "hours": 2, "year": 1},
    {"code": "ICC 0105.1", "title": "Information Management (Lab)", "hours": 1, "year": 1},
    {"code": "ICC 0106", "title": "Applications Development and Emerging Technologies (Lec)", "hours": 2, "year": 1},
    {"code": "ICC 0106.1", "title": "Applications Development and Emerging Technologies (Lab)", "hours": 1, "year": 1}
]

# Full-time faculty - set to exactly 4 courses
faculty = [{"id": f"FT{i}", "type": "Full-time", "max_classes": 4} for i in range(1, 7)]
print(f'faculty: {faculty}\n\n')

# Part-time faculty - set to a max of 3 courses and max 12 units
# The year level constraint is also applied
faculty += [{"id": f"PT{i}", "type": "Part-time", "max_classes": 3, "max_units": 12, "year_level": random.choice([1, 2, 3, 4])} for i in range(1, 5)]
print(f'faculty: {faculty}\n\n')

classrooms = ["Com Lab 1", "Com Lab 2", "Com Lab 3", "Com Lab 4"] + [f"GV 30{i}" for i in range(1, 8)] + [f"GCA 30{i}" for i in range(1, 6)]
print(f'classrooms: {classrooms}\n\n')

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
print(f'days_of_week: {days_of_week}\n\n')

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
print(f'timeslots: {timeslots}\n\n')

# Function to initialize faculty hours and courses taught
def initialize_faculty():
    for fac in faculty:
        fac["hours_taught"] = 0
        fac["courses_taught"] = []

# Adds ["hours_taught"] and ["courses_taught"] to faculty
initialize_faculty()
print(f'faculty: {faculty}\n\n')

"""
course,faculty,timeslot,classroom
CSC 0102,FT1,Monday 7:00 - 8:00 AM,Com Lab 1
"""
# Function to generate all possible combinations of courses, faculty, timeslots, and classrooms
def generate_combinations():
    combinations = []
    for course in courses:
        for fac in faculty:
            for timeslot in timeslots:
                for classroom in classrooms:
                    combinations.append([course["code"], fac["id"], timeslot, classroom])
    return combinations

combinations = generate_combinations()

# Create a dataframe from the combinations
df = pd.DataFrame(combinations, columns=["course", "faculty", "timeslot", "classroom"])
print(f'df: {df}\n\n')

# Just put '-' for classrooms to lessen the rows
df["classroom"] = "-"

# Remove duplicates
df.drop_duplicates(inplace=True)

# Put the dataframe into a csv file
df.to_csv("combinations.csv", index=False)

# Group by faculty
df = df.groupby("faculty")

# Show all columns
pd.set_option("display.max_columns", None)

for name, group in df:
    # Add color to print of `faculty` with its length
    print(f'\n\033[1;31m{name}\033[0m | Length: {len(group)}')
    print(group)


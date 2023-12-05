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

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
print(f'days_of_week: {days_of_week}\n\n')

# Function to calculate start and end times based on hours
def calculate_times(hours, start_hour):
    # Function to convert 24-hour format to 12-hour format with AM/PM
    def format_time(hour):
        if hour == 0 or hour == 24:
            return "12:00 AM"
        elif hour == 12:
            return "12:00 PM"
        else:
            am_pm = "AM" if hour < 12 else "PM"
            hour_12 = hour if hour <= 12 else hour - 12
            return f"{hour_12}:00 {am_pm}"

    end_hour = start_hour + hours
    start_time = format_time(start_hour)
    end_time = format_time(end_hour)
    timeslot = f"{start_time} - {end_time}"
    return timeslot

def generate_all_combinations(courses, days_of_week):
    all_combinations = []
    for course in courses:
        for day in days_of_week:
            start_hour = 7  # Start from 7 AM
            while start_hour + course["hours"] <= 21:  # Ensure end time does not exceed 9 PM
                timeslot = calculate_times(course["hours"], start_hour)
                all_combinations.append({"course": course["code"], "day": day, "timeslot": timeslot})
                start_hour += course["hours"]
    return all_combinations

# Generate all combinations
combinations = generate_all_combinations(courses, days_of_week)

# Add another column for start_time and end_time
for combination in combinations:
    start_time, end_time = combination["timeslot"].split(" - ")
    combination["start_time"] = start_time
    combination["end_time"] = end_time

# Add another column for hours
for combination in combinations:
    course = next(course for course in courses if course["code"] == combination["course"])
    combination["hours"] = course["hours"]

# Remove timeslot column
for combination in combinations:
    del combination["timeslot"]

# Convert to DataFrame
df = pd.DataFrame(combinations)

# Put the DataFrame into a CSV file
df.to_csv("combinations.csv", index=False)
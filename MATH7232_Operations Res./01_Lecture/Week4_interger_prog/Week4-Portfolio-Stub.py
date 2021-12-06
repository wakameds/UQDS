from gurobipy import *

# Part 1

# This is an example of using a dictionary to define a set and associated data
Products = {
    'Cars (Germany)': 10.3,
    'Cars (Japan)': 10.1,
    'Computers (USA)': 11.8,
    'Computers (Singapore)': 11.4,
    'Appliances (Europe)': 12.7,
    'Appliances (Asia)': 12.2,
    'Insurance (Germany)': 9.5,
    'Insurance (USA)': 9.9,
    'Short-term bonds': 3.6,
    'Medium-term bonds': 4.2
}

# Part 2

# Business as usual, downturn, upturn, crash
ScenarioProb = [0.8, 0.15, 0.04, 0.01]
S = range(len(ScenarioProb))

Year2Return = {
    'Cars (Germany)': [10.3, 5.1, 11.8, -30.0],
    'Cars (Japan)': [10.1, 4.4, 12.0, -35.0],
    'Computers (USA)': [11.8, 10.0, 12.5, 1.0],
    'Computers (Singapore)': [11.4, 11.0, 11.8, 2.0],
    'Appliances (Europe)': [12.7, 8.2, 13.4, -10.0],
    'Appliances (Asia)': [12.2, 8.0, 13.0, -12.0],
    'Insurance (Germany)': [9.5, 2.0, 14.7, -5.4],
    'Insurance (USA)': [9.9, 3.0, 12.9, -4.6],
    'Short-term bonds': [3.6, 4.2, 3.1, 5.9],
    'Medium-term bonds': [4.2, 4.7, 3.5, 6.3]
}

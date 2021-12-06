from gurobipy import *

# Sets
Nodes = [0,1,2,3,4,5,6,7,8,9]

Arcs = {
        'Line1': (0,4),
        'Line2': (1,3),
        'Line3': (2,3),
        'Line4': (3,4),
        'Unload1': (4,5),
        'Unload2': (4,5),
        'Bypass': (5,7),
        'Stacker1': (5,6),
        'Stacker2': (5,6),
        'Stacker3': (5,6),
        'Stacker4': (5,6),
        'Reclaimer1': (6,7),
        'Reclaimer2': (6,7),
        'Reclaimer3': (6,7),
        'Load1': (7,8),
        'Load2': (7,9)
    }

# Data
throughput = {
        'Line1': 100,
        'Line2': 60,
        'Line3': 60,
        'Line4': 100,
        'Unload1': 80,
        'Unload2': 80,
        'Bypass': 20,
        'Stacker1': 40,
        'Stacker2': 40,
        'Stacker3': 40,
        'Stacker4': 40,
        'Reclaimer1': 50,
        'Reclaimer2': 50,
        'Reclaimer3': 50,
        'Load1': 75,
        'Load2': 75
    }

maintain = {
    'Line3': 50,
    'Unload2': 15,
    'Bypass': 55,
    'Stacker1': 30,
    'Stacker2': 20,
    'Stacker3': 70,
    'Stacker4': 20,
    'Reclaim1': 35,
    'Reclaim2': 35,
    'Load1': 45
    }

m = Model("Coal Line Maintenance")


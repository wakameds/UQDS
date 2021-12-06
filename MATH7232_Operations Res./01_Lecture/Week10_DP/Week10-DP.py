# Strawberries

profit = [
    [0,3,7,9,12,13],
    [0,5,10,11,11,11],
    [0,4,6,11,12,12]
    ]

def strawberries(j,s):
    if j == 2:
        return (profit[j][s],s)
    else:
        return max((profit[j][a] + strawberries(j+1,s-a)[0],a,s-a)  for a in range(0,s+1))
    
# Want strawberries(0,5)

# Democracy

reps = [1.2, 1.4, 0.4]

def cities(j,s):
    if j == 2:
        return (abs(reps[j]-s),s)
        # return min(abs(reps[j] - a) for a in range(0,s+1))
    else:
        return min( (max(abs(reps[j] - a), cities(j+1,s-a)[0]),a,s-a) for a in range(0,s+1))
    
# Want cities(0,3)

# Minimum Altitude

edges = {
    ('A','B'): 10,
    ('A','D'): 6,
    ('A','C'): 7,
    ('B','E'): 9,
    ('C','E'): 7,
    ('D','E'): 11,
    ('D','F'): 7,
    ('E','G'): 8,
    ('E','H'): 7,
    ('E','I'): 10,
    ('F','G'): 8,
    ('F','H'): 6,
    ('F','I'): 7,
    ('G','J'): 13,
    ('H','J'): 8,
    ('I','J'): 9
    }

def joe(i):
    if i == 'J':
        return (0, 'Done')
    else:
        return min( (max(edges[road],joe(road[1])[0]), road[1])
                   for road in edges if road[0] == i)


# Betting Strategy
p = 0.4

def bets(j,s):
    if j == 3:
        if s >= 5:
            return (1,'Yay')
        else:
            return (0,'Sigh')
    else:
        return max( (p*bets(j+1,s+b)[0] + (1-p)*bets(j+1,s-b)[0],b) for b in range(0,s+1))

# Want bets(0,2) = (0.256,2)
# So bet $2 in first game
# If you win, bet $1 in second game
# If you get to the third game (with $4), bet $2 or $3











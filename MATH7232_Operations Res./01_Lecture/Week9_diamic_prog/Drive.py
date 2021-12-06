import math

Arcs = [
  ('X', 'A', 75),
  ('X', 'B', 105),
  ('A', 'C', 115),
  ('B', 'D', 100),
  ('C', 'D', 65),
  ('C', 'E', 95),
  ('D', 'C', 65),
  ('D', 'F', 80),
  ('E', 'G', 80),
  ('F', 'H', 75),
  ('G', 'I', 135),
  ('G', 'H', 40),
  ('H', 'G', 40),
  ('H', 'J', 125),
  ('I', 'J', 55),
  ('I', 'Y', 95),
  ('J', 'I', 55),
  ('J', 'Y', 60)]

Costs = {
  'A': (80,152),
  'B': (43,152),
  'C': (40,186),
  'D': (40,153),
  'E': (74,123),
  'F': (72,143),
  'G': (78,186),
  'H': (45,124),
  'I': (73,191),
  'J': (55,126),
  'Y': (0,120)}

FullFood = 240
FullFuel = 400
Orig = 'X'
Dest = 'Y'

UndoneLabels = set([(0,'X',FullFood,FullFuel,None)]) #start
AllLabels = set(UndoneLabels)

def AddLabel(cost,where,food,fuel,pred):
    global UndoneLabels
    for l in AllLabels:
        if l[1]==where and l[0]<=cost and l[2]>=food and l[3]>=fuel:
            return
    ### Check for and remove dominated labels in UndoneLabels
    Dominated = set([l for l in UndoneLabels if
                 l[1]==where and l[0]>=cost and l[2]<=food and l[3]<=fuel])
    UndoneLabels-=Dominated
    UndoneLabels.add((cost,where,food,fuel,pred))
    AllLabels.add((cost,where,food,fuel,pred))

while True:
    L = min(l for l in UndoneLabels)
    if L[1]=='Y':
        break
    UndoneLabels.remove(L)
    # Process all arcs that leave the node at L
    # Check they have enough food and fuel to make it
    for a in Arcs:
        if a[0]==L[1] and L[2]>=a[2] and L[3]>=a[2]:
            # Add four possible arcs
            if a[1] != 'Y':
                # Do nothing
                AddLabel(L[0],a[1],L[2]-a[2],L[3]-a[2],L)
                # Food
                AddLabel(L[0]+Costs[a[1]][0],a[1],FullFood,L[3]-a[2],L)
            # Fuel
            FuelCost = (Costs[a[1]][1]/100)*(FullFuel-(L[3]-a[2]))/10
            AddLabel(L[0]+FuelCost,a[1],L[2]-a[2],FullFuel,L)
            # Food+Fuel
            AddLabel(L[0]+Costs[a[1]][0]+FuelCost,a[1],FullFood,FullFuel,L)
            
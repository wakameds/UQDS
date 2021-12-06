from itertools import combinations

#Data#
#for Comm11
species = [1,3,4,9,11,17]
values = [10,17,22,35,44,53]
sizes = [3,5,7,9,11,13]
S = range(len(sizes))

#for comm12
Species12={
      0:[1,3,4,9,11,17],#S4
      1:[1,2,7,8,19],#S8
      2:[0,4,9,12,16],#S7
      3:[0,5,9,16],#S5
      4:[0,6,10,17],#S3
      5:[4,8,17],#S1
      6:[13,15,18],#S0
      7:[14,17],#S6
      8:[18]#S2
      }


#for comm13-15
Species={
        0:[13,15,18],
        1:[4,8,17],
        2:[18],
        3:[0,6,10,17],
        4:[1,3,4,9,11,17],
        5:[0,5,9,16],
        6:[14,17],
        7:[0,4,9,12,16],
        8:[1,2,7,8,19]
         }

Adjacent = {
        0:[1,3],
        1:[0,4],
        2:[6],
        3:[0,1,4],
        4:[1,3,5,7],
        5:[4,6],
        6:[2,5,8],
        7:[4],
        8:[6],
        }

Sites = range(9)


#Return saved species list and current state.
def SavedSpeciesList(s):
    SavedSpeciesList = []
    for i in Sites:
        if s[i] == 1:
            SavedSpeciesList = SavedSpeciesList+[j for j in Species[i] if j not in SavedSpeciesList]
    return SavedSpeciesList



#Return expected get species by action a and current state for com12.
def ExpectedSpecies12(s, a):
    SavedSpeciesList12 = []
    for i in Sites:
        if s[i] == 1:
            SavedSpeciesList12 = SavedSpeciesList12+[j for j in Species12[i] if j not in SavedSpeciesList12]
    return (len(list(set(Species12[a]) - set(SavedSpeciesList12))))



#Return expected get species at site i at the next month and current state.
def ExpectedSpecies(s, a):
    SavedSpeciesList = []
    for i in Sites:
        if s[i] == 1:
            SavedSpeciesList = SavedSpeciesList+[j for j in Species[i] if j not in SavedSpeciesList]
    return (len(list(set(Species[a]) - set(SavedSpeciesList))))



# The state is a 9-tuple where each element is
# -1 = lost, 0 = fragile, 1 = restored
# GenerateOutcomes returns a list of tuples with a probability
# in position 0 and a state as a tuple in position 1
def GenerateOutcomes(s, LossProb):
    ans = []
    tempSites = [j for j in Sites if s[j]==0]
    n = len(tempSites)
    for i in range(n+1):
        for tlist in combinations(tempSites, i):
            p = 1.0
            slist = list(s)
            for j in range(n):
                if tempSites[j] in tlist:
                    p *= LossProb[tempSites[j]]
                    slist[tempSites[j]] = -1
                else:
                    p *= 1-LossProb[tempSites[j]]
            ans.append((p, tuple(slist)))
    return ans



#Return newState and newSite we can go in the next for Comm12
def getNextState(s):
    newState = s.copy()
    for i in Sites:
        if newState[i] == 0:
            newState[i]=1
            newSite = i
            break
    return (newState, newSite)



#Return list of lists of [newState, newSite] with nextState and nextSite from state
def NextSite(s):
    newoutcomes = GenerateOutcomes(s, [0.2 for j in Sites])
    newoutcomes = [[newoutcomes[i][0], list(newoutcomes[i][1]), j] for i in range(len(newoutcomes))
    for j in Sites if newoutcomes[i][1][j] == 0]
    for i in range(len(newoutcomes)):
        newoutcomes[i][1][newoutcomes[i][2]] = 1 #will go the site in the next month
    return [[newoutcomes[i][1], newoutcomes[i][2]]for i in range(len(newoutcomes))]



#Return the number of lost adjacent site
def CheckNeighbour(s):
    #confirm site[i] is lost
    LostCount =[0,0,0,0,0,0,0,0,0]
    for i in Sites:
        if s[i] == -1:
            for j in range(len(Adjacent[i])):
                if s[Adjacent[i][j]] ==  0: #adjacent site j is fragile
                    LostCount[Adjacent[i][j]] = LostCount[Adjacent[i][j]] + 1
    return LostCount


#Return lost prob list by site         
def createProbList(s):
    ProbList = CheckNeighbour(s)
    for i in range(len(ProbList)):
        if ProbList[i] == 0:
            ProbList[i] = 0.2
        else:
            ProbList[i] = 0.2 + 0.05 *  ProbList[i]
    return ProbList
    


#Create lists with prob and newState and nextSite from newstate with lost prob 0.2
def newOutcomes12(s):
    nextSites = NextSite(s)
    newoutcomes = []
    for i in range(len(nextSites)):
        newoutcomes.append([GenerateOutcomes(nextSites[i][0], [0.2 for j in Sites]),nextSites[i][1]])    
    return newoutcomes



#Create lists with prob and newState and nextSite from newstate with considering adjecent site
def newOutcomes(s):
    nextSites = NextSite(s)
    outcomes = []
    for i in range(len(nextSites)):
        outcomes.append([GenerateOutcomes(nextSites[i][0], createProbList(s)),nextSites[i][1]])    
    return outcomes




#V11(s) is max value from loading a track of size "s"
_V11 = {}
def V11(s):
    if s < 3:
        return (0,'Full')
    else:
        if not s in _V11:
            _V11[s] = max((values[a] + V11(s-sizes[a])[0],'Species '+str(species[a]),s-sizes[a])
            for a in S if sizes[a] <= s)
        return _V11[s]



#V12 Return Expected Value    
def V12(s):
    if s.count(0)==0:
        return 0
    else:
        #take an action(chose fragile site)
        [newState, newSite] = getNextState(s)
        #Find the possible states following this action
        newoutcomes = GenerateOutcomes(newState, [0.2 for j in Sites])
        #return unique species from restoring the new site + expected future value
        return ExpectedSpecies12(s,newSite)+sum(newoutcomes[a][0]*V12(list(newoutcomes[a][1]))
                               for a in range(len(newoutcomes)))
    
    

#V13 returns maximised expected value
_V13 = {}
def V13(s):
    if s.count(0)==0:
        return (0,0)
    
    if tuple(s) not in _V13:
        #Take an action(chose fragile site)
        nextSites = NextSite(s)
        #Find the possible states following this action
        newoutcomes = newOutcomes12(s)
        #return unique species from restoring the new site + expected future value
        _V13[tuple(s)] = max((ExpectedSpecies(s, nextSites[a][1])+
             sum(newoutcomes[a][0][i][0]*V13(newoutcomes[a][0][i][1])[0]
             for i in range(len(newoutcomes[a][0]))), "S"+str(newoutcomes[a][1])) for a in range(len(newoutcomes))) 
    return _V13[tuple(s)]
    


#V14 returns maximised expected value
_V14 = {}
def V14(s):
    if s.count(0)==0:
        return (0,0)
    
    elif tuple(s) not in _V14:    
        #Take an action(chose fragile site)
        nextSites = NextSite(s)
        #Find the possible states following this action
        newoutcomes = newOutcomes(s)        
        #return unique species from restoring the new site + expected future value
        _V14[tuple(s)] = max((ExpectedSpecies(s, nextSites[a][1])+
                          sum(newoutcomes[a][0][i][0]*V14(newoutcomes[a][0][i][1])[0]
                          for i in range(len(newoutcomes[a][0]))),"S"+str(newoutcomes[a][1])) for a in range(len(newoutcomes)))
    return _V14[tuple(s)]



#V15 returns maximised probability of target species
_V15 = {}
def V15(s):
    if s.count(0) == 0:
        if len(set([1,4,9,14])-set(SavedSpeciesList(s))) == 0:
            return (1,0)
        elif len(set([1,4,9,14])-set(SavedSpeciesList(s))) != 0:
            return (0,0)

    if tuple(s) not in _V15:                           
        newoutcomes = newOutcomes(s)
        
        _V15[tuple(s)] = max((sum(newoutcomes[a][0][i][0]*V15(newoutcomes[a][0][i][1])[0]
          for i in range(len(newoutcomes[a][0]))),"S"+str(newoutcomes[a][1])) for a in range(len(newoutcomes)))
    return _V15[tuple(s)]




print("Comm11",V11(215))
print("Comm12", round(V12([0,0,0,0,0,0,0,0,0]),2))
print("Comm13", V13([0,0,0,0,0,0,0,0,0]))
print("Comm14", V14([0,0,0,0,0,0,0,0,0]))
print("Comm15", V15([0,0,0,0,0,0,0,0,0]))

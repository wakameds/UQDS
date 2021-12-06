from itertools import combinations

#Data
Sites = range(9)
Priority =["4", "8", "7", "5", "3", "1", "0", "6", "2"]

Species={
      "4":[1,3,4,9,11,17],
      "8":[1,2,7,8,19],
      "7":[0,4,9,12,16],
      "5":[0,5,9,16],
      "3":[0,6,10,17],
      "1":[4,8,17],
      "0":[13,15,18],
      "6":[14,17],
      "2":[18]
      }

#Return species at visited site, VS is list for visited sites.
def SavedSpeciesList(VS):
    SavedSpeciesList = []
    for i in Sites:
        if VS[i] == 1:
            SavedSpeciesList = SavedSpeciesList+[j for j in Species[Priority[i]] if j not in SavedSpeciesList]
    return (SavedSpeciesList)


#Return expected get species at site i. VS list for visited sites list.
def ExpectedSpecies(i, VS):
    return (len(list(set(Species[i]) - set(SavedSpeciesList(VS)))))


# The state is a 9-tuple where each element is
# -1 = lost, 0 = fragile, 1 = restored
# GenerateOutcomes returns a list of tuples with a probability
# in position 0 and a state as a tuple in position 1
def GenerateOutcomes(State, LossProb):
    ans = []
    tempSites = [j for j in Sites if State[j]==0]
    n = len(tempSites)
    for i in range(n+1):
        for tlist in combinations(tempSites, i):
            p = 1.0
            slist = list(State)
            for j in range(n):
                if tempSites[j] in tlist:
                    p *= LossProb[tempSites[j]]
                    slist[tempSites[j]] = -1
                else:
                    p *= 1-LossProb[tempSites[j]]
            ans.append((p, tuple(slist)))
    return ans

# example
outcomes = GenerateOutcomes([1,0,0,0,0,0,0,0,0], [0.2 for j in Sites])



def Comm12(state):
    if state == [0,0,0,0,0,0,0,0,0]:
        return len(Species[Priority[0]]) + Comm12([1,0,0,0,0,0,0,0,0])    
    else:
        outcomes = GenerateOutcomes(state, [0.2 for j in Sites])
        newOutcomes = [0 for i in range(len(outcomes))]
        
        for i in range(len(outcomes)):
            if 0 not in outcomes[i][1]:
                newOutcomes[i] = [outcomes[i][0], list(outcomes[i][1]), "NoZero"]   
            else:
                for n in Sites:
                    if outcomes[i][1][n] == 0:
                        lisoutcomes = list(outcomes[i][1])
                        lisoutcomes[n] = 1
                        break                       
                newOutcomes[i] = [outcomes[i][0], lisoutcomes, n]
                            
    return sum((newOutcomes[i][0]*(ExpectedSpecies(Priority[newOutcomes[i][2]], state)
                +Comm12(newOutcomes[i][1]))) for i in range(len(newOutcomes))
                if newOutcomes[i][2]!="NoZero")
                            

            
        




          

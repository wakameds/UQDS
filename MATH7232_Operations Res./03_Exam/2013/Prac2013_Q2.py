#Set
Targets = [1,2,3,4]
T = range(len(Targets))


# stage: game count(t)
# state: hit targets list(s) and next target(r)
# action: decide next target
# Value: maximised expected number of point

Points = [6, 4, 10, 7]
P = [0.8, 0.6, 0.9, 0.5]


def state(s,i):
    s[i]=1
    return s

state = [-1,-1,-1,-1]


_V={}
def V(t,s):
    if t == 9:
        return (s,'Finish')
    elif s == [1,1,1,1]:
        return 27
    else:
        for a in s:
            if a == -1:

    return _V[t,tuple(s)]

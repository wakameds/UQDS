#Stage(t) is game stage
#Data is score
#State is ptrob(s) at the stage(t)
#V(t,s) returns maxmised expected payoff
def prob(s):
    if s > 1:
        return 1
    elif s < 0:
        return 0
    else:
        return s


def V(t,s):
    if t == 9:
        return (s*5+(1-s)*1,"Defect")
    else:
        coop = s*3+V(t+1, prob(s+0.1))[0]
        defe = s*5 +(1-s)*1 + V(t+1, prob(s-0.2))[0]
    return max((coop,"Cooperate",s+0.1),(defe, "Defect",s-0.2))

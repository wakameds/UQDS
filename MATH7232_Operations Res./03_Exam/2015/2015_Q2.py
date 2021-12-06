# Stage: season (t)
# Action: the number of the hired or laid off operators
# State(s): hired operators at stage(t)
# Value: minimise the cost

Operators = [155,120,140,100,155]
cost1 = 2000
cost2 = 200
Maxreq = max(Operators)

_V = {}
def V(t,s):
    if t == 4:
        return (cost2*(s-155)**2,155-s)
    else:
        return min((cost1*(s+a-Operators[t])+cost2*a**2+V(t+1,s+a)[0],a) 
                    for a in range(Operators[t]-s, Maxreq-s+1))

s = 155
for t in range(5):
    total,a = V(t,s)
    s += a
    print("Season",t,"Hire",a,"Operators",s)

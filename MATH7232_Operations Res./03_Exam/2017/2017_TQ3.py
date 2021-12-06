
#stage: (t) season {1,2,3}
#State:(s)water height at the beginign of the season
#Action: (u) release from storage #meteres
#Valur: maxmise the revenue
#Want V(1,3)

q =[2,1,1] #cm
b =[50,100,150] #$/tonne

def V(i,s):
    if i == 3:
        return (0,0)
    else:
        return max((b[i]*(u+q[i]-0.1*(u+q[i])**2)+V(i+1, min(3,s+q[i]-u))[0],u,s+q[i]-u)
                   for u in range(s+1))
    
print("$",100*V(0,3)[0])

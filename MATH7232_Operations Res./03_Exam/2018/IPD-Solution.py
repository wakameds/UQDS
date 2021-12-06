_V = {}

#Stage: game
#State:probability
#action:corp or defe?
#V(t,p): gives maximum expected value from starting game t with probability p of cooperating

#prob is 0 <= p <= 1

def V(t,p):
	if t == 10:
		return (0,0)
	elif not (t,p) in _V:
		cooperate = p*(3 + V(t+1,min(1, p+0.1))[0]) + (1-p)*(0 + V(t+1,min(1, p+0.1))[0])
		defect = p*(5 + V(t+1,max(0, p-0.2))[0]) + (1-p)*(1 + V(t+1,max(0, p-0.2))[0])
		_V[t,p] = max((cooperate,'Coop'),(defect,'Def'))
	return _V[t,p]
		
p = 0.6
# Transitions are deterministic so we can generate sequence of actions
for t in range(10):
	payoff,decision = V(t,p)
	print(t,decision)
	if decision == 'Coop':
		p = min(1, p+0.1)
	else:
		p = max(0, p-0.2)
		
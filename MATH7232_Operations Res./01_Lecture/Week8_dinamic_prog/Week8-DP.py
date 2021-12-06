def square(x):
    return x*x

def factorial(n):
    f = 1
    for j in range(2,n+1):
        f = f*j
    return f

# recursive definition
def factorial2(n):
    if n == 0:
        return 1
    else:
        return n*factorial2(n-1)
    
passprob = [
    [.2,.3,.35,.38,.4],
    [.25,.3,.33,.35,.38],
    [.1,.3,.4,.45,.5]
    ]
# MinFail(j,s) is minimum probability of failing subjects j, ..., 2
# with s hours of study available
# We want MinFail(0,4)
#probability*Future probability
#Return probability, hour to min the prob, last time
def MinFail(j,s):
    if j == 2:
        return (1 - passprob[j][s],s,0)
    else:
        return min(((1-passprob[j][a])*MinFail(j+1,s-a)[0],a,s-a) for a in range(0,s+1)) #range

def MinFailSolution():
    s = 4
    for j in [0,1,2]:
        v = MinFail(j,s)
        print(v[1],'hours for subject',j)
        s = v[2]

# Chess(t,s) is max prob of winning match if we start game t with s points
# We want Chess(1,0)
def Chess(t,s):
    if t == 3: #Final stage game
        if s < 1:
            return (0,'Lost')
        elif s > 1:
            return (1,'Won')
        else:
            return (.45,'Bold')
    else:
        bold = .45*Chess(t+1,s+1)[0] + .55*Chess(t+1,s+0)[0] #win, lost
        conservative = .9*Chess(t+1,s+0.5)[0] + .1*Chess(t+1,s+0)[0] #draw, lost
        return max((bold,'Bold'),(conservative,'Conservative'))
    
# Play boldly on first game
# If you win, play conservatively on second game
# If you lose, play boldly on second game
# If tied on third game, play boldly

# Knapsack problem
sizes = [7,4,3]
values = [25,12,8]
J = range(len(sizes))

# Knap(s) is max value from packing a knapsack of size "s"
_Knap = {}
def Knap(s):
    if s < 3:
        return (0,'Full')
    else:
        if not s in _Knap:
            _Knap[s] = max((values[a] + Knap(s-sizes[a])[0],'Item '+str(a+1),s-sizes[a]) for a in J if sizes[a] <= s)
        return _Knap[s]
    
# Use memoization to improve efficiency
_Fib = {}  
def Fib(n):
    if not n in _Fib:
        if n <= 2:
            _Fib[n] = 1
        else:
            _Fib[n] = Fib(n-1) + Fib(n-2)
    return _Fib[n]
    
    
    
    
    
    
    
        
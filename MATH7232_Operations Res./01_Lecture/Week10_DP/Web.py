lam = {4:0.66, 8:0.77, 12:0.82}
beta = {4:0.01, 8:0.02, 12:0.03}


def alpha(s,w):
    return -0.125*w+0.005*w*s+0.4

def Vd(t,s):
    p = s-int(s)
    return p*V(t,int(s)+1)[0]+(1-p)*V(t,int(s))[0]

_V = {}

def V(t,s):
    if t == 9:
        return (s>=80, 'End')
    if s < 25:
        return (0, "Starve")
    if s >= 80:
        return (1, 'Eggs')
    if (t,s) not in _V:
        _V[t,s]= max((lam[w]*(1-beta[w])*Vd(t+1, s+6-alpha(s,w))+
          (1-lam[w])*(1-beta[w])*Vd(t+1, s-alpha(s,w)),w)
            for w in lam)
    return _V[t,s]
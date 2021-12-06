# Stage - Song # (t)
# States - Current song (i); # of skips (s)
# Action - Skip or not
# V = Max expected motivation value
# Want V(1,0,5)

m = [10, 5, 2, -2]
p = [0.5, 0.2, 0.1, 0.2]

_Spot = {}
def Spot(t, i, s):
    if t == 13:
        return (0,0)
    else:
        if (t,i,s) not in _Spot:
            if s == 0:
                yes = 0
                no = m[i] + p[0]*Spot(t+1, 0, s)[0] + p[1]*Spot(t+1, 1, s)[0] + p[2]*Spot(t+1, 2, s)[0] + p[3]*Spot(t+1, 3, s)[0]
            else:
                yes = p[0]*Spot(t, 0, s-1)[0] + p[1]*Spot(t, 1, s-1)[0] + p[2]*Spot(t, 2, s-1)[0] + p[3]*Spot(t, 3, s-1)[0]
                no = m[i] + p[0]*Spot(t+1, 0, s)[0] + p[1]*Spot(t+1, 1, s)[0] + p[2]*Spot(t+1, 2, s)[0] + p[3]*Spot(t+1, 3, s)[0]

            _Spot[t,i,s] = max((no, "No Skip"), (yes, "Skip"))

        return _Spot[t,i,s]
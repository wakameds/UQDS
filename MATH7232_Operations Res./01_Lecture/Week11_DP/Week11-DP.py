import pylab

# Advertising

revHigh = 800
revLow = 600

def advertise(t,s):
    if t == 5:
        return (0,'Done')
    else:
        if s == 'High':
            yes = .8*(revHigh-70+advertise(t+1, 'High')[0]) + .2*(revLow-70-80+advertise(t+1, 'Low')[0])
            no = .6*(revHigh+advertise(t+1, 'High')[0]) + .4*(revLow-80+advertise(t+1, 'Low')[0])
        else: # s == 'Low'
            yes = .6*(revHigh-70-80+advertise(t+1, 'High')[0]) + .4*(revLow-70+advertise(t+1, 'Low')[0])
            no = .2*(revHigh-80+advertise(t+1, 'High')[0]) + .8*(revLow+advertise(t+1, 'Low')[0])
        return max((yes,'Yes'),(no,'No'))

# Bird Song
           
psing = .004
pforage = .6
restfood = 3.6
foodpatch = 32

def singfood(i):
    return 12 + .002*i

def foragefood(i):
    return 8 + .007*i

def SongDash(t,x,m):
    i = int(x)
    p = x - i
    return p*Song(t,i+1,m)[0] + (1-p)*Song(t,i,m)[0]

def SongBlur(t,i,m):
    return 0.25*SongDash(t,i-6.4,m) + 0.5*SongDash(t,i,m) + 0.25*SongDash(t,i+6.4,m)

_Song = {}

def Song(t,i,m):
    if i <= 0:
        return (0,'Dead')
    elif t == 150:
        if m == 1:
            return (2,'Mate')
        else:
            return (1,'Lonely')
    else:
        if (t,i,m) not in _Song:
            if t >= 75:
                _Song[t,i,m] = (SongDash(t+1,i-restfood,m), 'ZZZZ')
            else:
                rest = SongDash(t+1,i-restfood,m)
                sing = psing*SongBlur(t+1,i-singfood(i),1)+ \
                        (1-psing)*SongBlur(t+1,i-singfood(i),m)
                forage = pforage*SongBlur(t+1,i-foragefood(i)+foodpatch,m)+\
                        (1-pforage)*SongBlur(t+1,i-foragefood(i),m)
                _Song[t,i,m] = max((rest,'ZZZZ'),(sing,'Sing'),(forage,'Forage'))
        return _Song[t,i,m]

def SingThreshold(t):
    i = 1
    while Song(t,i,0)[1] != 'Sing':
        i += 1
    return i

thresholds = [SingThreshold(t) for t in range(75)]
pylab.plot(range(75),thresholds)
pylab.xlabel('Time period')
pylab.ylabel('Food reserve required to sign')
pylab.show()










            
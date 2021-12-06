#Stage: distriction
#State: s the number of salesmen we can assigned
#Action: a assign salesmen
#Value: maximise profit
#Want: V(0,4)

P={
   0:[0,16,25,30,32],
   1:[0,12,17,21,22],
   2:[0,10,14,16,17]
   }


def V(n,s):
    if n ==3:
        return (0,0)
    else:
        return max((P[n][a]+V(n+1, max(s-a,0))[0],a) for a in range(s+1))
    
print(V(0,4))
    

#Stage: weeks
#state: condition of the previous week
#action: add or not
#Value: maxmise expected profit

adcost = 70
excost = 80

def profit(t,m):
    if t ==4:
        return (0,'End')
    else:
        if m =='High':
            ad = 0.8*(800-adcost+profit(t+1,'High')[0])+0.2*(600-adcost-excost+profit(t+1,'Low')[0])
            no_ad = 0.6*(800+profit(t+1,'High')[0])+0.4*(600-excost+profit(t+1,'Low')[0])
        elif m == 'Low':
            ad = 0.6*(800-adcost-excost+profit(t+1,'High')[0])+0.4*(600-adcost+profit(t+1,'Low')[0])
            no_ad = 0.2*(800-excost+profit(t+1,'High')[0])+0.8*(600+profit(t+1,'Low')[0])
        best = max((ad,"advatise"),(no_ad,"no_ad"))
    return best

print('If High',profit(0,'High'))
print('If Low',profit(0,'Low'))

print('If High',profit(1,'High'))
print('If Low',profit(1,'Low'))

print('If High',profit(2,'High'))
print('If Low',profit(2,'Low'))
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

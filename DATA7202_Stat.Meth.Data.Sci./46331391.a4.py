import numpy as np
import pandas as pd
import heapq
import random
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
inter = df.inter_arrival_time.values
service = df.service_time.values

class SimData:
    #for each customer
    def __init__(self, time_arrive, event_type, service_num ,serv_start, serv_end):
        self.m_time_arrive = time_arrive
        self.m_event_type = event_type
        self.m_service_num = service_num
        self.m_service_start = serv_start
        self.m_service_end = serv_end

    def Print(self):
        print(self.m_time_arrive, self.m_event_type, self.m_service_num, self.m_service_start, self.m_service_end)


T = 3000 #simulation time
t_current = 0
ell = []
priorityQueue = []
Queue1 = [] #physical queue for serv_1
Queue2 = [] #physical queue for serv_2
Queue3 = [] #physical queue for serv_3
Queue4 = [] #physical queue for serv_4
Queue5 = [] #physical queue for serv_5
Queue6 = [] #physical queue for serv_6
Queue7 = [] #physical queue for serv_7

#first arrival
t_current = random.choice(inter)
data = SimData(t_current,"ARRIVAL",-1,-1,-1)
heapq.heappush(priorityQueue,(t_current, data))

#create first state
time_arr = []

serv1busy = 0 #whether server1 is busy
serv2busy = 0 #whether server2 is busy
serv3busy = 0 #whether server3 is busy
serv4busy = 0 #whether server4 is busy
serv5busy = 0 #whether server5 is busy
serv6busy = 0 #whether server6 is busy
serv7busy = 0 #whether server7 is busy

#Iteration
while (t_current < T):
    obj = heapq.heappop(priorityQueue)
    t_current = obj[0]
    event = obj[1]

    #record queues length
    time_arr.append(t_current)
    Len = [len(Queue1),
           len(Queue2),
           len(Queue3),
           len(Queue4),
           len(Queue5),
           len(Queue6),
           len(Queue7)
           ]

    if (event.m_event_type == "ARRIVAL"):
        #decide service number
        event.m_service_num = Len.index(min(Len))

        #schedule the next arrival
        t_next = t_current + random.choice(inter)
        data = SimData(t_next, "ARRIVAL", -1, -1, -1) #set next customer
        heapq.heappush(priorityQueue, (t_next, data)) #time index: object

        #Service desk
        if event.m_service_num == 0:
            if serv1busy == 0:
                serv1busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))
            else:
                Queue1.append(event)

        elif event.m_service_num == 1:
            if serv2busy == 0:
                serv2busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))

            else:
                Queue2.append(event)

        elif event.m_service_num == 2:
            if serv3busy == 0:
                serv3busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))

            else:
                Queue3.append(event)

        elif event.m_service_num == 3:
            if serv4busy == 0:
                serv4busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))

            else:
                Queue4.append(event)

        elif event.m_service_num == 4:
            if serv5busy == 0:
                serv5busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))

            else:
                Queue5.append(event)

        elif event.m_service_num == 5:
            if serv6busy == 0:
                serv6busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))

            else:
                Queue6.append(event)

        elif event.m_service_num == 6:
            if serv7busy == 0:
                serv7busy = 1
                event.m_service_start = t_current
                event.m_service_end = t_current + random.choice(service)
                event.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (event.m_service_end, event))

            else:
                Queue7.append(event)

        continue


    #Departure
    elif (event.m_event_type == "DEPARTURE"):
        #handle departure from the second queue
        ell.append(event)

        if event.m_service_num == 0:
            #Waiting other customer
            serv1busy = 0
            if len(Queue1) != 0:
                serv1busy = 1
                obj_wait_inq1 = Queue1.pop(0)
                obj_wait_inq1.m_service_start = t_current
                obj_wait_inq1.m_service_end = t_current + random.choice(service)
                obj_wait_inq1.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq1.m_service_end, obj_wait_inq1))

        elif event.m_service_num == 1:
            serv2busy = 0
            if len(Queue2) != 0:
                serv2busy = 1
                obj_wait_inq2 = Queue2.pop(0)
                obj_wait_inq2.m_service_start = t_current
                obj_wait_inq2.m_service_end = t_current + random.choice(service)
                obj_wait_inq2.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq2.m_service_end, obj_wait_inq2))

        elif event.m_service_num == 2:
            serv3busy = 0
            if len(Queue3) != 0:
                serv3busy = 1
                obj_wait_inq3 = Queue3.pop(0)
                obj_wait_inq3.m_service_start = t_current
                obj_wait_inq3.m_service_end = t_current + random.choice(service)
                obj_wait_inq3.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq3.m_service_end, obj_wait_inq3))

        elif event.m_service_num == 3:
            serv4busy = 0
            if len(Queue4) != 0:
                serv4busy = 1
                obj_wait_inq4 = Queue4.pop(0)
                obj_wait_inq4.m_service_start = t_current
                obj_wait_inq4.m_service_end = t_current + random.choice(service)
                obj_wait_inq4.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq4.m_service_end, obj_wait_inq4))

        elif event.m_service_num == 4:
            serv5busy = 0
            if len(Queue5) != 0:
                serv5busy = 1
                obj_wait_inq5 = Queue5.pop(0)
                obj_wait_inq5.m_service_start = t_current
                obj_wait_inq5.m_service_end = t_current + random.choice(service)
                obj_wait_inq5.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq5.m_service_end, obj_wait_inq5))

        elif event.m_service_num == 5:
            serv6busy = 0
            if len(Queue6) != 0:
                serv6busy = 1
                obj_wait_inq6 = Queue6.pop(0)
                obj_wait_inq6.m_service_start = t_current
                obj_wait_inq6.m_service_end = t_current + random.choice(service)
                obj_wait_inq6.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq6.m_service_end, obj_wait_inq6))

        elif event.m_service_num == 6:
            serv7busy = 0
            if len(Queue7) != 0:
                serv7busy = 1
                obj_wait_inq7 = Queue7.pop(0)
                obj_wait_inq7.m_service_start = t_current
                obj_wait_inq7.m_service_end = t_current + random.choice(service)
                obj_wait_inq7.m_event_type = "DEPARTURE"
                heapq.heappush(priorityQueue, (obj_wait_inq7.m_service_end, obj_wait_inq7))

        continue

#Calculate the average waiting time
#burn-in
burn_in = int(T*0.3)
sampling_time = int((T-burn_in))

samples = []
for i in range(len(ell)):
    if (ell[i].m_time_arrive > 3000-sampling_time):
        samples.append(ell[i])


#sample waiting time
ell_w_times = []
for i in range(len(samples)):
    ell_w_times.append(samples[i].m_service_start - samples[i].m_time_arrive)


batch_samples = []
batch = 50
data_size = round(len(ell_w_times)/batch)
for i in range(0, len(ell_w_times), data_size):
    batch_samples.append(ell_w_times[i:i+data_size])

#less than 8 min
num = 8
waitcustomer = []
over8mincustomers = []

for i in range(batch):
    less8min = [element for element in batch_samples[i] if element < num]
    count_less8min = len(less8min)
    num_customers = len(batch_samples[i])
    count_over8min = num_customers - count_less8min
    waitcustomer.append(count_less8min/num_customers)
    over8mincustomers.append(count_over8min)



#the probability that the waiting time less than 8min
x = np.linspace(1,50)
plt.plot(x, waitcustomer)
plt.scatter(x, waitcustomer)
plt.xlabel('batch id')
plt.ylabel('probability')
plt.show()

#histogram of the number of the waiting users by min
plt.hist(ell_w_times)
plt.xlabel('time(min)')
plt.ylabel('customers')
plt.show()

tmeans = np.mean(ell_w_times)
tstds = np.std(ell_w_times)/np.sqrt(len(ell_w_times))
print(f'The number of customer waiting for less than 8min is {tmeans} min')
print(f'CI 95% is {tmeans - 1.96*tstds} to {tmeans + 1.96*tstds}')

#prob CI
pmeans = np.mean(waitcustomer)*100
pstds = np.std(waitcustomer)/np.sqrt(len(waitcustomer))*100
print(f'The number of customer waiting for less than 8min is {pmeans} %')
print(f'k = 7, CI 95% is {pmeans - 1.96*pstds} to {pmeans + 1.96*pstds}')


plt.plot(np.linspace(1,len(ell_w_times), len(ell_w_times)), ell_w_times)
plt.xlabel('customers')
plt.ylabel('wait time (min)')
plt.axhline(y=8, color='gray',linestyle='--')
plt.show()
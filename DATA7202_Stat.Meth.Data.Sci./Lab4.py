#Q1
#(a)
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sn
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

df = pd.read_csv("nerve.csv", index_col = 0)

# calculate the empirical cdf
X = df.values
X = np.ndarray.flatten(X)
ecdf = ECDF(X)

# plot ecdf
plt.plot(ecdf.x, ecdf.y)
plt.xlabel('x')
plt.ylabel('cdf')
plt.show()

#%%(b)
# Estimate P(0.4 <= X <= 0.6) = F(0.6) - F(0.4)
F06 = ecdf.y[ecdf.x <= 0.6]
F06 = F06[len(F06)-1]
F04 = ecdf.y[ecdf.x <= 0.4]
F04 = F04[len(F04)-1]
print("P(0.4 <= X <= 0.6) = ",F06 - F04)

#%%(c)
# Estimate skewness
np.random.seed(12345)
def skew(X):
    mu = np.mean(X)
    sigma = np.std(X)
    sk = np.mean(np.power(X-mu,3)/sigma**3)
    return sk

theta = skew(X)
print("skewness = ",theta)

#%%
# Estimate confidence interval with bootstrap method
N = 1000
ell = np.zeros(N)

for i in range(0,N):
    #sampling
    t_b = np.random.choice(X, size=len(X), replace=True) #bootstrap
    ell[i] = skew(t_b)

ell_mean = np.mean(ell)
ell_std = np.std(ell)

print("mean = ",theta, "CI = (",theta-1.96*ell_std , ", ", theta+1.96*ell_std,")")


#%%Q2
import numpy as np
import heapq #priority queue
import matplotlib.pyplot as plt

#############################################################
class SimData:

    #for each client
    def __init__(self, time_arrive, event_type, serv_1_start, serv_1_end, serv_2_start, serv_2_end):
        self.m_time_arrive = time_arrive
        self.m_event_type = event_type
        self.m_serv_1_start = serv_1_start
        self.m_serv_1_end = serv_1_end
        self.m_serv_2_start = serv_2_start
        self.m_serv_2_end = serv_2_end

    def Print(self):
        print(self.m_time_arrive, " (", self.m_event_type, ") : "
                ", ", self.m_serv_1_start, " - ", self.m_serv_1_end,
              ", ", self.m_serv_2_start, " - ", self.m_serv_2_end)
#############################################################


T = 20000 # simulation time
ell = []
priorityQueue = []
Queue1 = [] #physical queue for serv_1
Queue2 = [] #physical queue for serv_2

mu1 = 1 #mean of service1 (1/mu)
mu2 = 2 #mean of service2 (1/mu)
beta = 3 #mean of arrival time (beta is 1/lambda)

t_current = 0

# create first arrival
t_current = np.random.exponential(beta)
data = SimData(t_current,"ARRIVAL",-1,-1,-1,-1)
heapq.heappush(priorityQueue,(t_current, data))

#create first state
time_arr = []
ellq1_len = [] #size of queue1
ellq2_len = [] #size of queue2

serv1busy = 0 #whether server1 is busy
serv2busy = 0 #whether server2 is busy


# main loop
while (t_current < T):
    obj = heapq.heappop(priorityQueue)
    t_current = obj[0]
    event = obj[1]

    # record queues length
    time_arr.append(t_current)
    ellq1_len.append(len(Queue1))
    ellq2_len.append(len(Queue2))

    if (event.m_event_type == "ARRIVAL"):
        # handle arrival
        # schedule the next arrival
        t_next = t_current + np.random.exponential(beta)
        data = SimData(t_next, "ARRIVAL", -1, -1, -1, -1) #set next customer
        heapq.heappush(priorityQueue, (t_next, data)) #time index: object

        #server1 is not busy
        if (serv1busy == 0):
            serv1busy = 1
            event.m_serv_1_start = t_current
            event.m_serv_1_end = t_current + np.random.exponential(1/mu1)
            event.m_event_type = "DEPARTURE_SERV1"
            heapq.heappush(priorityQueue, (event.m_serv_1_end, event))
        else:
            Queue1.append(event)

        continue

    if (event.m_event_type == "DEPARTURE_SERV1"):
        serv1busy = 0
        # handle departure from the first queue
        if (serv2busy == 0):
            serv2busy = 1
            event.m_serv_2_start = t_current
            event.m_event_type = "DEPARTURE_SERV2"
            event.m_serv_2_end = t_current + np.random.exponential(1/mu2)
            heapq.heappush(priorityQueue, (event.m_serv_2_end, event))
        else:
            Queue2.append(event)

        if (len(Queue1) != 0):
            obj_wait_inq1 = Queue1.pop(0)
            obj_wait_inq1.m_serv_1_start = t_current
            obj_wait_inq1.m_event_type = "DEPARTURE_SERV1"
            obj_wait_inq1.m_serv_1_end = t_current + np.random.exponential(1/mu1)
            heapq.heappush(priorityQueue, (obj_wait_inq1.m_serv_1_end, obj_wait_inq1))
            serv1busy = 1
        continue

    if (event.m_event_type == "DEPARTURE_SERV2"):
        # handle departure from the second queue
        ell.append(event)
        serv2busy = 0
        if (len(Queue2) != 0):
            serv2busy = 1
            obj_wait_inq2 = Queue2.pop(0)
            obj_wait_inq2.m_serv_2_start = t_current
            obj_wait_inq2.m_event_type = "DEPARTURE_SERV2"
            obj_wait_inq2.m_serv_2_end = t_current + np.random.exponential(1/mu2)
            heapq.heappush(priorityQueue, (obj_wait_inq2.m_serv_2_end, obj_wait_inq2))
        continue



####################################################################################
# for event in ell:
#    event.Print()

####################################################################################
# Calculate the average waiting time in the system
####################################################################################

ell_w_time = np.zeros(len(ell) - 1000)

ell_w_timeQ1 = np.zeros(len(ell) - 1000)
ell_w_timeQ2 = np.zeros(len(ell) - 1000)

for i in range(0, len(ell) - 1000):
    event = ell[i + 1000]
    wait_t = event.m_serv_2_end - event.m_time_arrive
    ell_w_time[i] = wait_t

    wait_t1 = event.m_serv_1_start - event.m_time_arrive
    wait_t2 = event.m_serv_2_start - event.m_serv_1_end
    ell_w_timeQ1[i] = wait_t1
    ell_w_timeQ2[i] = wait_t2

tmean = np.mean(ell_w_time)
tstd = np.std(ell_w_time) / np.sqrt(len(ell_w_time))

print("95% CI for total time in the system (", tmean - 1.96 * tstd, " , ", tmean + 1.96 * tstd, ")")

plt.plot(time_arr, ellq1_len, label="Q1")
plt.plot(time_arr, ellq2_len, label="Q2")
plt.xlabel('time')
plt.ylabel('waiting customer')
plt.legend()
plt.show()


####################################################################################
# average waiting time in queus
####################################################################################
tmean = np.mean(ell_w_timeQ1)
tstd  = np.std(ell_w_timeQ1)/np.sqrt(len(ell_w_timeQ1))
print("95% CI for average waiting time in the first queue (",tmean - 1.96*tstd, " , ", tmean + 1.96*tstd, ")")

tmean = np.mean(ell_w_timeQ2)
tstd  = np.std(ell_w_timeQ2)/np.sqrt(len(ell_w_timeQ2))
print("95% CI for average waiting time in the second queue (",tmean - 1.96*tstd, " , ", tmean + 1.96*tstd, ")")



from gurobipy import *

def wordletters (w):
	letters = [0 for j in L]
	for c in w:
		k = ord(c) - 97
		if (k >= 0) and (k < 26):
			letters[k] = 1
	return letters

# word frequencies from http://www.wordfrequency.info
wordsfile = open('English.txt', 'r')

words = [w.strip().split(',') for w in wordsfile]

L = range(26)

letterdata = [wordletters(w[0]) for w in words]
usage = [int(w[1]) for w in words]

W = range(len(letterdata))

wordlengths = [sum(ls) for ls in letterdata]

m = Model('Minimum Alphabet')
Y = [m.addVar(vtype=GRB.BINARY) for j in L]
X = [m.addVar(vtype=GRB.BINARY) for k in W]

m.update()
m.setObjective(quicksum(usage[k]*X[k] for k in W), GRB.MAXIMIZE)

m.addConstr(quicksum(Y[j] for j in L) <= 22)

SingleLetterConstraint = {(k,j):
    m.addConstr(X[k]<=Y[j])
    for k in W for j in L if letterdata[k][j]>0}
#[m.addConstr(quicksum(letterdata[k][j]*Y[j] for j in L) >= wordlengths[k]*X[k])\
#for k in W]

m.optimize()

for j in L:
	if Y[j].x > 0.99:
		print (chr(65+j),end='')
	else:
		print (chr(97+j),end='')
print()
totalusage = sum(usage[k] for k in W)
bestusage = sum(usage[k] for k in W if X[k].x > 0.99)
print (bestusage/totalusage)

# print out the words available
#for k in W:
#	if X[k].x > 0.99:
#		print (words[k][0])

# Matching for a Dating Platform

import pandas as pd
import numpy as np
from numpy import *
import math
from math import *
import cvxpy as cvx
from cvxpy import *
data_url='https://raw.githubusercontent.com/ormarketing/OD/master/OD.csv'
df = pd.read_csv(data_url)#read the dataset
df

#Q3.1
#Creating separate matrices for male and female
f = matrix(df)[0:8,2:]
m = matrix(df)[8:,2:]
x = cvx.Variable((8,8),boolean = True) #Variable in vector form
c = (f@m.T)


#Objective function
objective = cvx.Maximize(sum(cvx.multiply(c,x)))

#Constraints
c1 = (cvx.sum(x,axis=0) == 1) #Constraint if  each person has one possible match
c2 = (cvx.sum(x,axis=1) == 1) #Constraint if  each person has one possible match
c3 = (cvx.multiply(c,x) >=0)

con = [c1,c2,c3]

prob = cvx.Problem(objective, con)
result = prob.solve()

#The optimal match scores are:
print(prob.value)

#The matched pairs table is:
print(x.value)

out = pd.DataFrame(x.value)
out.index = df['Unnamed: 0'][0:8]
out.columns = df['Unnamed: 0'][8:]
l = []
for i in range(len(out.index)):
    for j in range(len(out.columns)):
        # if the value is 1, print the row index and column index
        if out.iloc[i, j] == 1:
            l.append([out.columns[j],out.index[i]] )
l.sort()
print("The matched pairs are:")
l

#Q3.2
f = matrix(df)[0:8,2:]
m = matrix(df)[8:,2:]
x = cvx.Variable((8,8),boolean = True)
c = (f@m.T)


#Objective function
objective = cvx.Maximize(sum(cvx.multiply(c,x)))

#Constraints
c1 = (cvx.sum(x,axis=0) == 2) #Constraint if  each person has two possible matches
c2 = (cvx.sum(x,axis=1) == 2) #Constraint if  each person has two possible matches
c3 = (cvx.multiply(c,x) >=0)

con = [c1,c2,c3]

prob = cvx.Problem(objective, con)
result = prob.solve()

#The optimal match scores are:
print(prob.value)

#The matched pairs table is:
print(x.value)

out = pd.DataFrame(x.value)
out.index = df['Unnamed: 0'][0:8]
out.columns = df['Unnamed: 0'][8:]
l = []
for i in range(len(out.index)):
    for j in range(len(out.columns)):
        # if the value is 1, print the row index and column index
        if out.iloc[i, j] == 1:
            l.append([out.columns[j],out.index[i]] )

l.sort()
print("The matched pairs are:")
l

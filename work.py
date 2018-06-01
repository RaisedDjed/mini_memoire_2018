import random as r
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def UUniform(u, n):
    totalUtilisation = 2
    while totalUtilisation > 1:
        totalUtilisation = 0;
        taskSet = []
        for i in range(1, n):
            utilisation =  r.random()
            totalUtilisation += utilisation
            taskSet.append(utilisation)
    taskSet.append(1-totalUtilisation)
    for i in range(0, len(taskSet)):
        taskSet[i] = taskSet[i] * u
    return taskSet

def UuniFast(u, n):
    previous = u;
    taskSet = []
    for i in range(1, n):
       current = previous * r.random() ** (1/(n-i));
       taskSet.append(previous - current)
       previous = current
    taskSet.append(previous)
    return taskSet

def UuniFastDiscard(u, n, m):
    listTaskSet = []
    i = 0;
    while i < m:
        previous = u;
        taskSet = []
        j=1
        while j<n:
           current = previous * r.random() ** (1/(n-j));
           utilisation = previous - current
           taskSet.append(utilisation)
           previous = current
           j+=1;
           if (utilisation>1):
               j+=n
        taskSet.append(previous)
        if utilisation < 1 and previous < 1:
            listTaskSet.append(taskSet)
            i+=1
    return np.array(listTaskSet)

def StaffordRandFixedSum(n, u, nsets):
    """
    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
    EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    The views and conclusions contained in the software and documentation are
    those of the authors and should not be interpreted as representing official
    policies, either expressed or implied, of Paul Emberson, Roger Stafford or
    Robert Davis.
    Includes Python implementation of Roger Stafford's randfixedsum implementation
    http://www.mathworks.com/matlabcentral/fileexchange/9700
    Adapted specifically for the purpose of taskset generation with fixed
    total utilisation value
    Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
    any questions regarding this software.
    """
    if n < u:
        return None

    #deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = min(int(u), n - 1)
    s = u
    s1 = s - np.arange(k, k - n, -1.)
    s2 = np.arange(k + n, k, -1.) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    x = np.zeros((n, nsets))
    rt = np.random.uniform(size=(n - 1, nsets))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, nsets))  # rand position in simplex
    s = np.repeat(s, nsets)
    j = np.repeat(k + 1, nsets)
    sm = np.repeat(0, nsets)
    pr = np.repeat(1, nsets)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
        sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
        sm = sm + (1.0 - sx) * pr * s / (i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required
    x[n - 1, ...] = sm + pr * s

    #iterated in fixed dimension order but needs to be randomised
    #permute x row order within each column
    for i in range(0, nsets):
        x[..., i] = x[np.random.permutation(n), i]

    return x.T.tolist()

def RipollEtAl(m, maxComputeTime, maxSlackTime, maxDelay, u):
    listSet = []
    for nSet in range(0, m):
        newSet =[]
        current_utilisation=0
        while current_utilisation<u:
            c = r.random()*maxComputeTime
            d = c + r.random()*maxSlackTime
            p = d + r.random()*maxDelay
            newSet.append(c/p)
            current_utilisation+=c/p
        listSet.append(newSet)
    return listSet

def GoosensEtMack (u1, u2, d1, d2, o1, o2, M, V, U, n):
    current_utilisation = 0
    taskSet =[]
    i=0
    current_utilisation=0
    while current_utilisation<U and i<n:
        i+=1
        Ti = 1
        for j in range(0, len(M)):
            p = round(r.uniform(0, len(M[0])-1))
            Ti = Ti * M[j][p]
        Ci = max(1, (r.uniform(u1, u2) * Ti))
        Oi = round(r.uniform(o1, o2) * Ti)
        Di = round((Ti-Ci) *  r.uniform(d1, d2)) + Ci
        if current_utilisation+(Ci/Ti)<=1:
            taskSet.append(Ci/Ti)
            current_utilisation+=(Ci/Ti)
    return taskSet
            
def drawCurve(newList):
    s = [0.01 for n in range(len(newList[0]))]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(newList[0], newList[1], newList[2], s=s, c="black", marker="o")
    ax.view_init(30, 45)
    plt.show()
    
def oneDimensionBiasTest(listArg):
    newList=[]
    counter=0
    for j in range(0,3):
        newList.append([])
    for l in listArg:
        for i in range(0,3):
            newList[i].append(l[i])
    drawCurve(newList)

def UuniFastCreation():
    listItem=[]
    for i in range(0, 20000):
        listItem.append(UuniFast(0.98, 3))
    return listItem

def UuniFastProduction():
    listItem=UuniFastCreation()
    oneDimensionBiasTest(listItem)

def UuniformCreation():
    listItem=[]
    for i in range(0, 20000):
        listItem.append(UUniform(0.98, 3))
    return listItem

def UuniformProduction():
    listItem = UuniformCreation()
    oneDimensionBiasTest(listItem)
    
def UuniFastDiscardCreation():
    listItem=[]
    for i in range(0, 20000):
        listItem.append(UuniFastDiscard(0.98, 3, 1)[0])
    return listItem

def UuniFastDiscardProduction():
    listItem =UuniFastDiscardCreation 
    oneDimensionBiasTest(listItem)
    
def StaffordCreation():
    listItem=[]
    for i in range(0, 20000):
        listItem.append(StaffordRandFixedSum(3, 0.98, 1)[0])
    return listItem

def StaffordRandFixedSumProduction():
    listItem =StaffordCreation()
    oneDimensionBiasTest(listItem)

def RipollEtAlProduction():
    listItem=[]
    for i in range(0, 20000):
        j=5
        su=2
        while j!=3 or su>0.99:
            tmp = RipollEtAl(1,1,1,1,0.98)[0]
            j=len(tmp)
            su=sum(tmp)
        listItem.append(tmp)
    oneDimensionBiasTest(listItem)

def GoosensEtAlProduction():
    listItem=[]
    for i in range(0, 20000):
        j=5
        su=2
        while j!=3 or su>0.99 or su<0.97:
            tmp = GoosensEtMack (0, 1, 0, 1, 0, 0, M, 0, 0.98, 3)
            j=len(tmp)
            su=sum(tmp)
        listItem.append(tmp)
    oneDimensionBiasTest(listItem)

def RipollCreation():
    listItem =[]
    for i in range(0,20000):
        listItem.append(RipollEtAl(1, 1, 1, 0, 0.98)[0])
    return listItem

def testNonBiasedRipollEtAl():
    listItem =[]
    for i in range(0,20000):
        listItem.append(len(RipollEtAl(1, 1, 1, 0, 0.98)[0]))
    newList=[]
    for i in range(0, max(listItem)+1):
        newList.append(0)
    for i in listItem:
        newList[i]+=1
    plt.plot(newList, c="black")
    plt.xlabel('cardinality', fontsize=12)
    plt.ylabel('number of sets', fontsize=12)
    plt.show()

def GoosensCreation():
    listItem =[]
    for i in range(0,20000):
        listItem.append(GoosensEtMack (0, 1, 0, 1, 0, 0, M, 0, 0.98, 10))
    return listItem

def testNonBiasedGoosensEtMacq():
    listItem =[]
    for i in range(0,20000):
        listItem.append(len(GoosensEtMack (0, 1, 0, 1, 0, 0, M, 0, 0.98, 10)))
    newList=[]
    for i in range(0, max(listItem)+1):
        newList.append(0)
    for i in listItem:
        newList[i]+=1
    plt.plot(newList, c="black")
    plt.xlabel('cardinality', fontsize=12)
    plt.ylabel('number of sets', fontsize=12)
    plt.show()

def getCorrectListDelta(listItem):
    listDelta=[]
    listCounter=[]
    for i in range(0, 26):
        listCounter.append(0)
    for i in listItem:
        delta = (max(i)-min(i))/sum(i)
        for j in range(0, len (listCounter)):
            if delta*25>j and delta*25<j+1:
                listCounter[j]+=1
    return listCounter

def testDeltaUuniform():
    listItem=UuniformCreation()
    listCounter=getCorrectListDelta(listItem)
    listItem=UuniFastCreation()
    listCounter2=getCorrectListDelta(listItem)
    listItem=UuniFastDiscardCreation()
    listCounter3=getCorrectListDelta(listItem)

    listItem=StaffordCreation()
    listCounter4=getCorrectListDelta(listItem)
    
    listItem=RipollCreation()
    listCounter5=getCorrectListDelta(listItem)
    
    listItem=GoosensCreation()
    listCounter6=getCorrectListDelta(listItem)
    listItem=[]
    fig = plt.figure(figsize=(6, 4.4))
    for i in range(0,26):
        listItem.append(i/25.0)
    plt.plot(listItem, listCounter, c="red", label="Uuniform")
    plt.plot(listItem, listCounter2, c="yellow", label="UuniFast")
    plt.plot(listItem, listCounter3, c="orange", label="UuniFastDiscard")
    plt.plot(listItem, listCounter4, c="blue", label="Stafford")
    plt.plot(listItem, listCounter5, c="green", label="Ripoll et Al")
    plt.plot(listItem, listCounter6, c="black", label="Goosens and Macq")
    plt.xlabel('δ', fontsize=12)
    plt.legend(["Uuniform", "UuniFast", "UuniFastDiscard", "Stafford",
               "Ripoll et Al", "Goosens and Macq"], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
    plt.ylabel('number of sets', fontsize=12)
    fig.savefig('samplefigure', bbox_inches='tight')
    plt.show()
        

M = [[1,1,1,1,4,4,4,8,1],
     [1,3,3,3,3,9,9,27,27],
     [1,5,1,1,1,1,1,1,1],
     [1,7,7,7,1,1,1,1,1],
     [1,1,13,1,1,1,1,1,1],
     [1,1,1,17,17,1,1,1,1],
     [1,1,1,1,19,1,1,1,1]]

import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from math import pi
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

import Tools


if __name__ == "__main__":
    data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))
    print('data save location:',data_path)

    ##############################
    ## hyperparameters

    if 0: #Settings for Example 1:
        n = 100
        R = 2
        ExampleNumber = 1

        xMin,xMax = -0.5,6.5 #edges of window
        eta = 0.05 #distance of the z-points where s_nu is calculated from the real line
        N = 200  #number of points in plot
        BinNr = 40 #number of bars in histogram

    if 0: #Settings for Example 2:
        n = 200
        R = 4
        ExampleNumber = 2

        xMin,xMax = -0.5,8.5 #edges of window
        eta = 0.05 #distance of the z-points where s_nu is calculated from the real line
        N = 200  #number of points in plot
        BinNr = 40 #number of bars in histogram

    if 1: #Settings for Example 3:
        n = 100
        R = 4
        ExampleNumber = 3

        xMin,xMax = -0.5,26 #edges of window
        #eta = 0.1 #distance of the z-points where s_nu is calculated from the real line
        eta = 0.05 #distance of the z-points where s_nu is calculated from the real line
        N = 200  #number of points in plot
        BinNr = 40 #number of bars in histogram

    ##############################


    Y, A, B = Tools.ExampleMaker(n,ExampleNumber,R=R)

    d,_ = Y.shape
    print('d=',d,', n=',n)

    tau = Tools.getTau(A,B)
    
    S = Y@Y.H/n
    tS = Y.H@Y/n
    tSampEV,_ = np.linalg.eigh(tS)
    #print(tSampEV)

    zVec = np.linspace(xMin,xMax,N) + 1j*eta

    #preparations for finding determeinistic equivalents delA and delB
    MixMatrA = dict()
    MixMatrB = dict()
    for r in range(R):
        for s in range(R):
            MixMatrA[(r,s)] = A[r]@A[s].H
            MixMatrB[(r,s)] = B[s].H@B[r]

    s_nu = []
    i = 0
    for z in zVec:
        i+=1
        print(i,'/',len(zVec))

        #find empirical versions for a warm start
        empDelA,empDelB = Tools.getEmpiricalDeltas(Y,z,R,MixMatrA,MixMatrB)
        
        #find determeinistic equivalents delA and delB
        delA,delB = Tools.solveDualMP(z,A,B,Start=(empDelA,empDelB),MixMatrA=MixMatrA,MixMatrB=MixMatrB,eps=10**-4*R)
        
        #find the Stieltjes transform of \nu
        s_nu_z = -1/z-np.sum(delA*delB)
        s_nu.append(s_nu_z)

    plt.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(1,1,layout='constrained', figsize=(8, 5.6))
    ax.set_title(r'Predicted $\underline{\nu}$ vs. empirical $\hat{\underline{\nu}}$ for Example '+str(ExampleNumber)+'\n d='+str(d)+', n='+str(n)+', R='+str(R)+r', $\tau$='+str(round(tau,2)))


    ax.hist(tSampEV,bins=40,density=True,alpha=0.5,edgecolor='black')
    ax.plot(np.real(zVec),np.imag(s_nu)/pi, linewidth=2)

    plt.show()
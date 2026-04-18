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
        n_List = [10*i for i in range(1,11)]
        R = 2
        ExampleNumber = 1

        zPoints = [1.5+1j, 1.5+0.1j]

        N = 25
        M_maker = lambda d: np.eye(d)
        tM_maker = lambda n: np.ones((n,n))/np.sqrt(n)

    if 0: #Settings for Example 2:
        n_List = [10*i for i in range(1,11)]
        R = 4
        ExampleNumber = 2

        zPoints = [6+0.1j, 0.5+0.1j]
        N = 25
        M_maker = lambda d: np.diag([i/d for i in range(1,d+1)])
        tM_maker = lambda n: np.diag([1]*(n//2)+[0]*(n-n//2))

    if 1: #Settings for Example 3:
        n_List = [10*i for i in range(1,11)]
        R = 4
        ExampleNumber = 3

        zPoints = [0+0.1j,2+0.1j]
        N = 25
        M_maker = lambda d: np.diag([1]+[0]*(d-1))
        tM_maker = None

    n_List = np.array(n_List)
    ColorMap = dict()
    ColorMap[(0,'A')] = "#4C78A8"   # blue / light blue
    ColorMap[(0,'B')] = "#9ECAE9"
    ColorMap[(1,'A')] = "#E15759"   # red / soft pink-red
    ColorMap[(1,'B')] = "#FF9D9A"
    ColorMap[(2,'A')] = "#59A14F"   # green / light green
    ColorMap[(2,'B')] = "#8CD17D"

    ##############################

    DifferencesA = np.zeros((len(n_List), len(zPoints), N))
    DifferencesB = np.zeros((len(n_List), len(zPoints), N))
    tauMin = np.inf
    j=0
    for n in n_List:
        print('n=',n)
        for I in range(N):
            print((I+1),'/',N)
            Y, A, B = Tools.ExampleMaker(n,ExampleNumber,R=R)

            d,_ = Y.shape

            tau = Tools.getTau(A,B)
            if tau < tauMin:
                tauMin = tau

            M = M_maker(d)
            if ExampleNumber==3:
                tM = B[0]
            else:
                tM = tM_maker(n)
            
            S = Y@Y.H/n
            tS = Y.H@Y/n
            tSampEV,_ = np.linalg.eigh(tS)
            #print(tSampEV)

            #preparations for finding determeinistic equivalents delA and delB
            MixMatrA = dict()
            MixMatrB = dict()
            for r in range(R):
                for s in range(R):
                    MixMatrA[(r,s)] = A[r]@A[s].H
                    MixMatrB[(r,s)] = B[s].H@B[r]

            s_nu = []
            i = 0
            for z in zPoints:
                #find empirical versions for a warm start
                empDelA,empDelB = Tools.getEmpiricalDeltas(Y,z,R,MixMatrA,MixMatrB)
                
                #find determeinistic equivalents delA and delB
                delA,delB = Tools.solveDualMP(z,A,B,Start=(empDelA,empDelB),MixMatrA=MixMatrA,MixMatrB=MixMatrB,eps=10**-4*R)
                
                #Construct deterministic equivalents of the resolvents
                QA = np.eye(d,dtype='complex')
                QB = np.eye(n,dtype='complex')
                for r in range(R):
                    for s in range(R):
                        QA += delB[r,s]*MixMatrA[(r,s)]
                        QB += delA[r,s]*MixMatrB[(r,s)]
                R_detEquiv = -np.linalg.inv(QA)/z
                tR_detEquiv = -np.linalg.inv(QB)/z

                #Construct resolvents
                Rz = np.linalg.inv(S-z*np.eye(d))
                tRz = np.linalg.inv(tS-z*np.eye(n))

                #Find approximation error
                DiffA = np.trace(M@(Rz - R_detEquiv))/n
                DiffB = np.trace(tM@(tRz - tR_detEquiv))/n
                DifferencesA[j,i,I] = np.abs(DiffA)
                DifferencesB[j,i,I] = np.abs(DiffB)

                i += 1
                #print(i,'/',len(zPoints))

        j += 1

    offsets = np.linspace(-2, 2, 2*len(zPoints))
    plt.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(1,1,layout='constrained', figsize=(8, 5.6))
    ax.set_title(r'Estimation errors for Example '+str(ExampleNumber)+'\n R='+str(R)+r', $\tau$='+str(round(tau,2)))


    for i in range(len(zPoints)):
        MeansA = DifferencesA[:,i,:].mean(axis=1)
        lowerA = np.quantile(DifferencesA[:,i,:], 0.1, axis=1)
        upperA = np.quantile(DifferencesA[:,i,:], 0.9, axis=1)
        lowerA_err = MeansA - lowerA
        upperA_err = upperA - MeansA

        MeansB = DifferencesB[:,i,:].mean(axis=1)
        lowerB = np.quantile(DifferencesB[:,i,:], 0.1, axis=1)
        upperB = np.quantile(DifferencesB[:,i,:], 0.9, axis=1)
        lowerB_err = MeansB - lowerB
        upperB_err = upperB - MeansB


        
        ax.errorbar(n_List+offsets[2*i],MeansA,yerr=[lowerA_err, upperA_err],color=ColorMap[(i,'A')],fmt='o',capsize=4,label=r'$A$-Errors for point '+str(i+1))
        ax.errorbar(n_List+offsets[2*i+1],MeansB,yerr=[lowerB_err, upperB_err],color=ColorMap[(i,'B')],fmt='o',capsize=4,label=r'$B$-Errors for point '+str(i+1))
        if 0: #fit power law
            Ignore=2
            C,alpha = Tools.fit_power_law(n_List[Ignore:],MeansA[Ignore:])
            ax.plot(n_List,[C*n**(alpha) for n in n_List],color=ColorMap[(i,'A')],linestyle='dashed',alpha=0.3,label=r'fitted $\frac{C}{n^{\alpha}}, \alpha=$'+'{}'.format(np.round(-alpha,decimals=2)))

            C,alpha = Tools.fit_power_law(n_List[Ignore:],MeansB[Ignore:])
            ax.plot(n_List,[C*n**(alpha) for n in n_List],color=ColorMap[(i,'B')],linestyle='dashed',alpha=0.3,label=r'fitted $\frac{C}{n^{\alpha}}, \alpha=$'+'{}'.format(np.round(-alpha,decimals=2)))


    ax.semilogy()
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.show()
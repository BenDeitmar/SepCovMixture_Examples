import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from math import pi, ceil
#from scipy.optimize import minimize
#from scipy.optimize import approx_fprime
from scipy.optimize import curve_fit
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve

na = np.newaxis

def generateComplexNormalMatrix(d,n):
    X = np.random.normal(size=(d,n))
    Y = np.random.normal(size=(d,n))
    Z = np.matrix(X+1j*Y)/np.sqrt(2)
    return Z

def getHaarUnitary(d):
    Z = generateComplexNormalMatrix(d,d)
    _,U = np.linalg.eigh(Z@Z.H)
    return U

def getPermutationMatrix(n):
    perm = np.random.permutation(n)
    P = np.eye(n, dtype=int)[perm]
    return P

def getEmpiricalDeltas(Y,z,R,MixMatrA,MixMatrB):
    d,n = Y.shape

    Rz = np.linalg.inv(Y@Y.H/n-z*np.eye(d))
    tRz = np.linalg.inv(Y.H@Y/n-z*np.eye(n))

    empDelA = np.zeros((R,R),dtype='complex')
    empDelB = np.zeros((R,R),dtype='complex')
    for r in range(R):
        for s in range(R):
            empDelA[r,s] = np.trace(MixMatrA[(r,s)]@Rz)/n
            empDelB[r,s] = np.trace(MixMatrB[(r,s)]@tRz)/n

    return empDelA,empDelB


def traces_of_A_Binv(B, A_list, positive_definite=False):
    #B = np.asarray(B)
    n,_ = B.shape

    factor = lu_factor(B, overwrite_a=False, check_finite=False)
    solve_with_B = lambda A: lu_solve(factor, A, check_finite=False)

    traces = []
    for A in A_list:
        A = np.asarray(A)
        X = solve_with_B(A)
        traces.append(np.trace(X))

    return np.asarray(traces)

def solveDualMP(z,A,B,maxIt=3000,eps=10**-6,Start=None,MixMatrA=None,MixMatrB=None):
    R = len(A)
    d,_ = A[0].shape
    n,_ = B[0].shape

    last_delA = np.zeros((R,R),dtype='complex')
    last_delB = np.zeros((R,R),dtype='complex')
    if Start is None:
        delA = 1j*np.eye(R,dtype='complex')
        delB = 1j*np.eye(R,dtype='complex')
    else:
        delA = Start[0]
        delB = Start[1]

    if MixMatrA is None:
        MixMatrA = dict()
        for r in range(R):
            for s in range(R):
                MixMatrA[(r,s)] = A[r]@A[s].H
    if MixMatrB is None:
        MixMatrB = dict()
        for r in range(R):
            for s in range(R):
                MixMatrB[(r,s)] = B[s].H@B[r]

    IndexList = list(MixMatrA.keys())
    ProdListA = [MixMatrA[key] for key in IndexList]
    ProdListB = [MixMatrB[key] for key in IndexList]

    for i in range(maxIt):
        Diff = np.linalg.norm(last_delA-delA,ord='fro')+np.linalg.norm(last_delB-delB,ord='fro')
        #print(i,Diff)
        if Diff < eps:
            print('iteration successful after:',i)
            break;
        last_delA,last_delB = delA.copy(),delB.copy()

        QA = np.eye(d,dtype='complex')
        QB = np.eye(n,dtype='complex')
        for r in range(R):
            for s in range(R):
                QA += delB[r,s]*MixMatrA[(r,s)]
                QB += delA[r,s]*MixMatrB[(r,s)]

        TraceListA = traces_of_A_Binv(QA, ProdListA)
        TraceListB = traces_of_A_Binv(QB, ProdListB)

        for i in range(len(IndexList)):
            indexes = IndexList[i]
            r,s = indexes
            delA[r,s] = -TraceListA[i]/z/n
            delB[r,s] = -TraceListB[i]/z/n

    return delA, delB

def getTau(A,B):
    R = len(A)
    d,_ = A[0].shape
    n,_ = B[0].shape

    ASum = np.zeros((d,d),dtype='complex')
    BSum = np.zeros((n,n),dtype='complex')

    for r in range(R):
        ASum += A[r]@A[r].H
        BSum += B[r].H@B[r]

    ASum_eig,_ = np.linalg.eigh(ASum)
    BSum_eig,_ = np.linalg.eigh(BSum)

    GA = np.zeros((R,R),dtype='complex')
    GB = np.zeros((R,R),dtype='complex')

    for r in range(R):
        for s in range(R):
            GA[r,s] = np.trace(A[r]@A[s].H)/n
            GB[r,s] = np.trace(B[s].H@B[r])/n

    GA_eig,_ = np.linalg.eigh(GA)
    GB_eig,_ = np.linalg.eigh(GB)

    tau = min(np.min(ASum_eig),np.min(BSum_eig),np.min(GA_eig),np.min(GB_eig))

    return tau


def ExampleMaker(n,ExampleNumber,R=2):
    if ExampleNumber==1:
        R = 2
        d = 5*n

        #constructing V_d
        J = np.arange(d, dtype=np.float64)
        phase = (-2j*pi/d)*np.outer(J, J)
        Vd = np.exp(phase)/np.sqrt(d)

        #Defining A's and B's
        B1 = np.diag([1 for i in range(ceil(n/2))]+[0 for i in range(n-ceil(n/2))])
        B2 = np.diag([0 for i in range(ceil(n/2))]+[1 for i in range(n-ceil(n/2))])
        A1 = np.diag([1/3 for i in range(ceil(d/2))]+[0 for i in range(d-ceil(d/2))])
        A2 = Vd@np.diag([1/2 for i in range(ceil(d/2))]+[1 for i in range(d-ceil(d/2))])

        A = [np.matrix(A1),np.matrix(A2)]
        B = [np.matrix(B1),np.matrix(B2)]

        X = generateComplexNormalMatrix(d,n)

        Y = A1@X@B1 + A2@X@B2

    if ExampleNumber==2:
        d = n
        assert R<=n

        A1 = np.matrix(np.eye(d))
        B1 = np.matrix(np.eye(n,k=0))
        A = [A1]
        B = [B1]

        X = np.random.choice([-1,1],size=(d,n)).astype(complex)
        Y = A1@X@B1

        for r in range(1,R):
            Br = np.eye(n,k=r)
            Ur = getHaarUnitary(d)
            Ar = Ur@np.diag([i/d for i in range(1,d+1)])
            #Ar = np.vstack([np.ones((1, d)), np.zeros((d - 1, d))]).T
            #Ar = np.ones((d,d))/np.sqrt(d)
            #Ar = np.tile(np.eye(d)[r][:, None], (1, d))/np.sqrt(d)
            #Ar = np.eye(d)

            Y += Ar@X@Br

            B.append(np.matrix(Br))
            A.append(np.matrix(Ar))

    if ExampleNumber==3:
        d = 2*n
        X = np.sqrt(5/7)*np.random.standard_t(df=7, size=(d, n))
        Y = np.zeros((d,n),dtype='complex')

        A = []
        B = []
        for r in range(R):
            Ar = getPermutationMatrix(d)
            Br = getPermutationMatrix(n)
            Y += Ar@X@Br
            A.append(np.matrix(Ar))
            B.append(np.matrix(Br))

    return np.matrix(Y), A, B

def power_law_model(x, C, alpha):
    return C * x**alpha

def fit_power_law(dL, fL, alpha_bounds=(-np.inf, np.inf)):
    x = np.asarray(dL, dtype=float)
    y = np.asarray(fL, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise ValueError("Need at least two data points to fit.")

    # Initial guess using log–log fit on positive points
    pos = (x > 0) & (y > 0)
    if pos.sum() >= 2:
        X = np.log(x[pos])
        Y = np.log(y[pos])
        a, b = np.polyfit(X, Y, 1)
        C0 = float(np.exp(b))
        alpha0 = float(a)
        if not np.isfinite(C0) or C0 <= 0:
            C0 = 1.0
        if not np.isfinite(alpha0):
            alpha0 = 0.0
    else:
        C0, alpha0 = 1.0, 0.0

    # Bounds: C >= 0, alpha between alpha_bounds
    (amin, amax) = alpha_bounds
    popt, pcov = curve_fit(
        power_law_model,
        x, y,
        p0=[C0, alpha0],
        bounds=([0.0, amin], [np.inf, amax]),
        maxfev=10000
    )

    C_hat, alpha_hat = popt
    return C_hat, alpha_hat

if __name__ == "__main__":
    data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))
    print('data save location:',data_path)

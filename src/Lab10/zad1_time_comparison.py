
import math
import numpy as np
import matplotlib.pyplot as plt
import time


def crout(A):
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    for j in range(n):
        U[j][j] = 1

        for i in range(j, n):
            alpha = float(A[i][j])
            for k in range(j):
                alpha -= L[i][k]*U[k][j]
            L[i][j] = alpha

        for i in range(j+1, n):
            uu = float(A[j][i])
            for k in range(j):
                uu -= L[j][k]*U[k][i]

            if L[j][j] == 0:
                raise ZeroDivisionError("0 occurred on diagonal. Could not compute U")

            U[j][i] = uu/L[j][j]

    return L, U


def doolittle(A):
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = 1
        for i in range(j + 1):
            s1 = 0
            for k in range(i):
                s1 += U[k][j] * L[i][k]
            U[i][j] = A[i][j] - s1
        for i in range(j, n):
            s2 = 0
            for k in range(j):
                s2 += U[k][j] * L[i][k]
            L[i][j] = (A[i][j] - s2) / U[j][j]

    return L, U


def cholesky(A):
    n = len(A)
    L = [[0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if i == k:
                L[i][k] = math.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = 1 / L[k][k] * (A[i][k] - tmp_sum)

    return L, np.array(L).transpose()  # L, U


def solve(A, b, L, U):
    z = np.linalg.solve(L, b)
    x = np.linalg.solve(U, z)
    return x


def random_matrix(low, high, n):  # returns random symmetric and positive matrix
    A = np.random.randint(low, high, size=(n, n))
    return np.dot(A, A.transpose()), np.random.randint(low, high, size=(n, 1))


def compare(A, b):
    L1, U1 = crout(A)
    L2, U2 = doolittle(A)
    L3, U3 = cholesky(A)

    print(f"\nCrout:\nL:")
    for i in range(len(L1)):
        print(L1[i])
    print(f"U:")

    for i in range(len(U1)):
        print(U1[i])
    print(f"\nDoolittle:\nL:")

    for i in range(len(L2)):
        print(L2[i])
    print(f"U:")
    for i in range(len(U2)):
        print(U2[i])

    print(f"\nCholesky:\nL:")
    for i in range(len(L3)):
        print(L3[i])
    print(f"U:")
    for i in range(len(U3)):
        print(U3[i])

    print(f"\nSolve using Crout:\n {solve(A, b, L1, U1)}")
    print(f"Solve using Doolittle:\n {solve(A, b, L2, U2)}")
    print(f"Solve using Cholesky:\n {solve(A, b, L3, U3)}")


def compare_times():
    tab1 = []
    tab2 = []
    tab3 = []
    x_axis = []
    for n in range(299, 500):
        A, b = random_matrix(-1000, 1000, n)
        start_time_1 = time.time()
        crout(A)
        stop_time_1 = time.time()
        start_time_2 = time.time()
        doolittle(A)
        stop_time_2 = time.time()
        start_time_3 = time.time()
        cholesky(A)
        stop_time_3 = time.time()
        t1 = stop_time_1 - start_time_1
        t2 = stop_time_2 - start_time_2
        t3 = stop_time_3 - start_time_3
        print(f"\nMatrix {n}x{n}:")
        print(f"    Crout:     {t1}s")
        print(f"    Doolittle: {t2}s")
        print(f"    Cholesky:  {t3}s")
        tab1.append(t1)
        tab2.append(t2)
        tab3.append(t3)
        x_axis.append(n)

    plt.plot(x_axis, tab1, 'b.', label="Crout")
    plt.plot(x_axis, tab2, 'r.', label="Doolittle")
    plt.plot(x_axis, tab3, 'g.', label="Cholesky")
    plt.xlabel("Square matrix size")
    plt.ylabel("Time [s]")
    plt.title("Time of evaluation for LU decompositions")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    compare_times()
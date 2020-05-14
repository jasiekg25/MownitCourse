




# without pivoting
import numpy
from tabulate import tabulate
def forward_elimination(A, b):
    n = len(A)
    for k in range(0, n-1):
        for i in range(k+1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] = A[i][j] - factor * A[k][j]
            b[i] = b[i] - factor * b[k]
    return A, b


def back_substitution(A, b):
    n = len(A)
    x = [0 for _ in range(n)]
    x[n-1] = b[n-1] / A[n-1][n-1]
    for k in range(n-2, -1, -1):
        sums = b[k]
        for j in range(k+1, n):
            sums = sums - A[k][j] * x[j]
        x[k] = sums / A[k][k]
    return x


def without_pivot(A, b):
    for i in range(len(A)):
        if len(A) != len(A[i]):
            raise ZeroDivisionError('Division by zero will occur; pivoting currently not supported')
    A, b = forward_elimination(A, b)
    return back_substitution(A, b)


# with pivoting
def with_pivot(A, b):
    n = len(A)
    M = A
    i = 0

    for x in A:
        x.append(b[i])
        i += 1

    for k in range(n):
        for i in range(k, n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[k], M[i] = M[i], M[k]

        for j in range(k+1, n):
            q = M[j][k] / M[k][k]
            for m in range(k, n+1):
                M[j][m] -= q * M[k][m]

    x = [0 for _ in range(n)]

    x[n-1] = M[n-1][n] / M[n-1][n-1]
    for i in range(n-1, -1, -1):
        z = 0
        for j in range(i+1, n):
            z += M[i][j]*x[j]
        x[i] = (M[i][n] - z) / M[i][i]
    return x







import matplotlib.pyplot as plt
import time


def measure():
    tab1 = []
    tab2 = []
    x_axis = []
    for n in range(1, 500):
        A = numpy.random.randint(-10000, 10000, size=(n, n))
        b = numpy.random.rand(len(A))
        C = A.copy()
        d = b.copy()
        try:
            start_time_1 = time.time()
            with_pivot(A.tolist(), b.tolist())
            stop_time_1 = time.time()
            start_time_2 = time.time()
            without_pivot(C.tolist(), d.tolist())
            stop_time_2 = time.time()
            t1 = stop_time_1 - start_time_1
            t2 = stop_time_2 - start_time_2
            tab1.append(t1)
            tab2.append(t2)
            x_axis.append(n)
            print(f"{n}x{n} With pivot: {t1}")
            print(f"{n}x{n} Without pivot: {t2}\n")
        except ZeroDivisionError:
            pass
    return tab1, tab2, x_axis


tab1, tab2, x_axis = measure()

plt.plot(x_axis, tab1, 'b.', label="With pivoting")
plt.plot(x_axis, tab2, 'r.', label="Without pivoting")
plt.xlabel("Square matrix size")
plt.ylabel("Time [s]")
plt.title("Time of evaluation fro Gaussian elimination")
plt.legend()
plt.grid()
plt.show()


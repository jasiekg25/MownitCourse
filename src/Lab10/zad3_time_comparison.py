from src.Lab10.zad1_time_comparison import *
from scipy import linalg


def solve_time_comparison():
    tab1 = []
    tab2 = []
    tab3 = []
    tab4 = []
    tab5 = []
    x_axis = []
    for n in range(10, 350):
        A, b = random_matrix(-1000, 1000, n)
        start_time_1 = time.time()
        linalg.solve(A, b, crout(A))
        stop_time_1 = time.time()
        start_time_2 = time.time()
        linalg.solve(A, b, doolittle(A))
        stop_time_2 = time.time()
        start_time_3 = time.time()
        linalg.solve(A, b, cholesky(A))
        stop_time_3 = time.time()
        start_time_4 = time.time()
        np.linalg.solve(A, b)
        stop_time_4 = time.time()
        start_time_5 = time.time()
        linalg.solve(A, b)
        stop_time_5 = time.time()
        t1 = stop_time_1 - start_time_1
        t2 = stop_time_2 - start_time_2
        t3 = stop_time_3 - start_time_3
        t4 = stop_time_4 - start_time_4
        t5 = stop_time_5 - start_time_5
        print(f"\nMatrix {n}x{n}:")
        print(f"    Crout:     {t1}s")
        print(f"    Doolittle: {t2}s")
        print(f"    Cholesky:  {t3}s")
        print(f"    Numpy:     {t4}s")
        print(f"    Scipy:     {t5}s")
        tab1.append(t1)
        tab2.append(t2)
        tab3.append(t3)
        tab4.append(t4)
        tab5.append(t5)
        x_axis.append(n)

    plt.plot(x_axis, tab1, 'b.', label="Crout")
    plt.plot(x_axis, tab2, 'r.', label="Doolittle")
    plt.plot(x_axis, tab3, 'g.', label="Cholesky")
    plt.plot(x_axis, tab4, 'k.', label="Numpy")
    plt.plot(x_axis, tab5, 'y.', label="Scipy")
    plt.xlabel("Square matrix size")
    plt.ylabel("Time [s]")
    plt.title("Time of evaluation for LU decompositions")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    solve_time_comparison()
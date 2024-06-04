from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, diff, solve, Symbol
from scipy.stats import f

W = np.array([
    (2.24, 0.07), (1.61, -1.85), (4.66, -1.9), (3.52, -0.42), (1.4, -2.51), (3.88, -2.45), (4.44, -3.12), (4.38, -2.54),
    (1.08, -1.95), (3.97, -2.01), (3.43, 0.08), (3.39, -0.91), (2.48, 0.25), (4.27, -2.75), (0.85, 1.33), (3.13, 1),
    (2.92, 1.59), (4.19, -2.57), (0.79, 2.08), (3.31, 0.23), (0.96, 0.13), (2.07, 0), (3.4, 0.07), (4.02, -1.87),
    (4.18, -2.65), (4.26, -3.28), (3.11, 0.99), (3.03, 1.85), (4.74, -0.18), (1.89, -0.96), (2.7, 1.39), (4.57, -2.32),
    (1.4, -2.75), (2.84, 1.4), (3.91, -2.19), (4.54, -1.36), (2.41, 0.59), (0.91, 0.3), (0.77, 2.12), (4.8, 0.09),
])
xs = W[:, 0]
ys = W[:, 1]
x = symbols("x", real=True)


def find_polinomial_of_order(order: int) -> Symbol:
    order += 1

    a_s = symbols(", ".join([f"a{i}" for i in range(order)]), real=True)
    Dxb = 0
    for xi, yi in W:
        eq = 0
        for i in range(order):
            eq += a_s[i] * xi ** i
        Dxb += (yi - eq) ** 2

    Dxb_s = [diff(Dxb, a) for a in a_s]
    solved = solve(Dxb_s, a_s, dict=True)[0]

    Px: Symbol = 0
    for i, a in enumerate(a_s):
        Px += solved[a] * x ** i

    return Px


def find_ssq(Px: Symbol):
    Ssq = sum([(yi - Px.evalf(subs={x: xi})) ** 2 for xi, yi in W])
    Ssq /= len(W) - 2
    Ssq = sqrt(Ssq)
    Ssq *= 2

    return Ssq


def calculate_best_order() -> int:
    beta = 0.95
    Pxl = find_polinomial_of_order(1)
    sl = sum([(yi - Pxl.evalf(subs={x: xi})) ** 2 for xi, yi in W])
    sl /= len(W) - 2

    for i in range(2, 10):
        Pxc = find_polinomial_of_order(i)
        sc = sum([(yi - Pxc.evalf(subs={x: xi})) ** 2 for xi, yi in W])
        sc /= len(W) - i - 1

        s1 = max(sl, sc)
        s2 = min(sl, sc)

        Fn = s1 / s2
        Fcrit = f.ppf(q=beta, dfn=len(W)-2-i, dfd=len(W)-1-i)
        print(f"Ступінь {i} : Fn={Fn:.3f}, Fкрит={Fcrit:.3f}")
        if Fn > Fcrit:
            print("Fn > Fкрит")
            return i

        Pxl, sl = Pxc, sc


def main():
    order = calculate_best_order()
    Px = find_polinomial_of_order(order)
    print(f"Поліном: {Px}")

    x_plot = np.linspace(xs.min(), xs.max())
    y_plot = [Px.evalf(subs={x: x_p}) for x_p in x_plot]
    plt.plot(x_plot, y_plot, color='#58b970', label='Лінія регресії')

    Ssq = find_ssq(Px)

    y_plot1 = [yp + Ssq for yp in y_plot]
    plt.plot(x_plot, y_plot1, color='#0000ff', label='Довірча область', linestyle="dashed")

    y_plot2 = [yp - Ssq for yp in y_plot]
    plt.plot(x_plot, y_plot2, color='#0000ff', label='Довірча область', linestyle="dashed")

    plt.scatter(xs, ys, c='#ef5423', label='Точки')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

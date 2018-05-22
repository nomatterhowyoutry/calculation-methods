from numpy import *
from random import *

# def f(x):
#     return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def swarm(f, xmin, xmax, d, swarm=500, w=0.72, s=20, c1=1.19, c2=1.19, k=1):
    ''' xmin - left x boundary
        xmax - right x boundary
        d - dimension of a function
        f - function'''

    x = zeros((s, d))
    v = zeros((s, d))
    p = zeros((s, d))
    fitness = zeros(s)
    fp = zeros(s)

    for i in range(0, s):
        for j in range(0, d):
            x[i][j] = uniform(xmin, xmax)
            v[i][j] = (uniform(xmin, xmax) - x[i][j]) / 2
            p[i][j] = x[i][j]
        fitness[i] = f(x[i])
        fp[i] = fitness[i]

    gbest = 0
    for i in range(1, s):
        if fp[i] < fp[gbest]:
            gbest = i
    while k <= swarm:
        for i in range(0, s):
            for j in range(0, d):
                r1 = uniform(0, 1)
                r2 = uniform(0, 1)

                v[i][j] = w * v[i][j] + c1*r1*(p[i][j] - x[i][j])

                if (i != gbest):
                    v[i][j] = v[i][j] + c2*r2*(p[gbest][j] - x[i][j])

                x[i][j] = x[i][j] + v[i][j]

                if x[i][j] < xmin:
                    x[i][j] = xmin
                    v[i][j] = 0
                if x[i][j] > xmax:
                    x[i][j] = xmax
                    v[i][j] = 0

        for i in range(0, s):
            fitness[i] = f(x[i])
            if fitness[i] < fp[i]:
                fp[i] = fitness[i]
                for j in range(0, d):
                    p[i][j] = x[i][j]

        for i in range(0, s):
            if fp[i] < fp[gbest]:
                gbest = i
        k += 1

    # print('f = {}; x = {}'.format(fp[gbest], p[gbest]))
    return [p[gbest], fp[gbest]]

# print(swarm(f, -10, 10, 2))

# def f(x):
#     return 5*x**2 - 4*x + 1

# def df(x):
#     return 10*x - 4

def dichotomy(f, df, a, b, eps):
    ''' f - function
        df - diff function
        a - left boundary
        b - right boundary
        eps - accuracy'''

    while abs(b - a) > eps:
        c = (a + b) / 2
        if df(a)*df(c) < 0:
            b = c
        else:
            a = c

    return [c, f(c)]

# print(dichotomy(f, df, 0, 10, 0.1))

# def f(x):
#     return x**3 - 3*sin(x)

# def df(x):
#     return 3*(x**2 - cos(x))

def chord(f, df, a, b, eps):
    ''' f - function
        df - diff function
        a - left boundary
        b - right boundary
        eps - accuracy'''

    x = a - ( df(a) / (df(a) - df(b)) *(a-b))
    d_f = df(x)
    while abs(d_f) > eps:
        if d_f > 0:
            b = x
        else:
            a = x
        x = a - ( df(a) / (df(a) - df(b)) *(a-b))
        d_f = df(x)

    return [x, f(x)]

# print(chord(f, df, 0, 1, 10**(-4)))

# def f(x):
    # return 129*x[0]**2 - 256*x[0]*x[1] + 129*x[1]**2 - 51*x[0] - 149*x[1] - 27

def coordinate(f, x0, a, b, eps):
    ''' f - function
        x0 - starting points
        a - left boundary
        b - right boundary
        eps - accuracy'''

    def dich(x, i, a, b, eps):
        delta = eps / 10
        xl = x.copy()
        xr = x.copy()
        while abs(b - a) > eps:
            xl[i] = (a + b - delta) / 2
            xr[i] = (a + b + delta) / 2
            if f(xl) < f(xr):
                b = xr[i]
            else:
                a = xl[i]
        return (a + b) / 2

    def norm(x):
        n = len(x)
        m = abs(x[0])
        for i in range(0, n):
            if abs(x[i]) > m:
                m = abs(x[i])
        return m

    n = len(x0)
    x1 = zeros(n, dtype = float)
    for i in range(0, n):
        x1[i] = dich(x0, i, a, b, eps)

    while norm(x1 - x0) > eps:
        x0 = x1.copy()
        for i in range(0, n):
            x1[i] = dich(x0, i, a, b, eps)

    return [x1, f(x1)]

# print(coordinate(f, [4, 4], -10, 10, 10**-8))


# def f(x):
#     return 3*x[0]**2 - 3*x[0]*x[1] + 4*x[1]**2 - 2*x[0] + x[1]

# def g(x):
#     return array([-3*x[1] + 6*x[0] - 2, 8*x[1] - 3*x[0] + 1])

#gradient

def gradient(f, g, a_x, b_x, x0, eps):

    def norm(g):

        s = 0
        for i in range(0, len(g)):
            s += g[i] ** 2
        return s

    def dich(f, g, x, a, b, eps):
        ''' f - function
            g - gradient
            x - starting points
            a - left boundary
            b - right boundary,
            eps - accuracy'''
        
        delta = eps / 10

        while abs(b - a) > eps:
            alpha1 = (a + b - delta) / 2
            alpha2 = (a + b + delta) / 2
            f1 = f(x0 - alpha1 * g(x0))
            f2 = f(x0 - alpha2 * g(x0))
            if f1 < f2:
                b = alpha2
            else:
                a = alpha1

        return (a + b) / 2

    p = - g(x0)
    alpha = dich(f, g, x0, a_x, b_x, eps)
    x1 = x0 + alpha * p
    while norm(g(x1)) > eps:
        b = norm(g(x1))**2 / norm(g(x0))**2
        p = - g(x1) + b * p
        x0 = x1
        alpha = dich(f, g, x0, a_x, b_x, eps)
        x1 = x0 + alpha * p

    return x1, f(x1)


# print(gradient(f, g, -5, 6, array([0., 1.]), 10**-6))

# def f(x):
#     return 0.5 * (1 - x[0])**2 + 0.5 * (x[1] - x[0]**2)**2

# def jacobi(x):
#     return array([ [-1, 0], [-2*x[0], 1]])

# def f_a(x):
#        return array([1 - x[0] , x[1] - x[0] ** 2]).reshape((2,1))

#marquardt
def marquardt(f, f_a, j, x, eps, k_max = 50):
    ''' f - function
        f_a - function array
        j - jacobi matrix
        x - starting points
        eps - accuracy'''

    k = 0

    def getA(j):
       return dot(jacobi.T, jacobi)

    jacobi = j(x)
    v = 2
    m = (10**(-3))*(jacobi.T * jacobi).max()

    while k < k_max:

        A = getA(jacobi)
        g = dot(jacobi.T, f_a(x)).reshape((-1, 1))
        leftPartInverse = linalg.inv(A + m * eye(A.shape[0], A.shape[1]))
        d_lm = - dot(leftPartInverse, g)
        x_new = x + d_lm.reshape((-1))
        grain_numerator = (f(x) - f(x_new))
        gain_divisor = 0.5 * dot(d_lm.T, m*d_lm-g) + 1e-10
        gain = grain_numerator / gain_divisor
        if gain > 0:
            x = x_new
            jacobi = j(x)
            m = m * max(1 / 3, 1 - (2 * gain - 1) ** 3)
            v = 2
            if gain < eps:
                break
        else:
            m *= v
            v *= 2
        k += 1

    return x, f(x)

# x = array([-2, -2])

# eps = 10**-23

# print(marquardt(f, f_a, jacobi, x, eps))
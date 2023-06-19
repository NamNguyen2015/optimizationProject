import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

# Optimized function with constrain
def f(X):
    pen = 0
    if X[0] + X[1] < 2:
        pen = 500 + 1000 * (2 - X[0] - X[1])
    return np.sum(X) + pen


varbound = np.array([[0, 10]] * 3)

# Simple function with parameters
def f(X):
    return np.sum(X)


varbound = np.array([[0, 10]] * 3)

algorithm_param = {'max_num_iteration': 3000, \
                   'population_size': 100, \
                   'mutation_probability': 0.1, \
                   'elit_ratio': 0.01, \
                   'crossover_probability': 0.5, \
                   'parents_portion': 0.3, \
                   'crossover_type': 'uniform', \
                   'max_iteration_without_improv': None}

model = ga(function=f, \
           dimension=3, \
           variable_type='real', \
           variable_boundaries=varbound, \
           algorithm_parameters=algorithm_param)

# Ackley
"""https://en.wikipedia.org/wiki/Ackley_function"""
def f(X):
    dim = len(X)

    t1 = 0
    t2 = 0
    for i in range(0, dim):
        t1 += X[i] ** 2
        t2 += math.cos(2 * math.pi * X[i])

    OF = 20 + math.e - 20 * math.exp((t1 / dim) * -0.2) - math.exp(t2 / dim)

    return OF


varbound = np.array([[-32.768, 32.768]] * 2)

model = ga(function=f, dimension=2, variable_type='real', variable_boundaries=varbound)


# Weierstrass
""" http://infinity77.net/global_optimization/test_functions_nd_W.html"""

import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga


def f(X):
    dim = len(X)

    a = 0.5
    b = 3
    OF = 0
    for i in range(0, dim):
        t1 = 0
        for k in range(0, 21):
            t1 += (a ** k) * math.cos((2 * math.pi * (b ** k)) * (X[i] + 0.5))
        OF += t1
    t2 = 0
    for k in range(0, 21):
        t2 += (a ** k) * math.cos(math.pi * (b ** k))
    OF -= dim * t2

    return OF


varbound = np.array([[-0.5, 0.5]] * 2)

algorithm_param = {'max_num_iteration': 1000, \
                   'population_size': 100, \
                   'mutation_probability': 0.1, \
                   'elit_ratio': 0.01, \
                   'crossover_probability': 0.5, \
                   'parents_portion': 0.3, \
                   'crossover_type': 'uniform', \
                   'max_iteration_without_improv': None}

model = ga(function=f, dimension=2, \
           variable_type='real', \
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)
#model.run()
# Generate data points
x = np.linspace(-0.5, 0.5, 100)
y = np.linspace(-0.5, 0.5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(x)):
    for j in range(len(y)):
        Z[i][j] = f([X[i][j], Y[i][j]])

# Plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='rainbow')


# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.set_title('Weierstrass Function')

# Show the plot
plt.show()

model.run()

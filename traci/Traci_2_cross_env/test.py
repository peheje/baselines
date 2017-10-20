import numpy as np

from scipy.optimize import rosen, differential_evolution

bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
result = differential_evolution(rosen, bounds)
result.x, result.fun

print(result.x)
print(result.fun)

def relu(x, derivative=False):
    return np.where(derivative,
                    (x > 0) * 1.0 + (x <= 0) * 0.0,
                    np.maximum(x, 0))

f = relu

y = np.random.uniform(-1, 1, (3, 3))
print(y)
print(f(y))
print(f(y, True))

X = np.array([10, 20]).T
print(X)
l1 = np.array([[1, 2], [3, 4], [5, 6]]).T
print("---l1")
print(l1)
print("---l1 mutated")
print(l1)
print("---")
ol1 = f(np.dot(X, l1))
print(ol1)

l2 = np.array([[1, 2, 3], [4, 5, 6]]).T
ol2 = f(np.dot(ol1, l2))
print(ol2)

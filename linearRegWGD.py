import matplotlib.pyplot as plt
import numpy as np
import random

# Gradient of Simple Linear Regression y = b0 + b1*x


def ssr_gradient(x, y, b):
    result = b[0] + b[1]*x - y
    return result.mean(), (result*x).mean()


def gradient_descent(gradient, x, y, start, learning_rate, iteration_count=50, tolerance=1e-06):

    vector = start
    while iteration_count > 0:
        iteration_count -= 1
        update_value = -learning_rate * np.array(gradient(x, y, vector))
        if np.all(np.abs(update_value) <= tolerance):
            break
        vector += update_value
    return vector


x = np.array([x for x in range(30)])
y = np.array([x - random.randint(0, 2) for x in range(30)])

output = gradient_descent(ssr_gradient, x, y, np.array(
    [750, 7500], dtype='float64'), 0.0008, 100_000)
plt.plot(x, y, 'ro')
line = np.linspace(0, 30, 1000)
plt.plot(line, output[0]+output[1]*line)
plt.show()

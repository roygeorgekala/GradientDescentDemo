import matplotlib.pyplot as plt
import numpy as np
import random


def gradient_descent(gradient, start, learning_rate, iteration_count):

    vector = start
    while iteration_count > 0:
        iteration_count -= 1
        update_value = learning_rate * (-1*gradient(vector))
        vector += update_value
    return vector


x = np.array([x for x in range(50)])
y = np.array([x - random.randint(0, 2) for x in range(50)])

plt.plot(x, y, 'ro')
plt.show()

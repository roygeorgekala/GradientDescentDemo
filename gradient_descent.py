import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(gradient, start, learning_rate, iteration_count=50, tolerance=1e-06):

    vector = start
    while iteration_count > 0:
        iteration_count -= 1
        update_value = learning_rate * (-1*gradient(vector))
        if np.all(np.abs(update_value) <= tolerance):
            break
        vector += update_value
    return vector


print(gradient_descent(lambda x: 2*x, start=10, learning_rate=0.1))

import matplotlib.pyplot as plt
import numpy as np
import random


# Gradient of Simple Linear Regression y = b0 + b1*x
def ssr_gradient(x, y, b):
    result = b[0] + b[1]*x - y
    return result.mean(), (result*x).mean()


"""
This SSR Gradient can be explained simply as: 
    We are using Simple Linear Regression or a single dependent variable "F(x)" on a single indepedent "x". The SLR approach estimates the dependent variable F(X) to be equal to an equation of dependent variable X as F(x) = b0 + b1*X. The different equations to measure the error of the output can be taken as Sum of Squared Residuals(SSR), given by the equation Σᵢ (Yi - F(Xi))^2 where Y is the actual output, F(X) is the computed output, and we sum it for all the training values of X and corresponding Y. We can also use the Mean Squared Error = SSR/n  or the Cost function C = SSR/2n which is slighty easier mathematically. 
    To calculate the Gradient of the Cost function C = Σᵢ (Yi - b0 + b1*X)^2, we differentiate with respect to the 2 variables b0 and b1, with ∂C/∂b0 = (1/n)(Σᵢ(b0 + b1*Xi - Yi)) = mean(b0+b1*Xi-Yi). Similary, ∂C/∂b1 = (1/n)(Σᵢ(b0 + b1*Xi - Yi)Xi) = mean((b0+b1*Xi-Yi)Xi).
    These are the values being returned as the gradient function ssr_gradient.
"""

# General Gradient Descent Algorithm


def gradient_descent(gradient, x, y, start, learning_rate, iteration_count=50, tolerance=1e-06):

    vector = start  # Setting the vector that is going to updated via the equation
    while iteration_count > 0:  # iterating as many times as iteration count is set
        iteration_count -= 1

        # Weight Updation is done via ∇W = -η * ∇C
        update_value = -learning_rate * np.array(gradient(x, y, vector))

        # An Early break statement aiding in leaving if all of the update_values are less than the tolerance, default set to 0.000001
        if np.all(np.abs(update_value) <= tolerance):
            break
        vector += update_value  # update the vector using the update value
    return vector


RANGE = 30  # Number of points to be plotted

# Creating and plotting a data set to be close to X = Y equation but with deviations.
x = np.array([x for x in range(RANGE)])  # X values from 1 to RANGE.
y = np.array([x - random.randint(0, 2)
             for x in range(RANGE)])  # Y values with deviations
plt.plot(x, y, 'ro')  # Plotting of coordinates

output = gradient_descent(ssr_gradient, x, y, np.array(
    [750, 7500], dtype='float64'), 0.0008, 100_000)  # Running the GD algorithm to recieve the values of b0 and b1 from the x and y values passed.
# coordinates to draw a line representing best fit line
line = np.linspace(0, RANGE, 1000)
plt.plot(line, output[0]+output[1]*line)  # Plotting best fit line

plt.title("Simple Linear Regression using Gradient Descent Algorithm")
plt.show()

# Testing how accurate to X = Y for unseen values.
tester = int(input(
    "\n\n\n To test accuracy of the learning, input a X value to check closeness to X = Y equation: "))
print(output[0]+tester*output[1])

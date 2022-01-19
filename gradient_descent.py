import matplotlib.pyplot as plt
import numpy as np
import random

x = np.array([x for x in range(50)])
y = np.array([x - random.randint(0, 2) for x in range(50)])

plt.plot(x, y, 'ro')
plt.show()

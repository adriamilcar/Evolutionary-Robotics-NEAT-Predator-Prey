import numpy as np
import matplotlib.pyplot as plt


r = [0.12, 0.22, 0.25, 0.16, 0.2, 0.17, 0.22, 0.31, 0.22, 0.05, 0.11, 0.24, 0.2,0.23, 0.19]

h, b = np.histogram(r, density=True)
binWidth = b[1] - b[0]
plt.bar(b[:-1], h * binWidth, binWidth)
plt.title('R in evolved neural networks')
plt.xlabel('Bits')
plt.ylabel('p(R)')
plt.show()

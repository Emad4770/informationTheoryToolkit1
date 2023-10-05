import numpy as np
import matplotlib.pyplot as plt
from P0 import entropy_binary

p0_values = np.linspace(0, 1, 100)
print(p0_values)

#entropy for each p0 value
entropy_values = [entropy_binary(p0) for p0 in p0_values]

#plotting entropy function
plt.plot(p0_values, entropy_values)
plt.xlabel('p0')
plt.ylabel('Entropy')
plt.title('Entropy of Binary Random Variable')
plt.grid(True)
plt.show()

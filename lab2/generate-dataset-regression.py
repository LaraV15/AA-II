import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creates a regression dataset with 2 features, x1 and x2, and a target y

# Number of samples
N = 1000


# Randomly generate the features
X1 = np.random.rand(N)
X2 = np.random.rand(N)

# Calculate the target
Y = np.zeros(N)
for i in range(N):
  d = np.sqrt(X1[i] ** 4 + X2[i] ** 4) # A simple function to generate the target
  Y[i] = d

# Scale Y to be between 0 and 1
Y = (Y - Y.min()) / (Y.max() - Y.min())

# Plot the data with a color map
plt.scatter(X1, X2, c=Y, cmap='viridis', alpha=0.8)

# Axis labels
plt.xlabel('x1')
plt.ylabel('x2')

# Plot the color bar
plt.colorbar()

plt.show()

# Save the data to a file
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
data.to_csv('dataset-lab2-b.csv', index=False)

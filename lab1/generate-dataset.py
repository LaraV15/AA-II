import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creates a classification dataset with 2 features and 2 classes
# The border between the classes is a line w1 * x1 + w2 * x2 + b = 0
# x1 is the exams score and x2 is the assignments score

# Number of samples
N = 100

W1 = 0.7
W2 = 0.3
B = -0.5
NOISE = 0.1

# Randomly generate the features
X1 = np.random.rand(N)
X2 = np.random.rand(N)

# Calculate the target
Y = np.zeros(N)
for i in range(N):
  f = W1 * X1[i] + W2 * X2[i] + B + np.random.normal(0, NOISE)
  if W1 * X1[i] + W2 * X2[i] + B > 0:
    Y[i] = 1

# Plot the data in green and red
plt.scatter(X1[Y == 0], X2[Y == 0], color='red', alpha=0.5)
plt.scatter(X1[Y == 1], X2[Y == 1], color='green', alpha=0.5)

# Axis labels
plt.xlabel('Promedio Parciales')
plt.ylabel('Promedio TPs')

plt.show()

# Save the data to a file
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
data.to_csv('dataset.csv', index=False)

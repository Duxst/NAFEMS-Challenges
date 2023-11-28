# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Manually implement Latin Hypercube Sampling for 2D
np.random.seed(0)  # for reproducibility

# Number of samples
n_samples = 10

# Step 1: Divide each axis into n non-overlapping intervals
x_intervals = np.linspace(0, 1, n_samples + 1)
y_intervals = np.linspace(0, 1, n_samples + 1)

# Step 2: Randomly sample a point from each interval along each axis
x_samples = [np.random.uniform(low=x_intervals[i], high=x_intervals[i+1]) for i in range(n_samples)]
y_samples = [np.random.uniform(low=y_intervals[i], high=y_intervals[i+1]) for i in range(n_samples)]

# Shuffle one of the arrays to break the correlation
np.random.shuffle(y_samples)

# Step 3: Combine the points from the two axes to form n sample points in 2D space
sample_points = list(zip(x_samples, y_samples))

# Visualization
plt.figure(figsize=(8, 8))
plt.scatter(x_samples, y_samples, c='red', label='Sampled Points')
plt.grid(True, which='both')

# Draw the intervals
for i in x_intervals:
    plt.axhline(i, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(i, color='grey', linestyle='--', linewidth=0.5)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Latin Hypercube Sampling in 2D')
plt.legend()
plt.show()

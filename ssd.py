

#####This is just example data for the presentation#####

import numpy as np
import matplotlib.pyplot as plt

# Original dataset: scores of 10 students
original_data = np.array([56, 45, 67, 89, 34, 50, 29, 77, 89, 90])

# Number of bootstrap samples
n_bootstrap_samples = 1000

# Initialize an array to store bootstrap means
bootstrap_means = np.zeros(n_bootstrap_samples)

# Generate bootstrap samples and calculate their means
np.random.seed(42)  # for reproducibility
for i in range(n_bootstrap_samples):
    bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
    bootstrap_means[i] = np.mean(bootstrap_sample)

# Calculate the 95% confidence interval for the mean
conf_interval_lower = np.percentile(bootstrap_means, 2.5)
conf_interval_upper = np.percentile(bootstrap_means, 97.5)

# Calculate the original sample mean for comparison
original_mean = np.mean(original_data)

# Create the histogram of the bootstrap means in the context of the original dataset [56, 45, 67, 89, 34, 50, 29, 77, 89, 90]
plt.figure(figsize=(12, 6))
plt.hist(bootstrap_means, bins=30, density=True, alpha=0.75, color='blue')
plt.axvline(original_mean, color='red', linestyle='dashed', linewidth=2, label="Original Mean")
plt.axvline(conf_interval_lower, color='green', linestyle='dashed', linewidth=2, label="2.5th Percentile")
plt.axvline(conf_interval_upper, color='purple', linestyle='dashed', linewidth=2, label="97.5th Percentile")

# Add labels and title
plt.xlabel("Mean Test Score")
plt.ylabel("Density")
plt.title("Bootstrap Distribution of Mean Test Score")
plt.legend()

# Show the plot
plt.show()

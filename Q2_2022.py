
###########################Q2_2022#####################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Data
diameters = [0.0032, 0.0039, 0.0037,0.0035, 0.0031, 0.0040, 0.0038, 0.0038, 0.0040, 0.0037]
porosities = [0.375, 0.347, 0.329, 0.352, 0.388, 0.419, 0.404, 0.394, 0.352, 0.370]
lengths = [2.86, 3.13, 3.08, 3.12, 2.94, 2.90, 2.80, 3.05, 3.02, 3.04]

# Check for normality using a Shapiro-Wilk test
_, p_value = stats.shapiro(diameters)
if p_value > 0.05:
    print("Diameter data may follow a normal distribution.")
else:
    print("Diameter data does not appear to follow a normal distribution.")

# Check for normality using a Shapiro-Wilk test for porosity data
_, p_value = stats.shapiro(porosities)
if p_value > 0.05:
    print("Porosity data may follow a normal distribution.")
else:
    print("Porosity data does not appear to follow a normal distribution.")

# Check for normality using a Shapiro-Wilk test for length data
_, p_value = stats.shapiro(lengths)
if p_value > 0.05:
    print("Length data may follow a normal distribution.")
else:
    print("Length data does not appear to follow a normal distribution.")

# Function to create Q-Q plots
def qq_plot(data, distribution, title, subplot_position):
    plt.subplot(subplot_position)
    stats.probplot(data, dist=distribution, plot=plt)
    plt.title(title)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

# Create Figure 1: Histograms with fitted normal curves
plt.figure(figsize=(12, 6))

# Plot histograms with fitted normal distribution curves
plt.subplot(131)
plt.hist(diameters, bins=10, density=True, alpha=0.6, color='g', label='Data')
mu, std = stats.norm.fit(diameters)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal')
plt.xlabel('Diameter (D_p)')
plt.ylabel('Frequency')
plt.title('Histogram for Diameter Data')
plt.legend()

plt.subplot(132)
plt.hist(porosities, bins=10, density=True, alpha=0.6, color='g', label='Data')
mu, std = stats.norm.fit(porosities)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal')
plt.xlabel('Porosity (epsilon)')
plt.ylabel('Frequency')
plt.title('Histogram for Porosity Data')
plt.legend()

plt.subplot(133)
plt.hist(lengths, bins=10, density=True, alpha=0.6, color='g', label='Data')
mu, std = stats.norm.fit(lengths)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal')
plt.xlabel('Length (L)')
plt.ylabel('Frequency')
plt.title('Histogram for Length Data')
plt.legend()

plt.tight_layout()  # Ensure proper spacing between subplots

# Create Figure 2: Q-Q plots
plt.figure(figsize=(12, 6))

# Plot Q-Q plots for Diameter, Porosity, and Length data on the same set of subplots
qq_plot(diameters, 'norm', 'Q-Q Plot for Diameter Data', 131)
qq_plot(porosities, 'norm', 'Q-Q Plot for Porosity Data', 132)
qq_plot(lengths, 'norm', 'Q-Q Plot for Length Data', 133)

plt.tight_layout()  # Ensure proper spacing between subplots


# Variables of pressure drop equation
density = 1.225
viscosity = 1.81 * (10 ** (-5))
velocity = 0.35
target_pressure_drop = 15250

# Probability distributions
mu_diameter, std_diameter = np.mean(diameters), np.std(diameters)
mu_porosity, std_porosity = np.mean(porosities), np.std(porosities)
mu_length, std_length = np.mean(lengths), np.std(lengths)

def pressure_drop(length, diameter, porosity):
    result = (150*viscosity*length*((1-porosity)**2)*velocity)/((diameter**2)*(porosity**3)) + (1.75*length*density*(1-porosity)*(velocity**2))/((diameter)*(porosity**3))
    return result

# Number of Monte Carlo samples
N = 100000

# Initialize counters
count = 0

# Lists to store changes in Δp for each variable
changes_diameter = []
changes_porosity = []
changes_length = []

# Generate random samples and evaluate Δp for each sample
for _ in range(N):
    random_diameter = np.random.normal(mu_diameter, std_diameter)
    random_porosity = np.random.normal(mu_porosity, std_porosity)
    random_length = np.random.normal(mu_length, std_length)

    delta_p = pressure_drop(random_length, random_diameter, random_porosity)

    if delta_p > 15250:
        count += 1

    # Calculate changes in Δp for sensitivity analysis
    change_diameter = abs(pressure_drop(mu_length, random_diameter, mu_porosity) - delta_p)
    change_porosity = abs(pressure_drop(mu_length, mu_diameter, random_porosity) - delta_p)
    change_length = abs(pressure_drop(random_length, mu_diameter, mu_porosity) - delta_p)

    changes_diameter.append(change_diameter)
    changes_porosity.append(change_porosity)
    changes_length.append(change_length)

# Calculate probability of failure
probability = count / N

# Quantify uncertainty by looking at the changes in Δp for each variable
mean_change_diameter = np.mean(changes_diameter)
mean_change_porosity = np.mean(changes_porosity)
mean_change_length = np.mean(changes_length)

# Identify which variable contributes most to uncertainty
max_change = max(mean_change_diameter, mean_change_porosity, mean_change_length)
if max_change == mean_change_diameter:
    most_uncertain_variable = "Diameter"
elif max_change == mean_change_porosity:
    most_uncertain_variable = "Porosity"
else:
    most_uncertain_variable = "Length"

# Define your decision criteria (e.g., probability threshold)
decision_threshold = 0.05  # Adjust based on your risk tolerance

# Make a decision based on the probability of failure
if probability < decision_threshold:
    decision = "Acceptable"
else:
    decision = "Unacceptable"

# Print results
print(f"Probability of Δp exceeding 15,250 Pa: {probability:.6f}")
print("Most Uncertain Variable:", most_uncertain_variable)
print("Decision:", decision)

#Here we use bootstrapping to find the confidence interval for the probability of failure
# Number of bootstrapping samples
num_bootstraps = 10000

# Initialize a list to store bootstrapped probabilities
bootstrapped_probabilities = []

# Perform bootstrapping
for _ in range(num_bootstraps):
    resampled_diameters = np.random.normal(mu_diameter, std_diameter, N)
    resampled_porosities = np.random.normal(mu_porosity, std_porosity, N)
    resampled_lengths = np.random.normal(mu_length, std_length, N)

    delta_p = pressure_drop(resampled_lengths, resampled_diameters, resampled_porosities)

    # Calculate changes in Δp for sensitivity analysis
    change_diameter = abs(pressure_drop(mu_length, resampled_diameters, mu_porosity) - delta_p)
    change_porosity = abs(pressure_drop(mu_length, mu_diameter, resampled_porosities) - delta_p)
    change_length = abs(pressure_drop(resampled_lengths, mu_diameter, mu_porosity) - delta_p)

    bootstrapped_probabilities.append(np.mean(delta_p > 15250))

# Calculate the lower and upper percentiles for the confidence interval
confidence_level = 0.95
lower_percentile = (1 - confidence_level) / 2.0 * 100
upper_percentile = (1 + confidence_level) / 2.0 * 100

# Calculate confidence interval bounds
lower_bound = np.percentile(bootstrapped_probabilities, lower_percentile)
upper_bound = np.percentile(bootstrapped_probabilities, upper_percentile)
print("95% Confidence Interval (MC): ({}, {})".format(lower_bound, upper_bound))


plt.show()
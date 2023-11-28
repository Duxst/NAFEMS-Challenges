###########################Q1_2022#####################
import numpy as np
from pyDOE import lhs
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Define the data for R and S
R = np.array([503.252, 460.005, 485.503, 466.061, 475.449])
S = np.array([376.594, 278.222, 331.535, 330.774, 395.173, 394.203, 387.309, 361.754, 300.191, 381.09])

# Calculate the mean and standard deviation of R and S
mu_R, std_R = np.mean(R), np.std(R)
mu_S, std_S = np.mean(S), np.std(S)

# Calculate the mean and standard deviation of the difference g = R - S
mu_g, std_g = mu_R - mu_S, np.sqrt(std_R**2 + std_S**2)

# Generate a range of values around the mean of R, S, and g
values_R = np.linspace(mu_R - 3*std_R, mu_R + 3*std_R, 100)
values_S = np.linspace(mu_S - 3*std_S, mu_S + 3*std_S, 100)
values_g = np.linspace(mu_g - 3*std_g, mu_g + 3*std_g, 100)

# Calculate the PDF of the values
pdf_R = norm.pdf(values_R, mu_R, std_R)
pdf_S = norm.pdf(values_S, mu_S, std_S)
pdf_g = norm.pdf(values_g, mu_g, std_g)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(values_R, pdf_R, label='R distribution')
plt.plot(values_S, pdf_S, label='S distribution')
plt.plot(values_g, pdf_g, label='g distribution')
plt.legend()
plt.grid(True)
plt.title('Distributions of R, S, and g')
plt.legend()
plt.xlabel('Values')
plt.ylabel('Probability Density Function')

# Find P(g < 0)
analytical_prob_failure = norm.cdf(0, mu_g, std_g)

# Identify variable with the most uncertainty
most_uncertain = "R" if std_R > std_S else "S"

print(f"Probability that g is less than zero: {analytical_prob_failure:.4f}")
print(f"The variable contributing most to uncertainty is: {most_uncertain}")
print(f"Standard deviation of R: {std_R:.4f}")
print(f"Standard deviation of S: {std_S:.4f}")

# Monte Carlo simulation using direct sampling from lists R and S
num_trials = 100000
count_failure = 0
probabilities = []

for _ in range(num_trials):
    sample_R = np.random.choice(R)
    sample_S = np.random.choice(S)
    g_sample = sample_R - sample_S
    if g_sample < 0:
        count_failure += 1
    # Storing the cumulative probability for visualization
    probabilities.append(count_failure / (1 + _))

# Estimate the probability of failure
estimated_prob_failure = count_failure / (1 + num_trials)

print(f"Estimated probability of failure using Monte Carlo (direct sampling): {estimated_prob_failure:.4f}")

# Plot the convergence of Monte Carlo estimation with direct sampling
plt.figure(figsize=(10, 6))
plt.plot(probabilities)
plt.axhline(y=analytical_prob_failure, color='r', linestyle='--', label='Analytical Probability')
plt.xlabel("Number of Trials")
plt.ylabel("Probability of g < 0")
plt.title("Convergence of Monte Carlo Estimation with Direct Sampling")
plt.grid(True)
plt.legend()
plt.savefig("Convergence_Monte_Carlo_Direct_Sampling.png")

# Monte Carlo simulation with normal distribution sampling
num_trials = 100000
count_failure = 0
probabilities = []

for _ in range(num_trials):
    sample_R = np.random.normal(mu_R, std_R)
    sample_S = np.random.normal(mu_S, std_S)
    g_sample = sample_R - sample_S
    if g_sample < 0:
        count_failure += 1
    # Storing the cumulative probability for visualization
    probabilities.append(count_failure / (1 + _))

# Estimate the probability of failure
estimated_prob_failure = count_failure / num_trials

print(f"Estimated probability of failure using Monte Carlo: {estimated_prob_failure:.4f}")

# Plot the convergence of Monte Carlo estimation
plt.figure(figsize=(10, 6))
plt.axhline(y=analytical_prob_failure, color='r', linestyle='--', label='Analytical Probability')
plt.plot(probabilities)
plt.xlabel("Number of Trials")
plt.ylabel("Probability of g < 0")
plt.title("Convergence of Monte Carlo Estimation")
plt.legend()
plt.grid(True)
plt.savefig("Convergence_Monte_Carlo.png")

# Define number of LHS samples
n = 100000

# Generate LHS samples in the range [0, 1]
lhs_samples = lhs(2, samples=n)

# Map these samples to your distributions
R_lhs_samples = stats.norm(loc=mu_R, scale=std_R).ppf(lhs_samples[:, 0])
S_lhs_samples = stats.norm(loc=mu_S, scale=std_S).ppf(lhs_samples[:, 1])

# Compute g for each pair
g_values = R_lhs_samples - S_lhs_samples

# Estimate the probability that g < 0 using LHS
prob_g_less_than_0_lhs = np.mean(g_values < 0)

print(f"Estimated probability of g < 0 using Latin Hypercube Sampling (LHS): {prob_g_less_than_0_lhs:.4f}")

bootstrap_code = '''
import numpy as np

# Bootstrap the differences
num_samples = 100000
g_bootstrap_prob = []

mu_R, std_R = 485.054, 15.2762
mu_S, std_S = 362.564, 39.2509

for _ in range(num_samples):
    R_sample = np.random.normal(mu_R, std_R, 10000)
    S_sample = np.random.normal(mu_S, std_S, 10000)
    g_bootstrap = R_sample - S_sample
    g_bootstrap_prob.append(np.mean(g_bootstrap < 0))

# Calculate the lower and upper percentiles for the confidence interval
confidence_level = 0.95
lower_percentile = (1 - confidence_level) / 2.0 * 100
upper_percentile = (1 + confidence_level) / 2.0 * 100

# Calculate confidence interval bounds
lower_bound = np.percentile(g_bootstrap_prob, lower_percentile)
upper_bound = np.percentile(g_bootstrap_prob, upper_percentile)
lower_bound, upper_bound
'''
# Bootstrap the differences
num_samples = 100000
g_bootstrap_prob = []

for _ in range(num_samples):
    R_sample = np.random.normal(mu_R, std_R,10000)
    S_sample = np.random.normal(mu_S, std_S,10000)
    g_bootstrap = R_sample - S_sample
    g_bootstrap_prob.append(np.mean(g_bootstrap < 0))

# Calculate the lower and upper percentiles for the confidence interval
confidence_level = 0.95
lower_percentile = (1 - confidence_level) / 2.0 * 100
upper_percentile = (1 + confidence_level) / 2.0 * 100

# Calculate confidence interval bounds
lower_bound = np.percentile(g_bootstrap_prob, lower_percentile)
upper_bound = np.percentile(g_bootstrap_prob, upper_percentile)
print("95% Confidence Interval (MC): ({}, {})".format(lower_bound, upper_bound))
# Execute the bootstrap code to get lower and upper bounds
exec_globals = {}
exec(bootstrap_code, exec_globals)
lower_bound = exec_globals['lower_bound']
upper_bound = exec_globals['upper_bound']
g_bootstrap_prob = exec_globals['g_bootstrap_prob']

# Create the plot
plt.figure(figsize=(10, 6))
sns.histplot(g_bootstrap_prob, bins=50, kde=True, color='blue')
plt.axvline(x=lower_bound, color='r', linestyle='--', label=f'Lower 95% CI: {lower_bound:.4f}')
plt.axvline(x=upper_bound, color='g', linestyle='--', label=f'Upper 95% CI: {upper_bound:.4f}')
plt.xlabel('Probability that g < 0')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution with 95% Confidence Intervals')
plt.legend()

# Fit a normal distribution to the bootstrap probabilities for the main data
mu_main, std_main = norm.fit(g_bootstrap_prob)

# Create the plot
plt.figure(figsize=(10, 6))
sns.histplot(g_bootstrap_prob, bins=50, kde=False, color='blue', label='Bootstrap Probabilities')

# Plot the fitted normal distribution for the main data
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_main = norm.pdf(x, mu_main, std_main)
plt.plot(x, p_main * len(g_bootstrap_prob) * (xmax - xmin) / 100, 'k', linewidth=2, label='Fitted Normal')

# Shift the fitted normal to the lower and upper bounds of the 95% CI
p_lower = norm.pdf(x, lower_bound, std_main)
p_upper = norm.pdf(x, upper_bound, std_main)
plt.plot(x, p_lower * len(g_bootstrap_prob) * (xmax - xmin) / 100, 'r--', linewidth=2, label='Shifted to Lower 95% CI')
plt.plot(x, p_upper * len(g_bootstrap_prob) * (xmax - xmin) / 100, 'g--', linewidth=2, label='Shifted to Upper 95% CI')

# Add vertical lines for the 95% confidence interval
plt.axvline(x=lower_bound, color='r', linestyle='--', label=f'Lower 95% CI: {lower_bound:.4f}')
plt.axvline(x=upper_bound, color='g', linestyle='--', label=f'Upper 95% CI: {upper_bound:.4f}')

plt.xlabel('Probability that g < 0')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution with Shifted Fitted Normals at 95% Confidence Intervals')
plt.legend()

# Calculate the minimum and maximum values that make sense to plot for the CDF based on the bootstrapped data
cdf_xmin = min(min(g_bootstrap_prob), lower_bound, upper_bound) - 0.1
cdf_xmax = max(max(g_bootstrap_prob), lower_bound, upper_bound) + 0.1

# Generate x values for CDF within a sensible range
cdf_x = np.linspace(cdf_xmin, cdf_xmax, 100)
# Calculate the CDF of the normal distributions
cdf_main = norm.cdf(x, mu_main, std_main)
cdf_lower = norm.cdf(x, lower_bound, std_main)
cdf_upper = norm.cdf(x, upper_bound, std_main)

# Calculate the 98th percentile of each curve
percentile_98_main = norm.ppf(0.98, mu_main, std_main)
percentile_98_lower = norm.ppf(0.98, lower_bound, std_main)
percentile_98_upper = norm.ppf(0.98, upper_bound, std_main)

# Create the cumulative frequency plot
plt.figure(figsize=(10, 6))
plt.plot(x, cdf_main, 'k', linewidth=2, label='Fitted Normal CDF')
plt.plot(x, cdf_lower, 'r--', linewidth=2, label='Shifted to Lower 95% CI CDF')
plt.plot(x, cdf_upper, 'g--', linewidth=2, label='Shifted to Upper 95% CI CDF')

# Mark the 98th percentile of each curve
plt.axvline(x=percentile_98_main, color='k', linestyle='--', label=f'98th Percentile Main: {percentile_98_main:.4f}')
plt.axvline(x=percentile_98_lower, color='r', linestyle='--', label=f'98th Percentile Lower: {percentile_98_lower:.4f}')
plt.axvline(x=percentile_98_upper, color='g', linestyle='--', label=f'98th Percentile Upper: {percentile_98_upper:.4f}')

plt.xlabel('Probability that g < 0')
plt.ylabel('Cumulative Frequency')
plt.title('CDFs of Shifted Fitted Normals with 98th Percentile Markers')
plt.legend()
plt.grid(True)
plt.show()




# Function for Monte Carlo simulation to estimate the probability that g < 0
def monte_carlo_simulation(mu_R, std_R, mu_S, std_S, num_trials=100000):
    count_failure = 0
    for _ in range(num_trials):
        sample_R = np.random.normal(mu_R, std_R)
        sample_S = np.random.normal(mu_S, std_S)
        g_sample = sample_R - sample_S
        if g_sample < 0:
            count_failure += 1
    return count_failure / num_trials


# Initialize the mean and standard deviations for R and S
mu_R, std_R = np.mean([503.252, 460.005, 485.503, 466.061, 475.449]), np.std(
    [503.252, 460.005, 485.503, 466.061, 475.449])
mu_S, std_S = np.mean(
    [376.594, 278.222, 331.535, 330.774, 395.173, 394.203, 387.309, 361.754, 300.191, 381.09]), np.std(
    [376.594, 278.222, 331.535, 330.774, 395.173, 394.203, 387.309, 361.754, 300.191, 381.09])

# Perform sensitivity test by perturbing the standard deviations
perturb_factor = np.linspace(0.5, 1.5, 20)  # Factors to multiply with the standard deviations
sensitivity_R = []
sensitivity_S = []

for factor in perturb_factor:
    # Perturb the standard deviation of R and run Monte Carlo simulation
    new_std_R = std_R * factor
    prob_failure_R = monte_carlo_simulation(mu_R, new_std_R, mu_S, std_S)
    sensitivity_R.append(prob_failure_R)

    # Perturb the standard deviation of S and run Monte Carlo simulation
    new_std_S = std_S * factor
    prob_failure_S = monte_carlo_simulation(mu_R, std_R, mu_S, new_std_S)
    sensitivity_S.append(prob_failure_S)

# Plotting the sensitivity analysis results
plt.figure(figsize=(12, 6))
plt.plot(perturb_factor, sensitivity_R, label='Sensitivity to R', marker='o')
plt.plot(perturb_factor, sensitivity_S, label='Sensitivity to S', marker='x')
plt.axhline(y=monte_carlo_simulation(mu_R, std_R, mu_S, std_S), color='r', linestyle='--', label='Original Probability')
plt.xlabel('Perturbation Factor')
plt.ylabel('Probability that g < 0')
plt.title('Sensitivity Analysis')
plt.legend()
plt.grid(True)
plt.show()



#############################Q2_2021#####################

import numpy as np
from scipy.stats import norm, uniform, lognorm
import matplotlib.pyplot as plt
import scipy
# Distributions of input parameters
SM_mu, SM_std, SM_CoV = 17.2, 1.7, 0.1
SWBM_mu, SWBM_std, SWBM_CoV = 2.4, 0.1, 0.1
wave_BM1 = 1.9
wave_BM2 = 2.3
yield_stress_mu, yield_stress_std, yield_stress_CoV = 343, 17, 0.05

# Allowable yield stress
allowable_yield_stress = 0.90 * 315

# Number of simulations
num_simulations = 1000000

# Generate random samples from input parameter distributions
SM_samples = np.random.normal(SM_mu, SM_std, num_simulations)
SWBM_samples = np.random.normal(SWBM_mu, SWBM_std, num_simulations)
wave_BM_samples = np.random.uniform(wave_BM1, wave_BM2, num_simulations)
yield_stress_samples = np.random.lognormal(
    np.log(yield_stress_mu / np.sqrt(1 + yield_stress_CoV ** 2)),
    np.sqrt(np.log(1 + yield_stress_CoV ** 2)),
    num_simulations,
)

# Calculate the imposed bending stress for each simulation
bending_stress_samples = (SWBM_samples + wave_BM_samples) * 10 ** 3 / SM_samples

# Plot distribution of SM samples
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(SM_samples, bins=50, color='blue', alpha=0.7)
plt.xlabel('SM (m^3)')
plt.ylabel('Frequency')
plt.title('Distribution of SM samples')

# Plot distribution of SWBM samples
plt.subplot(2, 2, 2)
plt.hist(SWBM_samples, bins=50, color='red', alpha=0.7)
plt.xlabel('SWBM (GNm)')
plt.ylabel('Frequency')
plt.title('Distribution of SWBM samples')

# Plot distribution of wave_BM samples
plt.subplot(2, 2, 3)
plt.hist(wave_BM_samples, bins=50, color='green', alpha=0.7)
plt.xlabel('Wave BM (GNm)')
plt.ylabel('Frequency')
plt.title('Distribution of Wave BM samples')

# Plot distribution of bending stress samples
plt.subplot(2, 2, 4)
plt.hist(bending_stress_samples, bins=50, color='purple', alpha=0.7)
plt.xlabel('Bending Stress (MPa)')
plt.ylabel('Frequency')
plt.title('Distribution of Bending Stress samples')
plt.axvline(allowable_yield_stress, color='black', linestyle='--', label='Allowable Yield Stress')
plt.legend()
plt.tight_layout()



# Number of Monte Carlo samples
n_samples = 10000000

# Initialize failure count
failure_count = 0

# Given constants
mean_SM = 17.2  # m^3
std_SM = 1.7  # m^3
mean_SWBM = 2.4  # GNm
std_SWBM = 0.12  # GNm
WBM_lower = 1.9  # GNm
WBM_upper = 2.3  # GNm
mean_sigma_yield = 343  # MPa
std_sigma_yield = 17  # MPa

# Convert mean and std dev of sigma_yield for lognormal distribution
mean_log_sigma_yield = np.log(mean_sigma_yield)
std_log_sigma_yield = np.sqrt(np.log(1 + (std_sigma_yield / mean_sigma_yield) ** 2))

# Run Monte Carlo simulation
for _ in range(n_samples):
    # Generate random samples based on given distributions
    SM_sample = np.random.normal(mean_SM, std_SM)
    SWBM_sample = np.random.normal(mean_SWBM, std_SWBM)
    WBM_sample = np.random.uniform(WBM_lower, WBM_upper)
    sigma_yield_sample = np.random.lognormal(mean_log_sigma_yield, std_log_sigma_yield)

    # Calculate sigma_b for the current sample
    sigma_b_sample = ((SWBM_sample + WBM_sample) * 1e3) / SM_sample  # Convert GNm to MNm

    # Check if the sample exceeds the allowable yield stress (multiplied by safety factor of 0.9)
    if sigma_b_sample > sigma_yield_sample:
        failure_count += 1

# Calculate the estimated probability of failure
prob_failure = failure_count / n_samples

print("Estimated Monte Carlo probability of exceeding yield stress:", prob_failure)
plt.show()

#############################Q2_2021###############
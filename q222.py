
#####This is just a separate script to get the confidence intervals of the LHS model. For some reason It was taking too long to compute so i have tried using numba and multiprocessing to improve performance.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyDOE import lhs
from numba import jit
from multiprocessing import Pool, cpu_count
# Data
diameters = [0.0032, 0.0039, 0.0037, 0.0035, 0.0031, 0.0040, 0.0038, 0.0038, 0.0040, 0.0037]
porosities = [0.375, 0.347, 0.329, 0.352, 0.388, 0.419, 0.404, 0.394, 0.352, 0.370]
lengths = [2.86, 3.13, 3.08, 3.12, 2.94, 2.90, 2.80, 3.05, 3.02, 3.04]
# Variables of pressure drop equation
density = 1.225
viscosity = 1.81 * (10 ** (-5))
velocity = 0.35
target_pressure_drop = 15250

# Probability distributions
mu_diameter, std_diameter = np.mean(diameters), np.std(diameters)
mu_porosity, std_porosity = np.mean(porosities), np.std(porosities)
mu_length, std_length = np.mean(lengths), np.std(lengths)




@jit(nopython=True)
def pressure_drop(length, diameter, porosity):
    result = (150 * viscosity * length * ((1 - porosity) ** 2) * velocity) / ((diameter ** 2) * (porosity ** 3)) + (
            1.75 * length * density * (1 - porosity) * (velocity ** 2)) / ((diameter) * (porosity ** 3))
    return result





def worker_function(start, end, samples):
    count = 0
    for i in range(start, end):
        delta_p = pressure_drop(samples[i, 2], samples[i, 0], samples[i, 1])
        if delta_p > 15250:
            count += 1
    return count


def run_lhs_simulation_parallel(N, mu_length, mu_diameter, mu_porosity, std_length, std_diameter, std_porosity):
    lhs_samples = lhs(3, samples=N)

    lhs_diameters = norm.ppf(lhs_samples[:, 0], loc=mu_diameter, scale=std_diameter)
    lhs_porosities = norm.ppf(lhs_samples[:, 1], loc=mu_porosity, scale=std_porosity)
    lhs_lengths = norm.ppf(lhs_samples[:, 2], loc=mu_length, scale=std_length)

    samples = np.column_stack([lhs_diameters, lhs_porosities, lhs_lengths])

    num_workers = cpu_count()
    pool = Pool(num_workers)

    chunk_size = N // num_workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]

    results = pool.starmap(worker_function, [(start, end, samples) for start, end in ranges])

    pool.close()
    pool.join()

    total_count = sum(results)

    return total_count / N

def calculate_confidence_interval(probability, N, confidence_level=0.95):
    Z = norm.ppf(1 - (1 - confidence_level) / 2)  # Z-value for 95% confidence level
    interval = Z * np.sqrt((probability * (1 - probability)) / N)
    return probability - interval, probability + interval

if __name__ == '__main__':
    # Number of LHS samples
    N_LHS = 100000


    # Run the LHS simulation
    probability_LHS = run_lhs_simulation_parallel(N_LHS, mu_length, mu_diameter, mu_porosity, std_length, std_diameter,std_porosity)

    # Print results
    print(f"Probability of Δp exceeding 15,250 Pa using LHS: {probability_LHS}.6f")
    # Calculate 95% confidence intervals
    lower_bound, upper_bound = calculate_confidence_interval(probability_LHS, N_LHS)

    # Print results
    print(f"95% Confidence Interval: ({lower_bound}, {upper_bound})")

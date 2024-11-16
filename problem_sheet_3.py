"""Imports and setup."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import chisquare, poisson

# Load the data
data_path = Path("/Users/lucascurtin/Desktop/module_repos/s1/datasets")
mom_data_path = data_path / "mom_data.npy"
mom_data = np.load(mom_data_path)

# Calculate first and second moments
mu_1 = np.mean(mom_data)  # First moment (mean)
mu_2 = np.mean(mom_data**2)  # Second moment (not variance)

# Define fixed bounds for the integral
a, b = -1, 1


# Function to calculate d_k terms
def d_calc(k: int) -> float:
    """Calculates our d variable for a certain k.

    Args:
        k (int): Either d_1 or d_2 etc...

    Returns:
        float: d variable
    """
    return (1 / k) * (b**k - a**k)


# Calculate terms A, B, C, D, E, F
A = mu_1 * d_calc(2) - d_calc(3)
B = mu_1 * d_calc(3) - d_calc(4)
C = d_calc(2) - mu_1 * d_calc(1)

D = mu_2 * d_calc(2) - d_calc(4)
E = mu_2 * d_calc(3) - d_calc(5)
F = d_calc(3) - mu_2 * d_calc(1)

# Solve for alpha and beta
alpha = (C * E - B * F) / (A * E - B * D)
beta = (A * F - C * D) / (A * E - B * D)

# Calculate normalization constant N
N = 1 / (d_calc(1) + alpha * d_calc(2) + beta * d_calc(3))


# Define the PDF as a function
def pdf(x: float) -> float:
    """_Calculate the normalized PDF value at x using alpha, beta, and N.

    Args:
        x (float): Our x variable

    Returns:
        float: Probability
    """
    return N * (1 + alpha * x + beta * x**2)


# %%
# ? Accept-Reject Sampling
def accept_reject_sampling(pdf: pdf, num_samples: int) -> np.array:
    """Performs our accept_reject sampling.

    Args:
        pdf (pdf): Our pdf function
        num_samples (int): Number of samples

    Returns:
        np.array: Our collected outputs
    """
    x_min, x_max = a, b
    f_max = max(pdf(np.linspace(x_min, x_max, num_samples)))  # Estimate f_max

    rng = np.random.default_rng()
    samples = []
    i = 0
    while i < num_samples:
        x_i = rng.uniform(x_min, x_max)
        y = rng.uniform(0, f_max)
        if y <= pdf(x_i):
            logger.info((i / num_samples) * 100)
            samples.append(x_i)
            i += 1
    return np.array(samples)


# Generate a sample
num_samples = 1000
samples = accept_reject_sampling(pdf, num_samples)


# %%
# ? Plot the histogram of samples and the PDF
x_vals = np.linspace(a, b, 1000)
pdf_vals = pdf(x_vals)

integral = np.trapezoid(pdf_vals, x_vals)

logger.info(f"integral = {integral}")
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.5, label="Sampled Data")
plt.plot(x_vals, pdf_vals, label="PDF", color="black")
plt.title("Histogram of Sampled Data with PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()


# %%
# ? Define the log-likelihood function
def log_likelihood(params, data):
    """Calculate the negative log-likelihood for the given alpha and beta."""
    alpha, beta = params
    # Recalculate the normalization constant N for each alpha and beta
    N = 1 / (d_calc(1) + alpha * d_calc(2) + beta * d_calc(3))
    # Calculate the PDF values for the data points using these parameters
    pdf_vals = N * (1 + alpha * data + beta * data**2)
    # Avoid log(0) by replacing zero values with a small number
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    # Calculate the negative log-likelihood
    return -np.sum(np.log(pdf_vals))


# Use the observed samples to find the MLE for alpha and beta
initial_guess = [alpha, beta]  # Start with the moment-based estimates
result = minimize(log_likelihood, initial_guess, args=(samples,), method="L-BFGS-B")
alpha_mle, beta_mle = result.x

logger.info(f"MLE alpha = {alpha_mle}")
logger.info(f"MLE beta = {beta_mle}")

# Comparison of moment-based and MLE estimates
logger.info(f"Moment-based alpha: {alpha}, beta: {beta}")
# %%

# ? Question 3
# Below are the number of neutrino events detected in 10 second intervals by the
# Irvine- Michigan-Brookhaven experiment on 23rd February 1987. For context the
# “back- ground” should give a Poisson distribution of neutrino counts, but some
# kind of “sig- nal” producing excess neutrinos (e.g. a supernova like S1987a
# which was discovered around this time) would provide an excess of neutrinos in a short interval.
# No.ofevents: 0 1 2 3 4 56789
# No.ofintervals: 1042 860 307 78 15 3 0 0 0 1

events = np.linspace(0, 9, 10)
intervals = [1042, 860, 307, 78, 15, 3, 0, 0, 0, 1]


# Calculate observed mean and variance
mean = np.sum(events * intervals) / np.sum(intervals)
variance = np.sum(intervals * (events - mean) ** 2) / np.sum(intervals)

# Expected Poisson distribution based on observed mean
expected_counts = poisson.pmf(events, mean) * np.sum(intervals)

# Perform chi-square test
chi2_stat, p_value = chisquare(f_obs=intervals, f_exp=expected_counts)


# %%
events = np.linspace(0, 9, 10)
intervals = [1042, 860, 307, 78, 15, 3, 0, 0, 0, 0]


# Calculate observed mean and variance
mean = np.sum(events * intervals) / np.sum(intervals)
variance = np.sum(intervals * (events - mean) ** 2) / np.sum(intervals)

# Expected Poisson distribution based on observed mean
expected_counts = poisson.pmf(events, mean) * np.sum(intervals)

# Perform chi-square test
chi2_stat, p_value = chisquare(f_obs=intervals, f_exp=expected_counts)

# We suddenly no longer have a well fitting possion distribution
# and this is likely due to a signficant change i.e. the supernova

# %%

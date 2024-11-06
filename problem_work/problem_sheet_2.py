# %%
"""problem_sheet_2."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import multivariate_normal

# %%
# ? For a two-dimensional normal distribution with parameters, μ1 = 1, μ2 = 4, sigma1 = 3,
# ?sigma2 = 2, ro = 0.5 make plots of the conditional and marginal probabilities. Extension
# ?(don't waste too much time on this), what if this is now a 3D Gaussian? Can you
# ?think of ways of presenting the equivalent information?

mu = np.array([1, 4])
sigma1, sigma2 = 3, 2  # Standard deviations sigma1=3, sigma2=2
rho = 0.5  # Correlation coefficient
cov = np.array([[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]])

# ? Create a grid for plotting
x = np.linspace(-10, 12, 100)
y = np.linspace(-5, 13, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv = multivariate_normal(mu, cov)

Z = rv.pdf(pos)

# %%
# ? Plot the joint distribution as a contour plot
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, levels=30, cmap="viridis")
plt.colorbar(label="Density")
plt.title("Joint Distribution of 2D Gaussian")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# %%
# ? Accept Rejection, f(x)=cos^2(x)
# Define the data class
@dataclass
class FunctionData:
    """Holds all our lovely func stuff."""

    name: str
    func: callable


def f_a(x: float) -> float:
    """Calculates the square of the cosine of x."""
    return np.cos(x) ** 2


def f_b(x: float) -> float:
    """Calculates the sum of the sine and cosine of x, plus 2."""
    return np.sin(x) + np.cos(x) + 2


def f_c(x: float) -> float:
    """Combines sine, cosine, hyperbolic sine, and hyperbolic cosine of x."""
    return (np.sin(x) + np.cos(x)) / (np.sinh(x) + np.cosh(x)) + 25


def f_d(x: float) -> float:
    """Calculates the square of the exponential of x."""
    return np.exp(x) ** 2


def generate_binned_counts(func_data: FunctionData) -> pd.DataFrame:
    """Generates a DataFrame with binned counts for a given function over a specified range.

    This function applies Monte Carlo sampling to estimate the distribution of values from func
    between hardcoded values of `x_min=0` and `x_max=2`. It returns a DataFrame with binned counts,
    actual function values at bin midpoints, and normalized counts.

    Args:
        func_data (FunctionData): The FunctionData object containing function details.

    Returns:
        pd.DataFrame: A DataFrame with columns for binned counts (`count`), actual function values
                      at midpoints (`actual`), and normalized counts (`normalised_counts`).
    """
    x_min = 0
    x_max = 2 * np.pi
    num_samples = 1000
    num_bins = 50

    f_max = func_data.func(np.linspace(x_min, x_max, num_samples)).max()

    rng = np.random.default_rng()
    samples = []
    i = 0
    while i < num_samples:
        x_i = rng.uniform(x_min, x_max)
        y = rng.uniform(0, f_max)
        if y <= func_data.func(x_i):
            samples.append(x_i)
            logger.info(f"Sampling progress: {(i / num_samples) * 100:.2f}%")
            i += 1

    bins = np.linspace(x_min, x_max, num_bins + 1)
    val_counts = pd.Series(samples)

    # Create binned counts DataFrame
    return (
        pd.cut(val_counts, bins=bins)
        .value_counts(sort=False)
        .to_frame(name="count")
        .assign(
            midpoint=lambda df: pd.to_numeric(
                df.index.to_series().apply(lambda x: x.mid),
            ),
            actual=lambda df: func_data.func(df["midpoint"]),
            normalised_counts=lambda df: (df["count"] - df["count"].min())
            / (df["count"].max() - df["count"].min()),
            normalised_actual=lambda df: (df["actual"] - df["actual"].min())
            / (df["actual"].max() - df["actual"].min()),
        )
        .set_index("midpoint")[["normalised_counts", "normalised_actual"]]
    )


functions = [
    FunctionData(name="f_a", func=f_a),
    FunctionData(name="f_b", func=f_b),
    FunctionData(name="f_c", func=f_c),
    FunctionData(name="f_d", func=f_d),
]

funcs_concat = pd.concat(
    generate_binned_counts(func).assign(name=func.name) for func in functions
).reset_index()

funcs_table = funcs_concat.pivot_table(
    index="midpoint",
    columns="name",
    values=["normalised_actual", "normalised_counts"],
)
funcs_table.columns = funcs_table.columns.swaplevel(0, 1)
funcs_table.columns.names = ["name", "value"]

for name in funcs_table.columns.get_level_values("name").unique():
    funcs_table[name].plot()


# %%
def acceptance_efficiency(func: callable, sigma_range: int, num_samples: int) -> float:
    """Calculates our efficencies when sampling different x (normal).

    Args:
        func (callable): Our func
        sigma_range (int): Our range of sigma we sample
        num_samples (int): Our number of samples

    Returns:
        float: Our efficiencies
    """
    std_dev = 1
    x_min, x_max = -sigma_range * std_dev, sigma_range * std_dev

    rng = np.random.default_rng()
    x_samples = rng.uniform(x_min, x_max, num_samples)
    f_max = (
        1 if func == f_a else np.max(func(x_samples))
    )  # Assuming f_max=1 for f_a based on cos^2(x)
    y_samples = rng.uniform(0, f_max, num_samples)

    accepted_samples = y_samples <= func(x_samples)
    return np.sum(accepted_samples) / num_samples


sigma_ranges = np.arange(1, 9)
efficiency_results = []

for func_data in functions:
    func_efficiencies = [
        acceptance_efficiency(func_data.func, sigma, 1000) for sigma in sigma_ranges
    ]
    efficiency_results.append(
        pd.DataFrame(
            {
                "sigma_range": sigma_ranges,
                "effiency": func_efficiencies,
                "function": func_data.name,
            },
        ),
    )

efficiency_data = pd.concat(efficiency_results).pivot_table(
    index="sigma_range",
    columns="function",
    values="effiency",
)

efficiency_data.plot()
# %%

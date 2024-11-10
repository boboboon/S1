# %%
"""First problem sheet workthrough."""

import pickle
from pathlib import Path

import seaborn as sns
from loguru import logger

data_path = Path(
    "/Users/lucascurtin/Desktop/module_repos/s1/datasets",
)

with Path.open(data_path / "ps1.pkl", "rb") as file:
    data = pickle.load(file)

# %%
# ? Question 1

_ = sns.pairplot(data)
# %%

# Seaborn scatter plot with KDE in margins
sns.jointplot(
    data=data,
    x="vx",
    y="vz",
    s=10,
    kind="scatter",
    marginal_kws={"bins": 20, "fill": True},
)

# %%
# Calculate the correlation matrix
correlation_matrix = data.corr()

# Calculate the covariance matrix
covariance_matrix = data.cov()

# Display the correlation and covariance matrices
logger.info("Correlation Matrix:")
logger.info(correlation_matrix)

logger.info("\nCovariance Matrix:")
logger.info(covariance_matrix)
# %%

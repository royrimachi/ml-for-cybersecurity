# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ml-for-cybersecurity (3.14.x)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 1 – Supervised Learning
#
# ## Exercise using Random Forests

# %% [markdown]
# The dataset was obtained in the following URL (https://cicresearch.ca/CICDataset/CIC-IDS-2017/download.php). However, it is necessary to register in order to be able to download it.
#
# - Date of Download: 11-09-2026
# - License: No license provided
#

# %% [markdown]
# ## Part A – Data acquisition and hygiene
# #### Unzip contents

# %%
# # !unzip -n ../data/MachineLearningCSV.zip -d ../data/

# %% [markdown]
# #### Load data on CSV

# %%
import glob
import os

import pandas as pd

path = os.path.join("../data", "MachineLearningCVE")
all_files = glob.glob(os.path.join(path, "*.csv"))

df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

print("Dataset loaded successfully!")

# %%
df.head()

# %% [markdown]
# #### Strip column-name whitespace

# %%
df.columns

# %%
df.columns = df.columns.str.strip()

# %%
df.columns

# %% [markdown]
# #### Show shape

# %%
df.shape

# %% [markdown]
# #### Show dtypes

# %%
df.dtypes.value_counts()

# %%
df.dtypes.head(10)

# %% [markdown]
# #### Show label value counts

# %%
df["Label"].value_counts()

# %% [markdown]
# #### Audit Inf / NaN

# %%
raw_na = df.isna().sum()
raw_na[raw_na > 0]

# %%
df.isna().mean().loc[lambda x: x > 0] * 100

# %%
import numpy as np

is_inf_mask = df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf])
inf_counts = is_inf_mask.sum()

inf_counts[inf_counts > 0].head()

# %% [markdown]
# #### Replace Infinite values with NaN

# %%
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# %%
# Total NaN values
total_nan = df.isna().any(axis=1).sum()

# %%
(total_nan / len(df)) * 100

# %% [markdown]
# After reviewing the number of NaN values, we can see that it is only around $0.1\%$ of the total dataset. This percentage is not relevant and can be dropped.

# %%
df.dropna(subset=[col for col in df.columns if col != "Label"], inplace=True)

# %%
df.reset_index(drop=True, inplace=True)
df.shape

# %% [markdown]
# #### Hunt data leakage
#
# The data leakage could happen when we keep data that could be found as well on the test, validation or target dataset. In this case, we need to check which information could be found on the target dataset as well and avoid feeding it to the model as it could influence the classification.

# %%
df.columns

# %%
df.describe()

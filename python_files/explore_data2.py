# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset
#
# In this notebook, we will explore the **Midwest Survey** dataset from [skrub](https://skrub-data.org/).
#
# This dataset contains survey responses from people across the United States,
# asking them about their perception of the Midwest region.
#
# The goal is to predict the **Census Region** where a respondent lives,
# based on their survey answers.

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()

# X contains the features (the survey answers)
X = dataset.X
# y contains the target (the Census Region)
y = dataset.y

# added for plotting
import matplotlib.pyplot as plt

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?

# Use the `.shape` attribute to find out the number of rows and columns.

# %%
# Display the number of rows and columns
n_rows, n_cols = X.shape
print(f"Number of examples (rows): {n_rows}")
print(f"Number of features (columns): {n_cols}")

### Number of examples (rows): 2494 ###
### Number of features (columns): 28 ###

# %%
# You can also look at the first few rows of the dataset
X.head()

# %% [markdown]
# ## Question 2: What is the distribution of the target?
#
# The target variable `y` tells us the Census Region of each respondent.
# Let's see how many respondents belong to each region.

# %%
# Count how many respondents belong to each region
y_counts = y.value_counts()
print("Target distribution:")
print(y_counts)

# %%
# Visualize the target distribution with a bar plot
# hint: use barh
plt.figure(figsize=(6, 4))
y_counts.plot.barh()
plt.xlabel("Count")
plt.title("Census Region distribution")
plt.tight_layout()
plt.show()

### Target distribution: ###
# Census_Region
# East North Central    758
# West North Central    358
# Middle Atlantic       334
# South Atlantic        248
# Pacific               243
# Mountain              190
# West South Central    172
# East South Central     97
# New England            94

# %% [markdown]
# Is the target balanced (roughly the same number of examples per class) or imbalanced?

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?
#
# Let's look at the column names and their data types.

# %%
# List all column names
print("Columns:")
print(list(X.columns))

# %%
# Show data types for each column
print("\nData types:")
print(X.dtypes)

# %% [markdown]
# How many features are numerical? How many are categorical (text)?

# %% 

# %% 
from skrub import TableReport
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?
#
# Missing values can cause problems for machine learning models.
# Let's check if there are any.

# %%
# Check for NaN missing values
missing_counts = X.isna().sum()
print("Columns with missing values:")
print(missing_counts[missing_counts > 0])

### Columns with missing values: ###
# Series([], dtype: int64)

# %% [markdown]
# Missing values can sometimes be encoded differently. Let's look at some columns more closely.

# %%
# Look at unique values for the Household_Income column
col = "Household_Income"
if col in X.columns:
    print(f"\nUnique values for {col}:")
    print(X[col].unique()[:50])
else:
    print(f"\nColumn {col} not present in dataset.")

# %%
# Look at unique values for the Education column
col = "Education"
if col in X.columns:
    print(f"\nUnique values for {col}:")
    print(X[col].unique()[:50])
else:
    print(f"\nColumn {col} not present in dataset.")

# %% [markdown]
# Do you see a special value that could represent missing data?

# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?
#
# Let's explore this important feature.

# %%
# TODO: display the value counts for the column
# "How_much_do_you_personally_identify_as_a_Midwesterner"
col = "How_much_do_you_personally_identify_as_a_Midwesterner"
if col in X.columns:
    ident_counts = X[col].value_counts(dropna=False)
    print(f"\nValue counts for '{col}':")
    print(ident_counts)
else:
    ident_counts = None
    print(f"\nColumn '{col}' not found in the dataset.")

# %%
# TODO: make a bar plot of the results
if ident_counts is not None:
    plt.figure(figsize=(8, 4))
    ident_counts.plot.bar()
    plt.ylabel("Count")
    plt.title("How much do you personally identify as a Midwesterner")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Bonus: Explore another feature
#
# Pick another column and explore its distribution.
# For example: `Gender`, `Age`, or one of the
# "Do you consider X state as part of the Midwest" columns.

# %%
# TODO: explore a column of your choice
bonus_col = "Gender"
if bonus_col in X.columns:
    print(f"\nDistribution for {bonus_col}:")
    print(X[bonus_col].value_counts(dropna=False))
    plt.figure(figsize=(6, 3))
    X[bonus_col].value_counts().plot.bar()
    plt.title(bonus_col)
    plt.tight_layout()
    plt.show()
else:
    # try Age if Gender absent
    bonus_col = "Age"
    if bonus_col in X.columns:
        print(f"\nSummary statistics for {bonus_col}:")
        print(X[bonus_col].describe())
        plt.figure(figsize=(6, 3))
        X[bonus_col].hist(bins=20)
        plt.title(bonus_col)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo common bonus columns (Gender/Age) found in dataset.")
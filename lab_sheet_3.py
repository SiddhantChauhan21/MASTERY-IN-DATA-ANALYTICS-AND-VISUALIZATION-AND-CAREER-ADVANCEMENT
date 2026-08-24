import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm


pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

sns.set_style("whitegrid")

# Load Titanic dataset directly from Seaborn
df = sns.load_dataset("titanic")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("Missing Values:")
print(df.isnull().sum())

numeric_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

# Mean
mean_values = df[numeric_columns].mean()

# Median
median_values = df[numeric_columns].median()

# Mode
mode_values = df[numeric_columns].mode().iloc[0]

# Variance
variance_values = df[numeric_columns].var()

# Standard deviation
std_values = df[numeric_columns].std()

# Create a statistics table
statistics_table = pd.DataFrame({
    "Mean": mean_values,
    "Median": median_values,
    "Mode": mode_values,
    "Variance": variance_values,
    "Standard Deviation": std_values
})

print("Descriptive Statistics:")
print(statistics_table)

correlation_matrix = df[numeric_columns].corr(method="pearson")

print("Pearson Correlation Matrix:")
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Pearson Correlation Matrix - Titanic Dataset")
plt.tight_layout()
plt.show()

survived = df[df["survived"] == 1]["fare"].dropna()

not_survived = df[df["survived"] == 0]["fare"].dropna()

print("Number of survivors:", len(survived))
print("Number of non-survivors:", len(not_survived))

print("\nAverage Fare of Survivors:",
      survived.mean())

print("Average Fare of Non-Survivors:",
      not_survived.mean())

t_statistic, p_value = stats.ttest_ind(
    survived,
    not_survived,
    equal_var=False
)

print("Independent Sample t-test")
print("--------------------------------")
print("t-statistic =", t_statistic)
print("p-value =", p_value)

alpha = 0.05

if p_value < alpha:
    print("\nConclusion:")
    print("Reject H0.")
    print("There is a statistically significant difference")
    print("in average fare between survivors and non-survivors.")
else:
    print("\nConclusion:")
    print("Fail to reject H0.")
    print("There is no statistically significant difference")
    print("in average fare between survivors and non-survivors.")

    # Boxplot: Fare vs Survival

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="survived",
    y="fare",
    data=df
)

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.title("Fare Distribution by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Fare")

plt.show()

first_class = df[df["pclass"] == 1]["fare"].dropna()
second_class = df[df["pclass"] == 2]["fare"].dropna()
third_class = df[df["pclass"] == 3]["fare"].dropna()

# Perform One-Way ANOVA
f_statistic, anova_p_value = stats.f_oneway(
    first_class,
    second_class,
    third_class
)

print("One-Way ANOVA")
print("--------------------------------")
print("F-statistic =", f_statistic)
print("p-value =", anova_p_value)

if anova_p_value < 0.05:
    print("\nConclusion:")
    print("Reject H0.")
    print("There is a statistically significant difference")
    print("in average fare among passenger classes.")
else:
    print("\nConclusion:")
    print("Fail to reject H0.")
    print("There is no statistically significant difference")
    print("in average fare among passenger classes.")

    # Boxplot of Fare by Passenger Class

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="pclass",
    y="fare",
    data=df
)

plt.title("Fare Distribution Across Passenger Classes")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.show()

# Boxplot of Fare by Passenger Class

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="pclass",
    y="fare",
    data=df
)

plt.title("Fare Distribution Across Passenger Classes")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.show()


# Remove missing values
regression_data = df[["age", "fare"]].dropna()

# Independent variable
X = regression_data["age"]

# Dependent variable
Y = regression_data["fare"]

# Add constant
X_with_constant = sm.add_constant(X)

# Build regression model
regression_model = sm.OLS(
    Y,
    X_with_constant
).fit()

# Display results
print(regression_model.summary())

intercept = regression_model.params["const"]
slope = regression_model.params["age"]

r_squared = regression_model.rsquared

print("Regression Equation:")
print(
    f"Fare = {intercept:.2f} + "
    f"({slope:.2f} × Age)"
)

print("\nIntercept:")
print(intercept)

print("\nSlope / Regression Coefficient:")
print(slope)

print("\nP-values:")
print(regression_model.pvalues)

print("\n95% Confidence Intervals:")
print(regression_model.conf_int())

print("\nR-squared:")
print(r_squared)

print("\nAdjusted R-squared:")
print(regression_model.rsquared_adj)

# Plot regression line

plt.figure(figsize=(9, 6))

sns.regplot(
    x="age",
    y="fare",
    data=regression_data,
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "red"}
)

plt.title("Linear Regression: Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.show()
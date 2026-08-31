
# 1 & 2. Import libraries and load data


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("titanic")


# 3. Inspect data


print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# 4. Handle missing values


df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Remove deck because it has many missing values
df = df.drop(columns=["deck"])

print("\nMissing Values:")
print(df.isnull().sum())

# 5. Remove duplicates


print("\nDuplicates:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.reset_index(drop=True)


# 6. Standardize categorical values

df["sex"] = df["sex"].str.lower().str.strip()
df["embarked"] = df["embarked"].str.lower().str.strip()


# 9. Feature Engineering


df["FamilySize"] = df["sibsp"] + df["parch"] + 1

df["FamilyCategory"] = pd.cut(
    df["FamilySize"],
    bins=[0, 1, 4, np.inf],
    labels=["Alone", "Small", "Large"]
)

df["AgeGroup"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)


# 6. One-Hot Encoding


categorical_columns = [
    "sex",
    "embarked",
    "class",
    "who",
    "alone",
    "FamilyCategory",
    "AgeGroup"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# 7. Outlier Detection and Treatment


Q1 = df["fare"].quantile(0.25)
Q3 = df["fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nFare Lower Bound:", lower_bound)
print("Fare Upper Bound:", upper_bound)

# Cap outliers
df["fare"] = df["fare"].clip(
    lower=lower_bound,
    upper=upper_bound
)


# 8. Standardization


features_to_scale = [
    "age",
    "fare",
    "sibsp",
    "parch",
    "FamilySize"
]

scaler = StandardScaler()

df[features_to_scale] = scaler.fit_transform(
    df[features_to_scale]
)


# 10. Save cleaned dataset


df.to_csv("titanic_cleaned.csv", index=False)


# Final result


print("\nFinal Dataset Shape:", df.shape)
print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

print("\nFinal Dataset:")
print(df.head())

print("\nCleaned dataset saved as titanic_cleaned.csv")

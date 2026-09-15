import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load dataset
df = pd.read_csv("C:\\Users\\Siddhant\\Desktop\\Mall_Customers.csv")

# 2. Display basic information
print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Elbow Method
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# 7. K-Means with 5 clusters
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# 8. Cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# 9. Visualization
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    s=100
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='black',
    marker='X',
    s=250,
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# 10. Cluster analysis
print("\nCluster Analysis:")
print(
    df.groupby('Cluster')[
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    ].mean()
)

# 11. Silhouette Score
score = silhouette_score(X_scaled, df['Cluster'])
print("\nSilhouette Score:", score)



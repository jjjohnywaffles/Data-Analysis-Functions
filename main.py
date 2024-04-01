import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("merged.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Select the relevant columns for clustering
X = df[['price', 'beds']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"WCSS for {i} clusters: {kmeans.inertia_:.2f}")
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply k-means clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Calculate the averages for each cluster
df['cluster'] = y_kmeans
cluster_averages = df.groupby('cluster').mean()[['price', 'beds']]
print("Cluster averages:")
print(cluster_averages)

# Visualize the clustering results with a scatter plot
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('price')
plt.ylabel('number of beds (hosptial size)')
plt.show()
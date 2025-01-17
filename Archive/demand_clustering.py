import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the data
demand_df = pd.read_csv('Data/demand.csv')
products_df = pd.read_csv('Data/products.csv')

# Convert demand to pallets
demand_df = demand_df.merge(products_df[['SKU', 'SKUs/pallet']], on='SKU', how='left')
demand_df['Demand_Pallets'] = demand_df['Demand'] / demand_df['SKUs/pallet']

# Pivot the data to get store-month combinations
pivot_df = demand_df.pivot_table(
    values='Demand_Pallets',
    index='StoreID',
    columns='month',
    aggfunc='sum'
).fillna(0)

# Prepare data for DBSCAN
X = StandardScaler().fit_transform(pivot_df)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Add cluster labels to the original data
pivot_df['Cluster'] = clusters

# Plot results
plt.figure(figsize=(12, 8))

# We'll plot first two months as an example
months = pivot_df.columns[0:2]
colors = clusters

plt.scatter(pivot_df[months[0]], pivot_df[months[1]], 
           c=colors, cmap='viridis')
plt.xlabel(f'Demand in Pallets - {months[0]}')
plt.ylabel(f'Demand in Pallets - {months[1]}')
plt.title('DBSCAN Clustering of Store Demand Patterns')

# Add colorbar
plt.colorbar(label='Cluster')

# Save the plot
plt.savefig('demand_clusters.png')

# Print summary
print("\nClustering Results:")
print(f"Number of clusters found: {len(np.unique(clusters[clusters >= 0]))}")
print(f"Number of noise points: {len(clusters[clusters == -1])}")

# Save results to CSV
pivot_df.to_csv('demand_clusters_results.csv')

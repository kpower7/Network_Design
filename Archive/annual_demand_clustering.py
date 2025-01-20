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

# Calculate annual demand per store
annual_demand = demand_df.groupby('StoreID')['Demand_Pallets'].sum().reset_index()

# Prepare data for DBSCAN
X = StandardScaler().fit_transform(annual_demand[['Demand_Pallets']].values)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Add cluster labels to the data
annual_demand['Cluster'] = clusters

# Sort by demand to see the distribution
annual_demand_sorted = annual_demand.sort_values('Demand_Pallets', ascending=True)

# Plot results
plt.figure(figsize=(15, 8))

# Create scatter plot
plt.scatter(range(len(annual_demand_sorted)), 
           annual_demand_sorted['Demand_Pallets'],
           c=annual_demand_sorted['Cluster'], 
           cmap='viridis')

plt.xlabel('Store Index (sorted by demand)')
plt.ylabel('Annual Demand (Pallets)')
plt.title('DBSCAN Clustering of Annual Store Demand')

# Add colorbar
plt.colorbar(label='Cluster')

# Save the plot
plt.savefig('annual_demand_clusters.png')

# Print summary statistics
print("\nClustering Results:")
print(f"Number of clusters found: {len(np.unique(clusters[clusters >= 0]))}")
print(f"Number of noise points: {len(clusters[clusters == -1])}")

# Print cluster statistics
print("\nCluster Statistics:")
for cluster in sorted(annual_demand['Cluster'].unique()):
    cluster_data = annual_demand[annual_demand['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Number of stores: {len(cluster_data)}")
    print(f"Average annual demand (pallets): {cluster_data['Demand_Pallets'].mean():.2f}")
    print(f"Min annual demand (pallets): {cluster_data['Demand_Pallets'].min():.2f}")
    print(f"Max annual demand (pallets): {cluster_data['Demand_Pallets'].max():.2f}")

# Save results to CSV
annual_demand.sort_values('Demand_Pallets', ascending=False).to_csv('annual_demand_clusters.csv', index=False)

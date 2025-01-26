import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the data
demand_df = pd.read_csv('../Data/demand.csv')
products_df = pd.read_csv('../Data/products.csv')
stores_df = pd.read_csv('../Data/stores.csv')

# Convert demand to pallets
demand_df = demand_df.merge(products_df[['SKU', 'SKUs/pallet']], on='SKU', how='left')
demand_df['Demand_Pallets'] = demand_df['Demand'] / demand_df['SKUs/pallet']

# Calculate annual demand per store
annual_demand = demand_df.groupby('StoreID')['Demand_Pallets'].sum().reset_index()

# Merge with store locations
store_demand = stores_df.merge(annual_demand, on='StoreID', how='left')

# Prepare data for DBSCAN - using lat/long coordinates
X = store_demand[['Latitude', 'Longitude']].values

# Perform DBSCAN clustering
# eps is in degrees (approximately 50km at the equator)
dbscan = DBSCAN(eps=0.5, min_samples=3)
clusters = dbscan.fit_predict(X)

# Add cluster labels to the data
store_demand['Cluster'] = clusters

# Create visualization
plt.figure(figsize=(15, 10))

# Create scatter plot
scatter = plt.scatter(store_demand['Longitude'], 
                     store_demand['Latitude'],
                     c=store_demand['Cluster'],
                     s=store_demand['Demand_Pallets']/2,  # Size based on demand
                     cmap='viridis',
                     alpha=0.6)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Store Clusters\n(Circle size represents annual demand in pallets)')

# Add colorbar
plt.colorbar(scatter, label='Cluster')

# Save the plot
plt.savefig('geo_demand_clusters.png')

# Print summary statistics
print("\nClustering Results:")
print(f"Number of clusters found: {len(np.unique(clusters[clusters >= 0]))}")
print(f"Number of noise points: {len(clusters[clusters == -1])}")

# Print cluster statistics
print("\nCluster Statistics:")
for cluster in sorted(store_demand['Cluster'].unique()):
    cluster_data = store_demand[store_demand['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Number of stores: {len(cluster_data)}")
    print(f"Total annual demand (pallets): {cluster_data['Demand_Pallets'].sum():.2f}")
    print(f"Average annual demand per store (pallets): {cluster_data['Demand_Pallets'].mean():.2f}")
    print(f"Cities: {', '.join(cluster_data['City'].unique())}")

# Save results to CSV with detailed information
store_demand = store_demand.sort_values('Demand_Pallets', ascending=False)
store_demand.to_csv('geo_demand_clusters.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from folium import plugins
from geopy.distance import geodesic
import branca.colormap as cm

# Load the data
demand_df = pd.read_csv('Data/demand.csv')
products_df = pd.read_csv('Data/products.csv')
stores_df = pd.read_csv('Data/stores.csv')

# Convert demand to pallets
demand_df = demand_df.merge(products_df[['SKU', 'SKUs/pallet']], on='SKU', how='left')
demand_df['Demand_Pallets'] = demand_df['Demand'] / demand_df['SKUs/pallet']

# Calculate annual demand per store
annual_demand = demand_df.groupby('StoreID')['Demand_Pallets'].sum().reset_index()

# Merge with store locations
store_demand = stores_df.merge(annual_demand, on='StoreID', how='left')

# Prepare data for clustering
X = store_demand[['Latitude', 'Longitude']].values
weights = store_demand['Demand_Pallets'].values

# Function to calculate weighted sum of distances for a solution
def calculate_total_weighted_distance(X, weights, centers):
    total_distance = 0
    for i in range(len(X)):
        # Find minimum distance to any facility
        min_distance = float('inf')
        for center in centers:
            distance = geodesic((X[i][0], X[i][1]), (center[0], center[1])).miles
            min_distance = min(min_distance, distance)
        total_distance += min_distance * weights[i]
    return total_distance

# Function to create map with facilities and stores
def create_facility_map(store_demand, facilities, n_facilities, facility_assignments):
    # Create base map centered on US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Create colormap for facilities
    colors = ['red', 'blue', 'green', 'purple', 'orange'][:n_facilities]
    
    # Add stores to map
    max_demand = store_demand['Demand_Pallets'].max()
    min_demand = store_demand['Demand_Pallets'].min()
    
    for idx, row in store_demand.iterrows():
        # Scale circle size based on demand
        radius = 2000 * (row['Demand_Pallets'] - min_demand) / (max_demand - min_demand) + 1000
        
        # Create circle marker
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=colors[facility_assignments[idx]],
            fill=True,
            popup=f"Store: {row['StoreID']}<br>City: {row['City']}<br>Annual Demand: {row['Demand_Pallets']:.0f} pallets",
        ).add_to(m)
    
    # Add facilities to map
    for i, facility in enumerate(facilities):
        folium.Marker(
            location=[facility[0], facility[1]],
            icon=folium.Icon(color=colors[i], icon='star'),
            popup=f'Facility {i+1}'
        ).add_to(m)
        
        # Draw circles showing approximate service areas
        folium.Circle(
            location=[facility[0], facility[1]],
            radius=300000,  # 300km radius
            color=colors[i],
            fill=False,
            opacity=0.3
        ).add_to(m)
    
    return m

# Analyze both 2 and 3 facility scenarios
results = []
for n_facilities in [2, 3]:
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_facilities, random_state=42)
    kmeans.fit(X, sample_weight=weights)
    
    # Get facility locations (cluster centers)
    facilities = kmeans.cluster_centers_
    
    # Calculate total weighted distance
    total_distance = calculate_total_weighted_distance(X, weights, facilities)
    
    # Create and save map
    m = create_facility_map(store_demand, facilities, n_facilities, kmeans.labels_)
    m.save(f'DBSCAN/facility_locations_{n_facilities}_map.html')
    
    # Store results
    results.append({
        'n_facilities': n_facilities,
        'total_weighted_distance': total_distance,
        'facilities': facilities,
        'labels': kmeans.labels_
    })

# Print results
print("\nFacility Location Analysis:")
for result in results:
    print(f"\n{result['n_facilities']} Facility Solution:")
    print("Facility Locations:")
    for i, coords in enumerate(result['facilities']):
        print(f"Facility {i+1}: (Lat: {coords[0]:.4f}, Lon: {coords[1]:.4f})")
    print(f"Total Weighted Distance: {result['total_weighted_distance']:,.0f} mile-pallets")

    # Calculate demand served by each facility
    store_demand[f'Facility_{result["n_facilities"]}'] = result['labels']
    facility_demands = store_demand.groupby(f'Facility_{result["n_facilities"]}')['Demand_Pallets'].agg(['sum', 'count'])
    print("\nDemand Distribution:")
    for facility in range(result['n_facilities']):
        demand = facility_demands.loc[facility]
        print(f"Facility {facility+1}:")
        print(f"  Stores served: {demand['count']}")
        print(f"  Total annual demand: {demand['sum']:.0f} pallets")

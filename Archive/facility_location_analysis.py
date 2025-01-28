import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

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

# Function to get nearest city to coordinates
def get_nearest_city(lat, lon):
    try:
        geolocator = Nominatim(user_agent="my_agent")
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and location.raw.get('address'):
            address = location.raw['address']
            city = address.get('city', address.get('town', address.get('village', 'Unknown')))
            state = address.get('state', 'Unknown')
            return f"{city}, {state}"
    except:
        return f"({lat:.2f}, {lon:.2f})"
    return f"({lat:.2f}, {lon:.2f})"

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
    
    # Assign stores to nearest facility
    store_demand[f'Facility_{n_facilities}'] = kmeans.labels_
    
    # Get facility locations and their nearest cities
    facility_locations = [get_nearest_city(lat, lon) for lat, lon in facilities]
    
    # Store results
    results.append({
        'n_facilities': n_facilities,
        'total_weighted_distance': total_distance,
        'facilities': facilities,
        'facility_locations': facility_locations
    })

# Create visualizations
for result in results:
    n_facilities = result['n_facilities']
    facilities = result['facilities']
    
    plt.figure(figsize=(15, 10))
    
    # Plot stores with size based on demand
    scatter = plt.scatter(store_demand['Longitude'], 
                         store_demand['Latitude'],
                         c=store_demand[f'Facility_{n_facilities}'],
                         s=store_demand['Demand_Pallets']/2,
                         cmap='viridis',
                         alpha=0.6)
    
    # Plot facility locations
    plt.scatter(facilities[:, 1], 
               facilities[:, 0],
               c='red',
               marker='*',
               s=500,
               label='Facility Locations')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Optimal {n_facilities} Facility Locations\n(Circle size represents annual demand in pallets)')
    plt.legend()
    
    # Save the plot
    plt.savefig(f'DBSCAN/facility_locations_{n_facilities}.png')

# Print results
print("\nFacility Location Analysis:")
for result in results:
    print(f"\n{result['n_facilities']} Facility Solution:")
    print("Facility Locations:")
    for i, (loc, coords) in enumerate(zip(result['facility_locations'], result['facilities'])):
        print(f"Facility {i+1}: {loc} (Lat: {coords[0]:.4f}, Lon: {coords[1]:.4f})")
    print(f"Total Weighted Distance: {result['total_weighted_distance']:,.0f} mile-pallets")

    # Calculate demand served by each facility
    facility_demands = store_demand.groupby(f'Facility_{result["n_facilities"]}')['Demand_Pallets'].agg(['sum', 'count'])
    print("\nDemand Distribution:")
    for facility in range(result['n_facilities']):
        demand = facility_demands.loc[facility]
        print(f"Facility {facility+1}:")
        print(f"  Stores served: {demand['count']}")
        print(f"  Total annual demand: {demand['sum']:.0f} pallets")

# Save detailed results to CSV
store_demand.to_csv('DBSCAN/facility_location_results.csv', index=False)

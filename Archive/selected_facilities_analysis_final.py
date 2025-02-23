#%% [markdown]
# Selected Facilities Analysis - Final Configuration
## Phoenix, Houston, and Columbus Network Configuration
This notebook analyzes the three selected facilities based on the demand clusters:
1. Phoenix, AZ (Western Region) - FAC027 (52,000 sqft)
2. Houston, TX (Central Region) - FAC011 (30,000 sqft)
3. Columbus, OH (Eastern Region) - FAC035 (58,000 sqft)

Key advantages of this configuration:
- Columbus: Large facility (58,000 sqft) with moderate costs ($12.6/sqft) serving eastern population centers
- Houston: Strategic central location with low costs ($10.5/sqft)
- Phoenix: Good size (52,000 sqft) serving western markets with moderate costs ($12.9/sqft)

#%% [markdown]
## Import Required Libraries

#%%
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
%matplotlib inline

#%% [markdown]
## Load and Prepare Data

#%%
# Load all datasets
rates_df = pd.read_csv('Data/rates.csv')
stores_df = pd.read_csv('Data/stores.csv')
demand_df = pd.read_csv('Data/demand.csv')
products_df = pd.read_csv('Data/products.csv')
labor_df = pd.read_csv('Data/labor_cost.csv')
facilities_df = pd.read_csv('Data/facilities.csv')

# Calculate annual demand in pallets
demand_df = demand_df.merge(products_df[['SKU', 'SKUs/pallet']], on='SKU', how='left')
demand_df['Demand_Pallets'] = demand_df['Demand'] / demand_df['SKUs/pallet']
annual_demand = demand_df.groupby('StoreID')['Demand_Pallets'].sum().reset_index()
store_demand = stores_df.merge(annual_demand, on='StoreID', how='left')

#%% [markdown]
## Cluster Analysis

#%%
# Prepare data for clustering
X = store_demand[['Latitude', 'Longitude']].values
weights = store_demand['Demand_Pallets'].values

# Perform 3-cluster analysis
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X, sample_weight=weights)
store_demand['Cluster'] = kmeans.labels_

# Selected facilities
selected_facilities = {
    'Western': facilities_df[facilities_df['FacilityID'] == 'FAC027'].iloc[0],  # Phoenix
    'Central': facilities_df[facilities_df['FacilityID'] == 'FAC011'].iloc[0],  # Houston
    'Eastern': facilities_df[facilities_df['FacilityID'] == 'FAC035'].iloc[0]   # Columbus
}

# Calculate total facility costs
facility_costs = pd.DataFrame([
    {
        'Region': region,
        'Location': facility['Location'],
        'Size': facility['Size_SqFt'],
        'Total_Cost': facility['RentCost_SqFt'] + facility['UtilitiesOpsCost_SqFt']
    }
    for region, facility in selected_facilities.items()
])

print("\nFacility Cost Summary:")
display(facility_costs)

#%% [markdown]
## Create Network Map with Clusters

#%%
def create_network_map():
    """Create a map showing selected facilities and their clusters"""
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    colors = ['red', 'blue', 'green']
    region_colors = {'Western': 'red', 'Central': 'blue', 'Eastern': 'green'}
    
    # Add stores with cluster colors
    for idx, store in store_demand.iterrows():
        folium.CircleMarker(
            location=[store['Latitude'], store['Longitude']],
            radius=5,
            color=colors[store['Cluster']],
            fill=True,
            popup=f"Store: {store['StoreID']}<br>" +
                  f"City: {store['City']}, {store['State']}<br>" +
                  f"Annual Demand: {store['Demand_Pallets']:,.0f} pallets<br>" +
                  f"Cluster: {store['Cluster']}",
            fillOpacity=0.7
        ).add_to(m)
    
    # Add facilities with larger markers
    for region, facility in selected_facilities.items():
        city = facility['Location'].split(',')[0].strip()
        state = facility['Location'].split(',')[1].strip()
        matching_stores = stores_df[
            (stores_df['City'] == city) & 
            (stores_df['State'] == state)
        ]
        
        if len(matching_stores) > 0:
            folium.CircleMarker(
                location=[matching_stores.iloc[0]['Latitude'],
                         matching_stores.iloc[0]['Longitude']],
                radius=10,
                color=region_colors[region],
                fill=True,
                popup=f"Facility: {facility['Location']}<br>" +
                      f"Total Cost: ${facility['RentCost_SqFt'] + facility['UtilitiesOpsCost_SqFt']:.2f}/sqft<br>" +
                      f"Size: {facility['Size_SqFt']:,} sqft",
                fillOpacity=0.9
            ).add_to(m)
            
            # Add service area circle
            folium.Circle(
                location=[matching_stores.iloc[0]['Latitude'],
                         matching_stores.iloc[0]['Longitude']],
                radius=500000,  # 500km radius
                color=region_colors[region],
                fill=True,
                opacity=0.1,
                popup=f"{facility['Location']} Service Area"
            ).add_to(m)
    
    return m

network_map = create_network_map()
display(network_map)

#%% [markdown]
## Cluster Demand Analysis

#%%
# Calculate demand statistics by cluster
cluster_stats = store_demand.groupby('Cluster').agg({
    'Demand_Pallets': ['sum', 'mean'],
    'StoreID': 'count'
}).round(2)

# Fix the column names
cluster_stats.columns = ['Total_Demand', 'Avg_Demand', 'Store_Count']
print("\nCluster Statistics:")
display(cluster_stats)

# Create demand distribution visualization
plt.figure(figsize=(15, 5))

# Pie chart of total demand by cluster
plt.subplot(1, 2, 1)
plt.pie(cluster_stats['Total_Demand'], 
        labels=[f'Cluster {i}' for i in range(3)],
        autopct='%1.1f%%', 
        startangle=90)
plt.title('Distribution of Total Demand by Cluster')

# Bar chart of average demand by cluster
plt.subplot(1, 2, 2)
plt.bar(range(3), cluster_stats['Avg_Demand'])
plt.title('Average Demand by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Demand (Pallets)')
plt.xticks(range(3), [f'Cluster {i}' for i in range(3)])

plt.tight_layout()
plt.show()

#%% [markdown]
## Distance Analysis

#%%
def calculate_distances_to_facilities():
    """Calculate distances from each store to each facility"""
    facility_locations = {}
    
    # Get facility coordinates
    for region, facility in selected_facilities.items():
        city = facility['Location'].split(',')[0].strip()
        state = facility['Location'].split(',')[1].strip()
        matching_stores = stores_df[
            (stores_df['City'] == city) & 
            (stores_df['State'] == state)
        ]
        if len(matching_stores) > 0:
            facility_locations[region] = (
                matching_stores.iloc[0]['Latitude'],
                matching_stores.iloc[0]['Longitude']
            )
    
    # Calculate distances
    distances = pd.DataFrame()
    for region, coords in facility_locations.items():
        distances[f'Distance_to_{region}'] = store_demand.apply(
            lambda row: geodesic(
                (row['Latitude'], row['Longitude']),
                coords
            ).miles,
            axis=1
        )
    
    return distances

distances_df = calculate_distances_to_facilities()
store_demand = pd.concat([store_demand, distances_df], axis=1)

# Calculate average distances by cluster
avg_distances = store_demand.groupby('Cluster')[
    ['Distance_to_Western', 'Distance_to_Central', 'Distance_to_Eastern']
].mean().round(2)

print("\nAverage Distances (miles) by Cluster:")
display(avg_distances)

# Create distance visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=store_demand.melt(
    id_vars=['Cluster'],
    value_vars=['Distance_to_Western', 'Distance_to_Central', 'Distance_to_Eastern'],
    var_name='Facility',
    value_name='Distance'
))
plt.title('Distribution of Distances to Facilities by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Distance (miles)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% [markdown]
## Facility Size and Cost Analysis

#%%
# Create facility comparison visualization
plt.figure(figsize=(12, 5))

# Size comparison
plt.subplot(1, 2, 1)
plt.bar(facility_costs['Location'], facility_costs['Size'])
plt.title('Facility Sizes')
plt.xlabel('Location')
plt.ylabel('Square Feet')
plt.xticks(rotation=45)

# Cost comparison
plt.subplot(1, 2, 2)
plt.bar(facility_costs['Location'], facility_costs['Total_Cost'])
plt.title('Total Cost per Square Foot')
plt.xlabel('Location')
plt.ylabel('Cost ($/sqft)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%% [markdown]
## Key Findings

1. Facility Characteristics:
   - Columbus (Eastern): {selected_facilities['Eastern']['Size_SqFt']:,} sqft, ${selected_facilities['Eastern']['RentCost_SqFt'] + selected_facilities['Eastern']['UtilitiesOpsCost_SqFt']:.2f}/sqft
   - Houston (Central): {selected_facilities['Central']['Size_SqFt']:,} sqft, ${selected_facilities['Central']['RentCost_SqFt'] + selected_facilities['Central']['UtilitiesOpsCost_SqFt']:.2f}/sqft
   - Phoenix (Western): {selected_facilities['Western']['Size_SqFt']:,} sqft, ${selected_facilities['Western']['RentCost_SqFt'] + selected_facilities['Western']['UtilitiesOpsCost_SqFt']:.2f}/sqft

2. Demand Distribution:
   - Western Cluster: {(cluster_stats.loc[0, 'Total_Demand'] / cluster_stats['Total_Demand'].sum() * 100):.1f}% of total demand
   - Central Cluster: {(cluster_stats.loc[1, 'Total_Demand'] / cluster_stats['Total_Demand'].sum() * 100):.1f}% of total demand
   - Eastern Cluster: {(cluster_stats.loc[2, 'Total_Demand'] / cluster_stats['Total_Demand'].sum() * 100):.1f}% of total demand

3. Average Distances:
   - Western Cluster average distance to Phoenix: {avg_distances.loc[0, 'Distance_to_Western']:,.0f} miles
   - Central Cluster average distance to Houston: {avg_distances.loc[1, 'Distance_to_Central']:,.0f} miles
   - Eastern Cluster average distance to Columbus: {avg_distances.loc[2, 'Distance_to_Eastern']:,.0f} miles

4. Network Advantages:
   - Good geographic spread covering major US regions
   - Columbus and Phoenix provide large facility sizes (58k and 52k sqft)
   - Houston offers central location with lowest cost per sqft
   - All facilities have good transportation infrastructure

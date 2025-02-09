#%% [markdown]
# Manufacturing Facility Cost Analysis
## Overview
This notebook analyzes total costs for potential manufacturing facility locations, considering:
1. Labor costs
2. Shipping rates (FTL)
3. Rent and utilities
4. Distance to demand centers

#%% [markdown]
## Import Required Libraries

#%%
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
import matplotlib.pyplot as plt
import seaborn as sns

#%% [markdown]
## Load Data

#%%
# Load shipping rates
rates_df = pd.read_csv('Data/rates.csv')

# Load store locations and demand (from previous analysis)
stores_df = pd.read_csv('Data/stores.csv')
demand_df = pd.read_csv('Data/demand.csv')
products_df = pd.read_csv('Data/products.csv')

# Display sample of rates data
print("Shipping Rates Sample:")
display(rates_df.head())

#%% [markdown]
## Calculate Shipping Costs per Pallet-Mile

#%%
def calculate_pallet_mile_rate(row):
    """Calculate shipping cost per pallet per mile"""
    # Assuming standard 53' trailer can hold 26 pallets
    pallets_per_truck = 26
    
    # Calculate distance between origin and destination
    origin_coords = get_coords(row['Origin'])
    dest_coords = get_coords(row['Destination'])
    distance = geodesic(origin_coords, dest_coords).miles
    
    # Calculate rate per pallet-mile
    pallet_mile_rate = row['Cost_FTL'] / (pallets_per_truck * distance)
    return pallet_mile_rate

def get_coords(city_state):
    """Get coordinates for a city from our stores dataset"""
    city = city_state.split(',')[0].strip()
    state = city_state.split(',')[1].strip()
    
    # Find matching city in stores dataset
    city_data = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ].iloc[0]
    
    return (city_data['Latitude'], city_data['Longitude'])

# Calculate rates per pallet-mile
rates_df['Rate_Per_Pallet_Mile'] = rates_df.apply(calculate_pallet_mile_rate, axis=1)

# Display summary statistics
print("\nShipping Rate Statistics ($ per pallet-mile):")
display(rates_df['Rate_Per_Pallet_Mile'].describe())

# Visualize rate distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=rates_df, x='Rate_Per_Pallet_Mile', bins=20)
plt.title('Distribution of Shipping Rates ($ per pallet-mile)')
plt.xlabel('Rate ($/pallet-mile)')
plt.ylabel('Count')
plt.show()

#%% [markdown]
## Analyze Target Regions
Based on our previous clustering analysis, we'll focus on three regions:
1. Central California (around 37.35°N, 118.62°W)
2. West Virginia (around 38.88°N, 80.70°W)
3. Northern Texas (around 33.68°N, 97.72°W)

#%%
# Define our target regions
target_regions = {
    'Central California': (37.3506, -118.6246),
    'West Virginia': (38.8837, -80.6994),
    'Northern Texas': (33.6814, -97.7189)
}

# Calculate average shipping rates to major demand centers
def analyze_region_shipping(region_name, region_coords):
    """Analyze shipping costs from a region to major demand centers"""
    # Get nearest rate points
    rates_from_region = []
    
    for _, row in rates_df.iterrows():
        origin_coords = get_coords(row['Origin'])
        dest_coords = get_coords(row['Destination'])
        
        # Calculate distances
        dist_to_origin = geodesic(region_coords, origin_coords).miles
        dist_to_dest = geodesic(region_coords, dest_coords).miles
        
        # Use rate if either origin or destination is close to our region
        if dist_to_origin < 300 or dist_to_dest < 300:
            rates_from_region.append(row['Rate_Per_Pallet_Mile'])
    
    return np.mean(rates_from_region) if rates_from_region else None

# Analyze each region
print("\nRegional Shipping Rate Analysis:")
for region, coords in target_regions.items():
    avg_rate = analyze_region_shipping(region, coords)
    print(f"\n{region}:")
    print(f"Average shipping rate: ${avg_rate:.4f} per pallet-mile")

#%% [markdown]
## Visualize Regional Analysis

#%%
# Create a map showing regions and shipping rates
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add target regions
for region, coords in target_regions.items():
    # Add marker for region
    folium.Marker(
        location=[coords[0], coords[1]],
        popup=f'{region}',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add circle showing approximate service area
    folium.Circle(
        location=[coords[0], coords[1]],
        radius=300000,  # 300km radius
        color='red',
        fill=True,
        opacity=0.2
    ).add_to(m)

# Add shipping routes from rates_df
for _, row in rates_df.iterrows():
    origin_coords = get_coords(row['Origin'])
    dest_coords = get_coords(row['Destination'])
    
    # Create a line for the route
    folium.PolyLine(
        locations=[origin_coords, dest_coords],
        weight=2,
        color='blue',
        opacity=0.5,
        popup=f"${row['Rate_Per_Pallet_Mile']:.4f}/pallet-mile"
    ).add_to(m)

display(m)

#%% [markdown]
## Next Steps
To complete the total cost analysis, we need:
1. Labor cost data by region
2. Rent and utilities costs by region
3. More detailed shipping rate data around our target regions

Once we have this additional data, we can:
1. Calculate total operating costs for each region
2. Compare regions based on:
   - Fixed costs (rent, utilities)
   - Variable costs (labor, shipping)
   - Total annual costs
3. Create sensitivity analyses for different demand scenarios
4. Make recommendations for optimal facility locations

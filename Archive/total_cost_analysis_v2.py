#%% [markdown]
# Total Cost Analysis for Manufacturing Facility Locations
## Overview
This notebook performs a comprehensive cost analysis for potential manufacturing facility locations, considering:
1. Labor costs
2. Shipping rates (FTL)
3. Rent and utilities costs
4. Distance to demand centers

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
from scipy.stats import describe
import branca.colormap as cm

#%% [markdown]
## Load and Prepare Data

#%%
# Load all relevant datasets
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
## Create Cost Heatmaps

#%%
def create_cost_heatmap(title, locations, values, colormap='YlOrRd'):
    """Create a folium map with cost heatmap"""
    # Create base map
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Create color scale
    min_val = min(values)
    max_val = max(values)
    colormap = cm.LinearColormap(
        colors=colormap,
        vmin=min_val,
        vmax=max_val
    )
    
    # Add points with colors based on values
    for loc, val in zip(locations, values):
        folium.CircleMarker(
            location=loc,
            radius=10,
            popup=f'Cost: ${val:.2f}',
            color=colormap(val),
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add color scale
    colormap.add_to(m)
    colormap.caption = title
    
    return m

# Create labor cost heatmap
labor_locations = []
labor_costs = []
for _, row in labor_df.iterrows():
    city = row['Location'].split(',')[0].strip()
    state = row['Location'].split(',')[1].strip()
    matching_stores = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ]
    if len(matching_stores) > 0:
        labor_locations.append([matching_stores.iloc[0]['Latitude'], 
                              matching_stores.iloc[0]['Longitude']])
        labor_costs.append(row['Local Labor Cost (USD/hour)'])

labor_map = create_cost_heatmap('Labor Cost ($/hour)', labor_locations, labor_costs)
display(labor_map)

# Create facility cost heatmap
facility_locations = []
facility_costs = []
for _, row in facilities_df.iterrows():
    city = row['Location'].split(',')[0].strip()
    state = row['Location'].split(',')[1].strip()
    matching_stores = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ]
    if len(matching_stores) > 0:
        facility_locations.append([matching_stores.iloc[0]['Latitude'], 
                                 matching_stores.iloc[0]['Longitude']])
        facility_costs.append(row['RentCost_SqFt'] + row['UtilitiesOpsCost_SqFt'])

facility_map = create_cost_heatmap('Total Facility Cost ($/sqft)', 
                                 facility_locations, facility_costs, 'YlOrRd')
display(facility_map)

#%% [markdown]
## Analyze Shipping Costs

#%%
def calculate_shipping_costs():
    """Calculate and analyze shipping costs"""
    shipping_costs = []
    locations = []
    
    # Process each unique origin in rates_df
    for origin in rates_df['Origin'].unique():
        origin_rates = rates_df[rates_df['Origin'] == origin]
        
        # Get coordinates for origin
        city = origin.split(',')[0].strip()
        state = origin.split(',')[1].strip()
        matching_stores = stores_df[
            (stores_df['City'] == city) & 
            (stores_df['State'] == state)
        ]
        
        if len(matching_stores) > 0:
            # Calculate average shipping cost per mile for this origin
            avg_cost = origin_rates['Cost_FTL'].mean() / 26  # Assume 26 pallets per truck
            locations.append([matching_stores.iloc[0]['Latitude'],
                            matching_stores.iloc[0]['Longitude']])
            shipping_costs.append(avg_cost)
    
    return locations, shipping_costs

shipping_locations, shipping_costs = calculate_shipping_costs()
shipping_map = create_cost_heatmap('Average Shipping Cost ($/pallet)', 
                                 shipping_locations, shipping_costs, 'YlOrRd')
display(shipping_map)

#%% [markdown]
## Target Region Analysis
Based on our previous clustering analysis, we'll analyze costs around:
1. Central California (around 37.35°N, 118.62°W)
2. West Virginia (around 38.88°N, 80.70°W)
3. Northern Texas (around 33.68°N, 97.72°W)

#%%
target_regions = {
    'Central California': (37.3506, -118.6246),
    'West Virginia': (38.8837, -80.6994),
    'Northern Texas': (33.6814, -97.7189)
}

# Create map showing target regions with nearby facilities
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add target regions
for region, coords in target_regions.items():
    # Add marker for region
    folium.Marker(
        location=[coords[0], coords[1]],
        popup=f'{region}',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add circle showing service area
    folium.Circle(
        location=[coords[0], coords[1]],
        radius=300000,  # 300km radius
        color='red',
        fill=True,
        opacity=0.2,
        popup=f'{region} Service Area'
    ).add_to(m)

# Add existing facilities
for _, facility in facilities_df.iterrows():
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
            radius=8,
            color='blue',
            fill=True,
            popup=f"Facility: {facility['Location']}<br>" +
                  f"Rent: ${facility['RentCost_SqFt']}/sqft<br>" +
                  f"Utilities: ${facility['UtilitiesOpsCost_SqFt']}/sqft"
        ).add_to(m)

display(m)

#%% [markdown]
## Cost Analysis for Target Regions

#%%
def analyze_region_costs(region_name, region_coords):
    """Analyze all costs for a specific region"""
    radius_miles = 300  # Search radius in miles
    
    # 1. Analyze labor costs
    labor_costs = []
    for _, row in labor_df.iterrows():
        city = row['Location'].split(',')[0].strip()
        state = row['Location'].split(',')[1].strip()
        matching_stores = stores_df[
            (stores_df['City'] == city) & 
            (stores_df['State'] == state)
        ]
        if len(matching_stores) > 0:
            coords = (matching_stores.iloc[0]['Latitude'],
                     matching_stores.iloc[0]['Longitude'])
            dist = geodesic(region_coords, coords).miles
            if dist < radius_miles:
                labor_costs.append(row['Local Labor Cost (USD/hour)'])
    
    # 2. Analyze facility costs
    facility_costs = []
    for _, row in facilities_df.iterrows():
        city = row['Location'].split(',')[0].strip()
        state = row['Location'].split(',')[1].strip()
        matching_stores = stores_df[
            (stores_df['City'] == city) & 
            (stores_df['State'] == state)
        ]
        if len(matching_stores) > 0:
            coords = (matching_stores.iloc[0]['Latitude'],
                     matching_stores.iloc[0]['Longitude'])
            dist = geodesic(region_coords, coords).miles
            if dist < radius_miles:
                facility_costs.append(row['RentCost_SqFt'] + 
                                   row['UtilitiesOpsCost_SqFt'])
    
    # 3. Analyze shipping costs
    shipping_costs = []
    for origin in rates_df['Origin'].unique():
        city = origin.split(',')[0].strip()
        state = origin.split(',')[1].strip()
        matching_stores = stores_df[
            (stores_df['City'] == city) & 
            (stores_df['State'] == state)
        ]
        if len(matching_stores) > 0:
            coords = (matching_stores.iloc[0]['Latitude'],
                     matching_stores.iloc[0]['Longitude'])
            dist = geodesic(region_coords, coords).miles
            if dist < radius_miles:
                origin_rates = rates_df[rates_df['Origin'] == origin]
                avg_cost = origin_rates['Cost_FTL'].mean() / 26
                shipping_costs.append(avg_cost)
    
    return {
        'labor_cost': np.mean(labor_costs) if labor_costs else None,
        'facility_cost': np.mean(facility_costs) if facility_costs else None,
        'shipping_cost': np.mean(shipping_costs) if shipping_costs else None
    }

# Analyze costs for each region
results = {}
for region, coords in target_regions.items():
    results[region] = analyze_region_costs(region, coords)

# Create comparison visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ['labor_cost', 'facility_cost', 'shipping_cost']
titles = ['Labor Cost ($/hour)', 'Facility Cost ($/sqft)', 'Shipping Cost ($/pallet)']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    values = [results[region][metric] for region in results.keys()]
    axes[i].bar(list(results.keys()), values)
    axes[i].set_title(title)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#%% [markdown]
## Total Cost Analysis

#%%
# Define assumptions
working_hours_per_year = 2080  # 40 hours/week * 52 weeks
required_sqft = 50000  # Example facility size
workers_needed = 50  # Example number of workers
annual_total_pallets = store_demand['Demand_Pallets'].sum()

# Calculate annual costs for each region
annual_costs = {}
for region in results:
    if all(v is not None for v in results[region].values()):
        labor_cost = results[region]['labor_cost'] * workers_needed * working_hours_per_year
        facility_cost = results[region]['facility_cost'] * required_sqft
        shipping_cost = results[region]['shipping_cost'] * annual_total_pallets
        
        annual_costs[region] = {
            'Labor Cost': labor_cost,
            'Facility Cost': facility_cost,
            'Shipping Cost': shipping_cost,
            'Total Cost': labor_cost + facility_cost + shipping_cost
        }

# Create cost comparison visualization
cost_df = pd.DataFrame(annual_costs).T
cost_df_melted = cost_df.reset_index().melt(
    id_vars=['index'], 
    var_name='Cost Type', 
    value_name='Amount'
)

plt.figure(figsize=(12, 6))
sns.barplot(data=cost_df_melted, x='index', y='Amount', hue='Cost Type')
plt.title('Annual Cost Comparison by Region')
plt.xlabel('Region')
plt.ylabel('Annual Cost (USD)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print detailed cost breakdown
print("\nDetailed Annual Cost Breakdown (in millions USD):")
for region, costs in annual_costs.items():
    print(f"\n{region}:")
    for cost_type, amount in costs.items():
        print(f"{cost_type}: ${amount/1_000_000:.2f}M")

#%% [markdown]
## Conclusions and Recommendations

Based on our analysis:

1. Labor Costs:
   - Highest in [region with highest labor cost]
   - Lowest in [region with lowest labor cost]

2. Facility Costs:
   - Highest in [region with highest facility cost]
   - Lowest in [region with lowest facility cost]

3. Shipping Costs:
   - Most efficient from [region with lowest shipping cost]
   - Highest from [region with highest shipping cost]

4. Total Cost Comparison:
   - Most cost-effective region: [region with lowest total cost]
   - Most expensive region: [region with highest total cost]

Recommendations:
1. Primary consideration: [main recommendation]
2. Secondary options: [alternative options]
3. Trade-offs to consider: [list key trade-offs]

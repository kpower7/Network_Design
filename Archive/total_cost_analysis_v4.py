import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap

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

def create_cost_map(title, data_points, costs, filename):
    """Create a folium map with cost visualization and save to file"""
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Add markers with cost information
    for loc, cost in zip(data_points, costs):
        folium.CircleMarker(
            location=loc,
            radius=8,
            color='red',
            fill=True,
            popup=f'Cost: ${cost:.2f}',
            fillOpacity=0.7
        ).add_to(m)
    
    # Add heatmap layer
    heat_data = [[loc[0], loc[1], cost] for loc, cost in zip(data_points, costs)]
    HeatMap(heat_data).add_to(m)
    
    # Save map to HTML file
    m.save(filename)
    return m

# Create labor cost map
print("Creating labor cost map...")
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

labor_map = create_cost_map('Labor Costs', labor_locations, labor_costs, 'labor_cost_map.html')
print("Labor cost map saved as 'labor_cost_map.html'")

# Create facility cost map
print("\nCreating facility cost map...")
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

facility_map = create_cost_map('Facility Costs', facility_locations, facility_costs, 'facility_cost_map.html')
print("Facility cost map saved as 'facility_cost_map.html'")

# Target regions analysis
target_regions = {
    'Central California': (37.3506, -118.6246),
    'West Virginia': (38.8837, -80.6994),
    'Northern Texas': (33.6814, -97.7189)
}

# Create map with target regions and facilities
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

# Add existing facilities with cost information
for _, facility in facilities_df.iterrows():
    city = facility['Location'].split(',')[0].strip()
    state = facility['Location'].split(',')[1].strip()
    matching_stores = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ]
    if len(matching_stores) > 0:
        total_cost = facility['RentCost_SqFt'] + facility['UtilitiesOpsCost_SqFt']
        folium.CircleMarker(
            location=[matching_stores.iloc[0]['Latitude'],
                     matching_stores.iloc[0]['Longitude']],
            radius=8,
            color='blue',
            fill=True,
            popup=f"Facility: {facility['Location']}<br>" +
                  f"Total Cost: ${total_cost:.2f}/sqft"
        ).add_to(m)

m.save('target_regions_map.html')
print("\nTarget regions map saved as 'target_regions_map.html'")

# Cost analysis for target regions
def analyze_region_costs(region_name, region_coords):
    """Analyze all costs for a specific region"""
    radius_miles = 300
    
    # Analyze labor costs
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
    
    # Analyze facility costs
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
    
    return {
        'labor_cost': np.mean(labor_costs) if labor_costs else 0,
        'facility_cost': np.mean(facility_costs) if facility_costs else 0
    }

# Analyze each region
results = {}
for region, coords in target_regions.items():
    results[region] = analyze_region_costs(region, coords)

# Create comparison visualizations
plt.figure(figsize=(12, 6))
regions = list(results.keys())
labor_costs = [results[r]['labor_cost'] for r in regions]
facility_costs = [results[r]['facility_cost'] for r in regions]

x = np.arange(len(regions))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, labor_costs, width, label='Labor Cost ($/hour)')
rects2 = ax.bar(x + width/2, facility_costs, width, label='Facility Cost ($/sqft)')

ax.set_ylabel('Cost')
ax.set_title('Cost Comparison by Region')
ax.set_xticks(x)
ax.set_xticklabels(regions, rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig('cost_comparison.png')
print("\nCost comparison chart saved as 'cost_comparison.png'")

# Print detailed analysis
print("\nDetailed Cost Analysis by Region:")
for region in results:
    print(f"\n{region}:")
    print(f"Average Labor Cost: ${results[region]['labor_cost']:.2f}/hour")
    print(f"Average Facility Cost: ${results[region]['facility_cost']:.2f}/sqft")

# Calculate total annual costs (example with assumptions)
print("\nAnnual Cost Projections (with assumptions):")
workers = 50  # number of workers
hours_per_year = 2080  # 40 hours/week * 52 weeks
facility_size = 50000  # square feet

for region in results:
    annual_labor = results[region]['labor_cost'] * workers * hours_per_year
    annual_facility = results[region]['facility_cost'] * facility_size
    total = annual_labor + annual_facility
    
    print(f"\n{region}:")
    print(f"Annual Labor Cost: ${annual_labor:,.2f}")
    print(f"Annual Facility Cost: ${annual_facility:,.2f}")
    print(f"Total Annual Cost: ${total:,.2f}")

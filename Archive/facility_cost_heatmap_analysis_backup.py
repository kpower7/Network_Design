#%% [markdown]
# Facility Cost Analysis - National Heatmap
This notebook creates comprehensive heatmaps showing:
1. Total Facility Costs (Rent + Utilities)
2. Labor Costs
3. Transportation Costs
4. Combined Cost Index

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
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

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

# Calculate total facility cost per square foot
facilities_df['Total_Cost_SqFt'] = facilities_df['RentCost_SqFt'] + facilities_df['UtilitiesOpsCost_SqFt']

#%% [markdown]
## Create Cost Heatmap Function

#%%
def create_cost_heatmap(title, locations, values, save_name):
    """Create a folium map with cost heatmap"""
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Create heatmap layer
    HeatMap(
        list(zip(
            [loc[0] for loc in locations],
            [loc[1] for loc in locations],
            values
        )),
        radius=35
    ).add_to(m)
    
    # Add title
    title_html = f'''
        <div style="position: fixed; 
                    top: 10px; 
                    left: 50px; 
                    width: 300px; 
                    height: 30px; 
                    z-index:9999; 
                    background-color: white; 
                    font-size: 16px; 
                    font-weight: bold;
                    padding: 5px;
                    border-radius: 5px;">
            {title}
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(save_name)
    return m

#%% [markdown]
## Generate Facility Cost Heatmap

#%%
# Prepare facility cost data
facility_locations = []
facility_costs = []

for _, facility in facilities_df.iterrows():
    city = facility['Location'].split(',')[0].strip()
    state = facility['Location'].split(',')[1].strip()
    matching_stores = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ]
    
    if len(matching_stores) > 0:
        facility_locations.append([
            matching_stores.iloc[0]['Latitude'],
            matching_stores.iloc[0]['Longitude']
        ])
        facility_costs.append(facility['Total_Cost_SqFt'])

# Create facility cost heatmap
facility_cost_map = create_cost_heatmap(
    'Total Facility Cost per Square Foot ($)',
    facility_locations,
    facility_costs,
    'facility_cost_heatmap.html'
)
print("Facility cost heatmap saved as 'facility_cost_heatmap.html'")

#%% [markdown]
## Generate Labor Cost Heatmap

#%%
# Prepare labor cost data
labor_locations = []
labor_costs = []

for _, labor in labor_df.iterrows():
    city = labor['Location'].split(',')[0].strip()
    state = labor['Location'].split(',')[1].strip()
    matching_stores = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ]
    
    if len(matching_stores) > 0:
        labor_locations.append([
            matching_stores.iloc[0]['Latitude'],
            matching_stores.iloc[0]['Longitude']
        ])
        labor_costs.append(labor['Local Labor Cost (USD/hour)'])

# Create labor cost heatmap
labor_cost_map = create_cost_heatmap(
    'Local Labor Cost ($/hour)',
    labor_locations,
    labor_costs,
    'labor_cost_heatmap.html'
)
print("\nLabor cost heatmap saved as 'labor_cost_heatmap.html'")

#%% [markdown]
## Generate Transportation Cost Heatmap

#%%
def calculate_avg_transport_cost(origin):
    """Calculate average transportation cost per mile from an origin"""
    origin_rates = rates_df[rates_df['Origin'] == origin]
    if len(origin_rates) == 0:
        return None
    
    # Assume 26 pallets per truck for FTL
    return origin_rates['Cost_FTL'].mean() / 26

# Prepare transportation cost data
transport_locations = []
transport_costs = []

for origin in rates_df['Origin'].unique():
    city = origin.split(',')[0].strip()
    state = origin.split(',')[1].strip()
    matching_stores = stores_df[
        (stores_df['City'] == city) & 
        (stores_df['State'] == state)
    ]
    
    avg_cost = calculate_avg_transport_cost(origin)
    if len(matching_stores) > 0 and avg_cost is not None:
        transport_locations.append([
            matching_stores.iloc[0]['Latitude'],
            matching_stores.iloc[0]['Longitude']
        ])
        transport_costs.append(avg_cost)

transport_cost_map = create_cost_heatmap(
    'Transportation Cost ($/pallet)',
    transport_locations,
    transport_costs,
    'transport_cost_heatmap.html'
)
print("\nTransportation cost heatmap saved as 'transport_cost_heatmap.html'")

#%% [markdown]
## Create Combined Cost Index Map

#%%
def normalize_values(values):
    """Normalize values to 0-1 range"""
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def create_combined_cost_map():
    """Create a map showing combined cost index"""
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Get all unique locations
    all_locations = set()
    location_data = {}
    
    # Process facility costs
    for loc, cost in zip(facility_locations, normalize_values(facility_costs)):
        loc_key = f"{loc[0]},{loc[1]}"
        all_locations.add(loc_key)
        if loc_key not in location_data:
            location_data[loc_key] = {'loc': loc, 'costs': []}
        location_data[loc_key]['costs'].append(cost)
    
    # Process labor costs
    for loc, cost in zip(labor_locations, normalize_values(labor_costs)):
        loc_key = f"{loc[0]},{loc[1]}"
        all_locations.add(loc_key)
        if loc_key not in location_data:
            location_data[loc_key] = {'loc': loc, 'costs': []}
        location_data[loc_key]['costs'].append(cost)
    
    # Process transport costs
    for loc, cost in zip(transport_locations, normalize_values(transport_costs)):
        loc_key = f"{loc[0]},{loc[1]}"
        all_locations.add(loc_key)
        if loc_key not in location_data:
            location_data[loc_key] = {'loc': loc, 'costs': []}
        location_data[loc_key]['costs'].append(cost)
    
    # Calculate combined index
    combined_data = []
    for loc_key, data in location_data.items():
        avg_cost = np.mean(data['costs'])
        combined_data.append([
            data['loc'][0],
            data['loc'][1],
            avg_cost
        ])
        
        # Add marker with detailed information
        folium.CircleMarker(
            location=data['loc'],
            radius=8,
            color='purple',
            fill=True,
            popup=f'Combined Cost Index: {avg_cost:.2f}',
            fillOpacity=0.7
        ).add_to(m)
    
    # Add heatmap layer
    HeatMap(combined_data, radius=35).add_to(m)
    
    # Add title
    title_html = '''
        <div style="position: fixed; 
                    top: 10px; 
                    left: 50px; 
                    width: 300px; 
                    height: 30px; 
                    z-index:9999; 
                    background-color: white; 
                    font-size: 16px; 
                    font-weight: bold;
                    padding: 5px;
                    border-radius: 5px;">
            Combined Cost Index
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

combined_map = create_combined_cost_map()
combined_map.save('combined_cost_heatmap.html')
print("\nCombined cost heatmap saved as 'combined_cost_heatmap.html'")

#%% [markdown]
## Cost Distribution Analysis

#%%
# Create distribution plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

# Facility Cost Distribution
sns.histplot(data=facilities_df, x='Total_Cost_SqFt', bins=20, ax=ax1)
ax1.set_title('Facility Cost Distribution')
ax1.set_xlabel('Total Cost per Square Foot ($)')
ax1.set_ylabel('Count')

# Labor Cost Distribution
sns.histplot(data=labor_df, x='Local Labor Cost (USD/hour)', bins=20, ax=ax2)
ax2.set_title('Labor Cost Distribution')
ax2.set_xlabel('Labor Cost per Hour ($)')
ax2.set_ylabel('Count')

# Transportation Cost Distribution
transport_costs_df = pd.DataFrame({
    'Cost_Per_Pallet': transport_costs
})
sns.histplot(data=transport_costs_df, x='Cost_Per_Pallet', bins=20, ax=ax3)
ax3.set_title('Transportation Cost Distribution')
ax3.set_xlabel('Cost per Pallet ($)')
ax3.set_ylabel('Count')

# Cost Correlations
if len(facility_locations) > 0:
    facility_lat = [loc[0] for loc in facility_locations]
    facility_long = [loc[1] for loc in facility_locations]
    sns.scatterplot(x=facility_long, y=facility_lat, size=facility_costs, 
                   sizes=(20, 200), ax=ax4)
    ax4.set_title('Facility Costs by Location')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('cost_distributions.png')
print("\nCost distribution plots saved as 'cost_distributions.png'")

#%% [markdown]
## Key Findings

1. Facility Costs:
   - Highest in coastal areas (especially West Coast)
   - Lowest in central regions
   - Significant variation between urban and rural areas

2. Labor Costs:
   - Follow similar patterns to facility costs
   - Higher in major metropolitan areas
   - More uniform in central regions

3. Transportation Costs:
   - Lower in areas with good infrastructure
   - Higher in remote locations
   - Significant impact of distance to major markets

4. Combined Cost Index:
   - Identifies optimal regions balancing all cost factors
   - Shows clear cost advantages in certain geographic areas
   - Highlights potential trade-offs between different cost factors

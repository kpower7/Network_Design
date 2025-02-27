#%% [markdown]
# Facility Cost Analysis - National Heatmap
This notebook creates comprehensive heatmaps showing:
1. Total Facility Costs (Rent + Utilities)
2. Labor Costs
3. Combined Cost Index

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
display(facility_cost_map)

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
display(labor_cost_map)

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
    
    # Create state to labor cost mapping
    state_labor_costs = {}
    for _, row in labor_df.iterrows():
        state = row['Location'].split(',')[1].strip()
        if state not in state_labor_costs:
            state_labor_costs[state] = []
        state_labor_costs[state].append(row['Local Labor Cost (USD/hour)'])
    
    # Average labor costs for states with multiple cities
    state_labor_avg = {state: np.mean(costs) for state, costs in state_labor_costs.items()}
    
    # Process all facilities
    combined_data = []
    for _, facility in facilities_df.iterrows():
        state = facility['Location'].split(',')[1].strip()
        if state in state_labor_avg:
            facility_cost = facility['Total_Cost_SqFt']
            labor_cost = state_labor_avg[state]
            
            # Normalize both costs
            norm_facility = (facility_cost - min(facilities_df['Total_Cost_SqFt'])) / (max(facilities_df['Total_Cost_SqFt']) - min(facilities_df['Total_Cost_SqFt']))
            norm_labor = (labor_cost - min(labor_df['Local Labor Cost (USD/hour)'])) / (max(labor_df['Local Labor Cost (USD/hour)']) - min(labor_df['Local Labor Cost (USD/hour)']))
            
            # Calculate combined index
            combined_index = np.mean([norm_facility, norm_labor])
            
            # Get location coordinates
            matching_stores = stores_df[
                (stores_df['City'] == facility['Location'].split(',')[0].strip()) & 
                (stores_df['State'] == state)
            ]
            if len(matching_stores) > 0:
                lat = matching_stores.iloc[0]['Latitude']
                lon = matching_stores.iloc[0]['Longitude']
                
                combined_data.append({
                    'Location': facility['Location'],
                    'Combined_Index': combined_index,
                    'Facility_Cost': facility_cost,
                    'Labor_Cost': labor_cost,
                    'Latitude': lat,
                    'Longitude': lon
                })
                
                # Add marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color='purple',
                    fill=True,
                    popup=f"Location: {facility['Location']}<br>" +
                          f"Combined Index: {combined_index:.3f}<br>" +
                          f"Facility Cost: ${facility_cost:.2f}/sqft<br>" +
                          f"Labor Cost: ${labor_cost:.2f}/hr",
                    fillOpacity=0.7
                ).add_to(m)
    
    # Create and sort DataFrame
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values('Combined_Index')
    
    # Add heatmap layer
    heat_data = [[row['Latitude'], row['Longitude'], row['Combined_Index']] 
                 for _, row in combined_df.iterrows()]
    HeatMap(heat_data, radius=35).add_to(m)
    
    return m, combined_df

#%%
# Create and display combined cost map and table
combined_map, combined_df = create_combined_cost_map()
print("\nCombined Cost Index Analysis (Sorted from Lowest to Highest):")
display(combined_df)
display(combined_map)

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

# Combined Index Distribution
sns.histplot(data=combined_df, x='Combined_Index', bins=20, ax=ax3)
ax3.set_title('Combined Cost Index Distribution')
ax3.set_xlabel('Combined Index')
ax3.set_ylabel('Count')

# Facility vs Labor Cost Scatter
sns.scatterplot(data=combined_df, x='Facility_Cost', y='Labor_Cost', ax=ax4)
ax4.set_title('Facility Cost vs Labor Cost')
ax4.set_xlabel('Facility Cost ($/sqft)')
ax4.set_ylabel('Labor Cost ($/hr)')

plt.tight_layout()
plt.show()

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

3. Combined Cost Index:
   - Identifies optimal regions balancing both facility and labor costs
   - Shows clear cost advantages in certain geographic areas
   - Top 5 lowest-cost locations: {', '.join(combined_df.head(5)['Location'].tolist())}
